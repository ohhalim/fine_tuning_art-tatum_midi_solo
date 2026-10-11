#!/usr/bin/env python3
"""Why did the Mehldau Aria generations break? (issue #1728, docs/experiments/MEHLDAU_ARIA_ERROR_DIAG.md)

No training, no new sampling. Positions come from the saved #1726 generations before the model runs:
  E  the first error of each invalid stream (10: base 3, adapter 7). Every error token is a non-piano
     instrument token; 9 of them came where the grammar expected an onset or a dur (token_order) and
     1 between notes (non_piano_instrument)
  C  in the same stream, the last position before E with the same grammar state (a valid sample)
  S  every position before the first error, in all 16 streams
Allowed tokens depend on the grammar state, as in aria_token_validity: between notes a piano note
start, <T>, <D> or <E>; after a note start an onset not before the last onset of the segment; after
an onset a dur.
The same prefix is teacher-forced through base (LoRA off) and adapter (saved lora_u128.pt), and the
next-token distribution is read from raw logits at temperature 1, exactly as the generator did.

torch.multinomial(p, 1, generator=g) on CPU is argmax(p / Exp(1) noise) with one noise vector over the
vocab per call (checked 200/200 on torch 2.14.1 before writing this). Generation called it once per step,
so the noise of seeds 1 and 2 (already used, no new seeds) is replayed to check that the sampled token is
reproduced, and to ask which token the other model would have taken with the same noise.

Modes (one per process):
  --plan  input SHAs against the #1726 train report, validity recomputed and compared with the saved one,
          positions fixed -> plan.json. Loads the tokenizer only.
  --run   requires plan.json made from the same inputs; model + LoRA -> diag_report.json.

  mehldau_aria_error_diag.py <aria repo> <checkpoint> <data json> <manifest json> <run dir> <out dir> --plan|--run
"""
from __future__ import annotations

import json
import os
import signal
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402
from aria_lora import AriaLoRA, input_mismatches  # noqa: E402
from aria_token_validity import check as token_check, segment_state  # noqa: E402

RANK, ALPHA, SEED, TEMP = 16, 16, 0, 1.0          # same as the #1726 runner
WALL = {"--plan": 300, "--run": 900}
# reading rules, fixed before --run (docs/experiments/MEHLDAU_ARIA_ERROR_DIAG.md)
MAJORITY_E, MAJORITY_S, RATIO = 8, 13, 2.0        # of 10 error positions, of 16 streams, median ratio
TAIL_P = 0.01


def kind(t):
    return t[0] if isinstance(t, tuple) else t


def is_non_piano(t):
    return isinstance(t, tuple) and kind(t) not in ("piano", "onset", "dur", "prefix")


def is_allowed(t, state, last):
    if state == "note":
        return kind(t) == "piano" or t in ("<T>", "<D>", "<E>")
    if state == "onset":
        return kind(t) == "onset" and t[1] >= last
    return kind(t) == "dur"


def grammar_states(prompt_toks, new_toks, stop):
    """[(state, last onset in segment)] before each new token i < stop, following aria_token_validity."""
    last, expect = segment_state(prompt_toks)
    out = []
    for t in new_toks[:stop]:
        out.append((expect, last))
        k = kind(t)
        if expect == "note":
            if k == "<E>":
                break
            if k == "<T>":
                last = 0
            expect = "onset" if k == "piano" else "note"
        elif expect == "onset":
            last, expect = t[1], "dur"
        else:
            expect = "note"
    return out


def main() -> None:
    aria_repo, ckpt, data_path, manifest_path, run_dir, out_dir, mode = sys.argv[1:8]
    assert mode in WALL
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"mode": mode, "checks": {}, "phases": {}, "result": None, "stopped_at": None}

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    def on_alarm(*_):
        raise Stop(f"wall limit {WALL[mode]} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL[mode])
    report_path = os.path.join(out_dir, "plan.json" if mode == "--plan" else "diag_report.json")
    try:
        from ariautils.tokenizer import AbsTokenizer

        t0 = time.perf_counter()
        train = json.load(open(os.path.join(run_dir, "train_report.json")))
        gen_path, lora_path = os.path.join(run_dir, "generations.json"), os.path.join(run_dir, "lora_u128.pt")
        rec = {**{k: train["provenance"][k] for k in ("checkpoint_sha256", "data_sha256", "manifest_sha256")},
               "lora_file_sha256": train["lora_file_sha256"]}
        cur = {"checkpoint_sha256": file_sha(ckpt), "data_sha256": file_sha(data_path),
               "manifest_sha256": file_sha(manifest_path), "lora_file_sha256": file_sha(lora_path)}
        check("inputs_match_1726_train_report", not input_mismatches(rec, cur), mismatches=input_mismatches(rec, cur))
        log["inputs"] = {**cur, "generations_sha256": file_sha(gen_path)}
        gens = json.load(open(gen_path))
        saved = train["generation"]
        check("generations_match_report", len(gens) == len(saved) == 16 and all(
            {k: v for k, v in g.items() if k != "ids"} == s for g, s in zip(gens, saved)))
        prompts = {p["path"]: p["ids"] for p in json.load(open(data_path))["prompts"]}
        tok = AbsTokenizer()

        if mode == "--plan":
            streams, mism = [], []
            for n, g in enumerate(gens):
                pt, nt = tok.decode(list(prompts[g["prompt"]])), tok.decode(g["ids"])
                v = token_check(pt, nt)
                if v != g["validity"]:
                    mism.append(n)
                first = v["first_error"]
                states = grammar_states(pt, nt, (first["index"] + 1) if first else len(nt))
                row = {"stream": n, "prompt": g["prompt"], "seed": g["seed"], "adapter": g["adapter"],
                       "prompt_tokens": len(prompts[g["prompt"]]), "new_tokens": len(nt), "valid": v["valid"],
                       "S": list(range(len(states) - (1 if first else 0))), "states": states}
                if first:
                    e = first["index"]
                    c = max(i for i in range(e) if states[i][0] == states[e][0])
                    row.update({"E": e, "E_type": first["type"], "E_state": states[e][0], "E_token": str(nt[e]),
                                "C": c, "C_token": str(nt[c])})
                streams.append(row)
            check("validity_recomputed_equals_saved", not mism, mismatched_streams=mism)
            errs = [s for s in streams if not s["valid"]]
            types = [s["E_type"] for s in errs]
            check("ten_errors_all_non_piano_tokens", len(errs) == 10 and all(
                is_non_piano(tok.decode([gens[s["stream"]]["ids"][s["E"]]])[0]) for s in errs),
                  base=sum(not s["adapter"] for s in errs), adapter=sum(s["adapter"] for s in errs),
                  token_order=types.count("token_order"), non_piano_instrument=types.count("non_piano_instrument"),
                  E_states=[s["E_state"] for s in errs])
            check("allowed_rule_agrees_with_validity", all(
                is_allowed(t, *s["states"][i]) == (s["valid"] or i != s.get("E"))
                for s in streams for i, t in enumerate(tok.decode(gens[s["stream"]]["ids"])[:len(s["states"])])))
            log["streams"] = streams
            log["counts"] = {"E": len(errs), "C": len(errs), "S": sum(len(s["S"]) for s in streams)}
            log["result"] = "plan_done"
        else:
            import torch
            from safetensors.torch import load_file
            from aria.config import load_model_config
            from aria.model import ModelConfig, TransformerLM

            plan = json.load(open(os.path.join(out_dir, "plan.json")))
            check("plan_made_from_same_inputs", plan.get("result") == "plan_done" and plan["inputs"] == log["inputs"])
            check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
            dev_ = torch.device("mps")
            cfg_dict = load_model_config(name="medium")
            cfg = ModelConfig(**cfg_dict)
            cfg.set_vocab_size(tok.vocab_size)
            cfg.grad_checkpoint = True
            model = TransformerLM(cfg)
            model.load_state_dict(load_file(ckpt), strict=True)
            for p in model.parameters():
                p.requires_grad_(False)
            model.to(dev_)
            model.model.freqs_cis = None
            model.eval()
            lora = AriaLoRA(model, rank=RANK, alpha=ALPHA, seed=SEED, device=dev_)
            lora.load_state_dict(torch.load(lora_path))
            V = tok.vocab_size
            vocab = [tok.decode([i])[0] for i in range(V)]
            non_piano = torch.tensor([is_non_piano(t) for t in vocab])
            note_ok = torch.tensor([is_allowed(t, "note", 0) for t in vocab])
            dur_ok = torch.tensor([is_allowed(t, "dur", 0) for t in vocab])
            onset_val = torch.tensor([t[1] if kind(t) == "onset" else -1 for t in vocab])

            def allowed(state, last):
                return note_ok if state == "note" else dur_ok if state == "dur" else onset_val >= last

            log["vocab"] = {"size": V, "non_piano": int(non_piano.sum()), "allowed_between_notes": int(note_ok.sum()),
                            "dur": int(dur_ok.sum()), "onset": int((onset_val >= 0).sum())}
            log["phases"]["load"] = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure()}

            def next_dist(seq, on):
                """Exactly the generator's distribution: last-position logits / TEMP, softmax on MPS, then CPU."""
                lora.on = on
                with torch.no_grad():
                    lg = model(torch.tensor([seq], device=dev_))[0, -1].float() / TEMP
                    p = torch.softmax(lg, -1).cpu()
                lora.on = True
                return p

            def all_dists(seq, on):
                lora.on = on
                with torch.no_grad():
                    p = torch.softmax(model(torch.tensor([seq], device=dev_))[0].float() / TEMP, -1).cpu()
                lora.on = True
                return p

            # the replay assumption, on consecutive calls of one generator as in the #1726 generate()
            g1, g2, same = torch.Generator().manual_seed(2), torch.Generator().manual_seed(2), 0
            for k in range(32):
                p = torch.softmax(torch.randn(V, generator=torch.Generator().manual_seed(100 + k)) * 3, -1)
                same += torch.multinomial(p, 1, generator=g1).item() == int(torch.argmax(p / torch.empty(V).exponential_(1, generator=g2)))
            check("multinomial_is_exponential_race_over_steps", same == 32, matched=same, of=32, torch=torch.__version__)

            s0 = plan["streams"][0]
            seq0 = list(prompts[s0["prompt"]]) + gens[0]["ids"][:64]
            a, b = next_dist(seq0, True), next_dist(seq0, True)
            check("forward_deterministic", torch.equal(a, b))
            check("adapter_differs_from_base", (a - next_dist(seq0, False)).abs().max().item() > 0)

            def measure(p, sel, ok):
                ps = p[sel].item()
                return {"p_non_piano": p[non_piano].sum().item(), "p_allowed": p[ok].sum().item(),
                        "p_selected": ps, "selected_rank": int((p > ps).sum().item()) + 1,
                        "top_non_piano": [str(vocab[i]) for i in torch.topk(torch.where(non_piano, p, 0.0), 3).indices.tolist()]}

            t = time.perf_counter()
            points = []
            for s in plan["streams"]:
                if s["valid"]:
                    continue
                g, ids = gens[s["stream"]], gens[s["stream"]]["ids"]
                prompt = list(prompts[s["prompt"]])
                # replay the generator noise up to E (one Exp(1) vector over the vocab per step)
                gen = torch.Generator().manual_seed(g["seed"])
                noise = {}
                for step in range(s["E"] + 1):
                    q = torch.empty(V).exponential_(1, generator=gen)
                    if step in (s["C"], s["E"]):
                        noise[step] = q
                for where in ("C", "E"):
                    i = s[where]
                    seq, sel, q, ok = prompt + ids[:i], ids[i], noise[i], allowed(*s["states"][i])
                    pb, pa = next_dist(seq, False), next_dist(seq, True)
                    p_gen, p_other = (pa, pb) if s["adapter"] else (pb, pa)
                    pick_gen, pick_other = int(torch.argmax(p_gen / q)), int(torch.argmax(p_other / q))
                    points.append({
                        "stream": s["stream"], "prompt": s["prompt"], "seed": s["seed"], "generator": "adapter" if s["adapter"] else "base",
                        "where": where, "index": i, "state": s["states"][i][0], "token": str(vocab[sel]),
                        "base": measure(pb, sel, ok), "adapter": measure(pa, sel, ok),
                        "replay_reproduces_sample": pick_gen == sel,
                        "noise_rank_of_selected": int((q < q[sel]).sum().item()) + 1, "noise_of_selected": q[sel].item(),
                        "other_model_pick": str(vocab[pick_other]),
                        "other_model_pick_allowed": bool(ok[pick_other].item())})
            log["phases"]["E_C"] = {"wall_s": round(time.perf_counter() - t, 3), **rss(), **pressure()}
            log["points"] = points

            t = time.perf_counter()
            streams_s = []
            for s in plan["streams"]:
                if not s["S"]:
                    continue
                ids, prompt = gens[s["stream"]]["ids"], list(prompts[s["prompt"]])
                seq = prompt + ids[:max(s["S"]) + 1]
                pb, pa = all_dists(seq, False), all_dists(seq, True)
                rows = [len(prompt) - 1 + i for i in s["S"]]
                nb, na = pb[rows][:, non_piano].sum(-1), pa[rows][:, non_piano].sum(-1)
                ratio = (na / nb).tolist()
                streams_s.append({"stream": s["stream"], "generator": "adapter" if s["adapter"] else "base", "positions": len(rows),
                                  "median_ratio": statistics.median(ratio),
                                  "positions_adapter_higher": sum(1 for r in ratio if r > 1),
                                  "mean_p_non_piano_base": nb.mean().item(), "mean_p_non_piano_adapter": na.mean().item(),
                                  "max_p_non_piano_base": nb.max().item(), "max_p_non_piano_adapter": na.max().item()})
            log["phases"]["S"] = {"wall_s": round(time.perf_counter() - t, 3), **rss(), **pressure()}
            log["S"] = streams_s

            # reading rules (fixed before the run)
            e_pts = [p for p in points if p["where"] == "E"]
            r_e = [p["adapter"]["p_non_piano"] / p["base"]["p_non_piano"] for p in e_pts]

            def verdict(n_up, n_down, med, need):
                if n_up >= need and med >= RATIO:
                    return "adapter_higher"
                if n_down >= need and med <= 1 / RATIO:
                    return "adapter_lower"
                return "unclear"

            med_s = [x["median_ratio"] for x in streams_s]
            reproduced = all(p["replay_reproduces_sample"] for p in points)
            log["reading"] = {
                "R1_E": {"verdict": verdict(sum(r > 1 for r in r_e), sum(r < 1 for r in r_e), statistics.median(r_e), MAJORITY_E),
                         "adapter_higher": sum(r > 1 for r in r_e), "of": len(r_e), "median_ratio": statistics.median(r_e)},
                "R1_S": {"verdict": verdict(sum(m > 1 for m in med_s), sum(m < 1 for m in med_s), statistics.median(med_s), MAJORITY_S),
                         "streams_adapter_higher": sum(m > 1 for m in med_s), "of": len(med_s), "median_of_stream_medians": statistics.median(med_s)},
                "R2": {"errors_with_p_selected_below": TAIL_P,
                       "count": sum(1 for p in e_pts if p[p["generator"]]["p_selected"] < TAIL_P), "of": len(e_pts)},
                "R3": {"replay_reproduced_all": reproduced,
                       "adapter_errors_fixed_by_base": sum(1 for p in e_pts if p["generator"] == "adapter" and p["other_model_pick_allowed"]) if reproduced else None,
                       "adapter_errors": sum(1 for p in e_pts if p["generator"] == "adapter"),
                       "base_errors_fixed_by_adapter": sum(1 for p in e_pts if p["generator"] == "base" and p["other_model_pick_allowed"]) if reproduced else None,
                       "base_errors": sum(1 for p in e_pts if p["generator"] == "base")}}
            log["result"] = "run_done"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except RuntimeError as e:
        log["result"], log["stopped_at"] = "error", f"{type(e).__name__}: {e}"[:500]
    finally:
        signal.alarm(0)
        with open(report_path, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v["ok"] for k, v in log["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
