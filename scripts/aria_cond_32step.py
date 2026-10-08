#!/usr/bin/env python3
"""Aria chord-condition adapter: fixed 32-step synthetic contrast test (docs/experiments/ARIA_COND_32STEP.md).

Same frozen model and injection point as aria_cond_smoke.py, a new zero-initialised adapter,
exactly 32 Adam steps over 8 train families x Cmaj7/Cm7. Evaluation at step 0 and 32 only:
log-odds of the first discriminating pitch on 4 held-out prefixes under base, correct, opposite
and zero conditions, plus 8 short free generations. Run with the Aria venv:
aria_cond_32step.py <aria repo> <checkpoint> <out dir>
"""
from __future__ import annotations

import json
import os
import random
import signal
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract import chroma_per_position  # noqa: E402
from aria_cond_smoke import Stop, chord_pcs, file_sha, pressure, rss  # noqa: E402
from aria_cond_synth import CHORD, EVAL, TRAIN, VEL, example  # noqa: E402
from aria_t4b_sim import write_notes  # noqa: E402

STEPS, LR, WALL_LIMIT_S, MAX_TOKENS, GEN_TOKENS, GEN_SEED = 32, 1e-3, 900, 128, 24, 1


def main() -> None:
    aria_repo, ckpt, out_dir = sys.argv[1:4]
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"phases": {}, "checks": {}, "train_loss": [], "eval": {}, "train_families_eval": {},
           "generation": [], "decision": None, "result": None, "stopped_at": None}

    def phase(name, t0):
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(),
               "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
               "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()}
        log["phases"][name] = rec
        if rec["vm_pressure_level"] >= 4:
            raise Stop(f"memory pressure critical after {name}")

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    def on_alarm(*_):
        raise Stop(f"wall limit {WALL_LIMIT_S} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL_LIMIT_S)
    try:
        import torch
        from safetensors.torch import load_file
        from ariautils.midi import MidiDict
        from ariautils.tokenizer import AbsTokenizer
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        torch.manual_seed(0)
        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        dev = torch.device("mps")
        log["env"] = {"torch": torch.__version__, "start": pressure()}
        t0 = time.perf_counter()
        ckpt_sha = file_sha(ckpt)
        tok = AbsTokenizer()
        cfg = ModelConfig(**load_model_config(name="medium"))
        cfg.set_vocab_size(tok.vocab_size)
        model = TransformerLM(cfg)
        model.load_state_dict(load_file(ckpt), strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(dev)
        model.model.freqs_cis = None
        phase("load", t0)

        def base_param_sha():
            import hashlib
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        params_sha = base_param_sha()
        adapter = torch.nn.Linear(12, cfg.d_model, bias=False).to(dev)
        torch.nn.init.zeros_(adapter.weight)
        state = {"cond": None, "on": False}

        def pre_hook(_mod, args, kwargs):
            if not state["on"]:
                return None
            return (args[0] + adapter(state["cond"]),) + args[1:], kwargs

        model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)

        def run(ids, cond=None):
            state["on"], state["cond"] = cond is not None, cond
            return model(ids)

        conds = {q: chord_pcs(*CHORD[q]) for q in "PQ"}

        def cond_for(toks, q):
            if q is None:
                return torch.zeros(1, len(toks), 12, device=dev)
            return torch.tensor([chroma_per_position(toks, [(0, 10 ** 9, conds[q])])], dtype=torch.float32, device=dev)

        def build(fam, q):
            notes, cand = example(fam, q)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "x.mid")
                write_notes(notes, path)
                toks = tok.tokenize(MidiDict.from_midi(path))
            idx = next(i for i, x in enumerate(toks) if isinstance(x, tuple) and x[0] == "piano" and x[1] in cand.values())
            if toks[idx] != ("piano", cand[q], VEL) or len(toks) > MAX_TOKENS:
                raise Stop(f"unexpected tokens for {fam}{q}: {toks[idx]}, {len(toks)}")
            return {"fam": fam, "q": q, "toks": toks, "ids": torch.tensor([tok.encode(toks)], device=dev),
                    "pos": idx - 1, "tP": tok.encode([("piano", cand["P"], VEL)])[0],
                    "tQ": tok.encode([("piano", cand["Q"], VEL)])[0], "cand": cand}

        train = [build(f, q) for f in TRAIN for q in "PQ"]
        evals = {f: {q: build(f, q) for q in "PQ"} for f in EVAL}
        for f in TRAIN + EVAL:
            a, b = (evals[f]["P"], evals[f]["Q"]) if f in evals else [x for x in train if x["fam"] == f]
            check(f"prefix_identical_{f}", a["pos"] == b["pos"] and a["toks"][: a["pos"] + 1] == b["toks"][: b["pos"] + 1],
                  prefix_tokens=a["pos"] + 1, total_tokens=[len(a["toks"]), len(b["toks"])])

        def L_at(ex, q_cond, hook=True):
            n = ex["pos"] + 1
            ids = ex["ids"][:, :n]
            with torch.no_grad():
                logits = run(ids, cond_for(ex["toks"][:n], q_cond) if hook else None)
            lp = torch.log_softmax(logits[0, ex["pos"]].float(), -1)
            return {"L": (lp[ex["tP"]] - lp[ex["tQ"]]).item(), "logp_P": lp[ex["tP"]].item(), "logp_Q": lp[ex["tQ"]].item()}

        def evaluate(families, source):
            rows = []
            for f in families:
                ex = source[f]["P"] if isinstance(source, dict) else next(x for x in source if x["fam"] == f and x["q"] == "P")
                base, maj, mnr, zero = L_at(ex, None, hook=False), L_at(ex, "P"), L_at(ex, "Q"), L_at(ex, None)
                rows.append({"family": f, "candidates": ex["cand"], "base": base, "cond_Cmaj7": maj, "cond_Cm7": mnr,
                             "cond_zero": zero, "delta": maj["L"] - mnr["L"],
                             "examples": {"P": {"LO_correct": maj["L"], "LO_wrong": mnr["L"]},
                                          "Q": {"LO_correct": -mnr["L"], "LO_wrong": -maj["L"]}}})
            return rows

        t0 = time.perf_counter()
        log["eval"]["step0"] = evaluate(EVAL, evals)
        log["train_families_eval"]["step0"] = evaluate(TRAIN, train)
        z = max(abs(r["cond_Cmaj7"]["L"] - r["base"]["L"]) + abs(r["cond_Cm7"]["L"] - r["base"]["L"]) for r in log["eval"]["step0"])
        check("step0_zero_adapter_equals_base", z == 0.0, max_abs_L_diff=z)
        phase("eval_step0", t0)

        opt = torch.optim.Adam([adapter.weight], lr=LR)
        base_ids = {id(p) for p in model.parameters()}
        check("optimizer_has_no_base_param", sum(id(p) in base_ids for g in opt.param_groups for p in g["params"]) == 0)
        order = []
        rng = random.Random(0)
        for _ in range(STEPS // len(train)):
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx
        check("exactly_32_steps_planned", len(order) == STEPS)
        t0 = time.perf_counter()
        for step, i in enumerate(order, 1):
            ex = train[i]
            logits = run(ex["ids"], cond_for(ex["toks"], ex["q"]))
            loss = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ex["ids"][0, 1:])
            if not torch.isfinite(loss).item():
                raise Stop(f"non-finite loss at step {step}")
            opt.zero_grad()
            loss.backward()
            if not torch.isfinite(adapter.weight.grad).all().item():
                raise Stop(f"non-finite gradient at step {step}")
            opt.step()
            log["train_loss"].append({"step": step, "example": ex["fam"] + ex["q"], "loss": loss.item()})
        check("base_has_no_grad", all(p.grad is None for p in model.parameters()))
        phase("train_32_steps", t0)

        t0 = time.perf_counter()
        log["eval"]["step32"] = evaluate(EVAL, evals)
        log["train_families_eval"]["step32"] = evaluate(TRAIN, train)
        z = max(abs(r["cond_zero"]["L"] - r["base"]["L"]) for r in log["eval"]["step32"])
        log["checks"]["zero_vector_equals_base_after_training"] = {"ok": z == 0.0, "max_abs_L_diff": z, "info": "Linear has no bias"}
        phase("eval_step32", t0)

        rows = log["eval"]["step32"]
        deltas = [r["delta"] > 0 for r in rows]
        correct_pos = [r["examples"][q]["LO_correct"] > 0 for r in rows for q in "PQ"]
        if all(deltas) and all(correct_pos):
            decision = "pass"
        elif any(deltas):
            decision = "partial"
        else:
            decision = "fail"
        log["decision"] = {"result": decision, "pairs_delta_positive": sum(deltas), "of_pairs": len(deltas),
                           "examples_correct_preferred": sum(correct_pos), "of_examples": len(correct_pos),
                           "P_correct_preferred": sum(r["examples"]["P"]["LO_correct"] > 0 for r in rows),
                           "Q_correct_preferred": sum(r["examples"]["Q"]["LO_correct"] > 0 for r in rows)}

        # eval-set CE (information only)
        with torch.no_grad():
            ce = {}
            for name, pick in (("correct", lambda ex: ex["q"]), ("wrong", lambda ex: "Q" if ex["q"] == "P" else "P"), ("base", None)):
                vals = []
                for f in EVAL:
                    for q in "PQ":
                        ex = evals[f][q]
                        lg = run(ex["ids"], cond_for(ex["toks"], pick(ex))) if pick else run(ex["ids"])
                        vals.append(torch.nn.functional.cross_entropy(lg[0, :-1].float(), ex["ids"][0, 1:]).item())
                ce[name] = sum(vals) / len(vals)
        log["eval_ce_mean_info_only"] = ce

        # short free generation, information only
        t0 = time.perf_counter()
        for f in EVAL:
            ex = evals[f]["P"]
            prefix = ex["ids"][:, : ex["pos"] + 1]
            for q in "PQ":
                g = torch.Generator().manual_seed(GEN_SEED)
                ids, toks, new = prefix.clone(), list(ex["toks"][: ex["pos"] + 1]), []
                for _ in range(GEN_TOKENS):
                    with torch.no_grad():
                        logits = run(ids, cond_for(toks, q))[0, -1].float().cpu()
                    nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).item()
                    t = tok.decode([nxt])[0]
                    new.append(t)
                    toks.append(t)
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
                    if t == "<E>":
                        break
                pitches = [x[1] for x in new if isinstance(x, tuple) and x[0] == "piano"]
                log["generation"].append({"family": f, "condition": CHORD[q][0] + CHORD[q][1], "tokens": len(new),
                                          "pitches": pitches, "first_pitch": pitches[0] if pitches else None,
                                          "count_E": sum(p % 12 == 4 for p in pitches), "count_Eb": sum(p % 12 == 3 for p in pitches),
                                          "count_B": sum(p % 12 == 11 for p in pitches), "count_Bb": sum(p % 12 == 10 for p in pitches)})
        phase("generation", t0)

        t0 = time.perf_counter()
        path = os.path.join(out_dir, "adapter_synthetic_diagnostic_only.pt")
        torch.save({"weight": adapter.weight.detach().cpu(), "note": "synthetic_diagnostic_only; never reuse or deploy"}, path)
        ex = evals[EVAL[0]]["P"]
        with torch.no_grad():
            a = run(ex["ids"], cond_for(ex["toks"], "P")).float()
            w = adapter.weight.detach().clone()
            adapter.weight.data.copy_(torch.load(path)["weight"].to(dev))
            b = run(ex["ids"], cond_for(ex["toks"], "P")).float()
        check("reload_logits", (a - b).abs().max().item() <= 1e-6 and torch.equal(w, adapter.weight.detach()),
              max_abs_diff=(a - b).abs().max().item(), adapter_sha256=file_sha(path))
        check("base_params_unchanged", base_param_sha() == params_sha)
        check("checkpoint_file_unchanged", file_sha(ckpt) == ckpt_sha)
        phase("save_reload_hash", t0)
        log["result"] = "completed"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except Exception as e:
        log["result"], log["stopped_at"] = "stopped", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"], "decision": log["decision"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
