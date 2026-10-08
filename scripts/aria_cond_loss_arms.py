#!/usr/bin/env python3
"""Aria chord-condition adapter: three loss-position arms, 32 steps each (docs/experiments/ARIA_COND_LOSS_ARMS.md).

A = every next-token target, B = targets from the first discriminating note on, C = B without
the <E> target. Same frozen model, injection point, data, order and optimizer for all arms; new
zero-initialised adapters. Evaluation at step 0 and 32 only. Run with the Aria venv:
aria_cond_loss_arms.py <aria repo> <checkpoint> <previous 32-step dir> <out dir>
"""
from __future__ import annotations

import hashlib
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
from aria_cond_mass import CORRECT, PC_GROUPS, group_of  # noqa: E402
from aria_cond_smoke import Stop, chord_pcs, file_sha, pressure, rss  # noqa: E402
from aria_cond_synth import CHORD, EVAL, TRAIN, VEL, example, first_discriminating_index, target_mask  # noqa: E402
from aria_t4b_sim import write_notes  # noqa: E402

ARMS, STEPS, LR, WALL_LIMIT_S, MAX_TOKENS, GEN_TOKENS, GEN_SEED, REPRO_TOL = "ABC", 32, 1e-3, 900, 128, 24, 1, 1e-4
GROUPS = ["pc_E_B", "pc_Eb_Bb", "pc_C_G", "pc_other", "eos", "time", "onset", "dur", "other_instrument", "other"]
OPPOSITE = {"pc_E_B": "pc_Eb_Bb", "pc_Eb_Bb": "pc_E_B"}


def grammar_check(prefix, new):
    """Token order after the prefix: (piano, p, v) -> onset -> dur, <T> and <E> only between notes;
    onsets non-decreasing inside a 5 s segment."""
    expect, last_onset, errors = "note", max([x[1] for x in prefix if isinstance(x, tuple) and x[0] == "onset"] or [0]), []
    for i, t in enumerate(new):
        kind = t[0] if isinstance(t, tuple) else t
        if expect == "note":
            if kind == "<T>":
                last_onset = 0
            elif kind == "<E>":
                break
            elif kind != "piano":
                errors.append((i, str(t)))
            else:
                expect = "onset"
        elif expect == "onset":
            if kind != "onset":
                errors.append((i, str(t)))
            elif t[1] < last_onset:
                errors.append((i, f"onset {t[1]} < {last_onset}"))
            else:
                last_onset = t[1]
            expect = "dur" if kind == "onset" else expect
        else:
            if kind != "dur":
                errors.append((i, str(t)))
            expect = "note"
    return errors


def main() -> None:
    aria_repo, ckpt, prev_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"phases": {}, "checks": {}, "train_loss": {a: [] for a in ARMS}, "eval": {}, "eval_ce": {},
           "generation": [], "repro_A": None, "description": None, "result": None, "stopped_at": None}
    t_start = time.perf_counter()

    def phase(name, t0):
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(),
               "mps_current_allocated_bytes": torch.mps.current_allocated_memory()}
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

        def base_param_sha():
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        params_sha = base_param_sha()
        adapters = {a: torch.nn.Linear(12, cfg.d_model, bias=False).to(dev) for a in ARMS}
        for ad in adapters.values():
            torch.nn.init.zeros_(ad.weight)
        state = {"cond": None, "adapter": None}

        def pre_hook(_mod, args, kwargs):
            if state["adapter"] is None:
                return None
            return (args[0] + state["adapter"](state["cond"]),) + args[1:], kwargs

        model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)

        def run(ids, cond=None, adapter=None):
            state["cond"], state["adapter"] = cond, adapter
            return model(ids)

        vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
        gidx = {g: torch.tensor([i for i, t in enumerate(vocab) if group_of(t) == g], device=dev) for g in GROUPS}
        conds = {q: chord_pcs(*CHORD[q]) for q in "PQ"}

        def cond_for(toks, q):
            return torch.tensor([chroma_per_position(toks, [(0, 10 ** 9, conds[q])])], dtype=torch.float32, device=dev)

        def build(fam, q):
            notes, cand = example(fam, q)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "x.mid")
                write_notes(notes, path)
                toks = tok.tokenize(MidiDict.from_midi(path))
            first = first_discriminating_index(toks, cand)
            if toks[first] != ("piano", cand[q], VEL) or len(toks) > MAX_TOKENS:
                raise Stop(f"unexpected tokens for {fam}{q}")
            return {"fam": fam, "q": q, "toks": toks, "ids": torch.tensor([tok.encode(toks)], device=dev), "first": first,
                    "pos": first - 1, "cand": cand, "tP": tok.encode([("piano", cand["P"], VEL)])[0],
                    "tQ": tok.encode([("piano", cand["Q"], VEL)])[0],
                    "masks": {a: torch.tensor(target_mask(toks, first, a), device=dev) for a in ARMS}}

        train = [build(f, q) for f in TRAIN for q in "PQ"]
        evals = {f: {q: build(f, q) for q in "PQ"} for f in EVAL}
        phase("load_build", t0)

        def dist(ex, q, adapter):
            n = ex["pos"] + 1
            with torch.no_grad():
                logits = run(ex["ids"][:, :n], cond_for(ex["toks"][:n], q) if adapter is not None else None, adapter)
            prob = torch.softmax(logits[0, ex["pos"]].float(), -1)
            lp = torch.log(prob)
            mass = {g: prob[gidx[g]].sum().item() for g in GROUPS}
            return {"L_vel80": (lp[ex["tP"]] - lp[ex["tQ"]]).item(), "mass": mass, "sum": sum(mass.values())}

        def evaluate(adapter):
            rows = []
            for f in EVAL:
                ex = evals[f]["P"]
                for q in "PQ":
                    d = dist(ex, q, adapter)
                    good = CORRECT[q]
                    rows.append({"family": f, "condition": CHORD[q][0] + CHORD[q][1], "L_vel80": d["L_vel80"],
                                 "M": d["mass"][good], "opposite": d["mass"][OPPOSITE[good]],
                                 "M_minus_opposite": d["mass"][good] - d["mass"][OPPOSITE[good]],
                                 "eos": d["mass"]["eos"], "C_G": d["mass"]["pc_C_G"], "mass": d["mass"], "sum": d["sum"]})
            return rows

        def eval_ce(adapter):
            out = {}
            with torch.no_grad():
                for name, drop_eos in (("with_eos", False), ("without_eos", True)):
                    vals = []
                    for f in EVAL:
                        for q in "PQ":
                            ex = evals[f][q]
                            mask = ex["masks"]["C" if drop_eos else "B"]
                            lg = run(ex["ids"], cond_for(ex["toks"], q) if adapter is not None else None, adapter)
                            ce = torch.nn.functional.cross_entropy(lg[0, :-1].float(), ex["ids"][0, 1:], reduction="none")
                            vals.append((ce * mask).sum().item() / mask.sum().item())
                    out[name] = sum(vals) / len(vals)
            return out

        t0 = time.perf_counter()
        log["eval"]["base"] = evaluate(None)
        log["eval_ce"]["base"] = eval_ce(None)
        z = evaluate(adapters["A"])
        check("step0_zero_adapter_equals_base",
              max(abs(a["L_vel80"] - b["L_vel80"]) for a, b in zip(z, log["eval"]["base"])) == 0.0)
        check("group_sums_one", all(abs(r["sum"] - 1) < 1e-4 for r in log["eval"]["base"]))
        phase("eval_step0", t0)

        order, rng = [], random.Random(0)
        for _ in range(STEPS // len(train)):
            idx = list(range(len(train)))
            rng.shuffle(idx)
            order += idx
        check("exactly_32_steps_planned", len(order) == STEPS)
        base_ids = {id(p) for p in model.parameters()}
        for arm in ARMS:
            t0 = time.perf_counter()
            ad = adapters[arm]
            opt = torch.optim.Adam([ad.weight], lr=LR)
            check(f"optimizer_{arm}_has_no_base_param", sum(id(p) in base_ids for g in opt.param_groups for p in g["params"]) == 0)
            for step, i in enumerate(order, 1):
                ex = train[i]
                logits = run(ex["ids"], cond_for(ex["toks"], ex["q"]), ad)
                ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ex["ids"][0, 1:], reduction="none")
                mask = ex["masks"][arm]
                loss = (ce * mask).sum() / mask.sum()
                if not torch.isfinite(loss).item():
                    raise Stop(f"non-finite loss {arm} step {step}")
                opt.zero_grad()
                loss.backward()
                if not torch.isfinite(ad.weight.grad).all().item():
                    raise Stop(f"non-finite gradient {arm} step {step}")
                opt.step()
                log["train_loss"][arm].append({"step": step, "example": ex["fam"] + ex["q"], "loss": loss.item(),
                                               "targets": int(mask.sum().item())})
            check(f"base_has_no_grad_after_{arm}", all(p.grad is None for p in model.parameters()))
            log["eval"][arm] = evaluate(ad)
            log["eval_ce"][arm] = eval_ce(ad)
            check(f"group_sums_one_{arm}", all(abs(r["sum"] - 1) < 1e-4 for r in log["eval"][arm]))
            phase(f"train_eval_{arm}", t0)

        # A must reproduce the earlier 32-step run
        with open(os.path.join(prev_dir, "result.json")) as f:
            prev = json.load(f)
        with open(os.path.join(prev_dir, "mass.json")) as f:
            prev_mass = json.load(f)
        diffs = []
        for r in log["eval"]["A"]:
            old = next(x for x in prev["eval"]["step32"] if x["family"] == r["family"])["cond_" + r["condition"]]["L"]
            om = next(x for x in prev_mass["rows"] if x["family"] == r["family"] and x["condition"] == r["condition"])["mass_raw"]
            diffs += [abs(r["L_vel80"] - old)] + [abs(r["mass"][g] - om[g]) for g in ("pc_E_B", "pc_Eb_Bb", "eos", "pc_C_G")]
        log["repro_A"] = {"max_abs_diff": max(diffs), "tolerance": REPRO_TOL, "ok": max(diffs) <= REPRO_TOL}

        # description per the preregistered rules
        desc = {}
        for arm in ARMS:
            rows = log["eval"][arm]
            desc[arm] = {"M_gt_opposite": sum(r["M"] > r["opposite"] for r in rows),
                         "M_gt_opposite_Cmaj7": sum(r["M"] > r["opposite"] for r in rows if r["condition"] == "Cmaj7"),
                         "M_gt_opposite_Cm7": sum(r["M"] > r["opposite"] for r in rows if r["condition"] == "Cm7"),
                         "mean_M": sum(r["M"] for r in rows) / len(rows), "sum_M": sum(r["M"] for r in rows),
                         "mean_eos": sum(r["eos"] for r in rows) / len(rows), "mean_C_G": sum(r["C_G"] for r in rows) / len(rows)}
        base_rows = log["eval"]["base"]
        desc["base"] = {"mean_M": sum(r["M"] for r in base_rows) / len(base_rows), "mean_eos": sum(r["eos"] for r in base_rows) / len(base_rows),
                        "mean_C_G": sum(r["C_G"] for r in base_rows) / len(base_rows)}
        paired = {}
        for name, (x, y) in {"B_minus_A": ("B", "A"), "C_minus_B": ("C", "B"), "C_minus_A": ("C", "A")}.items():
            d = [a["M"] - b["M"] for a, b in zip(log["eval"][x], log["eval"][y])]
            pos = [v for v in d if v > 0]
            paired[name] = {"per_case": d, "positive": len(pos), "of": len(d), "mean": sum(d) / len(d),
                            "positive_Cmaj7": sum(v > 0 for v, r in zip(d, log["eval"][x]) if r["condition"] == "Cmaj7"),
                            "positive_Cm7": sum(v > 0 for v, r in zip(d, log["eval"][x]) if r["condition"] == "Cm7"),
                            "largest_share_of_positive_sum": (max(pos) / sum(pos)) if pos else None,
                            "eos_diff_per_case": [a["eos"] - b["eos"] for a, b in zip(log["eval"][x], log["eval"][y])]}
        desc["paired_M"] = paired
        log["description"] = desc

        # short free generation, paired common random numbers
        t0 = time.perf_counter()
        for arm in ARMS:
            for f in EVAL:
                ex = evals[f]["P"]
                for q in "PQ":
                    g = torch.Generator().manual_seed(GEN_SEED)
                    ids, toks, new = ex["ids"][:, : ex["pos"] + 1].clone(), list(ex["toks"][: ex["pos"] + 1]), []
                    for _ in range(GEN_TOKENS):
                        with torch.no_grad():
                            logits = run(ids, cond_for(toks, q), adapters[arm])[0, -1].float().cpu()
                        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=g).item()
                        t = tok.decode([nxt])[0]
                        new.append(t)
                        toks.append(t)
                        ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
                        if t == "<E>":
                            break
                    pitches = [x[1] for x in new if isinstance(x, tuple) and x[0] == "piano"]
                    log["generation"].append({"arm": arm, "family": f, "condition": CHORD[q][0] + CHORD[q][1],
                                              "tokens": len(new), "first_token_eos": new[0] == "<E>", "pitches": pitches,
                                              "grammar_errors": grammar_check(ex["toks"][: ex["pos"] + 1], new),
                                              "count_E": sum(p % 12 == 4 for p in pitches), "count_Eb": sum(p % 12 == 3 for p in pitches),
                                              "count_B": sum(p % 12 == 11 for p in pitches), "count_Bb": sum(p % 12 == 10 for p in pitches)})
        phase("generation", t0)

        t0 = time.perf_counter()
        ex = evals[EVAL[0]]["P"]
        for arm in ARMS:
            path = os.path.join(out_dir, f"adapter_{arm}_synthetic_diagnostic_only.pt")
            torch.save({"weight": adapters[arm].weight.detach().cpu(), "note": "synthetic_diagnostic_only; never deploy"}, path)
            re = torch.nn.Linear(12, cfg.d_model, bias=False).to(dev)
            re.weight.data.copy_(torch.load(path)["weight"].to(dev))
            with torch.no_grad():
                a = run(ex["ids"], cond_for(ex["toks"], "P"), adapters[arm]).float()
                b = run(ex["ids"], cond_for(ex["toks"], "P"), re).float()
            check(f"reload_{arm}", (a - b).abs().max().item() <= 1e-6, max_abs_diff=(a - b).abs().max().item(), sha256=file_sha(path))
        check("base_params_unchanged", base_param_sha() == params_sha)
        check("checkpoint_file_unchanged", file_sha(ckpt) == ckpt_sha)
        phase("save_reload_hash", t0)
        log["total_wall_s"] = round(time.perf_counter() - t_start, 1)
        log["result"] = "completed" if log["repro_A"]["ok"] else "completed_judgment_withheld"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except Exception as e:
        log["result"], log["stopped_at"] = "stopped", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"], "repro_A": log["repro_A"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
