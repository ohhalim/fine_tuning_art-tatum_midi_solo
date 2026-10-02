#!/usr/bin/env python3
"""Judge the repeat-weight pilot (docs/experiments/REPEAT_WEIGHT_PILOT.md, #1602).

Arms A (W=1) and B (W=3), both at update 518. Free rollouts from the same
positions, natural 256-token context and seeds as #1600, at temperature 1.0
(judged) and 0.6 (reported); fresh12 crop CE per song. No snapshot selection.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from scripts.context_mismatch_diag import (  # noqa: E402
    MAX_SEQ, PREFIX, TARGET_STEPS, bootstrap_ci, pick_positions, session_line, strip_tail, tail,
)
from scripts.decoding_gap_diag import GEN_TOKENS, SEEDS, mean_defined, measures, song_flag_rate, song_rates  # noqa: E402

ARMS = {"A": "outputs/repeat_pilot/export_A/checkpoint_update518.pt",
        "B": "outputs/repeat_pilot/export_B/checkpoint_update518.pt"}
TEMPS = (1.0, 0.6)
FRESH = "data/tvm/holdout_tatum_fresh12/val"
VAL = "data/tvm/holdout_tatum_val12/val"


def arm_stats(cases, t) -> dict:
    rows = [(c["song"], m) for c in cases for m in c["rollouts"][str(t)]]
    valid = [m["valid"] for _, m in rows]
    out = {k: mean_defined(song_rates(rows, k)) for k in ("interval", "variant", "exact", "repetition")}
    out["copy_rate"] = mean_defined(song_flag_rate(rows))
    out["valid_rate"] = sum(valid) / len(valid) if valid else None
    out["notes"] = statistics.mean(m["notes"] for _, m in rows) if rows else None
    return out


def paired_ir(cases_a, cases_b, t) -> dict:
    ra = song_rates([(c["song"], m) for c in cases_a for m in c["rollouts"][str(t)]], "interval")
    rb = song_rates([(c["song"], m) for c in cases_b for m in c["rollouts"][str(t)]], "interval")
    diffs = [rb[s] - ra[s] for s in rb if s in ra and ra[s] is not None and rb[s] is not None]
    return {"mean": statistics.mean(diffs) if diffs else None, "ci95": bootstrap_ci(diffs), "songs": len(diffs)}


def judge(s) -> dict:
    """Fixed rule (plan)."""
    a, b, real = s["rollout"]["A"]["1.0"], s["rollout"]["B"]["1.0"], s["real"]
    d = s["paired_ir_1.0"]
    ok = {
        "ir_gain": bool(d["ci95"]) and d["ci95"][0] > 0 and b["interval"] is not None
        and a["interval"] is not None and b["interval"] >= 2 * a["interval"],
        "ce_guard": s["ce"]["fresh12_B_minus_A"] is not None and s["ce"]["fresh12_B_minus_A"] <= 0.02,
        "repetition_guard": b["repetition"] is not None and real["repetition"] is not None
        and b["repetition"] <= real["repetition"] + 0.15,
        "copy_guard": b["copy_rate"] is not None and b["copy_rate"] <= max(0.25, 2 * (real["copy_rate"] or 0.0)),
        "valid_guard": b["valid_rate"] is not None and b["valid_rate"] >= 0.9,
    }
    return {**ok, "pass": all(ok.values()),
            "note": "no runtime default changes; use only after listening, in a separate issue"}


def song_ce(model, files):
    import numpy as np
    import torch
    import torch.nn.functional as F
    from scripts.run_mehldau_update_budget_diag import fixed_crops
    from scripts.validate_style_distance import load
    from utilities.constants import TOKEN_PAD

    out = {}
    with torch.no_grad():
        for f in files:
            total = n = 0
            for c in fixed_crops([np.asarray(load(f))], 512):
                c = np.asarray(c)
                x = torch.tensor(c[:-1]).unsqueeze(0)
                y = torch.tensor(c[1:])
                keep = y != TOKEN_PAD
                total += F.cross_entropy(model(x)[0][keep], y[keep], reduction="sum").item()
                n += int(keep.sum())
            out[f.name] = total / n
    return out


def run(out_dir: Path) -> dict:
    import torch
    from scripts.generate import generate_once, load_model_with_lora
    from scripts.run_resident_model_probe import validate_generated_token_block
    from scripts.train_qlora import merge_lora_for_inference
    from scripts.validate_style_distance import load

    out_dir.mkdir(parents=True, exist_ok=True)
    fresh = sorted((ROOT / FRESH).glob("*.npy"))
    val = sorted((ROOT / VAL).glob("*.npy"))
    songs = [(f.name, [int(t) for t in load(f)]) for f in fresh]
    positions = {name: pick_positions(toks) for name, toks in songs}
    cases, ce, real_rows = {}, {}, []
    for arm, ck in ARMS.items():
        model = load_model_with_lora(lora_path=str((ROOT / ck).parent), checkpoint_path=str(ROOT / ck),
                                     prefer_full_checkpoint=True, max_sequence=MAX_SEQ)
        merge_lora_for_inference(model)
        model.eval()
        t0, arm_cases = time.time(), []
        for name, toks in songs:
            for pi, (i, target) in enumerate(positions[name]):
                primer = tail(strip_tail(toks[:i]), PREFIX)
                sline = session_line(toks, i)
                case = {"song": name, "i": i, "rollouts": {}}
                if arm == "A":
                    real_rows.append((name, measures(target, sline)))
                for t in TEMPS:
                    outs = []
                    for seed in SEEDS:
                        torch.manual_seed(seed * 1000 + pi)
                        gen, _ = generate_once(model=model, primer=torch.tensor(primer, dtype=torch.long),
                                               target_length=min(MAX_SEQ, len(primer) + GEN_TOKENS),
                                               strip_primer=True, temperature=t, top_k=32, top_p=0.95,
                                               grammar_mask=True, target_duration_seconds=TARGET_STEPS / 100,
                                               return_metadata=True, use_kv_cache=True)
                        gen = [int(x) for x in gen]
                        mm = measures(gen, sline)
                        mm["valid"] = bool(validate_generated_token_block(
                            gen, lookahead_ms=TARGET_STEPS * 10, allow_rest_bar=True)["valid"])
                        outs.append(mm)
                    case["rollouts"][str(t)] = outs
                arm_cases.append(case)
        cases[arm] = arm_cases
        ce[arm] = {"fresh12": song_ce(model, fresh), "val12": song_ce(model, val), "wall_s": round(time.time() - t0, 1)}
        print(arm, "done", ce[arm]["wall_s"], flush=True)
        del model
    real = {k: mean_defined(song_rates(real_rows, k)) for k in ("interval", "variant", "exact", "repetition")}
    real["copy_rate"] = mean_defined(song_flag_rate(real_rows))
    ce_diff = [ce["B"]["fresh12"][s] - ce["A"]["fresh12"][s] for s in ce["A"]["fresh12"]]
    s = {"real": real,
         "rollout": {arm: {str(t): arm_stats(cases[arm], t) for t in TEMPS} for arm in ARMS},
         "paired_ir_1.0": paired_ir(cases["A"], cases["B"], 1.0),
         "paired_ir_0.6": paired_ir(cases["A"], cases["B"], 0.6),
         "ce": {"fresh12_A": statistics.mean(ce["A"]["fresh12"].values()),
                "fresh12_B": statistics.mean(ce["B"]["fresh12"].values()),
                "fresh12_B_minus_A": statistics.mean(ce_diff), "fresh12_B_minus_A_ci95": bootstrap_ci(ce_diff),
                "val12_A": statistics.mean(ce["A"]["val12"].values()),
                "val12_B": statistics.mean(ce["B"]["val12"].values())}}
    s["verdict"] = judge(s)
    report = {"schema": "repeat_pilot_eval_v1", "arms": ARMS, "musical_quality_verified": False,
              "style_verified": False, "summary": s}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out_dir / "cases.json").write_text(json.dumps(cases) + "\n")
    print(json.dumps(s, indent=1), flush=True)
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    os.environ.setdefault("FORCE_CPU", "1")
    run(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
