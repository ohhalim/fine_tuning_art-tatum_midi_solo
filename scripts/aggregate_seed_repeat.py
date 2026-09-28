#!/usr/bin/env python3
"""M-S1: is B (out_proj+qkv) better than A (out_proj) for every seed?

Pre-registered in docs/experiments/SEED_REPEAT_AND_MEHLDAU_APPLY.md. Reads one
snapshot-eval report and one runtime sweep report per (arm, seed).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def final_row(report: dict) -> dict:
    return max(report["rows"], key=lambda r: r["update"])


def runtime_row(sweep: dict) -> dict:
    model = next(iter(sweep["models"].values()))
    run = next(iter(model["bpms"].values()))["runs"][0]
    return {"gen_ms_p95": run.get("gen_ms_p95"), "fallback": run.get("fallback_bars"),
            "misses": run.get("deadline_misses")}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, required=True,
                    help="contains eval_{arm}_s{seed}/report.json and runtime_{arm}_s{seed}/sweep_report.json")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows = {}
    for arm in ("a", "b"):
        for seed in seeds:
            ev = json.loads((args.root / f"eval_{arm}_s{seed}" / "report.json").read_text())
            rt = json.loads((args.root / f"runtime_{arm}_s{seed}" / "sweep_report.json").read_text())
            r = final_row(ev)
            per = r["per_seed"]
            rows[(arm, seed)] = {
                "arm": arm, "seed": seed, "update": r["update"], "lora_targets": ev.get("lora_targets"),
                "d_ce_val": r["d_ce_target_val"], "d_ce_train": r["d_ce_target_train"],
                "d_ce_generic": r["d_ce_generic"], "specialisation_val": r["specialisation_val"],
                "grammar_valid_rate": (sum(p["grammar_valid"] for p in per) / len(per)) if per else None,
                "copy16_max": max((p["copy16"] or 0) for p in per) if per else None,
                "copy8_mean": statistics.mean((p["copy8"] or 0) for p in per) if per else None,
                "notes_mean": statistics.mean(p["notes"] for p in per) if per else None,
                **runtime_row(rt)}
    diffs = [rows[("b", s)]["specialisation_val"] - rows[("a", s)]["specialisation_val"] for s in seeds]
    b_generic = statistics.mean(rows[("b", s)]["d_ce_generic"] for s in seeds)
    robust = all(d < 0 for d in diffs) and statistics.mean(diffs) <= -0.01 and b_generic <= 0.02
    warnings = [f"{k[0]} s{k[1]}: {w}" for k, v in rows.items() for w in (
        (["grammar_valid < 100%"] if v["grammar_valid_rate"] is not None and v["grammar_valid_rate"] < 1 else [])
        + (["copy16 > 0.10"] if (v["copy16_max"] or 0) > 0.10 else [])
        + (["fallback > 0"] if (v["fallback"] or 0) > 0 else []))]
    out = {"schema": "seed_repeat_v1", "seeds": seeds,
           "rows": [rows[(a, s)] for a in ("a", "b") for s in seeds],
           "paired_specialisation_diff_b_minus_a": diffs,
           "mean_diff": statistics.mean(diffs), "b_mean_d_ce_generic": b_generic,
           "verdict": {"rule": "all diffs < 0, mean diff <= -0.01, B mean generic <= +0.02",
                       "robust": robust},
           "warnings": warnings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for r in out["rows"]:
        print(f"{r['arm']} s{r['seed']} spec {r['specialisation_val']:+.4f} gen {r['d_ce_generic']:+.4f} "
              f"valid {r['grammar_valid_rate']} copy16max {r['copy16_max']} p95 {r['gen_ms_p95']} fb {r['fallback']}")
    print("diffs", [round(d, 4) for d in diffs], "mean", round(out["mean_diff"], 4),
          "B generic", round(b_generic, 4), "robust", robust, "warnings", warnings)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
