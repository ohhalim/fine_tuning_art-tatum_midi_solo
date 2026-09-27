#!/usr/bin/env python3
"""Compare continuous-runtime reports between arms (S2, docs/experiments/RUNTIME_STALL_CAUSE.md).

Each arm is a directory searched for continuous_report.json. Reports the
per-run mean of >10 ms dispatches, lateness percentiles, misses, fallback and
steady-state generation p95, plus the pre-registered adoption check.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def arm_stats(directory: Path) -> dict:
    runs = []
    for path in sorted(directory.rglob("continuous_report.json")):
        r = json.loads(path.read_text())
        lat = r["dispatch_attempt_lateness_summary_ms"]
        steady = [b["generation_ms"] for b in r["bars_detail"]
                  if b["bar_index"] > 0 and b.get("generation_ms") is not None]
        runs.append({"over_10ms": lat["over_10ms"], "p50": lat["p50"], "p99": lat["p99"],
                     "max": lat["maximum"], "misses": r["scheduler_dispatch_deadline_miss_count"],
                     "fallback": r["production"]["fallback_bar_count"],
                     "gen_p95": float(np.percentile(steady, 95)) if steady else None,
                     "qos": r.get("thread_qos")})
    return {
        "runs": len(runs),
        "over_10ms_per_run": statistics.mean(r["over_10ms"] for r in runs),
        "lateness_p50_median": statistics.median(r["p50"] for r in runs),
        "lateness_p99_median": statistics.median(r["p99"] for r in runs),
        "lateness_max": max(r["max"] for r in runs),
        "runs_with_miss": sum(r["misses"] > 0 for r in runs),
        "fallback_total": sum(r["fallback"] for r in runs),
        "gen_p95_median": statistics.median(r["gen_p95"] for r in runs if r["gen_p95"] is not None),
        "qos_applied_runs": sum(bool(r["qos"] and r["qos"].get("applied")) for r in runs),
        "per_run": runs,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--treatment", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    base, treat = arm_stats(args.baseline), arm_stats(args.treatment)
    reduction = (1 - treat["over_10ms_per_run"] / base["over_10ms_per_run"]
                 if base["over_10ms_per_run"] else None)
    gen_ratio = treat["gen_p95_median"] / base["gen_p95_median"]
    adopt = bool(reduction is not None and reduction >= 0.5 and treat["fallback_total"] == 0
                 and gen_ratio <= 1.1)
    out = {"schema": "runtime_arm_compare_v1", "baseline": base, "treatment": treat,
           "over_10ms_reduction": reduction, "gen_p95_ratio": gen_ratio,
           "adopt": adopt}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for name, arm in (("baseline", base), ("treatment", treat)):
        print(name, {k: (round(v, 3) if isinstance(v, float) else v)
                     for k, v in arm.items() if k != "per_run"})
    print("over_10ms reduction", reduction, "gen_p95 ratio", round(gen_ratio, 3), "adopt", adopt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
