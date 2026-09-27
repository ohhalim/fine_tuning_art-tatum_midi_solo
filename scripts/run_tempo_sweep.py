#!/usr/bin/env python3
"""R1: at which tempo does a checkpoint keep up in the continuous runtime?

Runs ``scripts/run_continuous_jazz.py`` once per (model, bpm, seed), each in its
own process and strictly one after another, then aggregates the reports.
A tempo "holds" for a model when every run has zero fallback bars, zero
generation errors and zero scheduler deadline misses (pre-registered in
docs/experiments/TATUM_REALTIME_TEMPO.md).

Per-bar generation percentiles exclude bar 0, whose time includes cold start.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def summarize_report(report: dict) -> dict:
    prod = report["production"]
    bar_s = 240.0 / report["bpm"]
    steady = [b["generation_ms"] for b in report["bars_detail"]
              if b["bar_index"] > 0 and b.get("generation_ms") is not None]
    p95 = float(np.percentile(steady, 95)) if steady else None
    return {
        "bpm": report["bpm"], "run_completed": report["run_completed"],
        "completed_bars": report["completed_bars"],
        "model_bars": prod["model_bar_count"], "fallback_bars": prod["fallback_bar_count"],
        "errors": prod["error_count"], "discarded_late": prod["discarded_late_count"],
        "deadline_misses": report["scheduler_dispatch_deadline_miss_count"],
        "bar_ms": bar_s * 1000,
        "gen_ms_bar0": next((b["generation_ms"] for b in report["bars_detail"]
                             if b["bar_index"] == 0), None),
        "gen_ms_p50": float(np.median(steady)) if steady else None,
        "gen_ms_p95": p95,
        "gen_ms_max": float(max(steady)) if steady else None,
        "p95_over_bar": (p95 / (bar_s * 1000)) if p95 is not None else None,
        "played_notes": report.get("played_note_count"),
        "lateness_ms": report.get("dispatch_attempt_lateness_summary_ms"),
        "stall_trace": report.get("stall_trace"),
    }


def holds(runs: list[dict]) -> bool:
    return bool(runs) and all(r["run_completed"] and r["fallback_bars"] == 0 and r["errors"] == 0
                              and r["deadline_misses"] == 0 for r in runs)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT",
                    help="CHECKPOINT=FALLBACK runs --fallback-only (no generation)")
    ap.add_argument("--bpms", default="128,160,200,240")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--primer", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--common", default="", help="flags added to every run, fallback included")
    ap.add_argument("--no-capture", action="store_true",
                    help="omit --capture (scheduler lateness is still measured internally)")
    ap.add_argument("--extra", default="--chord-primer --chord-blocks-per-bar 2",
                    help="extra run_continuous_jazz flags, same for every run")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"output dir not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "FORCE_CPU": os.environ.get("FORCE_CPU", "1")}
    results: dict = {"schema": "tempo_sweep_v1", "bars": args.bars, "extra": args.extra,
                     "capture": not args.no_capture, "common": args.common,
                     "primer": str(args.primer), "force_cpu": env["FORCE_CPU"], "models": {}}
    for spec in args.model:
        name, ckpt = spec.split("=", 1)
        results["models"][name] = {"checkpoint": ckpt, "bpms": {}}
        for bpm in (int(b) for b in args.bpms.split(",")):
            runs = []
            for seed in (int(s) for s in args.seeds.split(",")):
                out = args.output_dir / name / f"bpm{bpm}_seed{seed}"
                model_args = (["--fallback-only"] if ckpt == "FALLBACK" else
                              ["--checkpoint", ckpt, "--conditioning-midi", str(args.primer)])
                extra = [] if ckpt == "FALLBACK" else args.extra.split()
                cmd = [args.python, str(ROOT / "scripts/run_continuous_jazz.py"), *model_args,
                       "--bars", str(args.bars), "--bpm", str(bpm), "--seed", str(seed),
                       *([] if args.no_capture else ["--capture"]),
                       "--output-dir", str(out), *extra, *args.common.split()]
                proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
                (args.output_dir / name).mkdir(parents=True, exist_ok=True)
                (args.output_dir / name / f"bpm{bpm}_seed{seed}.log").write_text(
                    proc.stdout + proc.stderr)
                report_path = out / "continuous_report.json"
                if proc.returncode != 0 or not report_path.exists():
                    runs.append({"bpm": bpm, "seed": seed, "run_completed": False,
                                 "fallback_bars": None, "errors": None, "deadline_misses": None,
                                 "returncode": proc.returncode})
                    print(f"{name} {bpm} BPM seed {seed}: FAILED rc={proc.returncode}", flush=True)
                    continue
                row = {"seed": seed, **summarize_report(json.loads(report_path.read_text()))}
                runs.append(row)
                lat = row.get("lateness_ms") or {}
                gen = (f"p50 {row['gen_ms_p50']:.0f} p95 {row['gen_ms_p95']:.0f}"
                       if row["gen_ms_p50"] is not None else "no generation")
                print(f"{name} {bpm} BPM seed {seed}: fallback {row['fallback_bars']} "
                      f"miss {row['deadline_misses']} {gen} / bar {row['bar_ms']:.0f} ms | "
                      f"lateness p99 {lat.get('p99', float('nan')):.2f} max "
                      f"{lat.get('maximum', float('nan')):.2f} ms", flush=True)
            results["models"][name]["bpms"][str(bpm)] = {"runs": runs, "holds": holds(runs)}
        held = [int(b) for b, v in results["models"][name]["bpms"].items() if v["holds"]]
        results["models"][name]["max_holding_bpm"] = max(held) if held else None
    (args.output_dir / "sweep_report.json").write_text(json.dumps(results, indent=2) + "\n")
    for name, m in results["models"].items():
        print(name, "max holding BPM:", m["max_holding_bpm"],
              {b: v["holds"] for b, v in m["bpms"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
