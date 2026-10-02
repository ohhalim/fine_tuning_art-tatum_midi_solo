#!/usr/bin/env python3
"""Runtime coherence check: candidate runtime runs vs the default runtime (docs/experiments/RUNTIME_PATTERN_CACHE.md).

Motif reuse and boundary ratio as in ``coherence_metrics`` (per-file block
length), chord-tone of played notes paired by progression and seed, fallback
and deadline misses per run.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

PROGS = {"iiVI": ("ii_V_I_C", 16, 128), "blues": ("blues_F", 12, 120), "minor": ("minor_ii_V_i_C", 16, 128)}
REAL_MOTIF_REUSE = {"tatum": 0.073, "mehldau": 0.094}       # docs/experiments/COHERENCE_GOAL.md
REAL_BOUNDARY = {"tatum": 1.277, "mehldau": 0.964}


def baseline_dir(model, tag, seed):
    return (ROOT / f"outputs/showcase_v1/runs/{PROGS[tag][0]}_{model}" if seed == 42
            else ROOT / f"outputs/ctx_hist/A_{model}_{tag}_s{seed}")


def chord_tone(report) -> float:
    from inference.app.fallback import parse_chord
    chords, hits, tot = report["chords"], 0, 0
    for i, b in enumerate(report["played_bars"]):
        root, iv = parse_chord(chords[i % len(chords)])
        pcs = {(root + k) % 12 for k in iv}
        for n in b["notes"]:
            tot += 1
            hits += (n[0] % 12) in pcs
    return hits / tot if tot else None


def judge(candidate: dict, baseline: dict, model: str) -> dict:
    real = REAL_MOTIF_REUSE[model]
    ok = {
        "motif_reuse_half_of_real": candidate["motif_reuse"] is not None and candidate["motif_reuse"] >= 0.5 * real,
        "chord_tone_guard": candidate["chord_tone_diff_mean"] is not None and candidate["chord_tone_diff_mean"] >= -0.03,
        "no_fallback": candidate["fallback_total"] == 0,
    }
    return {**ok, "pass": all(ok.values())}


def collect(dirs) -> dict:
    from scripts.coherence_metrics import report_line, summarize
    items, fb, miss, rows = [], 0, 0, []
    for d in dirs:
        p = d / "continuous_report.json"
        r = json.loads(p.read_text())
        line, block, _ = report_line(p)
        items.append((line, block))
        fb += r["production"]["fallback_bar_count"]
        miss += r["scheduler_dispatch_deadline_miss_count"]
        rows.append({"run": d.name, "chord_tone": chord_tone(r), "fallback": r["production"]["fallback_bar_count"],
                     "misses": r["scheduler_dispatch_deadline_miss_count"],
                     "cache": r.get("pattern_cache_stats")})
    s = summarize(items)
    return {"motif_reuse": s["motif_reuse"], "boundary_ratio": s["boundary_ratio"], "fallback_total": fb,
            "misses_total": miss, "runs": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tatum")
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    seeds = [int(s) for s in args.seeds.split(",")]
    pairs = [(tag, s) for tag in PROGS for s in seeds]
    cand_dirs = [args.candidate_root / f"R_{args.model}_{tag}_s{s}" for tag, s in pairs]
    base_dirs = [baseline_dir(args.model, tag, s) for tag, s in pairs]
    cand, base = collect(cand_dirs), collect(base_dirs)
    diffs = [c["chord_tone"] - b["chord_tone"] for c, b in zip(cand["runs"], base["runs"])]
    cand["chord_tone_diff_mean"] = statistics.mean(diffs)
    cand["chord_tone_diffs"] = diffs
    out = {"schema": "runtime_coherence_check_v1", "model": args.model, "real_motif_reuse": REAL_MOTIF_REUSE[args.model],
           "real_boundary_ratio": REAL_BOUNDARY[args.model], "baseline": base, "candidate": cand,
           "verdict": judge(cand, base, args.model), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"baseline motif reuse {f(base['motif_reuse'])} boundary {f(base['boundary_ratio'])} fallback {base['fallback_total']}")
    print(f"candidate motif reuse {f(cand['motif_reuse'])} boundary {f(cand['boundary_ratio'])} fallback {cand['fallback_total']} "
          f"misses {cand['misses_total']} chord-tone diff {cand['chord_tone_diff_mean']:+.3f}")
    print("verdict", out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
