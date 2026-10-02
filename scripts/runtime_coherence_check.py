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
# Same definitions without (0, 0, 0) grams and their share in the real songs (#1612).
REAL_REUSE_NO_SAME_NOTE = {"tatum": 0.0722, "mehldau": 0.0900}
REAL_SAME_NOTE_SHARE = {"tatum": 0.0012, "mehldau": 0.0100}


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


def judge(candidate: dict, baseline: dict, model: str, same_note_guard: bool = False) -> dict:
    """``same_note_guard`` (from #1612 on): reuse is counted without (0, 0, 0) grams and the
    same-note share must stay ≤ max(0.02, 3 × the real share); earlier units ran without it."""
    if same_note_guard:
        real, reuse = REAL_REUSE_NO_SAME_NOTE[model], candidate.get("motif_reuse_no_same_note")
    else:
        real, reuse = REAL_MOTIF_REUSE[model], candidate["motif_reuse"]
    ok = {
        "motif_reuse_half_of_real": reuse is not None and reuse >= 0.5 * real,
        "chord_tone_guard": candidate["chord_tone_diff_mean"] is not None and candidate["chord_tone_diff_mean"] >= -0.03,
        "no_fallback": candidate["fallback_total"] == 0,
    }
    if same_note_guard:
        share = candidate.get("same_note_share")
        ok["same_note_guard"] = share is not None and share <= max(0.02, 3 * REAL_SAME_NOTE_SHARE[model])
    return {**ok, "pass": all(ok.values())}


def collect(dirs) -> dict:
    from scripts.coherence_metrics import motif_reuse, report_line, summarize
    items, fb, miss, rows = [], 0, 0, []
    ns_r = ns_t = same = grams = 0
    for d in dirs:
        p = d / "continuous_report.json"
        r = json.loads(p.read_text())
        line, block, _ = report_line(p)
        items.append((line, block))
        a, b = motif_reuse(line, skip_same_note=True)
        ns_r, ns_t = ns_r + a, ns_t + b
        g = [tuple(line[i + k + 1][1] - line[i + k][1] for k in range(3)) for i in range(len(line) - 3)]
        same, grams = same + sum(1 for x in g if not any(x)), grams + len(g)
        fb += r["production"]["fallback_bar_count"]
        miss += r["scheduler_dispatch_deadline_miss_count"]
        rows.append({"run": d.name, "chord_tone": chord_tone(r), "fallback": r["production"]["fallback_bar_count"],
                     "misses": r["scheduler_dispatch_deadline_miss_count"],
                     "cache": r.get("pattern_cache_stats")})
    s = summarize(items)
    return {"motif_reuse": s["motif_reuse"], "motif_reuse_no_same_note": ns_r / ns_t if ns_t else None,
            "same_note_share": same / grams if grams else None,
            "boundary_ratio": s["boundary_ratio"], "fallback_total": fb,
            "misses_total": miss, "runs": rows}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="tatum")
    ap.add_argument("--candidate-root", type=Path, required=True)
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--same-note-guard", action="store_true",
                    help="judge reuse without (0,0,0) grams and cap the same-note share (from #1612 on)")
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
           "verdict": judge(cand, base, args.model, args.same_note_guard), "same_note_guard": args.same_note_guard,
           "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    f = lambda v: "-" if v is None else f"{v:.3f}"
    print(f"baseline motif reuse {f(base['motif_reuse'])} boundary {f(base['boundary_ratio'])} fallback {base['fallback_total']}")
    print(f"candidate reuse without same-note {f(cand['motif_reuse_no_same_note'])} same-note share {f(cand['same_note_share'])}")
    print(f"candidate motif reuse {f(cand['motif_reuse'])} boundary {f(cand['boundary_ratio'])} fallback {cand['fallback_total']} "
          f"misses {cand['misses_total']} chord-tone diff {cand['chord_tone_diff_mean']:+.3f}")
    print("verdict", out["verdict"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
