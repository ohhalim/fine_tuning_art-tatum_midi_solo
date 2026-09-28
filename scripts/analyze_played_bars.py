#!/usr/bin/env python3
"""Per-bar observations of what the continuous runtime actually played.

Reads ``continuous_report.json`` files (with ``played_bars``) and records, per
run: fallback ratio, generation latency, empty bars, bars with a silent span of
at least half a bar (gaps), exact and transposed bar repeats, cross-bar pitch
4-gram reuse, and pitch jumps across bar boundaries. Observations only: no
quality, style or listening claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np


def max_silence(notes, bar_s: float) -> float:
    """Longest span inside [0, bar_s] with no sounding note."""
    spans = sorted((max(0.0, s), min(bar_s, e)) for _, s, e in notes if e > 0 and s < bar_s)
    longest, cursor = 0.0, 0.0
    for s, e in spans:
        if s > cursor:
            longest = max(longest, s - cursor)
        cursor = max(cursor, e)
    return max(longest, bar_s - cursor)


def bar_features(played_bars, bar_s: float) -> dict:
    seqs = [[p for p, _, _ in b["notes"]] for b in played_bars]
    empty = sum(1 for s in seqs if not s)
    gaps = sum(1 for b in played_bars if max_silence(b["notes"], bar_s) >= bar_s / 2)
    exact, transposed, seen, seen_iv = 0, 0, [], []
    reuse = []
    grams_seen: set = set()
    for s in seqs:
        iv = tuple(b - a for a, b in zip(s, s[1:]))
        if s and tuple(s) in seen:
            exact += 1
        elif len(s) > 1 and iv in seen_iv:
            transposed += 1
        grams = {tuple(s[i:i + 4]) for i in range(len(s) - 3)}
        if grams and grams_seen:
            reuse.append(len(grams & grams_seen) / len(grams))
        grams_seen |= grams
        if s:
            seen.append(tuple(s))
        if len(s) > 1:
            seen_iv.append(iv)
    jumps = [abs(b[0] - a[-1]) for a, b in zip(seqs, seqs[1:]) if a and b]
    means = [float(np.mean(s)) for s in seqs if s]
    return {"bars": len(seqs), "notes_per_bar": [len(s) for s in seqs],
            "empty_bars": empty, "half_bar_gap_bars": gaps,
            "exact_repeat_bars": exact, "transposed_repeat_bars": transposed,
            "cross_bar_4gram_reuse_mean": float(np.mean(reuse)) if reuse else None,
            "boundary_jump_median": float(statistics.median(jumps)) if jumps else None,
            "boundary_jump_max": max(jumps) if jumps else None,
            "bar_mean_pitch_sd": float(np.std(means)) if means else None}


def run_row(report: dict) -> dict:
    prod = report["production"]
    bar_s = 240.0 / report["bpm"]
    steady = [b["generation_ms"] for b in report["bars_detail"]
              if b["bar_index"] > 0 and b.get("generation_ms") is not None]
    row = {"bars": report["bars"], "completed_bars": report["completed_bars"],
           "fallback_bars": prod["fallback_bar_count"],
           "fallback_ratio": prod["fallback_bar_count"] / report["bars"],
           "errors": prod["error_count"], "deadline_misses": report["scheduler_dispatch_deadline_miss_count"],
           "gen_ms_p50": float(np.median(steady)) if steady else None,
           "gen_ms_p95": float(np.percentile(steady, 95)) if steady else None,
           "gen_ms_max": float(max(steady)) if steady else None,
           "played_notes": report.get("played_note_count")}
    if "played_bars" in report:
        row.update(bar_features(report["played_bars"], bar_s))
    return row


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", type=Path, action="append", required=True, metavar="MODE=DIR")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"schema": "played_bars_v1", "musical_quality_verified": False, "style_verified": False,
           "listening_done": False, "runs": []}
    for spec in args.sweep:
        mode, d = str(spec).split("=", 1)
        for rep in sorted(Path(d).glob("*/bpm*_seed*/continuous_report.json")):
            model = rep.parent.parent.name
            seed = int(rep.parent.name.split("seed")[-1])
            out["runs"].append({"mode": mode, "model": model, "seed": seed,
                                **run_row(json.loads(rep.read_text()))})
    agg = {}
    for r in out["runs"]:
        agg.setdefault((r["mode"], r["model"]), []).append(r)
    out["summary"] = []
    for (mode, model), rows in sorted(agg.items()):
        def mean(k):
            vals = [r[k] for r in rows if r.get(k) is not None]
            return float(np.mean(vals)) if vals else None
        out["summary"].append({"mode": mode, "model": model, "runs": len(rows),
                               **{k: mean(k) for k in ("fallback_ratio", "gen_ms_p50", "gen_ms_p95",
                                                       "gen_ms_max", "empty_bars", "half_bar_gap_bars",
                                                       "exact_repeat_bars", "transposed_repeat_bars",
                                                       "cross_bar_4gram_reuse_mean", "boundary_jump_median",
                                                       "bar_mean_pitch_sd", "played_notes")},
                               "errors": sum(r["errors"] for r in rows),
                               "deadline_misses": sum(r["deadline_misses"] for r in rows)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for s in out["summary"]:
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
