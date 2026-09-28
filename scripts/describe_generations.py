#!/usr/bin/env python3
"""Descriptive table for generated token sequences (TATUM_VS_MEHLDAU.md §5).

Validity (grammar, empty, notes open at the end, duplicate note-on), exact
16-gram copy against each artist's training songs, and note/rhythm/register
statistics. Descriptive only: not a quality or style judgement.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))


def cluster_sizes(starts, window_s: float) -> list[int]:
    """Notes per onset cluster, clustered like ``measure_run_density.cluster_onsets``
    (a new cluster starts once a note is ``window_s`` or more after the cluster's first onset)."""
    sizes, first = [], None
    for s in sorted(starts):
        if first is None or s - first >= window_s:
            sizes.append(1)
            first = s
        else:
            sizes[-1] += 1
    return sizes


def describe_tokens(tokens, train_sets: dict[str, set]) -> dict:
    from scripts.eval_mehldau_snapshots import copy_rate, free_generation_validity
    from scripts.measure_run_density import CLUSTER_S, cluster_onsets, run_stats
    from scripts.style_distance import tokens_to_notes

    v = free_generation_validity(tokens)
    notes = tokens_to_notes(tokens)
    starts = [n.start for n in notes]
    onsets = cluster_onsets(starts)
    gaps = [b - a for a, b in zip(onsets, onsets[1:])]
    pitches = [n.pitch for n in notes]
    rs = run_stats(starts)
    sizes = cluster_sizes(starts, CLUSTER_S)
    return {
        **v, "empty": len(notes) == 0,
        **{f"copy16_{k}": copy_rate(tokens, s, 16) for k, s in train_sets.items()},
        "notes": len(notes),
        "pitch_min": min(pitches) if pitches else None, "pitch_max": max(pitches) if pitches else None,
        "pitch_range": (max(pitches) - min(pitches)) if pitches else None,
        "ioi_median_ms": round(statistics.median(gaps) * 1000, 1) if gaps else None,
        # share of onset clusters holding 2+ notes (a chord onset)
        "chord_onset_share": (sum(n >= 2 for n in sizes) / len(sizes)) if sizes else None,
        # share of notes that join an earlier onset's cluster (was mislabelled
        # chord_onset_share before the #1500 review)
        "simultaneous_note_share": (1 - len(onsets) / len(starts)) if starts else None,
        "run_ratio": rs["run_ratio"], "onsets_per_s": rs["onsets_per_s"],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=TOKENS_JSON:UPDATE")
    ap.add_argument("--train-set", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    from scripts.eval_mehldau_snapshots import ngram_set
    from scripts.validate_style_distance import load

    train_sets = {}
    for spec in args.train_set:
        name, d = spec.split("=", 1)
        grams = set()
        for f in sorted(Path(d).glob("*.npy")):
            grams |= ngram_set(load(f), 16)
        train_sets[name] = grams
    out = {"schema": "generation_descriptors_v1", "musical_quality_verified": False,
           "style_verified": False, "models": {}}
    for spec in args.model:
        name, rest = spec.split("=", 1)
        path, update = rest.rsplit(":", 1)
        seqs = json.loads(Path(path).read_text())[update]
        rows = [describe_tokens(t, train_sets) for t in seqs]
        # Keys from every row: an empty first sample (None stats) must not drop
        # the statistics of the valid samples after it.
        numeric = sorted({k for r in rows for k, v in r.items()
                          if isinstance(v, (int, float)) and not isinstance(v, bool)})
        summary = {k: statistics.mean(r[k] for r in rows if r[k] is not None)
                   for k in numeric if any(r[k] is not None for r in rows)}
        summary["grammar_valid"] = sum(r["grammar_valid"] for r in rows)
        summary["empty"] = sum(r["empty"] for r in rows)
        summary["sequences"] = len(rows)
        for k in train_sets:
            summary[f"copy16_{k}_max"] = max((r[f"copy16_{k}"] or 0) for r in rows)
        out["models"][name] = {"source": path, "update": int(update), "summary": summary, "per_sequence": rows}
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summary.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
