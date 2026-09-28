#!/usr/bin/env python3
"""Share of each song's exact 16-grams found in a reference token corpus.

Catches other transcriptions / recordings of the same performance that a
whole-sequence hash misses. Report only; nothing is excluded here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.validate_style_distance import load

N = 16
_P = np.uint64(1_000_003)


def ngram_hashes(tokens: np.ndarray, n: int = N) -> np.ndarray:
    """64-bit polynomial hash of every length-n window (vectorised)."""
    t = np.asarray(tokens, dtype=np.uint64)
    if len(t) < n:
        return np.zeros(0, dtype=np.uint64)
    h = np.zeros(len(t) - n + 1, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for i in range(n):
            h = h * _P + t[i: len(t) - n + 1 + i] + np.uint64(1)
    return h


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reference", type=Path, required=True, help="dir with train/ val/ npy")
    ap.add_argument("--songs", type=Path, action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--flag", type=float, default=0.2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    ref = np.unique(np.concatenate([ngram_hashes(load(f))
                                    for f in sorted(args.reference.glob("*/*.npy"))]))
    out = {"schema": "near_duplicate_v1", "n": N, "flag": args.flag,
           "reference": str(args.reference), "reference_unique_ngrams": int(len(ref)), "sets": {}}
    for spec in args.songs:
        name, d = str(spec).split("=", 1)
        rows = []
        for f in sorted(Path(d).rglob("*.npy")):
            h = np.unique(ngram_hashes(load(f)))
            share = float(np.isin(h, ref).mean()) if len(h) else 0.0
            rows.append({"file": str(f), "ngrams": int(len(h)), "share_in_reference": share})
        shares = [r["share_in_reference"] for r in rows]
        out["sets"][name] = {"songs": len(rows), "max_share": max(shares), "mean_share": float(np.mean(shares)),
                             "flagged": [r for r in rows if r["share_in_reference"] > args.flag], "rows": rows}
        print(name, "songs", len(rows), "max", round(max(shares), 4), "mean", round(float(np.mean(shares)), 4),
              "flagged", len(out["sets"][name]["flagged"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
