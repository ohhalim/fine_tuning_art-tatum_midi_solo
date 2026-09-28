#!/usr/bin/env python3
"""Derived personalisation indices from the 3x3 cross evaluation (TATUM_VS_MEHLDAU.md §4).

For each adapter: self gain (own-artist dCE), specialisation (self - generic),
and specificity (self - other artist). CIs come from resampling songs of each
column independently (the columns have different songs). Macro (song-level)
deltas are used so each song weighs equally.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def diff_ci(a, b, iters: int = 2000, seed: int = 0) -> list[float]:
    """95% CI of mean(a) - mean(b), resampling a and b independently."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ma = a[rng.integers(0, len(a), size=(iters, len(a)))].mean(1)
    mb = b[rng.integers(0, len(b), size=(iters, len(b)))].mean(1)
    d = ma - mb
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def indices(cols: dict, self_col: str, other_col: str) -> dict:
    s, o, g = (cols[c]["per_song_d_ce"] for c in (self_col, other_col, "generic"))
    return {"self_gain_macro": float(np.mean(s)), "self_gain_token": cols[self_col]["d_token_ce"],
            "self_gain_positive": cols[self_col]["d_token_ce"] < 0 and float(np.mean(s)) < 0,
            "specialisation_macro": float(np.mean(s) - np.mean(g)), "specialisation_ci95": diff_ci(s, g),
            "specificity_macro": float(np.mean(s) - np.mean(o)), "specificity_ci95": diff_ci(s, o),
            "other_gain_macro": float(np.mean(o)), "generic_gain_macro": float(np.mean(g))}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cross", type=Path, required=True)
    ap.add_argument("--tatum-col", default="tatum_fresh12")
    ap.add_argument("--mehldau-col", default="mehldau_val2")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    models = json.loads(args.cross.read_text())["models"]
    out = {"schema": "cross_artist_indices_v1", "columns": {"tatum": args.tatum_col, "mehldau": args.mehldau_col},
           "tatum_adapter": indices(models["tatum_adapter"]["columns"], args.tatum_col, args.mehldau_col),
           "mehldau_adapter": indices(models["mehldau_adapter"]["columns"], args.mehldau_col, args.tatum_col)}
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for k in ("tatum_adapter", "mehldau_adapter"):
        v = out[k]
        print(k, f"self {v['self_gain_macro']:+.4f} spec {v['specialisation_macro']:+.4f} "
              f"[{v['specialisation_ci95'][0]:+.4f},{v['specialisation_ci95'][1]:+.4f}] "
              f"specificity {v['specificity_macro']:+.4f} "
              f"[{v['specificity_ci95'][0]:+.4f},{v['specificity_ci95'][1]:+.4f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
