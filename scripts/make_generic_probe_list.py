#!/usr/bin/env python3
"""Pick generic jazz reference/probe files from jazz_full/train, excluding target artists.

Excludes every jazz_full file whose token sequence equals a song in any of the
given artist datasets (train+val). Same sampling as V1: seed 0, first half
reference, second half probe.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.validate_style_distance import MAIN_REPO, load, seq_hash


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exclude-dataset", type=Path, action="append", required=True)
    ap.add_argument("--jazz-dir", type=Path, default=MAIN_REPO / "data/jazz_full/train")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    excluded_hashes = {seq_hash(load(f)) for d in args.exclude_dataset
                       for split in ("train", "val") for f in (d / split).glob("*.npy")}
    files = sorted(args.jazz_dir.glob("*.npy"))
    excluded = [f.name for f in files if seq_hash(load(f)) in excluded_hashes]
    pool = [f for f in files if f.name not in set(excluded)]
    picked = random.Random(args.seed).sample(pool, args.n)
    half = args.n // 2
    out = {"schema": "generic_probe_list_v1", "jazz_dir": str(args.jazz_dir),
           "excluded_datasets": [str(d) for d in args.exclude_dataset],
           "jazz_excluded": excluded, "seed": args.seed,
           "generic_reference_files": [f.name for f in picked[:half]],
           "generic_probe_files": [f.name for f in picked[half:]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(f"excluded {len(excluded)}, pool {len(pool)}, picked {len(picked)} -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
