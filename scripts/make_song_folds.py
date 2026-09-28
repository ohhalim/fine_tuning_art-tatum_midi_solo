#!/usr/bin/env python3
"""Split a tokenized train set into song-level K folds (fold k: held-out songs as val).

The original val split is left out on purpose so it stays unused by any selection.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path


def fold_assignment(names: list[str], k: int, seed: int) -> dict[str, int]:
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    return {name: i % k for i, name in enumerate(ordered)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        ap.error(f"output not empty: {args.output}")
    files = sorted(args.train_dir.glob("*.npy"))
    assign = fold_assignment([f.name for f in files], args.k, args.seed)
    for fold in range(args.k):
        for split in ("train", "val"):
            (args.output / f"fold{fold}" / split).mkdir(parents=True, exist_ok=True)
        for f in files:
            split = "val" if assign[f.name] == fold else "train"
            shutil.copy2(f, args.output / f"fold{fold}" / split / f.name)
    (args.output / "folds.json").write_text(json.dumps(
        {"train_dir": str(args.train_dir), "k": args.k, "seed": args.seed,
         "held_out": {str(i): sorted(n for n, a in assign.items() if a == i) for i in range(args.k)}},
        indent=2) + "\n")
    print({i: sum(a == i for a in assign.values()) for i in range(args.k)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
