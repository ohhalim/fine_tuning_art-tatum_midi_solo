#!/usr/bin/env python3
"""Copy a tokenized train/val dataset, dropping every sequence found in the excluded datasets.

Used to build a base pretrain set that has never seen the adaptation target
(docs/experiments/MEHLDAU_CLEAN_BASE.md). Matching is by exact token-sequence hash.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_style_distance import load, seq_hash


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--exclude-dataset", type=Path, action="append", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        ap.error(f"output not empty: {args.output}")
    excluded = {seq_hash(load(f)) for d in args.exclude_dataset
                for split in ("train", "val") for f in (d / split).glob("*.npy")}
    summary = {"source": str(args.source), "excluded_datasets": [str(d) for d in args.exclude_dataset],
               "excluded_hashes": len(excluded), "splits": {}}
    for split in ("train", "val"):
        (args.output / split).mkdir(parents=True, exist_ok=True)
        kept, dropped = 0, []
        for f in sorted((args.source / split).glob("*.npy")):
            if seq_hash(load(f)) in excluded:
                dropped.append(f.name)
                continue
            shutil.copy2(f, args.output / split / f.name)
            kept += 1
        summary["splits"][split] = {"kept": kept, "dropped": dropped}
    (args.output / "manifest.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({s: {"kept": v["kept"], "dropped": len(v["dropped"])}
                      for s, v in summary["splits"].items()}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
