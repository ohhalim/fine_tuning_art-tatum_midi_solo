#!/usr/bin/env python3
"""Data splits for the Tatum vs Mehldau comparison (docs/experiments/TATUM_VS_MEHLDAU.md §2)."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.validate_style_distance import load, seq_hash

MAIN = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def pick(names: list[str], k: int, seed: int) -> list[str]:
    return sorted(random.Random(seed).sample(sorted(names), k))


def copy(files, dest: Path) -> list[dict]:
    dest.mkdir(parents=True, exist_ok=True)
    out = []
    for f in files:
        shutil.copy2(f, dest / f.name)
        out.append({"file": f.name, "source": str(f), "sha1": seq_hash(load(f)), "tokens": int(len(load(f)))})
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tatum", type=Path, default=ROOT / "data/tatum_full")
    ap.add_argument("--mehldau", type=Path, default=MAIN / "data/mehldau_full")
    ap.add_argument("--output", type=Path, default=ROOT / "data/tvm")
    args = ap.parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        ap.error(f"output not empty: {args.output}")
    t_train = sorted((args.tatum / "train").glob("*.npy"))
    names = [f.name for f in t_train]
    train16 = pick(names, 16, 0)
    fresh12 = pick([n for n in names if n not in set(train16)], 12, 1)
    by = {f.name: f for f in t_train}
    tatum_manifest = json.loads((args.tatum / "manifest.json").read_text())
    src = {s["file"]: s["source"] for s in tatum_manifest["songs"]}
    m = {"schema": "tvm_splits_v1", "tatum_train16_seed": 0, "tatum_fresh12_seed": 1, "sets": {}}
    m["sets"]["tatum16_train"] = copy([by[n] for n in train16], args.output / "tatum16" / "train")
    m["sets"]["holdout_tatum_fresh12"] = copy([by[n] for n in fresh12], args.output / "tatum16" / "val")
    copy([by[n] for n in fresh12], args.output / "holdout_tatum_fresh12" / "val")
    m["sets"]["holdout_tatum_val12"] = copy(sorted((args.tatum / "val").glob("*.npy")),
                                           args.output / "holdout_tatum_val12" / "val")
    m["sets"]["mehldau16_train"] = copy(sorted((args.mehldau / "train").glob("*.npy")),
                                        args.output / "mehldau16" / "train")
    m["sets"]["holdout_mehldau_val2"] = copy(sorted((args.mehldau / "val").glob("*.npy")),
                                            args.output / "mehldau16" / "val")
    copy(sorted((args.mehldau / "val").glob("*.npy")), args.output / "holdout_mehldau_val2" / "val")
    for key in ("tatum16_train", "holdout_tatum_fresh12", "holdout_tatum_val12"):
        for row in m["sets"][key]:
            row["title"] = src.get(f"{'val' if key == 'holdout_tatum_val12' else 'train'}/{row['file']}")
    m["summary"] = {k: {"songs": len(v), "tokens": sum(r["tokens"] for r in v)} for k, v in m["sets"].items()}
    (args.output / "manifest.json").write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(m["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
