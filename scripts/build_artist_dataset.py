#!/usr/bin/env python3
"""Tokenize one artist's MIDI folder into a song-level train/val split.

Uses the encoder that built ``data/jazz_full`` (archive/scripts/preprocess_jazz.py)
so token sequences are directly comparable, and records for every song whether
the identical sequence is already in ``jazz_full`` (base pretrain overlap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "music_transformer", ROOT / "music_transformer" / "third_party",
          ROOT / "archive" / "scripts"):
    sys.path.insert(0, str(p))

import numpy as np

MAIN_REPO = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def seq_hash(tokens) -> str:
    return hashlib.sha1(np.asarray(tokens, dtype=np.int64).ravel().tobytes()).hexdigest()


def song_split(names: list[str], val_fraction: float, seed: int) -> set[str]:
    """Deterministic song-level val set."""
    ordered = sorted(names)
    random.Random(seed).shuffle(ordered)
    n_val = max(1, round(len(ordered) * val_fraction))
    return set(ordered[:n_val])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artist-dir", type=Path, required=True)
    ap.add_argument("--jazz-full", type=Path, default=MAIN_REPO / "data/jazz_full")
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"output dir not empty: {args.output_dir}")

    from preprocess_jazz import process_midi_file

    base_hashes = {}
    for split in ("train", "val"):
        for f in sorted((args.jazz_full / split).glob("*.npy")):
            base_hashes[seq_hash(np.load(f))] = f"{split}/{f.name}"

    files = sorted(p for ext in (".mid", ".midi") for p in args.artist_dir.rglob(f"*{ext}"))
    encoded, skipped = {}, []
    for f in files:
        tokens = process_midi_file(f)
        rel = str(f.relative_to(args.artist_dir))
        if tokens is None:
            skipped.append(rel)
        else:
            encoded[rel] = np.asarray(tokens, dtype=np.int32)
    val_names = song_split(list(encoded), args.val_fraction, args.seed)

    manifest = {"schema": "artist_dataset_v1", "artist_dir": str(args.artist_dir),
                "encoder": "archive/scripts/preprocess_jazz.py:process_midi_file",
                "val_fraction": args.val_fraction, "seed": args.seed,
                "skipped": skipped, "songs": []}
    for split in ("train", "val"):
        (args.output_dir / split).mkdir(parents=True, exist_ok=True)
    counters = {"train": 0, "val": 0}
    for rel in sorted(encoded):
        split = "val" if rel in val_names else "train"
        name = f"{counters[split]:05d}.npy"
        counters[split] += 1
        np.save(args.output_dir / split / name, encoded[rel])
        h = seq_hash(encoded[rel])
        manifest["songs"].append({"source": rel, "split": split, "file": f"{split}/{name}",
                                  "tokens": int(len(encoded[rel])), "sha1": h,
                                  "in_jazz_full": base_hashes.get(h)})
    overlap = sum(1 for s in manifest["songs"] if s["in_jazz_full"])
    manifest["summary"] = {"songs": len(encoded), "train": counters["train"], "val": counters["val"],
                           "in_jazz_full": overlap, "max_token": int(max(t.max() for t in encoded.values()))}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(manifest["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
