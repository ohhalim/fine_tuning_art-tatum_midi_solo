#!/usr/bin/env python3
"""V1: can the descriptor distance tell Mehldau from generic jazz at all?

Pre-registered in docs/experiments/MEHLDAU_STYLE_SHIFT.md. Primary unit is one
deterministic 1024-token chunk per song (the middle of the song), because
generated sequences are about that long. Whole-song accuracy is reported too.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

import numpy as np

from scripts.style_distance import (FEATURES, distance, feature_counts, per_feature_distance,
                                    pool, tokens_to_notes)

MAIN_REPO = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def load(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=True).ravel().astype(np.int64)


def seq_hash(tokens: np.ndarray) -> str:
    return hashlib.sha1(tokens.astype(np.int64).tobytes()).hexdigest()


def middle_chunk(tokens: np.ndarray, length: int) -> np.ndarray:
    start = max(0, (len(tokens) - length) // 2)
    return tokens[start : start + length]


def classify(probes_m, probes_g, ref_g, features=FEATURES):
    """Leave-one-out for Mehldau probes; generic probes against all Mehldau."""
    ref_m_all = pool(probes_m)
    hits_m = []
    for i, c in enumerate(probes_m):
        ref_m = pool(x for j, x in enumerate(probes_m) if j != i)
        hits_m.append(distance(c, ref_m, features) < distance(c, ref_g, features))
    hits_g = [distance(c, ref_g, features) < distance(c, ref_m_all, features) for c in probes_g]
    acc_m, acc_g = float(np.mean(hits_m)), float(np.mean(hits_g))
    return {"mehldau_acc": acc_m, "generic_acc": acc_g, "balanced_acc": (acc_m + acc_g) / 2,
            "n_mehldau": len(hits_m), "n_generic": len(hits_g)}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mehldau-dir", type=Path, default=MAIN_REPO / "data/mehldau_full")
    p.add_argument("--jazz-dir", type=Path, default=MAIN_REPO / "data/jazz_full/train")
    p.add_argument("--n-generic", type=int, default=200)
    p.add_argument("--chunk", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)

    mehldau = [load(f) for split in ("train", "val")
               for f in sorted((args.mehldau_dir / split).glob("*.npy"))]
    m_hashes = {seq_hash(t) for t in mehldau}
    jazz_files = sorted(args.jazz_dir.glob("*.npy"))
    excluded = [f.name for f in jazz_files if seq_hash(load(f)) in m_hashes]
    pool_files = [f for f in jazz_files if f.name not in set(excluded)]
    rng = random.Random(args.seed)
    picked = rng.sample(pool_files, args.n_generic)
    half = args.n_generic // 2
    ref_files, probe_files = picked[:half], picked[half:]

    out = {"schema": "style_distance_validity_v1", "seed": args.seed, "chunk": args.chunk,
           "mehldau_songs": len(mehldau), "jazz_excluded_as_mehldau": excluded,
           "generic_reference_files": [f.name for f in ref_files],
           "generic_probe_files": [f.name for f in probe_files]}
    for unit in ("chunk", "song"):
        def feats(tokens):
            t = middle_chunk(tokens, args.chunk) if unit == "chunk" else tokens
            return feature_counts(tokens_to_notes(t))
        probes_m = [feats(t) for t in mehldau]
        ref_g = pool(feats(load(f)) for f in ref_files)
        probes_g = [feats(load(f)) for f in probe_files]
        res = {"all_features": classify(probes_m, probes_g, ref_g)}
        res["per_feature_exploratory"] = {
            f: classify(probes_m, probes_g, ref_g, (f,))["balanced_acc"] for f in FEATURES}
        res["reference_gap_per_feature"] = per_feature_distance(pool(probes_m), ref_g)
        out[unit] = res
        print(unit, json.dumps(res["all_features"]), flush=True)
        print("  per feature", {k: round(v, 3) for k, v in res["per_feature_exploratory"].items()})
    passed = out["chunk"]["all_features"]["balanced_acc"] >= 0.75
    out["verdict"] = {"threshold": 0.75, "primary_unit": "chunk", "passed": passed}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("verdict", out["verdict"], "excluded", len(excluded))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
