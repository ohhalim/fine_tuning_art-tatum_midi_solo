#!/usr/bin/env python3
"""V1b: song-grouped cross-validated classifier, Mehldau vs generic jazz chunks.

Pre-registered in docs/experiments/MEHLDAU_STYLE_SHIFT.md. Uses the same generic
sample as V1 (seed 0, 200 songs) and the same six histograms. Writes the
classifier fitted on all data so V2 can score generations with it.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

import numpy as np

from scripts.style_distance import LogisticModel, feature_counts, feature_vector, tokens_to_notes
from scripts.validate_style_distance import MAIN_REPO, load, seq_hash


def chunks(tokens: np.ndarray, length: int, cap: int) -> list[np.ndarray]:
    out = [tokens[s : s + length] for s in range(0, len(tokens) - length + 1, length)]
    return out[:cap] if out else [tokens]


def balanced_acc(y, p) -> float:
    pred = p >= 0.5
    return float((np.mean(pred[y == 1]) + np.mean(~pred[y == 0])) / 2)


def auc(y, p) -> float:
    pos, neg = p[y == 1], p[y == 0]
    return float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mehldau-dir", type=Path, default=MAIN_REPO / "data/mehldau_full")
    ap.add_argument("--jazz-dir", type=Path, default=MAIN_REPO / "data/jazz_full")
    ap.add_argument("--n-generic", type=int, default=200)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--mehldau-cap", type=int, default=8)
    ap.add_argument("--generic-cap", type=int, default=2)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model-output", type=Path, required=True)
    args = ap.parse_args(argv)

    mehldau = [load(f) for split in ("train", "val")
               for f in sorted((args.mehldau_dir / split).glob("*.npy"))]
    m_hashes = {seq_hash(t) for t in mehldau}
    located = {split: [f.name for f in sorted((args.jazz_dir / split).glob("*.npy"))
                       if seq_hash(load(f)) in m_hashes] for split in ("train", "val")}
    # Same generic sample as V1: seed 0 over jazz_full/train minus Mehldau.
    pool_files = [f for f in sorted((args.jazz_dir / "train").glob("*.npy"))
                  if f.name not in set(located["train"])]
    generic_files = random.Random(args.seed).sample(pool_files, args.n_generic)

    X, y, song = [], [], []
    for i, t in enumerate(mehldau):
        for c in chunks(t, args.chunk, args.mehldau_cap):
            X.append(feature_vector(feature_counts(tokens_to_notes(c)))); y.append(1); song.append(f"m{i}")
    for f in generic_files:
        for c in chunks(load(f), args.chunk, args.generic_cap):
            X.append(feature_vector(feature_counts(tokens_to_notes(c)))); y.append(0); song.append(f"g{f.stem}")
    X, y, song = np.array(X), np.array(y), np.array(song)

    songs = sorted(set(song))
    rng = random.Random(args.seed)
    m_songs = [s for s in songs if s.startswith("m")]
    g_songs = [s for s in songs if s.startswith("g")]
    rng.shuffle(m_songs); rng.shuffle(g_songs)
    fold_of = {s: i % args.folds for i, s in enumerate(m_songs)}
    fold_of.update({s: i % args.folds for i, s in enumerate(g_songs)})
    folds = np.array([fold_of[s] for s in song])
    p_cv = np.zeros(len(y))
    for k in range(args.folds):
        tr, te = folds != k, folds == k
        p_cv[te] = LogisticModel(lam=args.lam).fit(X[tr], y[tr]).predict_proba(X[te])

    res = {"schema": "style_classifier_validity_v1", "mehldau_located_in_jazz_full": located,
           "n_chunks": {"mehldau": int(y.sum()), "generic": int((1 - y).sum())},
           "n_songs": {"mehldau": len(m_songs), "generic": len(g_songs)},
           "chunk_balanced_acc": balanced_acc(y, p_cv), "chunk_auc": auc(y, p_cv),
           "mehldau_chunk_recall": float(np.mean(p_cv[y == 1] >= 0.5)),
           "generic_chunk_specificity": float(np.mean(p_cv[y == 0] < 0.5)),
           "mean_p_mehldau": {"mehldau": float(p_cv[y == 1].mean()), "generic": float(p_cv[y == 0].mean())},
           "per_mehldau_song_mean_p": {s: float(p_cv[song == s].mean()) for s in sorted(m_songs)}}
    res["verdict"] = {"threshold": 0.75, "passed": res["chunk_balanced_acc"] >= 0.75}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(res, indent=2) + "\n")
    final = LogisticModel(lam=args.lam).fit(X, y)
    args.model_output.write_text(json.dumps({"schema": "style_classifier_model_v1",
                                             "chunk": args.chunk, **final.to_dict()}) + "\n")
    print(json.dumps({k: res[k] for k in ("n_chunks", "chunk_balanced_acc", "chunk_auc",
                                          "mehldau_chunk_recall", "generic_chunk_specificity",
                                          "mean_p_mehldau", "verdict", "mehldau_located_in_jazz_full")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
