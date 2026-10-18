#!/usr/bin/env python3
"""Rough per-note gain from the condition PCs with a PC-only model (issue #1760, docs/experiments/COND_ADAPTER_MIN.md,
diagnostic 2). Data only, no Aria.

A: PC distribution of the window notes of the 8 training songs. B: A times alpha on the condition PCs, renormalized;
alpha fitted by maximum likelihood on the training songs (grid 1.00..4.00). Reported: mean log-likelihood gain per
window note (B - A, nats) on the 4 development songs and on all 189 songs, with the real and with the wrong set.

  cond_pc_gain.py <#1756 export dir> <split json> <report json>
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cond_adapter_data import load_song, song_examples, wrong_payload  # noqa: E402


def items(export_dir, song_idxs):
    """(window PCs, real set, wrong set) per request with a defined wrong set."""
    out = []
    for idx in song_idxs:
        exs, _ = song_examples(*load_song(export_dir, idx), song=idx)
        for ex in exs:
            w = wrong_payload(ex["payload"])
            if w is None or not ex["window_pcs"]:
                continue
            out.append((ex["window_pcs"], {c for c in range(12) if ex["payload"][0][c]}, {c for c in range(12) if w[0][c]}))
    return out


def loglik(data, base, alpha, which):
    total, n = 0.0, 0
    for pcs, real, wrong in data:
        s = real if which == "real" else wrong
        z = sum(base[c] * (alpha if c in s else 1.0) for c in range(12))
        for p in pcs:
            total += math.log(base[p] * (alpha if p in s else 1.0) / z)
            n += 1
    return total, n


def main() -> None:
    export_dir, split_path, report_path = sys.argv[1:4]
    split = json.load(open(split_path))
    train = items(export_dir, [s["idx"] for s in split["train"]])
    counts = [1e-9] * 12
    for pcs, _, _ in train:
        for p in pcs:
            counts[p] += 1
    tot = sum(counts)
    base = [c / tot for c in counts]
    grid = [1 + i / 100 for i in range(301)]
    alpha = max(grid, key=lambda a: loglik(train, base, a, "real")[0])
    res = {}
    for name, idxs in (("dev", [s["idx"] for s in split["dev"]]), ("all189", list(range(189)))):
        data = items(export_dir, idxs)
        la, n = loglik(data, base, 1.0, "real")
        lb, _ = loglik(data, base, alpha, "real")
        lw, _ = loglik(data, base, alpha, "wrong")
        res[name] = {"requests": len(data), "notes": n, "gain_real_nats_per_note": round((lb - la) / n, 5),
                     "gain_wrong_nats_per_note": round((lw - la) / n, 5)}
    report = {"issue": 1760, "step": "diagnostic 2", "alpha": alpha, "train_requests": len(train), "results": res}
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
