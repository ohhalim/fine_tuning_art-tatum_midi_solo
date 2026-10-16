#!/usr/bin/env python3
"""Does the recent condition carry more PC information than sounding? (issue #1760, docs/experiments/COND_ADAPTER_MIN.md,
diagnostic 3). The PC-only model of cond_pc_gain.py, same requests, only the condition set changes. No Aria.

  cond_pc_gain_recent.py <#1756 export dir> <split json> <report json>
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_contract import block_start  # noqa: E402
from cond_adapter_data import load_song, song_examples  # noqa: E402
from lh_role_proxy import assign_g3, online_condition, recent_condition  # noqa: E402


def items(export_dir, song_idxs):
    """(window PCs, sounding set, recent set) per eligible warm request where both sets are non-empty."""
    out = []
    for idx in song_idxs:
        tokens, ids, hum, t0 = load_song(export_dir, idx)
        roles = assign_g3(hum)
        exs, _ = song_examples(tokens, ids, hum, t0, song=idx)
        for ex in exs:
            if not ex["window_pcs"] or ex["k"] == 0:
                continue
            s = block_start(ex["k"], t0)
            snd = {n.pitch % 12 for n in online_condition(hum, roles, s)}
            rec = {n.pitch % 12 for n in recent_condition(hum, roles, s)}
            if snd and rec:
                out.append((ex["window_pcs"], snd, rec))
    return out


def loglik(data, base, alpha, col):
    total, n = 0.0, 0
    for row in data:
        pcs, s = row[0], row[col]
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
    base = [c / sum(counts) for c in counts]
    grid = [1 + i / 100 for i in range(301)]
    alphas = {name: max(grid, key=lambda a, col=col: loglik(train, base, a, col)[0]) for name, col in (("sounding", 1), ("recent", 2))}
    res = {}
    for name, idxs in (("dev", [s["idx"] for s in split["dev"]]), ("all189", list(range(189)))):
        data = items(export_dir, idxs)
        la, n = loglik(data, base, 1.0, 1)
        res[name] = {"requests": len(data), "notes": n,
                     "mean_set_size": {"sounding": round(sum(len(r[1]) for r in data) / len(data), 3),
                                       "recent": round(sum(len(r[2]) for r in data) / len(data), 3)},
                     **{f"gain_{c}_nats_per_note": round((loglik(data, base, alphas[c], col)[0] - la) / n, 5)
                        for c, col in (("sounding", 1), ("recent", 2))}}
    report = {"issue": 1760, "step": "diagnostic 3", "alpha": alphas, "train_requests": len(train), "results": res}
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
