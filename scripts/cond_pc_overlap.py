#!/usr/bin/env python3
"""Do the condition PCs overlap the window's right-hand PCs more than a wrong (tritone) set does? (issue #1760,
docs/experiments/COND_ADAPTER_MIN.md, diagnostic). Data only, no model, no verdict.

  cond_pc_overlap.py <#1756 export dir> <report json>
"""
from __future__ import annotations

import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cond_adapter_data import load_song, song_examples, wrong_payload  # noqa: E402


def main() -> None:
    export_dir, report_path = sys.argv[1:3]
    acc = collections.defaultdict(lambda: {"requests": 0, "notes": 0, "in_real": 0, "in_wrong": 0, "chance": 0.0,
                                           "req_real": 0.0, "req_wrong": 0.0})
    for idx in range(189):
        tokens, ids, hum, t0 = load_song(export_dir, idx)
        exs, _ = song_examples(tokens, ids, hum, t0, song=idx)
        for ex in exs:
            w = wrong_payload(ex["payload"])
            if w is None:
                continue
            real = {c for c in range(12) if ex["payload"][0][c]}
            wrong = {c for c in range(12) if w[0][c]}
            pcs = ex["window_pcs"]
            if not pcs:
                continue
            key = "1" if len(real) == 1 else "2" if len(real) == 2 else "3+"
            for k in (key, "all"):
                a = acc[k]
                a["requests"] += 1
                a["notes"] += len(pcs)
                a["in_real"] += sum(p in real for p in pcs)
                a["in_wrong"] += sum(p in wrong for p in pcs)
                a["chance"] += len(real) / 12 * len(pcs)
                a["req_real"] += sum(p in real for p in pcs) / len(pcs)
                a["req_wrong"] += sum(p in wrong for p in pcs) / len(pcs)
    out = {}
    for k, a in acc.items():
        out[k] = {"requests": a["requests"], "notes": a["notes"],
                  "pooled_in_real": round(a["in_real"] / a["notes"], 4), "pooled_in_wrong": round(a["in_wrong"] / a["notes"], 4),
                  "pooled_chance": round(a["chance"] / a["notes"], 4),
                  "request_mean_in_real": round(a["req_real"] / a["requests"], 4),
                  "request_mean_in_wrong": round(a["req_wrong"] / a["requests"], 4)}
    report = {"issue": 1760, "step": "diagnostic", "doc": "docs/experiments/COND_ADAPTER_MIN.md", "by_set_size": out}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
