#!/usr/bin/env python3
"""Pair check for the #1819 v2 note-cap runs (NOTE_CAP.md 11–12절, Astra review of #1835).

For each seed x scenario pair (nocap vs cap12, deterministic left hand), compare where the two arms first differ against
where the cap first set a floor (first block with capped_samples > 0):
  - block rows: (tokens, notes, S) per warm block of producer.json
  - sent note-ons: (pitch, velocity, due_ms) in order from sender.json; first differing note-on's block
Token-level identity is not checked (only the final committed token hash is kept).

  note_cap_pairs.py <run dir> <out json>
"""
from __future__ import annotations

import json
import os
import sys


def first_block_diff(a_blocks, b_blocks):
    for x, y in zip(a_blocks, b_blocks):
        if (x["tokens"], x.get("notes"), x["S"]) != (y["tokens"], y.get("notes"), y["S"]):
            return x["block"]
    return None


def note_ons(snd):
    return [(r["pitch"], r["velocity"], r["due_ms"], r.get("block")) for r in snd["sent"] if r["kind"] == "on"]


def first_note_on_diff(a, b):
    for x, y in zip(a, b):
        if x[:3] != y[:3]:
            return min(v for v in (x[3], y[3]) if v is not None) if (x[3] is not None or y[3] is not None) else None
    return None


def pair(run_dir, seed, scenario):
    out = {}
    for arm in ("nocap", "cap12"):
        d = os.path.join(run_dir, f"v2_{arm}_s{seed}", scenario)
        out[arm] = (json.load(open(os.path.join(d, "producer.json"))), json.load(open(os.path.join(d, "sender.json"))))
    (pa, sa), (pb, sb) = out["nocap"], out["cap12"]
    first_cap = next((y["block"] for y in pb["blocks"] if y.get("capped_samples")), None)
    fb, fn = first_block_diff(pa["blocks"], pb["blocks"]), first_note_on_diff(note_ons(sa), note_ons(sb))
    ok = lambda f: f is None or (first_cap is not None and f >= first_cap)  # noqa: E731
    return {"seed": seed, "scenario": scenario, "first_cap_block": first_cap, "first_block_row_diff": fb,
            "first_note_on_diff_block": fn, "p0_block_rows": ok(fb), "p0_note_on": ok(fn)}


def main() -> None:
    run_dir, out_path = sys.argv[1:3]
    rows = [pair(run_dir, s, sc) for s in range(1, 9) for sc in ("A_normal", "B_crop_early")]
    rep = {"pairs": rows, "p0_block_rows_all": all(r["p0_block_rows"] for r in rows),
           "p0_note_on_all": all(r["p0_note_on"] for r in rows)}
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "pairs"}))


if __name__ == "__main__":
    main()
