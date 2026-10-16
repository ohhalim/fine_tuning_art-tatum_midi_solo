#!/usr/bin/env python3
"""Features of the reference clips, liked vs not (issue #1762, docs/experiments/REFERENCE_CLIPS_DECISION.md).
Descriptive only: medians and ranges, no test.

  reference_clip_features.py <reference_candidates dir> <out json>
"""
from __future__ import annotations

import json
import os
import statistics
import sys

LIKED = ("01", "04", "06", "13", "16", "17", "18", "22")
NOT = ("02", "05", "07", "08", "09", "10", "11", "12", "14", "15")
LOW, HIGH, CHORD_MS, CHORD_MIN, LINE_GAP_MS = 48, 84, 50.0, 3, 250.0


def groups(onsets):
    """Onset groups: notes starting within CHORD_MS of the group's first onset. onsets: sorted (ms, pitch)."""
    out, cur = [], []
    for on, p in onsets:
        if cur and on - cur[0][0] > CHORD_MS:
            out.append(cur)
            cur = []
        cur.append((on, p))
    if cur:
        out.append(cur)
    return out


def features(notes, seconds):
    """notes: [(pitch, on_ms, off_ms)]."""
    ons = sorted((on, p) for p, on, _ in notes)
    gs = groups(ons)
    chords = [g for g in gs if len(g) >= CHORD_MIN]
    line = [g[0] for g in gs if len(g) < CHORD_MIN]          # single notes and dyads (their first note); chords left out
    runs, cur = [], 1
    for a, b in zip(line, line[1:]):
        if b[0] - a[0] <= LINE_GAP_MS:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    if line:
        runs.append(cur)
    steps = [abs(b[1] - a[1]) for a, b in zip(line, line[1:]) if b[0] - a[0] <= LINE_GAP_MS]
    s = sorted(runs)
    return {"notes": len(notes), "low_share": round(sum(p < LOW for _, p in ons) / len(ons), 4),
            "high_share": round(sum(p >= HIGH for _, p in ons) / len(ons), 4),
            "comping_per_s": round(len(chords) / seconds, 3),
            "lick_len_median": statistics.median(s) if s else 0, "lick_len_p90": s[min(len(s) - 1, int(0.9 * len(s)))] if s else 0,
            "semitone_step_share": round(sum(x == 1 for x in steps) / len(steps), 4) if steps else None,
            "notes_per_s": round(len(notes) / seconds, 3)}


def load(path):
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    notes = [(int(n.pitch), float(n.start) * 1000, float(n.end) * 1000) for i in pm.instruments if not i.is_drum for n in i.notes]
    seconds = max(o for _, _, o in notes) / 1000 - min(on for _, on, _ in notes) / 1000
    return notes, seconds


def main() -> None:
    clip_dir, out_path = sys.argv[1:3]
    per = {}
    for cid in LIKED + NOT:
        notes, sec = load(os.path.join(clip_dir, f"{cid}.mid"))
        per[cid] = {"group": "liked" if cid in LIKED else "not", **features(notes, sec)}
    summary = {}
    for grp in ("liked", "not"):
        rows = [v for v in per.values() if v["group"] == grp]
        summary[grp] = {k: {"median": statistics.median(r[k] for r in rows), "min": min(r[k] for r in rows), "max": max(r[k] for r in rows)}
                        for k in ("low_share", "high_share", "comping_per_s", "lick_len_median", "lick_len_p90", "semitone_step_share", "notes_per_s")}
    out = {"issue": 1762, "params": {"low": LOW, "high": HIGH, "chord_ms": CHORD_MS, "chord_min": CHORD_MIN, "line_gap_ms": LINE_GAP_MS},
           "clips": per, "summary": summary}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
