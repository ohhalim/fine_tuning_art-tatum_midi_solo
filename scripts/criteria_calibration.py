#!/usr/bin/env python3
"""Prefilter metrics on liked clips vs rejected outputs (issue #1762, docs/experiments/CRITERIA_CALIBRATION.md).

Descriptive only. All melodic metrics are measured on the top-line proxy (solo_line.top_notes, G3 and up);
the comping density on all notes of two-hand sets only.

  criteria_calibration.py <out json>
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..")):
    sys.path.insert(0, p)

REF = "/Users/ohhalim/git_box/listening/reference_candidates"
SETS = {
    "L": {"files": [f"{REF}/{c}.mid" for c in ("01", "04", "06", "13", "16", "17", "18", "22")], "two_hand": True},
    "N": {"files": [f"{REF}/{c}.mid" for c in ("02", "05", "07", "08", "09", "10", "11", "12", "14", "15")], "two_hand": True},
    "R1": {"files": sorted(glob.glob("/Users/ohhalim/git_box/jps_compare_run1/midi/*_gen.mid")), "two_hand": True},
    "R2": {"files": sorted(glob.glob("/Users/ohhalim/git_box/realchords_pq_run1/listening/*.mid")), "two_hand": False,
           "from_s": 60 / 128 * 4},        # bar 1 is the fixed human primer (8 notes); measure the generated bars only
}
EXPECTED = {"L": 8, "N": 10, "R1": 16, "R2": 4}
UNRATED = {"R2": ["P_s2.mid"]}             # the user did not rate P_s2; it is not counted as a rejection
SOLO_SPLIT, GAP_S, MIN_RUN, IOI_RANGE, EVEN_TOL = 55, 0.25, 8, (0.09, 0.26), 0.10
METRICS = ("P1_line_notes_per_s", "P2_lick_runs_per_10s", "P3_longest_lick_run", "P4_even_ioi_share", "P5_longest_repeat_s", "P6_comping_per_s")


def runs(line):
    out, cur = [], []
    for n in line:
        if cur and n.start - cur[-1].end > GAP_S:
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


def metrics(path, two_hand, from_s=0.0):
    import pretty_midi
    from inference.control.solo_line import top_notes
    from cmt_continuous_analysis import events, longest_repeat
    from reference_clip_features import CHORD_MIN, groups
    pm = pretty_midi.PrettyMIDI(path)
    notes = sorted((n for i in pm.instruments if not i.is_drum for n in i.notes if n.start >= from_s - 0.001),          # 1 ms: MIDI read-back puts 1.875 s at 1.874996 s
                   key=lambda n: (n.start, n.pitch))
    t0, t1 = min(n.start for n in notes), max(n.end for n in notes)
    seconds = max(t1 - t0, 1e-6)
    line = [n for n in top_notes(notes) if n.pitch >= SOLO_SPLIT]
    rs = [r for r in runs(line) if len(r) >= MIN_RUN]
    licks, even, total = [], 0, 0
    for r in rs:
        iois = [b.start - a.start for a, b in zip(r, r[1:])]
        med = statistics.median(iois)
        if IOI_RANGE[0] <= med <= IOI_RANGE[1]:
            licks.append(len(r))
        even += sum(abs(x - med) <= EVEN_TOL * med for x in iois)
        total += len(iois)
    ev = events(line)
    out = {"seconds": round(seconds, 2), "line_notes": len(line),
           "P1_line_notes_per_s": round(len(line) / seconds, 3),
           "P2_lick_runs_per_10s": round(10 * len(licks) / seconds, 3),
           "P3_longest_lick_run": max(licks, default=0),
           "P4_even_ioi_share": round(even / total, 3) if total else None,
           "P5_longest_repeat_s": round(longest_repeat(ev), 2)}
    if two_hand:
        gs = groups(sorted((n.start * 1000, n.pitch) for n in notes))
        out["P6_comping_per_s"] = round(sum(len(g) >= CHORD_MIN for g in gs) / seconds, 3)
    return out


def separation(sets_vals, metric):
    """L range, and how many items of each other set fall outside it (descriptive, rule fixed before the run)."""
    lv = [v for v in sets_vals["L"] if v is not None]
    if not lv:
        return None
    lo, hi = min(lv), max(lv)
    res = {"L_range": [lo, hi]}
    for k in ("N", "R1", "R2"):
        vs = [v for v in sets_vals.get(k, []) if v is not None]
        if vs:
            out = sum(1 for v in vs if v < lo or v > hi)
            res[k] = {"outside": out, "n": len(vs), "majority_outside": out * 2 > len(vs)}
    return res


def file_sha(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main() -> None:
    out_path = sys.argv[1]
    manifest = {}
    for name, cfg in SETS.items():
        if len(cfg["files"]) != EXPECTED[name]:
            sys.exit(f"{name}: expected {EXPECTED[name]} files, found {len(cfg['files'])}")
        manifest[name] = {"from_s": cfg.get("from_s", 0.0),
                          "files": {os.path.basename(f): file_sha(f) for f in cfg["files"]}}
    per, vals = {}, {}
    for name, cfg in SETS.items():
        per[name] = {os.path.basename(f): metrics(f, cfg["two_hand"], cfg.get("from_s", 0.0)) for f in cfg["files"]}
        rated = {k: v for k, v in per[name].items() if k not in UNRATED.get(name, [])}
        vals[name] = {m: [r.get(m) for r in rated.values()] for m in METRICS}
    summary = {}
    for m in METRICS:
        summary[m] = {"median": {k: (round(statistics.median([v for v in vals[k][m] if v is not None]), 3)
                                     if any(v is not None for v in vals[k][m]) else None) for k in SETS},
                      "range": {k: ([min(v for v in vals[k][m] if v is not None), max(v for v in vals[k][m] if v is not None)]
                                    if any(v is not None for v in vals[k][m]) else None) for k in SETS},
                      "separation": separation({k: vals[k][m] for k in SETS}, m)}
    report = {"issue": 1762, "version": 2, "manifest": manifest, "unrated_excluded_from_summary": UNRATED,
              "sets": {k: {"n": len(v["files"]), "two_hand": v["two_hand"]} for k, v in SETS.items()},
              "per_item": per, "summary": summary}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for m in METRICS:
        print(m, json.dumps(summary[m]["median"]), json.dumps(summary[m]["separation"]))


if __name__ == "__main__":
    main()
