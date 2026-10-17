#!/usr/bin/env python3
"""Aggregate count of lick windows in WJazzD for the two #1782 targets (issue #1784, docs/experiments/WJAZZD_LICK_COVERAGE.md).

Read-only: no audio, no MIDI, no melody or chord sequence leaves this script; the report holds aggregates only.
Windows come from wjazzd_retrieval.build_windows (#1718). A window is a lick window when one run of its notes
passes inference/control/lick_bank.lick_from_run.

  wjazzd_lick_coverage.py <wjazzd.db> <report json>
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sqlite3
import statistics
import sys
from types import SimpleNamespace

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..")):
    sys.path.insert(0, p)

TARGETS = {"change": ((0, 0, "m7"), (4, 5, "7")), "hold": ((0, 0, "maj7"),)}
# post-hoc subsets (chosen after the main result; WJAZZD_LICK_COVERAGE.md section 8)
SUBSETS = {"tempo_90_140": lambda i: bool(i["avgtempo"]) and 90 <= i["avgtempo"] <= 140,
           "tempo_90_140_and_not_swing": lambda i: bool(i["avgtempo"]) and 90 <= i["avgtempo"] <= 140
           and "SWING" not in (i["rhythmfeel"] or "") and (i["rhythmfeel"] or "") not in ("TWOBEAT", "BALLAD")}
OFFBEAT = (0.35, 0.85)


def window_notes(rows, start, end):
    """Notes (onset, pitch, duration) starting inside [start, end) as objects lick_bank understands."""
    return [SimpleNamespace(start=o, end=o + d, pitch=p, velocity=80) for o, p, d in rows if start <= o < end]


def is_lick_window(notes):
    from inference.control.lick_bank import lick_from_run, runs
    return any(lick_from_run(r) is not None for r in runs(notes))


def offbeat_positions(notes, beat_times, end):
    """Beat-relative positions in [0.35, 0.85) of notes inside the window."""
    edges = list(beat_times) + [end]
    out = []
    for n in notes:
        for b in range(len(beat_times)):
            if edges[b] <= n.start < edges[b + 1]:
                pos = (n.start - edges[b]) / (edges[b + 1] - edges[b])
                if OFFBEAT[0] <= pos < OFFBEAT[1]:
                    out.append(pos)
                break
    return out


def summarize(wins, info):
    solos = {w["melid"] for w in wins}
    pos = [p for w in wins for p in w["offbeat"]]
    by_feel = collections.defaultdict(list)
    for w in wins:
        by_feel[info[w["melid"]]["rhythmfeel"]] += w["offbeat"]
    tempos = [info[m]["avgtempo"] for m in solos if info[m]["avgtempo"]]
    return {"windows": len(wins), "solo_bars": len({(w["melid"], w["bar"]) for w in wins}), "solos": len(solos),
            "performers": len({info[m]["performer"] for m in solos}),
            "instrument": dict(collections.Counter(info[m]["instrument"] for m in solos).most_common()),
            "style": dict(collections.Counter(info[m]["style"] for m in solos).most_common()),
            "rhythmfeel": dict(collections.Counter(info[m]["rhythmfeel"] for m in solos).most_common()),
            "piano_solos": sum(1 for m in solos if info[m]["instrument"] == "p"),
            "avgtempo_median": round(statistics.median(tempos), 1) if tempos else None,
            "offbeat_notes": len(pos), "offbeat_pos_median": round(statistics.median(pos), 3) if pos else None,
            "offbeat_pos_median_by_rhythmfeel": {k: (round(statistics.median(v), 3), len(v)) for k, v in sorted(by_feel.items()) if v}}


def posthoc(c, info, out_path):
    import wjazzd_retrieval as wr
    out = {}
    for sname, f in SUBSETS.items():
        mel = [m for m in sorted(info) if f(info[m])]
        res = {}
        for tname, rel in TARGETS.items():
            wins, pos = [], []
            for m in mel:
                beats = c.execute("select onset, bar, beat, chord from beats where melid=? order by onset", (m,)).fetchall()
                rows = c.execute("select onset, pitch, duration from melody where melid=? order by onset", (m,)).fetchall()
                for w in wr.build_windows(beats, m):
                    if w["eligible"] and w["rel"] == rel:
                        notes = window_notes(rows, w["start"], w["end"])
                        if is_lick_window(notes):
                            wins.append(m)
                            pos += offbeat_positions(notes, w["beat_times"], w["end"])
            res[tname] = {"lick_windows": len(wins), "solos": len(set(wins)), "performers": len({info[m]["performer"] for m in set(wins)}),
                          "instrument": dict(collections.Counter(info[m]["instrument"] for m in set(wins)).most_common()),
                          "rhythmfeel": dict(collections.Counter(info[m]["rhythmfeel"] for m in set(wins)).most_common()),
                          "offbeat_pos_median": round(statistics.median(pos), 3) if pos else None, "offbeat_notes": len(pos)}
        out[sname] = {"solos_in_subset": len(mel), **res}
    with open(out_path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)


def main() -> None:
    db, report_path = sys.argv[1:3]
    import wjazzd_retrieval as wr
    h = hashlib.sha256()
    with open(db, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    c = sqlite3.connect(db)
    info = {r[0]: {"performer": r[1], "instrument": r[2], "style": r[3], "rhythmfeel": r[4], "avgtempo": r[5]}
            for r in c.execute("select melid, performer, instrument, style, rhythmfeel, avgtempo from solo_info")}
    found = {k: {"progression": [], "lick": []} for k in TARGETS}
    eligible = 0
    for m in sorted(info):
        beats = c.execute("select onset, bar, beat, chord from beats where melid=? order by onset", (m,)).fetchall()
        rows = c.execute("select onset, pitch, duration from melody where melid=? order by onset", (m,)).fetchall()
        for w in wr.build_windows(beats, m):
            if not w["eligible"]:
                continue
            eligible += 1
            for name, rel in TARGETS.items():
                if w["rel"] != rel:
                    continue
                notes = window_notes(rows, w["start"], w["end"])
                rec = {"melid": m, "bar": w["bar"], "offbeat": []}
                found[name]["progression"].append(rec)
                if is_lick_window(notes):
                    found[name]["lick"].append(rec | {"offbeat": offbeat_positions(notes, w["beat_times"], w["end"])})
    report = {"issue": 1784, "db_sha256": h.hexdigest(), "solos_total": len(info), "eligible_windows": eligible,
              "targets": {k: {"rel": TARGETS[k], "progression_only": {kk: v for kk, v in summarize(found[k]["progression"], info).items()
                                                                        if kk in ("windows", "solo_bars", "solos")},
                              "lick": summarize(found[k]["lick"], info)} for k in TARGETS},
              "corpus_rhythmfeel": dict(collections.Counter(v["rhythmfeel"] for v in info.values()).most_common()),
              "corpus_instrument": dict(collections.Counter(v["instrument"] for v in info.values()).most_common())}
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: (v["progression_only"]["windows"], v["lick"]["windows"], v["lick"]["solos"]) for k, v in report["targets"].items()}))
    if "--posthoc" in sys.argv:
        posthoc(c, info, sys.argv[sys.argv.index("--posthoc") + 1])


if __name__ == "__main__":
    main()
