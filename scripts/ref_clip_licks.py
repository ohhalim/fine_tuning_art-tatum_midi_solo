#!/usr/bin/env python3
"""Every real lick in the preferred reference clips, top-line proxy only (issue #1809, docs/experiments/REF_CLIP_LICKS.md).

Not a model output. For each clip, every phrase of the top-line proxy that starts inside the clip window and passes
lick_bank.lick_from_run is written as is (original note IDs, pitches, start/end, velocity; no transposition, no time
change), rendered, and listed with its speed. Clip 18's first lick was heard in #1786 and is marked so.

  ref_clip_licks.py <clips.json> <midi root> <run dir> <report json>
"""
from __future__ import annotations

import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for p in (HERE, os.path.join(HERE, "..")):
    sys.path.insert(0, p)
import bebop_fragment_retrieval as bf  # noqa: E402

CLIP_IDS = ("01", "04", "06", "13", "16", "17", "18", "22")   # the clips the user picked (#1762, REFERENCE_CLIPS_DECISION.md)
HEARD = {("18", 0)}                                           # (clip, lick index) already heard in #1786


def clip_licks(notes, window):
    """[(run as [(id, truncated note)], lick dict)] for every lick phrase starting inside the window, and run counts."""
    from inference.control.lick_bank import lick_from_run, runs
    line, trunc, _, _ = bf.split_song(notes)
    id_of = {id(t): nid for nid, t in line}
    out, counted = [], {"runs_8plus": 0, "too_long": 0, "other_reject": 0}
    for run in runs([t for _, t in line]):
        if not window[0] <= run[0].start < window[1] or len(run) < 8:
            continue
        counted["runs_8plus"] += 1
        lick = lick_from_run(run)
        if lick is None:
            counted["too_long" if len(run) > 24 else "other_reject"] += 1
            continue
        out.append(([(id_of[id(t)], t) for t in run], lick))
    return out, counted


def speed(run):
    """Notes per second over the lick (first to last onset) and the median gap."""
    ons = [t.start for _, t in run]
    gaps = [b - a for a, b in zip(ons, ons[1:])]
    span = ons[-1] - ons[0]
    return {"notes": len(run), "span_s": round(span, 3), "notes_per_s": round((len(run) - 1) / span, 2) if span > 0 else None,
            "median_ioi_s": round(statistics.median(gaps), 4)}


def melody_rows(notes, run):
    nb = bf.notes_by_id(notes)
    t0 = run[0][1].start
    return [{"id": nid, "pitch": nb[nid].pitch, "start": round(float(nb[nid].start) - t0, 4), "end": round(float(nb[nid].end) - t0, 4),
             "velocity": nb[nid].velocity} for nid, _ in run]


def main() -> None:
    clips_path, midi_root, run_dir, report_path = sys.argv[1:5]
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    clips = {c["id"]: c for c in json.load(open(clips_path))["clips"]}
    report = {"issue": 1809, "clips_json_sha256": bf.file_hash(clips_path), "clips": [], "licks": []}
    for cid in CLIP_IDS:
        c = clips[cid]
        src = os.path.join(midi_root, c["source"])
        notes = bf.load_notes(src)
        found, counted = clip_licks(notes, (c["start_s"], c["end_s"]))
        report["clips"].append({"id": cid, "title": c["title"], "source": c["source"], "source_sha256": bf.file_hash(src),
                                "window": [c["start_s"], c["end_s"]], "licks": len(found), **counted})
        for k, (run, _) in enumerate(found):
            rows = melody_rows(notes, run)
            name = f"clip{cid}_lick{k + 1}.mid"
            f = bf.write_midi(rows, os.path.join(run_dir, name))
            bf.render(os.path.join(run_dir, name))
            pitches = [r["pitch"] for r in rows]
            report["licks"].append({"clip": cid, "index": k + 1, "file": name, "audio": name[:-4] + ".m4a",
                                    "t0_s": round(float(run[0][1].start), 4), "t0_in_clip_s": round(float(run[0][1].start) - c["start_s"], 3),
                                    "pitch_min": min(pitches), "pitch_max": max(pitches), "heard_in_1786": (cid, k) in HEARD,
                                    "roundtrip_diffs": f["roundtrip_diffs"], "max_time_err_ms": f["max_time_err_ms"], **speed(run)})
    report["checks"] = {"roundtrip_all_zero": all(x["roundtrip_diffs"] == 0 for x in report["licks"]),
                        "licks_total": len(report["licks"])}
    with open(report_path, "w") as fo:
        json.dump(report, fo, ensure_ascii=False, indent=1)
    print(json.dumps({c["id"]: c["licks"] for c in report["clips"]}), json.dumps(report["checks"]))


if __name__ == "__main__":
    main()
