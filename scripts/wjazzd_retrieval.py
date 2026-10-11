#!/usr/bin/env python3
"""Source-conditioned retrieval baseline: 2-bar WJazzD fragments matched on the planned chords (issue #1718).

No model, no training. Targets: the first eligible 2-bar window of the first 4 validation songs of
the pilot split (data_summary.json order). Bank: every eligible 2-bar window of the 24 pilot train
songs (bar starts, 1-bar stride).

Rules fixed before the run (issue #1718, discussion log):
- a window is 8 consecutive 4/4 beats of bars >= 0, [start, end) where end is the next beat's onset;
  without that next onset the window is dropped
- the chord on every beat (including the chord already sounding at the window start) must map
  exactly to a quality family; consecutive equal chords are merged before comparing
- match: same chord changes relative to the first root, compared by beat ordinal: same change beats,
  same root steps, same qualities
- the fragment's DB note rows keep their pitches, transposed only; one time scale for the whole
  window (target seconds / source seconds); per-boundary timing error is recorded, never warped away
- notes starting before the window are left out and counted; notes running past the end are cut at
  the end and counted; a window with no note, or a transposition leaving 0..127, is not a candidate
- deterministic order: |log scale|, |transpose|, melid, bar; up to 3 per target; none gives no_match
Run with the Aria venv: wjazzd_retrieval.py <aria repo> <wjazzd.db> <pilot data_summary json> <out dir> <report json>
"""
from __future__ import annotations

import collections
import json
import math
import os
import sqlite3
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wjazzd_pilot_data import key, status  # noqa: E402

TARGET_SONGS, TOP_K = 4, 3
PATTERN = [1, 2, 3, 4, 1, 2, 3, 4]


def build_windows(beats, melid=None):
    """beats: [(onset, bar, beat, chord written at this beat or '')] in time order."""
    carried, cur = [], None
    for onset, bar, beat, chord in beats:
        cur = chord or cur
        carried.append((onset, bar, beat, cur))
    out = []
    for i in range(len(carried) - 8):              # the 9th beat must exist for the window end
        seq = carried[i:i + 8]
        if seq[0][2] != 1 or seq[0][1] < 0 or [b[2] for b in seq] != PATTERN or seq[4][1] != seq[0][1] + 1:
            continue
        chords = [b[3] for b in seq]
        eligible = all(c is not None and status(c) == "exact" for c in chords)
        changes = []
        for j, c in enumerate(chords):
            if c is None:
                break
            k = key(c)
            if not changes or changes[-1][1] != k:
                changes.append((j, k))
        rel = None
        if eligible:
            root0 = changes[0][1][0]
            rel = tuple((j, (k[0] - root0) % 12, k[1]) for j, k in changes)
        out.append({"melid": melid, "bar": seq[0][1], "start": seq[0][0], "end": carried[i + 8][0],
                    "beat_times": [b[0] for b in seq], "chords": chords, "eligible": eligible, "rel": rel,
                    "root0": changes[0][1][0] if eligible else None})
    return out


def transpose_amount(target_root, cand_root):
    return ((target_root - cand_root + 6) % 12) - 6


def place(notes, src, tgt, transpose):
    """Source DB notes [(onset, pitch, dur)] moved into the target window; counts the boundary notes."""
    scale = (tgt["end"] - tgt["start"]) / (src["end"] - src["start"])
    carry_in = sum(1 for o, _, d in notes if o < src["start"] < o + d)
    adopted, rows, cropped, cropped_s = [], [], 0, 0.0
    for o, p, d in notes:
        if not (src["start"] <= o < src["end"]):
            continue
        end = o + d
        if end > src["end"]:
            cropped += 1
            cropped_s += end - src["end"]
            end = src["end"]
        adopted.append((int(p) + transpose, (o - src["start"]) * scale, (end - src["start"]) * scale))
        rows.append({"db_onset": o, "db_pitch": p, "db_duration": d, "cropped_duration": end - o})
    in_range = all(0 <= p <= 127 for p, _, _ in adopted)
    return {"scale": scale, "notes": adopted, "rows": rows, "carry_in_excluded": carry_in, "cropped_at_end": cropped,
            "cropped_seconds": cropped_s, "in_range": in_range}


def quantize_for_export(notes, window_s):
    """Same grid as aria_t4b_sim.write_notes (tick = round(seconds * 1000), 1 tick = 1 ms).
    A note whose onset tick is not below its end tick is excluded and reported, never lengthened or
    moved. End ticks never pass the window end tick. Returns kept [(i, pitch, on, off)], excluded and
    checks (negative onsets, order, same-pitch overlaps among kept notes)."""
    end_tick = round(window_s * 1000)
    kept, excluded = [], []
    for i, (p, s, e) in enumerate(notes):
        on, off = round(s * 1000), min(round(e * 1000), end_tick)
        if on >= off:
            excluded.append({"index": i, "pitch": p, "on_tick": on, "off_tick": off, "seconds": [s, e]})
        else:
            kept.append((i, p, on, off))
    last_off = {}
    overlap = 0
    for _, p, on, off in sorted(kept, key=lambda k: (k[2], k[1])):
        if p in last_off and on < last_off[p]:
            overlap += 1
        last_off[p] = max(off, last_off.get(p, 0))
    checks = {"negative_onsets": sum(1 for k in kept if k[2] < 0), "past_window_end": sum(1 for k in kept if k[3] > end_tick),
              "same_pitch_overlaps": overlap, "window_end_tick": end_tick}
    return kept, excluded, checks


def boundary_errors(src, tgt, scale):
    """For every chord change after the first: (source change time scaled) - (target change time), seconds."""
    out = []
    for j, _, _ in tgt["rel"][1:]:
        moved = (src["beat_times"][j] - src["start"]) * scale
        want = tgt["beat_times"][j] - tgt["start"]
        out.append({"beat": j, "error_s": moved - want})
    return out


def rank(cands):
    return sorted(cands, key=lambda c: (abs(math.log(c["scale"])), abs(c["transpose"]), c["melid"], c["bar"]))


def match(tgt, bank_by_rel, notes_of):
    """Candidates in the deterministic order; empty or out-of-range windows are not candidates."""
    cands, excluded = [], collections.Counter()
    for w in bank_by_rel.get(tgt["rel"], []):
        tr = transpose_amount(tgt["root0"], w["root0"])
        pl = place(notes_of(w["melid"], w["start"], w["end"]), w, tgt, tr)
        if not pl["notes"]:
            excluded["empty_window"] += 1
            continue
        if not pl["in_range"]:
            excluded["out_of_midi_range"] += 1
            continue
        cands.append({**w, "transpose": tr, **pl, "boundary_errors": boundary_errors(w, tgt, pl["scale"])})
    return rank(cands), excluded


def main() -> None:
    aria_repo, db, summary_path, out_dir, report_path = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    from aria_t4b_sim import read_notes, write_notes
    from aria_token_validity import check as token_check
    tok = AbsTokenizer()
    os.makedirs(out_dir, exist_ok=True)
    c = sqlite3.connect(db)
    with open(summary_path) as f:
        summ = json.load(f)
    train = [s["melid"] for s in summ["songs"] if s["split"] == "train"]
    targets = [s["melid"] for s in summ["songs"] if s["split"] == "validation"][:TARGET_SONGS]
    beats_of = lambda m: c.execute("select onset, bar, beat, chord from beats where melid=? order by onset", (m,)).fetchall()

    def notes_of(m, start, end):
        return c.execute("select onset, pitch, duration from melody where melid=? and onset<? and onset+duration>? order by eventid",
                         (m, end, start)).fetchall()

    t0 = time.perf_counter()
    bank = [w for m in train for w in build_windows(beats_of(m), m) if w["eligible"]]
    by_rel = collections.defaultdict(list)
    for w in bank:
        by_rel[w["rel"]].append(w)
    index_s = time.perf_counter() - t0
    report = {"targets_fixed_before_matching": targets, "target_order": "data_summary.json order, split == validation, first 4",
              "bank_songs": train, "bank_windows": len(bank), "distinct_relative_progressions": len(by_rel),
              "index_build_s": round(index_s, 3), "search_time_scope": "matching and placement only, index build excluded",
              "rank_order": "|log scale|, |transpose|, melid, bar", "results": []}
    for m in targets:
        ws = [w for w in build_windows(beats_of(m), m) if w["eligible"]]
        if not ws:
            report["results"].append({"melid": m, "result": "no_eligible_target_window"})
            continue
        tgt = ws[0]                                    # fixed before looking at any match
        ts = time.perf_counter()
        ranked, excluded = match(tgt, by_rel, notes_of)
        search_s = time.perf_counter() - ts
        top = ranked[:TOP_K]
        entry = {"melid": m, "target_bar": tgt["bar"], "target_chords": tgt["chords"], "target_rel": tgt["rel"],
                 "target_seconds": round(tgt["end"] - tgt["start"], 4), "bank_rel_matches": len(by_rel.get(tgt["rel"], [])),
                 "candidates_after_exclusion": len(ranked), "excluded": dict(excluded), "search_s": round(search_s, 6),
                 "result": "returned" if top else "no_match", "candidates": []}
        for r, cnd in enumerate(top, 1):
            path = os.path.join(out_dir, f"target{m}_rank{r}_src{cnd['melid']}_bar{cnd['bar']}.mid")
            notes = [(p, s, e, 80) for p, s, e in cnd["notes"]]
            write_notes(notes, path)
            back = read_notes(path)
            midi_ok = len(back) == len(notes) and all(
                bp == p and abs(bs - s) <= 0.001 and abs(be - e) <= 0.001
                for (bp, bs, be, _), (p, s, e, _) in zip(sorted(back), sorted(notes)))
            toks = tok.tokenize(MidiDict.from_midi(path))
            n_head = next(i for i, t in enumerate(toks) if t == "<S>") + 1
            tv = token_check(toks[:n_head], toks[n_head:])
            overlap = [o["source_bar"] for o in entry["candidates"]
                       if o["source_melid"] == cnd["melid"] and abs(o["source_bar"] - cnd["bar"]) < 2]
            entry["candidates"].append({
                "rank": r, "source": "WJazzD DB melody rows", "source_melid": cnd["melid"], "source_bar": cnd["bar"],
                "source_time_s": [round(cnd["start"], 3), round(cnd["end"], 3)], "transpose": cnd["transpose"],
                "time_scale": round(cnd["scale"], 4), "notes": len(cnd["notes"]),
                "carry_in_excluded": cnd["carry_in_excluded"], "cropped_at_end": cnd["cropped_at_end"],
                "cropped_seconds": round(cnd["cropped_seconds"], 3),
                "boundary_errors_s": [{"beat": b["beat"], "error_s": round(b["error_s"], 4)} for b in cnd["boundary_errors"]],
                "pitch_range": [min(p for p, *_ in cnd["notes"]), max(p for p, *_ in cnd["notes"])],
                "midi_reread_matches": midi_ok, "aria_token_layer": tv,
                "overlaps_earlier_candidate_bars": overlap, "source_chords": cnd["chords"]})
        report["results"].append(entry)
        print(m, entry["result"], "rel matches", entry["bank_rel_matches"], "after exclusion", len(ranked), dict(excluded),
              [(x["source_melid"], x["source_bar"], x["transpose"], x["time_scale"]) for x in entry["candidates"]])
    report["coverage"] = {"targets": len(targets), "returned": sum(1 for r in report["results"] if r.get("result") == "returned")}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=str)
    print(report["coverage"], "bank windows", len(bank), "index", report["index_build_s"], "s")


if __name__ == "__main__":
    main()
