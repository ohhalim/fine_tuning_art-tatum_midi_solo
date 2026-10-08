#!/usr/bin/env python3
"""Validator for chord–performance paired takes (docs/experiments/PAIRED_CONTRACT.md).

A take is a directory holding take.json (metadata and the chord plan) and raw.mid (the DAW
export, untouched: tempo track, solo track, comp track, count-in included). The validator never
edits notes. A failed check means the contract is broken; a flag keeps the recording as it is
and marks it for a person to look at. Passing every check is a technical result only: it never
makes a harmony label correct, and `musically_verified` stays false without an independent
human review.

Usage: paired_validate.py <take dir> [--write-processed]
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

import mido

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from inference.control.chord_label import QUALITIES  # noqa: E402

SCHEMA = "paired_take_v1"
PC = {"C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "Fb": 4, "F": 5, "E#": 5,
      "F#": 6, "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11}
REQUIRED = ["schema", "take_id", "progression_id", "split", "source", "source_type", "rights_status",
            "performer", "label_basis", "ppq", "time_signature", "tempo_map", "count_in_beats", "bars",
            "tracks", "chords", "raw_sha256"]
LABEL_BASIS = {"planned", "performer_confirmed"}
SPLITS = {"train_candidate", "heldout_candidate"}
TICK_TOL = 1


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_midi(path: str):
    """Notes per track name as (pitch, on, off, velocity) in ticks, plus tempo and meter events.
    A note-off closes the earliest open note of that pitch; overlapping same-pitch notes are kept."""
    mid = mido.MidiFile(path)
    tracks, overlaps, tempos, meters = {}, collections.Counter(), [], []
    for tr in mid.tracks:
        t, open_, notes = 0, collections.defaultdict(list), []
        for msg in tr:
            t += msg.time
            if msg.type == "set_tempo":
                tempos.append((t, msg.tempo))
            elif msg.type == "time_signature":
                meters.append((t, msg.numerator, msg.denominator))
            elif msg.type == "note_on" and msg.velocity > 0:
                if open_[msg.note]:
                    overlaps[tr.name] += 1
                open_[msg.note].append((t, msg.velocity))
            elif msg.type in ("note_off", "note_on") and open_[msg.note]:
                on, vel = open_[msg.note].pop(0)
                notes.append((msg.note, on, t, vel))
        for pitch, stack in open_.items():
            notes.extend((pitch, on, None, vel) for on, vel in stack)     # never closed
        if notes:
            tracks[tr.name] = sorted(notes, key=lambda n: (n[1], n[0]))
    return mid.ticks_per_beat, tracks, overlaps, sorted(tempos), meters


def tick_to_sec(tick: int, tempos, ppq: int) -> float:
    sec, last_t, us = 0.0, 0, 500000
    for t, tempo in tempos:
        if t >= tick:
            break
        sec += (t - last_t) * us / ppq / 1e6
        last_t, us = t, tempo
    return sec + (tick - last_t) * us / ppq / 1e6


def sec_to_tick(sec: float, tempos, ppq: int) -> int:
    acc, last_t, us = 0.0, 0, 500000
    for t, tempo in tempos:
        seg = (t - last_t) * us / ppq / 1e6
        if acc + seg > sec:
            break
        acc += seg
        last_t, us = t, tempo
    return round(last_t + (sec - acc) * ppq * 1e6 / us)


def chord_pcs(chord) -> tuple[int, set[int], set[int]]:
    root = PC[chord["root"]]
    tones, required, tensions = QUALITIES[chord["quality"]]
    return root, {(root + i) % 12 for i in tones | tensions}, {(root + i) % 12 for i in required}


def validate(take_dir: str, write_processed: bool = False) -> dict:
    with open(os.path.join(take_dir, "take.json")) as f:
        meta = json.load(f)
    raw = os.path.join(take_dir, "raw.mid")
    checks, flags = {}, []

    def check(name, ok, detail=""):
        checks[name] = {"ok": bool(ok), "detail": detail}

    missing = [k for k in REQUIRED if k not in meta]
    check("metadata_fields", not missing and meta.get("schema") == SCHEMA,
          f"missing {missing}" if missing else meta.get("schema"))
    if missing:
        return {"take_id": meta.get("take_id"), "checks": checks, "flags": flags, "technical_pass": False,
                "musically_verified": False, "corpus_eligible": False}
    check("label_basis", meta["label_basis"] in LABEL_BASIS, meta["label_basis"])
    check("split", meta["split"] in SPLITS, meta["split"])
    raw_sha = sha256(raw)
    check("raw_hash", raw_sha == meta["raw_sha256"], raw_sha[:16])

    ppq, tracks, overlaps, tempos, meters = read_midi(raw)
    check("ppq", ppq == meta["ppq"], f"midi {ppq}, take.json {meta['ppq']}")
    want_tempos = [(round(e["beat"] * ppq), round(60e6 / e["bpm"])) for e in meta["tempo_map"]]
    check("tempo_map_preserved", [(t, us) for t, us in tempos] == want_tempos and tempos and tempos[0][0] == 0,
          f"midi {tempos}, take.json {want_tempos}")
    num, den = meta["time_signature"]
    check("time_signature", len(meters) == 1 and meters[0] == (0, num, den), f"midi {meters}")
    solo_name, comp_name = meta["tracks"]["solo"], meta["tracks"]["comp"]
    check("tracks_present", solo_name in tracks and comp_name in tracks, f"midi tracks {sorted(tracks)}")
    if not checks["tracks_present"]["ok"]:
        return {"take_id": meta["take_id"], "checks": checks, "flags": flags, "technical_pass": False,
                "musically_verified": False, "corpus_eligible": False}
    solo, comp = tracks[solo_name], tracks[comp_name]

    # chord plan: beats after the count-in, tiling [0, bars * beats per bar) with no gap or overlap
    offset = round(meta["count_in_beats"] * ppq)
    total_beats = meta["bars"] * num * 4 / den
    chords = sorted(meta["chords"], key=lambda c: c["onset_beat"])
    plan_errors = []
    edge = 0.0
    for c in chords:
        if c["onset_beat"] != edge:
            plan_errors.append(f"{'gap' if c['onset_beat'] > edge else 'overlap'} at beat {edge}")
        if c["end_beat"] <= c["onset_beat"]:
            plan_errors.append(f"end <= onset at beat {c['onset_beat']}")
        if c["root"] not in PC or c.get("bass", c["root"]) not in PC or c["quality"] not in QUALITIES:
            plan_errors.append(f"unknown root/bass/quality at beat {c['onset_beat']}")
        if not c.get("voicing") or not all(0 <= p <= 127 for p in c["voicing"]):
            plan_errors.append(f"missing or invalid voicing at beat {c['onset_beat']}")
        edge = c["end_beat"]
    if edge != total_beats:
        plan_errors.append(f"plan ends at beat {edge}, take has {total_beats}")
    check("chord_plan_coverage", not plan_errors, "; ".join(plan_errors))
    if plan_errors:
        return {"take_id": meta["take_id"], "checks": checks, "flags": flags, "technical_pass": False,
                "musically_verified": False, "corpus_eligible": False}
    segs = [(offset + round(c["onset_beat"] * ppq), offset + round(c["end_beat"] * ppq), c) for c in chords]
    end_tick = offset + round(total_beats * ppq)

    for on, _, c in segs:
        _, allowed, required = chord_pcs(c)
        pcs = {p % 12 for p in c["voicing"]}
        if not required <= pcs or not pcs <= allowed:
            flags.append({"flag": "voicing_outside_planned_quality", "tick": on, "chord": f"{c['root']}{c['quality']}"})

    # comp: every chord change must sound within one tick of the planned tick; comp pitches vs plan
    misaligned = []
    for on, end, c in segs:
        inside = [n for n in comp if on - TICK_TOL <= n[1] < end - TICK_TOL]
        if not inside or min(abs(n[1] - on) for n in inside) > TICK_TOL:
            misaligned.append(on)
            continue
        played = {n[0] % 12 for n in inside if abs(n[1] - on) <= TICK_TOL}
        if played != {p % 12 for p in c["voicing"]}:
            flags.append({"flag": "comp_differs_from_plan", "tick": on, "played_pcs": sorted(played),
                          "planned_pcs": sorted({p % 12 for p in c["voicing"]})})
    check("comp_alignment", not misaligned, f"chord changes without a comp onset within {TICK_TOL} tick: {misaligned}")
    stray = [n for n in comp if n[1] < offset - TICK_TOL or n[1] >= end_tick]
    if stray:
        flags.append({"flag": "comp_outside_take", "count": len(stray)})

    # solo: well-formed notes, nothing edited; disagreement with the chord is reported, never corrected
    bad = [n for n in solo + comp if n[2] is None or n[2] <= n[1] or not 1 <= n[3] <= 127]
    check("notes_well_formed", not bad, f"{len(bad)} notes with end <= start, unclosed, or bad velocity")
    if overlaps[solo_name] or overlaps[comp_name]:
        flags.append({"flag": "same_pitch_overlap_kept", "solo": overlaps[solo_name], "comp": overlaps[comp_name]})
    outside = [n for n in solo if n[1] < offset or n[1] >= end_tick]
    if outside:
        flags.append({"flag": "solo_outside_take", "count": len(outside)})
    nonchord = 0
    for p, on, _, _ in solo:
        seg = next((c for s, e, c in segs if s <= on < e), None)
        if seg is not None and p % 12 not in chord_pcs(seg)[1]:
            nonchord += 1
    flags.append({"flag": "solo_onsets_outside_chord_and_tensions", "count": nonchord, "of": len(solo),
                  "note": "information only, notes are not corrected"})

    # tempo conversion round trip on every event tick
    ticks = sorted({t for n in solo + comp for t in (n[1], n[2]) if t is not None} | {s for s, _, _ in segs})
    worst = max(abs(sec_to_tick(tick_to_sec(t, tempos, ppq), tempos, ppq) - t) for t in ticks)
    check("tempo_round_trip", worst <= TICK_TOL, f"worst {worst} tick")

    # processed = count-in removed; must map back to raw exactly
    proc = {name: [(p, on - offset, off - offset, v) for p, on, off, v in notes if off is not None]
            for name, notes in ((solo_name, solo), (comp_name, comp))}
    back = {name: sorted((p, on + offset, off + offset, v) for p, on, off, v in ns) for name, ns in proc.items()}
    same = all(back[name] == sorted(n for n in notes if n[2] is not None)
               for name, notes in ((solo_name, solo), (comp_name, comp)))
    check("count_in_round_trip", same, f"offset {offset} ticks")
    processed_sha = None
    if write_processed and all(c["ok"] for c in checks.values()):
        path = os.path.join(take_dir, "processed.mid")
        write_processed_midi(path, ppq, tempos, (num, den), offset, proc)
        processed_sha = sha256(path)

    technical = all(c["ok"] for c in checks.values())
    review = meta.get("independent_review") or {}
    verified = bool(technical and review.get("verdict") == "pass" and review.get("reviewer")
                    and review.get("reviewer") != meta["performer"])
    return {"take_id": meta["take_id"], "progression_id": meta["progression_id"], "split": meta["split"],
            "raw_sha256": raw_sha, "processed_sha256": processed_sha, "count_in_offset_ticks": offset,
            "checks": checks, "flags": flags, "technical_pass": technical, "musically_verified": verified,
            "corpus_eligible": technical and meta["source_type"] != "synthetic_fixture"}


def write_processed_midi(path, ppq, tempos, meter, offset, tracks):
    mid = mido.MidiFile(ticks_per_beat=ppq)
    conductor = mido.MidiTrack()
    start_tempo = [us for t, us in tempos if t <= offset][-1]
    events = [(0, mido.MetaMessage("set_tempo", tempo=start_tempo)),
              (0, mido.MetaMessage("time_signature", numerator=meter[0], denominator=meter[1]))]
    events += [(t - offset, mido.MetaMessage("set_tempo", tempo=us)) for t, us in tempos if t > offset]
    _append(conductor, events)
    mid.tracks.append(conductor)
    for name, notes in tracks.items():
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        ev = []
        for p, on, off, v in notes:
            ev.append((on, 1, mido.Message("note_on", note=p, velocity=v)))
            ev.append((off, 0, mido.Message("note_off", note=p, velocity=0)))
        _append(tr, [(t, m) for t, _, m in sorted(ev, key=lambda e: (e[0], e[1]))])
        mid.tracks.append(tr)
    mid.save(path)


def _append(track, events):
    last = 0
    for t, msg in events:
        track.append(msg.copy(time=t - last))
        last = t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("take_dir")
    ap.add_argument("--write-processed", action="store_true")
    args = ap.parse_args()
    report = validate(args.take_dir, args.write_processed)
    with open(os.path.join(args.take_dir, "validation.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("take_id", "technical_pass", "musically_verified", "corpus_eligible")}))
    sys.exit(0 if report["technical_pass"] else 1)


if __name__ == "__main__":
    main()
