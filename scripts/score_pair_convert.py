#!/usr/bin/env python3
"""Score head → aligned melody and chord annotations (docs/experiments/SCORE_PAIR_PILOT.md).

The MusicXML file is the source of truth. Output per song: derived.mid (tempo track and the
melody, velocity fixed because the score has none) and conversion.json (provenance, the
measure → absolute beat map, notes, chord segments, features found, unsupported items). Repeats
are kept as written, never unrolled. A chord the converter cannot map stays unmapped; nothing
is filled with a default or with the previous chord. Usage:
score_pair_convert.py <xml> <out dir>
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import sys
import xml.etree.ElementTree as ET
from fractions import Fraction

import mido

PPQ = 480
STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# MusicXML kind → quality of inference/control/chord_label.py; extended kinds keep their raw kind
KIND = {"major": "maj", "minor": "min", "dominant": "7", "major-seventh": "maj7", "minor-seventh": "m7",
        "half-diminished": "m7b5", "diminished-seventh": "dim7", "major-sixth": "6", "minor-sixth": "m6",
        "major-minor": "mMaj7", "dominant-ninth": "7", "dominant-11th": "7", "dominant-13th": "7",
        "major-ninth": "maj7", "major-13th": "maj7", "minor-ninth": "m7", "minor-11th": "m7", "minor-13th": "m7"}
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def pc(step: str, alter: str | None) -> int:
    return (STEP[step] + int(float(alter or 0))) % 12


def parse(xml_path: str) -> dict:
    root = ET.parse(xml_path).getroot()
    parts = root.findall("part")
    unsupported, features = [], collections.Counter()
    if len(parts) != 1:
        unsupported.append({"where": "score", "item": f"{len(parts)} parts", "reason": "only one melody part is read"})
    divisions, start = None, Fraction(0)
    meter, measures, tempos, meters, notes, harmonies = None, [], [], [], [], []
    open_ties = {}
    for mi, m in enumerate(parts[0].findall("measure")):
        number = m.get("number")
        cursor, longest, last_onset = Fraction(0), Fraction(0), None
        for el in m:
            if el.tag == "attributes":
                if el.findtext("divisions"):
                    divisions = int(el.findtext("divisions"))
                t = el.find("time")
                if t is not None:
                    meter = (int(t.findtext("beats")), int(t.findtext("beat-type")))
                    meters.append({"measure": number, "beat": float(start + cursor), "num": meter[0], "den": meter[1]})
                    features["time_signature"] += 1
            elif el.tag in ("direction", "sound"):
                for s in ([el] if el.tag == "sound" else el.iter("sound")):
                    if s.get("tempo"):
                        tempos.append({"measure": number, "beat": float(start + cursor), "bpm": float(s.get("tempo"))})
                for tag in ("segno", "coda", "words"):
                    for w in el.iter(tag):
                        features[tag] += 1
                        if tag != "words":
                            unsupported.append({"where": f"measure {number}", "item": tag, "reason": "kept as written, not followed"})
            elif el.tag == "harmony":
                off = Fraction(int(el.findtext("offset") or 0), divisions)
                if el.find("offset") is not None:
                    features["harmony_offset"] += 1
                r = el.find("root")
                kind = el.findtext("kind")
                h = {"measure": number, "xml_measure_index": mi, "onset": start + cursor + off,
                     "offset_divisions": int(el.findtext("offset") or 0), "kind": kind,
                     "kind_text": el.find("kind").get("text") if el.find("kind") is not None else None,
                     "root": None, "bass": None, "quality": None,
                     "degrees": [{"value": d.findtext("degree-value"), "alter": d.findtext("degree-alter"),
                                  "type": d.findtext("degree-type")} for d in el.findall("degree")]}
                if r is not None:
                    h["root"] = NAMES[pc(r.findtext("root-step"), r.findtext("root-alter"))]
                b = el.find("bass")
                if b is not None:
                    h["bass"] = NAMES[pc(b.findtext("bass-step"), b.findtext("bass-alter"))]
                    features["slash_bass"] += 1
                if kind == "none":
                    features["no_chord"] += 1
                elif kind in KIND and r is not None:
                    h["quality"] = KIND[kind]
                else:
                    unsupported.append({"where": f"measure {number}", "item": f"harmony kind {kind!r}", "reason": "no quality mapping; left unmapped"})
                if h["degrees"]:
                    features["harmony_degree"] += 1
                harmonies.append(h)
            elif el.tag == "backup":
                cursor -= Fraction(int(el.findtext("duration")), divisions)
                features["backup"] += 1
            elif el.tag == "forward":
                cursor += Fraction(int(el.findtext("duration")), divisions)
                features["forward"] += 1
            elif el.tag == "barline":
                for tag in ("repeat", "ending"):
                    if el.find(tag) is not None:
                        features[tag] += 1
                        unsupported.append({"where": f"measure {number}", "item": tag, "reason": "repeat policy: as written, not unrolled"})
            elif el.tag == "note":
                if el.find("grace") is not None:
                    features["grace"] += 1
                    unsupported.append({"where": f"measure {number}", "item": "grace note", "reason": "no duration; skipped"})
                    continue
                if el.findtext("voice") not in (None, "1"):
                    features["voice_other_than_1"] += 1
                if el.find("time-modification") is not None:
                    features["tuplet_note"] += 1
                dur = Fraction(int(el.findtext("duration")), divisions)
                onset = last_onset if el.find("chord") is not None else start + cursor
                if el.find("chord") is not None:
                    features["chord_note_in_melody"] += 1
                else:
                    cursor += dur
                last_onset = onset
                longest = max(longest, cursor)
                if el.find("rest") is not None:
                    continue
                p = el.find("pitch")
                pitch = (int(p.findtext("octave")) + 1) * 12 + STEP[p.findtext("step")] + int(float(p.findtext("alter") or 0))
                ties = {t.get("type") for t in el.findall("tie")}
                if "stop" in ties and pitch in open_ties and open_ties[pitch]["end"] == onset:
                    n = open_ties.pop(pitch)
                    n["end"] = onset + dur
                    n["tied_parts"] += 1
                    features["tie_merged"] += 1
                else:
                    if "stop" in ties:
                        unsupported.append({"where": f"measure {number}", "item": f"tie stop on {pitch} without matching start", "reason": "kept as its own note"})
                    n = {"pitch": pitch, "onset": onset, "end": onset + dur, "measure": number, "tied_parts": 1}
                    notes.append(n)
                if "start" in ties:
                    open_ties[pitch] = n
            longest = max(longest, cursor)
        nominal = Fraction(meter[0] * 4, meter[1])
        measures.append({"measure": number, "xml_index": mi, "start_beat": float(start), "length_beats": float(longest),
                         "nominal_beats": float(nominal), "implicit": m.get("implicit") == "yes"})
        if longest != nominal:
            features["irregular_measure"] += 1
            if mi == 0:
                features["pickup"] += 1
        start += longest
    for pitch in open_ties:
        unsupported.append({"where": "end", "item": f"tie start on {pitch} never stopped", "reason": "note ends at its own duration"})
    if len(tempos) > 1:
        features["tempo_change"] = len(tempos) - 1
    if len(meters) > 1:
        features["time_signature_change"] = len(meters) - 1
    # chord segments: harmony onset to next onset; before the first harmony nothing is labeled
    harmonies.sort(key=lambda h: h["onset"])
    segs = []
    if harmonies and harmonies[0]["onset"] > 0:
        segs.append({"onset": Fraction(0), "end": harmonies[0]["onset"], "label": "unlabeled"})
    for i, h in enumerate(harmonies):
        end = harmonies[i + 1]["onset"] if i + 1 < len(harmonies) else start
        if end == h["onset"]:
            unsupported.append({"where": f"measure {h['measure']}", "item": "two harmonies at one onset", "reason": "earlier one has zero length"})
        segs.append(dict(h, end=end, label="no_chord" if h["kind"] == "none" else "chord"))
    return {"divisions_seen": divisions, "total_beats": start, "measures": measures, "tempos": tempos, "meters": meters,
            "notes": notes, "segments": segs, "features": dict(features), "unsupported": unsupported}


def tick(beat: Fraction) -> int:
    return round(beat * PPQ)


def write_midi(path: str, parsed: dict) -> None:
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    cond = mido.MidiTrack()
    ev = [(tick(Fraction(t["beat"]).limit_denominator(10000)), mido.MetaMessage("set_tempo", tempo=round(60e6 / t["bpm"]))) for t in parsed["tempos"]]
    ev += [(tick(Fraction(m["beat"]).limit_denominator(10000)), mido.MetaMessage("time_signature", numerator=m["num"], denominator=m["den"])) for m in parsed["meters"]]
    _append(cond, sorted(ev, key=lambda e: e[0]))
    mid.tracks.append(cond)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name="Melody(score head)", time=0))
    ev = []
    for n in parsed["notes"]:
        ev.append((tick(n["onset"]), 1, mido.Message("note_on", note=n["pitch"], velocity=80)))
        ev.append((tick(n["end"]), 0, mido.Message("note_off", note=n["pitch"], velocity=0)))
    _append(tr, [(t, m) for t, _, m in sorted(ev, key=lambda e: (e[0], e[1]))])
    mid.tracks.append(tr)
    mid.save(path)


def _append(track, events):
    last = 0
    for t, msg in events:
        track.append(msg.copy(time=t - last))
        last = t


def read_back(path: str):
    mid = mido.MidiFile(path)
    out, open_ = [], collections.defaultdict(list)
    for tr in mid.tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                open_[msg.note].append(t)
            elif msg.type in ("note_off", "note_on") and open_[msg.note]:
                out.append((msg.note, open_[msg.note].pop(0), t))
    return sorted(out)


def main() -> None:
    xml_path, out_dir = sys.argv[1], sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    parsed = parse(xml_path)
    exact = PPQ % parsed["divisions_seen"] == 0
    midi_path = os.path.join(out_dir, "derived.mid")
    write_midi(midi_path, parsed)
    want = sorted((n["pitch"], tick(n["onset"]), tick(n["end"])) for n in parsed["notes"])
    got = read_back(midi_path)
    missing = sorted((collections.Counter(want) - collections.Counter(got)).elements())
    added = sorted((collections.Counter(got) - collections.Counter(want)).elements())
    worst = max(abs(tick(n[k]) - n[k] * PPQ) for n in parsed["notes"] for k in ("onset", "end"))
    mapped = sum(1 for s in parsed["segments"] if s["label"] == "chord" and s["quality"])
    chords = sum(1 for s in parsed["segments"] if s["label"] == "chord")
    f = lambda x: float(x)
    report = {
        "schema": "score_pair_v1", "source_type": "score_head", "label_basis": "score_annotation",
        "rights_status": "unknown", "provenance": "bebopnet-code resources/xmls; theme melody (head), not an improvisation, not a piano performance",
        "source": os.path.abspath(xml_path), "source_sha256": sha256(xml_path),
        "converter": "scripts/score_pair_convert.py", "converter_sha256": sha256(os.path.abspath(__file__)),
        "repeat_policy": "as_written", "ppq": PPQ, "divisions": parsed["divisions_seen"], "ticks_exact_for_divisions": exact,
        "derived_midi": midi_path, "derived_midi_sha256": sha256(midi_path),
        "velocity": "fixed 80, the score has no dynamics per note",
        "total_beats": f(parsed["total_beats"]), "measures": parsed["measures"], "tempos": parsed["tempos"], "meters": parsed["meters"],
        "notes": [{"pitch": n["pitch"], "onset_beat": f(n["onset"]), "end_beat": f(n["end"]), "onset_tick": tick(n["onset"]),
                   "end_tick": tick(n["end"]), "measure": n["measure"], "tied_parts": n["tied_parts"]} for n in parsed["notes"]],
        "chords": [{k: (f(v) if isinstance(v, Fraction) else v) for k, v in s.items()} for s in parsed["segments"]],
        "features": parsed["features"], "unsupported": parsed["unsupported"],
        "midi_check": {"notes": len(want), "missing": missing, "added": added, "worst_tick_error": f(worst)},
        "chord_coverage": {"chord_segments": chords, "mapped": mapped, "no_chord": parsed["features"].get("no_chord", 0),
                           "unlabeled_segments": sum(1 for s in parsed["segments"] if s["label"] == "unlabeled")},
        "eligibility": {"pipeline_test": not missing and not added and worst <= 1, "training": False, "musical": False},
        "independent_recount": None,
    }
    with open(os.path.join(out_dir, "conversion.json"), "w") as fh:
        json.dump(report, fh, indent=1, ensure_ascii=False)
    print(os.path.basename(xml_path), "notes", len(want), "missing", len(missing), "added", len(added),
          "chords", f"{mapped}/{chords}", "unsupported", len(parsed["unsupported"]), "features", parsed["features"])


if __name__ == "__main__":
    main()
