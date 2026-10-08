#!/usr/bin/env python3
"""Independent recount for score_pair_convert.py (docs/experiments/SCORE_PAIR_PILOT.md).

Reads the source MusicXML with music21 (a separate parser, not the converter's code) and checks
the converter's derived.mid notes and conversion.json chords against it: pitch, onset and end in
ticks after tie merging, chord onset, root, bass and quality. Writes the result into
conversion.json under "independent_recount". Run with music21 on PYTHONPATH:
score_pair_recount.py <out dir>
"""
from __future__ import annotations

import collections
import json
import os
import sys

import mido
from music21 import converter, harmony, note

PPQ = 480
# music21 chordKind → quality of inference/control/chord_label.py
M21_QUALITY = {"major": "maj", "minor": "min", "dominant-seventh": "7", "major-seventh": "maj7",
               "minor-seventh": "m7", "half-diminished-seventh": "m7b5", "diminished-seventh": "dim7",
               "major-sixth": "6", "minor-sixth": "m6", "minor-major-seventh": "mMaj7",
               "dominant-ninth": "7", "dominant-11th": "7", "dominant-13th": "7", "major-ninth": "maj7",
               "major-13th": "maj7", "minor-ninth": "m7", "minor-11th": "m7", "minor-13th": "m7"}


def midi_notes(path):
    out, open_ = [], collections.defaultdict(list)
    for tr in mido.MidiFile(path).tracks:
        t = 0
        for msg in tr:
            t += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                open_[msg.note].append(t)
            elif msg.type in ("note_off", "note_on") and open_[msg.note]:
                out.append((msg.note, open_[msg.note].pop(0), t))
    return sorted(out)


def main() -> None:
    out_dir = sys.argv[1]
    path = os.path.join(out_dir, "conversion.json")
    with open(path) as f:
        conv = json.load(f)
    score = converter.parse(conv["source"])
    part = score.parts[0].stripTies()
    flat = part.flatten()
    ref = sorted((n.pitch.midi, round(float(n.offset) * PPQ), round(float(n.offset + n.quarterLength) * PPQ))
                 for n in flat.getElementsByClass(note.Note))
    got = midi_notes(conv["derived_midi"])
    missing = sorted((collections.Counter(ref) - collections.Counter(got)).elements())
    added = sorted((collections.Counter(got) - collections.Counter(ref)).elements())

    cs = [c for c in flat.getElementsByClass(harmony.ChordSymbol)]
    ref_chords = [{"tick": round(float(c.offset) * PPQ), "root": c.root().pitchClass,
                   "bass": c.bass().pitchClass if c.bass() is not None else None,
                   "kind": c.chordKind, "quality": M21_QUALITY.get(c.chordKind)} for c in cs]
    names = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
    mine = [s for s in conv["chords"] if s["label"] in ("chord", "no_chord")]
    rows, bad = [], 0
    for i in range(max(len(ref_chords), len(mine))):
        r = ref_chords[i] if i < len(ref_chords) else None
        m = mine[i] if i < len(mine) else None
        diff = []
        if r is None or m is None:
            diff.append("count")
        else:
            if round(m["onset"] * PPQ) != r["tick"]:
                diff.append("onset")
            if names.index(m["root"]) != r["root"]:
                diff.append("root")
            mbass = names.index(m["bass"] or m["root"])
            if r["bass"] is not None and mbass != r["bass"]:
                diff.append("bass")
            if m["quality"] != r["quality"]:
                diff.append("quality")
        bad += bool(diff)
        rows.append({"i": i, "measure": m and m["measure"], "converter": m and {"tick": round(m["onset"] * PPQ), "root": m["root"],
                     "bass": m["bass"], "kind": m["kind"], "quality": m["quality"]},
                     "music21": r and dict(r, root=names[r["root"]], bass=None if r["bass"] is None else names[r["bass"]]), "diff": diff})
    conv["independent_recount"] = {
        "tool": "music21 " + __import__("music21").__version__, "scope": "all measures",
        "notes": {"music21": len(ref), "derived_midi": len(got), "missing_in_derived": missing, "added_in_derived": added},
        "chords": {"music21": len(ref_chords), "converter": len(mine), "rows_with_difference": bad,
                   "unmapped_music21_kinds": sorted({c["kind"] for c in ref_chords if c["quality"] is None})},
        "chord_rows": rows,
    }
    with open(path, "w") as f:
        json.dump(conv, f, indent=1, ensure_ascii=False)
    print(os.path.basename(out_dir), "notes", len(ref), "vs", len(got), "missing", len(missing), "added", len(added),
          "| chords", len(ref_chords), "vs", len(mine), "diff rows", bad,
          [ (r["measure"], r["diff"]) for r in rows if r["diff"]][:10])


if __name__ == "__main__":
    main()
