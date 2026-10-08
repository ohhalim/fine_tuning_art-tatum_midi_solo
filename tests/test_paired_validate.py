"""Paired take validator (scripts/paired_validate.py) on a synthetic fixture.

The fixture is written to a temporary directory for parser and alignment tests only. It is
marked source_type synthetic_fixture, so it can never be corpus eligible, and it is never
stored under data/.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import mido

from scripts.paired_validate import read_midi, sec_to_tick, sha256, tick_to_sec, validate

PPQ, COUNT_IN = 480, 4
CHORDS = [
    {"onset_beat": 0, "end_beat": 4, "root": "D", "quality": "m7", "bass": "D", "voicing": [50, 53, 57, 60]},
    {"onset_beat": 4, "end_beat": 8, "root": "G", "quality": "7", "bass": "G", "voicing": [55, 59, 62, 65]},
]
TEMPO_MAP = [{"beat": 0, "bpm": 120}, {"beat": 8, "bpm": 100}]     # tempo changes inside the take
SOLO = [(62, 0.0, 0.5), (65, 0.5, 1.0), (69, 1.0, 1.5), (72, 1.5, 2.5), (71, 4.0, 4.75), (67, 4.75, 6.0)]


def build(d, chords=CHORDS, tempo_map=TEMPO_MAP, solo=SOLO, comp_shift=0, label_basis="planned",
          extra_solo=(), meta_tempo_map=None):
    mid = mido.MidiFile(ticks_per_beat=PPQ)
    conductor = mido.MidiTrack()
    events = [(round(e["beat"] * PPQ), mido.MetaMessage("set_tempo", tempo=round(60e6 / e["bpm"]))) for e in tempo_map]
    events.insert(1, (0, mido.MetaMessage("time_signature", numerator=4, denominator=4)))
    last = 0
    for t, m in events:
        conductor.append(m.copy(time=t - last))
        last = t
    mid.tracks.append(conductor)

    def track(name, notes):
        tr = mido.MidiTrack()
        tr.append(mido.MetaMessage("track_name", name=name, time=0))
        ev = []
        for p, on, off in notes:
            ev.append((on, 1, mido.Message("note_on", note=p, velocity=80)))
            ev.append((off, 0, mido.Message("note_off", note=p, velocity=0)))
        last = 0
        for t, _, m in sorted(ev, key=lambda e: (e[0], e[1])):
            tr.append(m.copy(time=t - last))
            last = t
        mid.tracks.append(tr)

    tick = lambda beat: round((COUNT_IN + beat) * PPQ)
    track("Solo", [(p, tick(a), tick(b)) for p, a, b in solo] + list(extra_solo))
    track("Chords", [(p, tick(c["onset_beat"]) + comp_shift, tick(c["end_beat"])) for c in chords for p in c["voicing"]])
    raw = os.path.join(d, "raw.mid")
    mid.save(raw)
    meta = {"schema": "paired_take_v1", "take_id": "fixture_t1", "progression_id": "fixture", "split": "train_candidate",
            "source": "tests/test_paired_validate.py", "source_type": "synthetic_fixture", "rights_status": "generated_test",
            "performer": "none", "label_basis": label_basis, "ppq": PPQ, "time_signature": [4, 4],
            "tempo_map": meta_tempo_map or tempo_map, "count_in_beats": COUNT_IN, "bars": 2,
            "tracks": {"solo": "Solo", "comp": "Chords"}, "chords": chords,
            "raw_sha256": sha256(raw), "independent_review": None}
    with open(os.path.join(d, "take.json"), "w") as f:
        json.dump(meta, f)
    return d


class PairedValidateTest(unittest.TestCase):
    def run_case(self, **kw):
        with tempfile.TemporaryDirectory() as d:
            build(d, **kw)
            return validate(d, write_processed=True), d

    def test_valid_fixture_passes_but_is_never_verified_or_corpus_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            build(d)
            r = validate(d, write_processed=True)
            self.assertTrue(r["technical_pass"], r["checks"])
            self.assertFalse(r["musically_verified"])
            self.assertFalse(r["corpus_eligible"])
            ppq, tracks, _, tempos, _ = read_midi(os.path.join(d, "processed.mid"))
            self.assertEqual(tracks["Solo"][0][1], 0)                       # count-in removed
            self.assertEqual(tempos, [(0, 500000), (8 * PPQ - COUNT_IN * PPQ, 600000)])

    def test_chord_gap_and_overlap_fail(self) -> None:
        gap = [dict(CHORDS[0]), dict(CHORDS[1], onset_beat=5)]
        self.assertFalse(self.run_case(chords=gap)[0]["checks"]["chord_plan_coverage"]["ok"])
        over = [dict(CHORDS[0], end_beat=5), dict(CHORDS[1])]
        self.assertFalse(self.run_case(chords=over)[0]["checks"]["chord_plan_coverage"]["ok"])

    def test_comp_alignment_allows_one_tick_only(self) -> None:
        self.assertTrue(self.run_case(comp_shift=1)[0]["checks"]["comp_alignment"]["ok"])
        self.assertFalse(self.run_case(comp_shift=2)[0]["checks"]["comp_alignment"]["ok"])

    def test_same_pitch_overlap_is_flagged_and_kept(self) -> None:
        t = round((COUNT_IN + 0.25) * PPQ)
        r, _ = self.run_case(extra_solo=[(62, t, t + 200)])
        self.assertTrue(r["technical_pass"], r["checks"])
        self.assertIn("same_pitch_overlap_kept", [f["flag"] for f in r["flags"]])

    def test_inferred_labels_and_tempo_mismatch_fail(self) -> None:
        self.assertFalse(self.run_case(label_basis="inferred")[0]["technical_pass"])
        r, _ = self.run_case(meta_tempo_map=[{"beat": 0, "bpm": 120}])
        self.assertFalse(r["checks"]["tempo_map_preserved"]["ok"])

    def test_zero_length_note_fails(self) -> None:
        t = round((COUNT_IN + 2) * PPQ)
        self.assertFalse(self.run_case(extra_solo=[(80, t, t)])[0]["checks"]["notes_well_formed"]["ok"])

    def test_edited_raw_breaks_hash(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            build(d)
            with open(os.path.join(d, "raw.mid"), "ab") as f:
                f.write(b"\0")
            self.assertFalse(validate(d)["checks"]["raw_hash"]["ok"])

    def test_tempo_conversion_round_trip_across_tempo_change(self) -> None:
        tempos = [(0, 500000), (8 * PPQ, 600000)]
        for t in range(0, 16 * PPQ, 7):
            self.assertLessEqual(abs(sec_to_tick(tick_to_sec(t, tempos, PPQ), tempos, PPQ) - t), 1)


if __name__ == "__main__":
    unittest.main()
