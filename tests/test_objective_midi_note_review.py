from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from scripts.review_midi_note_objectives import analyze_review_manifest, pitch_name


class ObjectiveMidiNoteReviewTest(unittest.TestCase):
    def write_midi(self, path: Path, notes: list[tuple[int, int, int]]) -> None:
        midi = mido.MidiFile(ticks_per_beat=120)
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        meta.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        midi.tracks.append(meta)
        track = mido.MidiTrack()
        track.append(mido.Message("program_change", program=0, time=0))
        events: list[tuple[int, int, int]] = []
        for start, duration, pitch in notes:
            events.append((start, 1, pitch))
            events.append((start + duration, 0, pitch))
        events.sort(key=lambda event: (event[0], event[1]))
        previous = 0
        for tick, on, pitch in events:
            delta = tick - previous
            previous = tick
            velocity = 70 if on else 0
            track.append(mido.Message("note_on", note=pitch, velocity=velocity, time=delta))
        midi.tracks.append(track)
        midi.save(path)

    def test_pitch_name(self) -> None:
        self.assertEqual(pitch_name(60), "C4")
        self.assertEqual(pitch_name(61), "C#4")

    def test_stepwise_chromatic_candidate_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / "chromatic.mid"
            self.write_midi(
                midi_path,
                [(index * 60, 60, 60 + index) for index in range(8)],
            )
            report = analyze_review_manifest(
                {
                    "chord_progression": ["Cmaj7"],
                    "candidates": [
                        {
                            "mode": "straight_grid",
                            "review_rank": 1,
                            "sample_index": 1,
                            "review_midi_path": str(midi_path),
                        }
                    ],
                }
            )

        candidate = report["candidates"][0]

        self.assertIn("too_stepwise_or_scalar", candidate["objective_flags"])
        self.assertIn("chromatic_walk", candidate["objective_flags"])
        self.assertEqual(candidate["metrics"]["off_sixteenth_grid_count"], 0)

    def test_overlap_candidate_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            midi_path = Path(tmp) / "overlap.mid"
            self.write_midi(
                midi_path,
                [
                    (0, 120, 60),
                    (60, 120, 64),
                    (180, 60, 67),
                ],
            )
            report = analyze_review_manifest(
                {
                    "chord_progression": ["Cmaj7"],
                    "candidates": [
                        {
                            "mode": "data_motif",
                            "review_rank": 1,
                            "sample_index": 1,
                            "review_midi_path": str(midi_path),
                        }
                    ],
                }
            )

        candidate = report["candidates"][0]

        self.assertIn("overlap_polyphonic", candidate["objective_flags"])
        self.assertGreater(candidate["metrics"]["polyphonic_tick_ratio"], 0)


if __name__ == "__main__":
    unittest.main()
