from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.review_stage_b_margin_recovered_focused_context import (
    build_focused_context_decision,
    validate_decision,
)


def write_solo(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    solo = pretty_midi.Instrument(program=0, is_drum=False, name="Solo")
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        solo.notes.append(pretty_midi.Note(velocity=88, pitch=pitch, start=start, end=start + 0.2))
    midi.instruments.append(solo)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def write_context(path: Path, solo_path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    chord = pretty_midi.Instrument(program=4, is_drum=False, name="Chord Guide")
    bass = pretty_midi.Instrument(program=32, is_drum=False, name="Bass Root Guide")
    solo_source = pretty_midi.PrettyMIDI(str(solo_path))
    solo = solo_source.instruments[0]
    solo.name = "Solo - test"
    for pitch in [48, 51, 55, 58]:
        chord.notes.append(pretty_midi.Note(velocity=48, pitch=pitch, start=0.0, end=4.0))
    bass.notes.append(pretty_midi.Note(velocity=52, pitch=36, start=0.0, end=4.0))
    midi.instruments.extend([chord, bass, solo])
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def sample_package(root: Path, *, pitches: list[int], dead_air: float) -> dict:
    solo_path = root / "midi" / "candidate.mid"
    context_path = root / "context" / "candidate_context.mid"
    write_solo(solo_path, pitches)
    write_context(context_path, solo_path)
    return {
        "output_dir": str(root),
        "candidates": [
            {
                "candidate_id": "margin_recovered_rank_2_seed_31_sample_5",
                "review_files": {
                    "midi_path": str(solo_path),
                    "context_midi_path": str(context_path),
                },
                "source_metrics": {
                    "dead_air_ratio": dead_air,
                    "onset_coverage_ratio": 0.5,
                    "sustained_coverage_ratio": 0.7,
                },
                "listening": {"decision": "keep"},
                "focused_package_transform": {
                    "context_chords": ["Cm7", "Fm7"],
                    "context_bpm": 124,
                    "context_bars": 2,
                },
            }
        ],
    }


class MarginRecoveredFocusedContextDecisionTest(unittest.TestCase):
    def test_low_pitch_variety_candidate_needs_followup(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = sample_package(root, pitches=[60, 62, 63, 65, 60, 62, 63, 65, 60, 62, 63, 65], dead_air=0.45)

            report = build_focused_context_decision(package, output_dir=root / "decision")
            summary = validate_decision(
                report,
                expected_candidate_id="margin_recovered_rank_2_seed_31_sample_5",
                expected_decision="needs_followup",
            )

            candidate = report["candidates"][0]
            self.assertEqual(summary["decisions"], ["needs_followup"])
            self.assertIn("low_pitch_variety", candidate["decision_flags"])
            self.assertIn("dead_air_needs_review", candidate["decision_flags"])
            self.assertTrue(candidate["context_summary"]["has_chord_guide"])

    def test_cleaner_candidate_can_move_to_focused_listening(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pitches = [60, 62, 63, 65, 67, 69, 70, 72, 74, 76, 77, 79]
            package = sample_package(root, pitches=pitches, dead_air=0.20)

            report = build_focused_context_decision(package, output_dir=root / "decision")
            summary = validate_decision(
                report,
                expected_candidate_id="margin_recovered_rank_2_seed_31_sample_5",
                expected_decision="keep_for_focused_listening",
            )

            self.assertEqual(summary["decisions"], ["keep_for_focused_listening"])


if __name__ == "__main__":
    unittest.main()
