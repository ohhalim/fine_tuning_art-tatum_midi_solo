from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import (
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    SoloYieldRepairedTop8ObjectiveFailureReviewError,
    build_objective_failure_review,
    validate_report,
)


def write_midi(path: Path, pitches: list[int]) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    instrument = pretty_midi.Instrument(program=0)
    for index, pitch in enumerate(pitches):
        start = index * 0.25
        instrument.notes.append(
            pretty_midi.Note(velocity=90, pitch=int(pitch), start=start, end=start + 0.12)
        )
    midi.instruments.append(instrument)
    midi.write(str(path))


def package_report(root: Path, *, quality_claim: bool = False) -> dict:
    paths = [root / f"candidate_{index}.mid" for index in range(1, 4)]
    write_midi(paths[0], [61, 63, 66, 68, 73, 75, 78, 80])
    write_midi(paths[1], [60, 64, 67, 71, 65, 69, 72, 75])
    write_midi(paths[2], [62, 66, 69, 73, 77, 74, 70, 68])
    return {
        "schema_version": "music_transformer_solo_yield_listening_package_v1",
        "output_dir": str(root / "package"),
        "candidate_count": 3,
        "candidates": [
            {
                "review_index": 1,
                "case_label": "minor_backdoor",
                "chords": "Cm7,F7,Bbmaj7,Ebmaj7",
                "review_midi_path": str(paths[0]),
                "review_wav_path": str(root / "candidate_1.wav"),
                "note_count": 28,
                "unique_pitch_count": 8,
                "dead_air_ratio": 0.72,
                "direction_change_ratio": 0.30,
                "syncopated_onset_ratio": 0.75,
                "chord_tone_ratio": 0.44,
                "tension_ratio": 0.16,
                "score": 230.0,
            },
            {
                "review_index": 2,
                "case_label": "major_ii_v_turnaround",
                "chords": "Dm7,G7,Cmaj7,A7",
                "review_midi_path": str(paths[1]),
                "review_wav_path": str(root / "candidate_2.wav"),
                "note_count": 33,
                "unique_pitch_count": 8,
                "dead_air_ratio": 0.51,
                "direction_change_ratio": 0.55,
                "syncopated_onset_ratio": 0.82,
                "chord_tone_ratio": 0.44,
                "tension_ratio": 0.12,
                "score": 231.0,
            },
            {
                "review_index": 3,
                "case_label": "dominant_cycle",
                "chords": "Em7,A7,Dmaj7,G7",
                "review_midi_path": str(paths[2]),
                "review_wav_path": str(root / "candidate_3.wav"),
                "note_count": 31,
                "unique_pitch_count": 8,
                "dead_air_ratio": 0.66,
                "direction_change_ratio": 0.58,
                "syncopated_onset_ratio": 0.78,
                "chord_tone_ratio": 0.44,
                "tension_ratio": 0.14,
                "score": 232.0,
            },
        ],
        "readiness": {
            "musical_quality_claimed": quality_claim,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
    }


class MusicTransformerSoloYieldRepairedTop8ObjectiveFailureReviewTest(unittest.TestCase):
    def test_builds_objective_failure_review_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = build_objective_failure_review(
                package_report(root),
                output_dir=root / "review",
            )
        summary = validate_report(report, min_candidates=3)

        self.assertEqual(summary["schema_version"], SCHEMA_VERSION)
        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["failed_candidate_count"], 3)
        self.assertGreaterEqual(summary["final_landing_not_chord_tone_count"], 2)
        self.assertGreaterEqual(summary["package_low_chord_tone_ratio_count"], 3)
        self.assertEqual(summary["selected_next_target"], SELECTED_TARGET)
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaises(SoloYieldRepairedTop8ObjectiveFailureReviewError):
                build_objective_failure_review(
                    package_report(root, quality_claim=True),
                    output_dir=root / "review",
                )

    def test_rejects_missing_midi_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            report = package_report(root)
            Path(report["candidates"][0]["review_midi_path"]).unlink()
            with self.assertRaises(SoloYieldRepairedTop8ObjectiveFailureReviewError):
                build_objective_failure_review(report, output_dir=root / "review")


if __name__ == "__main__":
    unittest.main()
