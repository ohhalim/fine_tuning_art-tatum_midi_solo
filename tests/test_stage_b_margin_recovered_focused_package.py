from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pretty_midi

from scripts.build_stage_b_margin_recovered_focused_package import (
    MarginRecoveredFocusedPackageError,
    build_margin_recovered_focused_package,
    validate_package,
)


def write_test_midi(path: Path) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=124)
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="test_solo")
    piano.notes.extend(
        [
            pretty_midi.Note(velocity=88, pitch=62, start=0.0, end=0.25),
            pretty_midi.Note(velocity=88, pitch=65, start=0.5, end=0.75),
        ]
    )
    midi.instruments.append(piano)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))


def sample_review_notes(root: Path) -> dict:
    run_dir = root / "outputs" / "stage_b_generation_probe" / "seed31"
    midi_path = run_dir / "samples" / "stage_b_sample_5.mid"
    write_test_midi(midi_path)
    (run_dir / "report.json").write_text(
        """{
  "request": {
    "bpm": 124,
    "bars": 2,
    "chord_progression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"]
  }
}
""",
        encoding="utf-8",
    )
    return {
        "schema_version": "stage_b_margin_recovered_listening_notes_v1",
        "candidates": [
            {
                "candidate_id": "margin_recovered_rank_2_seed_31_sample_5",
                "review_metadata": {"review_rank": 2, "seed": 31, "sample_index": 5},
                "review_files": {"midi_path": str(midi_path)},
                "source_metrics": {"note_count": 19, "unique_pitch_count": 4, "tension_ratio": 0.2},
                "listening": {
                    "status": "reviewed",
                    "timing": "acceptable",
                    "phrase": "strong",
                    "phrase_quality": "strong",
                    "chord_fit": "proxy",
                    "jazz_vocabulary": "acceptable",
                    "decision": "keep",
                    "notes": "proxy keep",
                },
                "objective_review": {"objective_flags": []},
            }
        ],
    }


class MarginRecoveredFocusedPackageTest(unittest.TestCase):
    def test_build_package_creates_context_and_copies_solo_context_pair(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            package = build_margin_recovered_focused_package(
                sample_review_notes(root),
                output_dir=root / "focused",
            )
            summary = validate_package(
                package,
                expected_candidate_id="margin_recovered_rank_2_seed_31_sample_5",
                min_candidates=1,
            )

            self.assertEqual(summary["candidate_count"], 1)
            self.assertEqual(summary["copied_midi_files"], 2)
            candidate = package["candidates"][0]
            self.assertEqual(candidate["objective_first_16_notes"][0]["pitch"], 62)
            self.assertTrue(Path(candidate["review_files"]["midi_path"]).exists())
            self.assertTrue(Path(candidate["review_files"]["context_midi_path"]).exists())

    def test_validate_rejects_unexpected_candidate(self) -> None:
        package = {
            "candidates": [
                {
                    "candidate_id": "other",
                    "review_files": {"midi_path": "x", "context_midi_path": "y"},
                }
            ]
        }

        with self.assertRaises(MarginRecoveredFocusedPackageError):
            validate_package(package, expected_candidate_id="expected", min_candidates=1)


if __name__ == "__main__":
    unittest.main()
