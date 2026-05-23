from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from scripts.build_clean_review_package import build_clean_review_package, markdown_report


class CleanReviewPackageTest(unittest.TestCase):
    def sample_review_manifest(self, midi_path: str, context_path: str) -> dict:
        return {
            "output_dir": "outputs/review",
            "candidates": [
                {
                    "mode": "data_motif_phrase_recovery",
                    "review_rank": 1,
                    "sample_index": 1,
                    "review_midi_path": midi_path,
                    "context_midi_path": context_path,
                },
                {
                    "mode": "data_motif_guide_tones",
                    "review_rank": 1,
                    "sample_index": 1,
                    "review_midi_path": "warning.mid",
                    "context_midi_path": "warning_context.mid",
                },
            ],
        }

    def sample_objective_report(self) -> dict:
        return {
            "report_path": "outputs/objective/objective_midi_note_review.json",
            "candidates": [
                {
                    "candidate_id": "data_motif_phrase_recovery_rank_1_sample_1",
                    "objective_bucket": "clean",
                    "objective_flags": [],
                    "objective_priority_score": 100,
                    "metrics": {
                        "note_count": 63,
                        "unique_pitch_count": 24,
                        "stepwise_interval_ratio": 0.1,
                        "chromatic_interval_ratio": 0.05,
                        "large_leap_interval_ratio": 0.3,
                        "unresolved_large_leap_ratio": 0.0,
                        "chord_tone_ratio": 0.5,
                        "tension_ratio": 0.48,
                        "outside_ratio": 0.0,
                        "root_tone_ratio": 0.0,
                    },
                },
                {
                    "candidate_id": "data_motif_guide_tones_rank_1_sample_1",
                    "objective_bucket": "warning",
                    "objective_flags": ["unresolved_large_leaps"],
                    "objective_priority_score": 84,
                    "metrics": {"note_count": 63},
                },
            ],
        }

    def test_build_clean_review_package_filters_clean_allowed_modes(self) -> None:
        package = build_clean_review_package(
            self.sample_review_manifest("clean.mid", "clean_context.mid"),
            self.sample_objective_report(),
            output_dir=Path("outputs/clean"),
            allowed_modes={"data_motif_phrase_recovery"},
            copy_files=False,
        )

        self.assertEqual(package["candidate_count"], 1)
        candidate = package["candidates"][0]
        self.assertEqual(candidate["candidate_id"], "data_motif_phrase_recovery_rank_1_sample_1")
        self.assertEqual(candidate["objective_flags"], [])
        self.assertEqual(candidate["metrics"]["unresolved_large_leap_ratio"], 0.0)

    def test_build_clean_review_package_can_copy_files(self) -> None:
        with TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            midi_path = tmp_path / "clean.mid"
            context_path = tmp_path / "clean_context.mid"
            midi_path.write_bytes(b"midi")
            context_path.write_bytes(b"context")

            package = build_clean_review_package(
                self.sample_review_manifest(str(midi_path), str(context_path)),
                self.sample_objective_report(),
                output_dir=tmp_path / "package",
                allowed_modes={"data_motif_phrase_recovery"},
                copy_files=True,
            )

            candidate = package["candidates"][0]
            self.assertTrue(Path(candidate["review_midi_path"]).exists())
            self.assertTrue(Path(candidate["context_midi_path"]).exists())
            self.assertIn("package/midi", candidate["review_midi_path"])
            self.assertIn("package/context_midi", candidate["context_midi_path"])

    def test_markdown_report_lists_review_paths(self) -> None:
        package = build_clean_review_package(
            self.sample_review_manifest("clean.mid", "clean_context.mid"),
            self.sample_objective_report(),
            output_dir=Path("outputs/clean"),
            allowed_modes={"data_motif_phrase_recovery"},
            copy_files=False,
        )
        markdown = markdown_report(package)

        self.assertIn("data_motif_phrase_recovery_rank_1_sample_1", markdown)
        self.assertIn("clean_context.mid", markdown)


if __name__ == "__main__":
    unittest.main()
