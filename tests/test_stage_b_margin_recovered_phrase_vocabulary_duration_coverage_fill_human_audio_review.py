from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review import (
    DurationCoverageFillHumanAudioReviewFillError,
    build_review_fill,
    validate_review_fill,
)


def human_audio_boundary() -> dict:
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary_v1",
        "review_items": [
            {
                "role": "source_constrained_partial",
                "candidate_id": "source_candidate",
            },
            {
                "role": "duration_coverage_fill_keep",
                "candidate_id": "duration_fill_candidate",
            },
        ],
        "human_audio_boundary": {
            "status": "pending",
            "preference_claimed": False,
        },
    }


class StageBMarginRecoveredPhraseVocabularyDurationCoverageFillHumanAudioReviewTest(unittest.TestCase):
    def test_keeps_pending_without_review_input(self) -> None:
        report = build_review_fill(
            human_audio_boundary(),
            None,
            output_dir=Path("outputs/review_fill"),
        )
        summary = validate_review_fill(
            report,
            expected_candidate_id="duration_fill_candidate",
            require_pending_without_input=True,
            require_no_preference_without_input=True,
        )

        self.assertFalse(summary["review_input_present"])
        self.assertEqual(summary["fill_status"], "pending_review_input")
        self.assertEqual(summary["human_audio_status"], "pending")
        self.assertEqual(summary["preference"], "pending")
        self.assertFalse(summary["preference_claimed"])
        self.assertIn("human_audio_preference", report["not_proven"])

    def test_accepts_valid_review_input(self) -> None:
        report = build_review_fill(
            human_audio_boundary(),
            {
                "candidate_id": "duration_fill_candidate",
                "reviewer": "reviewer-a",
                "audio_render_used": True,
                "preference": "duration_coverage_fill_keep",
                "timing": "duration_coverage_fill_keep",
                "phrase": "tie",
                "vocabulary": "duration_coverage_fill_keep",
                "notes": "validated external review input",
            },
            output_dir=Path("outputs/review_fill"),
        )
        summary = validate_review_fill(
            report,
            expected_candidate_id="duration_fill_candidate",
            require_pending_without_input=True,
            require_no_preference_without_input=True,
        )

        self.assertTrue(summary["review_input_present"])
        self.assertEqual(summary["fill_status"], "review_input_applied")
        self.assertEqual(summary["human_audio_status"], "reviewed")
        self.assertEqual(summary["preference"], "duration_coverage_fill_keep")
        self.assertTrue(summary["preference_claimed"])

    def test_rejects_review_without_audio_render(self) -> None:
        with self.assertRaises(DurationCoverageFillHumanAudioReviewFillError):
            build_review_fill(
                human_audio_boundary(),
                {
                    "candidate_id": "duration_fill_candidate",
                    "reviewer": "reviewer-a",
                    "audio_render_used": False,
                    "preference": "duration_coverage_fill_keep",
                    "timing": "duration_coverage_fill_keep",
                    "phrase": "tie",
                    "vocabulary": "duration_coverage_fill_keep",
                },
                output_dir=Path("outputs/review_fill"),
            )

    def test_rejects_candidate_mismatch(self) -> None:
        with self.assertRaises(DurationCoverageFillHumanAudioReviewFillError):
            build_review_fill(
                human_audio_boundary(),
                {
                    "candidate_id": "other_candidate",
                    "reviewer": "reviewer-a",
                    "audio_render_used": True,
                    "preference": "duration_coverage_fill_keep",
                    "timing": "duration_coverage_fill_keep",
                    "phrase": "tie",
                    "vocabulary": "duration_coverage_fill_keep",
                },
                output_dir=Path("outputs/review_fill"),
            )


if __name__ == "__main__":
    unittest.main()
