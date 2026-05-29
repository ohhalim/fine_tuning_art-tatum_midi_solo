from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_duration_coverage_outside_soloing_repair_user_listening_review import (
    StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError,
    build_user_listening_review_fill,
    validate_user_listening_review_fill,
)


def audio_review_package(*, preference_claim: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package_v1",
        "candidate_id": "outside_soloing_repair_candidates",
        "audio_review_boundary": {
            "status": "ready_for_user_listening_review",
            "render_attempted": True,
            "rendered_audio_file_count": 2,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": False,
            "human_audio_preference_claimed": preference_claim,
            "broad_model_quality_claimed": False,
        },
        "review_items": [
            {
                "role": "outside_repair_sample_seed_155_contour_resolution",
                "candidate_id": "outside_155",
                "sample_seed": 155,
                "wav_file": {"path": "outputs/audio_155.wav", "exists": True},
                "metrics": {
                    "repaired_chord_tone_ratio": 1.0,
                    "repaired_max_non_chord_tone_run": 0,
                },
            },
            {
                "role": "outside_repair_sample_seed_131_contour_resolution",
                "candidate_id": "outside_131",
                "sample_seed": 131,
                "wav_file": {"path": "outputs/audio_131.wav", "exists": True},
                "metrics": {
                    "repaired_chord_tone_ratio": 1.0,
                    "repaired_max_non_chord_tone_run": 0,
                },
            },
        ],
    }


class StageBDurationCoverageFillOutsideSoloingRepairUserListeningReviewTest(unittest.TestCase):
    def test_keeps_pending_without_review_input_and_allows_objective_followup(self) -> None:
        report = build_user_listening_review_fill(
            audio_review_package(),
            None,
            output_dir=Path("outputs/review_fill"),
        )
        summary = validate_user_listening_review_fill(
            report,
            expected_boundary="outside_soloing_repair_audio_review_pending",
            require_pending_without_input=True,
            require_no_preference_without_input=True,
            require_objective_auto_progress_allowed=True,
        )

        self.assertFalse(summary["review_input_present"])
        self.assertEqual(summary["fill_status"], "pending_review_input")
        self.assertEqual(summary["user_listening_status"], "pending_review_input")
        self.assertEqual(summary["overall_decision"], "pending")
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertTrue(summary["objective_auto_progress_allowed"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertIn("human_audio_preference", report["not_proven"])

    def test_accepts_valid_review_input(self) -> None:
        report = build_user_listening_review_fill(
            audio_review_package(),
            {
                "candidate_id": "outside_soloing_repair_candidates",
                "reviewer": "user",
                "audio_render_used": True,
                "overall_decision": "keep_both",
                "timing": "improved",
                "phrase": "acceptable",
                "vocabulary": "improved",
                "assessment": "both repaired candidates are easier to follow",
                "candidate_reviews": [
                    {
                        "role": "outside_repair_sample_seed_155_contour_resolution",
                        "decision": "keep",
                    },
                    {
                        "role": "outside_repair_sample_seed_131_contour_resolution",
                        "decision": "keep",
                    },
                ],
            },
            output_dir=Path("outputs/review_fill"),
        )
        summary = validate_user_listening_review_fill(
            report,
            expected_boundary="outside_soloing_repair_audio_review_applied",
            require_pending_without_input=True,
            require_no_preference_without_input=True,
            require_objective_auto_progress_allowed=True,
        )

        self.assertTrue(summary["review_input_present"])
        self.assertEqual(summary["fill_status"], "review_input_applied")
        self.assertEqual(summary["user_listening_status"], "reviewed")
        self.assertEqual(summary["overall_decision"], "keep_both")
        self.assertTrue(summary["human_audio_preference_claimed"])

    def test_rejects_existing_preference_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError):
            build_user_listening_review_fill(
                audio_review_package(preference_claim=True),
                None,
                output_dir=Path("outputs/review_fill"),
            )


if __name__ == "__main__":
    unittest.main()
