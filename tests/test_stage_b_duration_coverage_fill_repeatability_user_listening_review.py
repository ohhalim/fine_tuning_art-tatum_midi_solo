from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_duration_coverage_repeatability_user_listening_review import (
    StageBDurationCoverageRepeatabilityUserListeningReviewError,
    build_repeatability_user_listening_review,
    validate_repeatability_user_listening_review,
)


def audio_review_package(*, preference_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_audio_review_package_v1",
        "candidate_id": "duration_coverage_fill_repeatability_sources",
        "audio_review_boundary": {
            "technical_wav_validation": True,
            "human_audio_preference_claimed": preference_claimed,
            "broad_model_quality_claimed": False,
        },
        "review_items": [
            {
                "role": "repeatability_sample_seed_155_duration_fill",
                "candidate_id": "candidate_155",
                "source_candidate_id": "source_155",
                "sample_seed": 155,
                "midi_file": {"path": "sample_155.mid"},
                "wav_file": {
                    "path": "sample_155.wav",
                    "exists": True,
                    "duration_seconds": 6.622,
                    "sample_rate": 44100,
                    "sha256": "a" * 64,
                },
                "metrics": {
                    "selected_dead_air_ratio": 0.3333333333333333,
                    "selected_focused_unique_pitch_count": 12,
                },
            },
            {
                "role": "repeatability_sample_seed_131_duration_fill",
                "candidate_id": "candidate_131",
                "source_candidate_id": "source_131",
                "sample_seed": 131,
                "midi_file": {"path": "sample_131.mid"},
                "wav_file": {
                    "path": "sample_131.wav",
                    "exists": True,
                    "duration_seconds": 6.866,
                    "sample_rate": 44100,
                    "sha256": "b" * 64,
                },
                "metrics": {
                    "selected_dead_air_ratio": 0.35294117647058826,
                    "selected_focused_unique_pitch_count": 13,
                },
            },
        ],
    }


class StageBDurationCoverageFillRepeatabilityUserListeningReviewTest(unittest.TestCase):
    def test_records_reject_all_without_keep_or_quality_claim(self) -> None:
        report = build_repeatability_user_listening_review(
            audio_review_package(),
            output_dir=Path("outputs/review"),
            reviewer="user",
            overall_decision="reject_all",
            candidate_decision="needs_followup",
            timing="outside_or_unclear",
            phrase="outside_or_unclear",
            vocabulary="outside_or_unclear",
            assessment="both candidates sound difficult and outside-soloing-like",
            notes="single user listening review",
        )
        summary = validate_repeatability_user_listening_review(
            report,
            expected_boundary="repeatability_audio_review_needs_followup",
            expected_overall_decision="reject_all",
            require_no_keep_claim=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["reviewed_audio_file_count"], 2)
        self.assertEqual(summary["candidate_decision"], "needs_followup")
        self.assertFalse(summary["repeatability_human_audio_keep_claimed"])
        self.assertFalse(summary["broad_model_quality_claimed"])
        self.assertIn("repeatability_human_audio_keep", report["not_proven"])

    def test_rejects_source_package_with_preference_claim(self) -> None:
        with self.assertRaises(StageBDurationCoverageRepeatabilityUserListeningReviewError):
            build_repeatability_user_listening_review(
                audio_review_package(preference_claimed=True),
                output_dir=Path("outputs/review"),
                reviewer="user",
                overall_decision="reject_all",
                candidate_decision="needs_followup",
                timing="outside_or_unclear",
                phrase="outside_or_unclear",
                vocabulary="outside_or_unclear",
                assessment="both candidates need follow-up",
                notes="",
            )


if __name__ == "__main__":
    unittest.main()
