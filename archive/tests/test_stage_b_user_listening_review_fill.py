from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_duration_coverage_user_listening_review import (
    StageBDurationCoverageUserListeningReviewFillError,
    build_user_listening_review_fill,
    validate_user_listening_review_fill,
)


def audio_render_report(*, preference_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_duration_coverage_fill_local_audio_render_attempt_v1",
        "candidate_id": "duration_fill_candidate",
        "audio_render_boundary": {
            "render_attempted": True,
            "technical_wav_validation": True,
            "human_audio_preference_claimed": preference_claimed,
        },
        "rendered_audio_files": [
            {
                "role": "source_constrained_partial",
                "candidate_id": "source_candidate",
                "source_midi_path": "source.mid",
                "wav_file": {
                    "path": "source.wav",
                    "exists": True,
                    "duration_seconds": 6.474,
                    "sample_rate": 44100,
                    "sha256": "a" * 64,
                },
            },
            {
                "role": "duration_coverage_fill_keep",
                "candidate_id": "duration_fill_candidate",
                "source_midi_path": "fill.mid",
                "wav_file": {
                    "path": "fill.wav",
                    "exists": True,
                    "duration_seconds": 6.474,
                    "sample_rate": 44100,
                    "sha256": "b" * 64,
                },
            },
        ],
    }


class StageBDurationCoverageFillUserListeningReviewFillTest(unittest.TestCase):
    def test_records_fill_preference_without_broad_quality_claim(self) -> None:
        report = build_user_listening_review_fill(
            audio_render_report(),
            output_dir=Path("outputs/user_review"),
            reviewer="user",
            preference="duration_coverage_fill_keep",
            timing="duration_coverage_fill_keep",
            phrase="duration_coverage_fill_keep",
            vocabulary="duration_coverage_fill_keep",
            source_assessment="source sounds like random notes and is hard to understand",
            fill_assessment="fill sounds much more jazz-like as soloing",
            notes="single user listening review after WAV render",
        )
        summary = validate_user_listening_review_fill(
            report,
            expected_preference="duration_coverage_fill_keep",
            require_human_audio_preference=True,
            require_no_broad_quality_claim=True,
        )

        self.assertEqual(summary["review_status"], "reviewed")
        self.assertEqual(summary["preference"], "duration_coverage_fill_keep")
        self.assertTrue(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["broad_model_quality_claimed"])
        self.assertIn("multi_reviewer_preference", report["not_proven"])

    def test_rejects_already_claimed_source_report(self) -> None:
        with self.assertRaises(StageBDurationCoverageUserListeningReviewFillError):
            build_user_listening_review_fill(
                audio_render_report(preference_claimed=True),
                output_dir=Path("outputs/user_review"),
                reviewer="user",
                preference="duration_coverage_fill_keep",
                timing="duration_coverage_fill_keep",
                phrase="duration_coverage_fill_keep",
                vocabulary="duration_coverage_fill_keep",
                source_assessment="source rejected",
                fill_assessment="fill preferred",
                notes="",
            )


if __name__ == "__main__":
    unittest.main()
