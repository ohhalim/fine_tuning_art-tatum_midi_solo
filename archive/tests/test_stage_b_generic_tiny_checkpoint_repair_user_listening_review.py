from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_generic_tiny_checkpoint_repair_user_listening_review import (
    StageBGenericTinyCheckpointRepairUserListeningReviewError,
    build_user_listening_review,
    validate_user_listening_review,
)


def audio_render_report(*, quality_claimed: bool = False) -> dict:
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt_v1",
        "audio_render_boundary": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_local_audio_render_attempt",
            "render_attempted": True,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": quality_claimed,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
        },
        "rendered_audio_files": [
            {
                "review_rank": 1,
                "sample_seed": 47,
                "sample_index": 6,
                "source_midi_path": "rank_01.mid",
                "wav_file": {
                    "path": "rank_01.wav",
                    "exists": True,
                    "duration_seconds": 8.491,
                    "sample_rate": 44100,
                    "sha256": "a" * 64,
                },
            },
            {
                "review_rank": 2,
                "sample_seed": 45,
                "sample_index": 4,
                "source_midi_path": "rank_02.mid",
                "wav_file": {
                    "path": "rank_02.wav",
                    "exists": True,
                    "duration_seconds": 10.657,
                    "sample_rate": 44100,
                    "sha256": "b" * 64,
                },
            },
        ],
    }


class StageBGenericTinyCheckpointRepairUserListeningReviewTest(unittest.TestCase):
    def test_records_reject_all_plunk_and_stop_without_quality_claim(self) -> None:
        report = build_user_listening_review(
            audio_render_report(),
            output_dir=Path("outputs/review"),
            reviewer="user",
            overall_decision="reject_all",
            candidate_decision="reject",
            primary_failure="plunk_and_stop",
            timing="too_short_or_stiff",
            phrase="fragmented",
            vocabulary="not_musical",
            assessment="all candidates only plunk briefly and end",
            notes="single user listening review",
        )
        summary = validate_user_listening_review(
            report,
            expected_boundary="generic_tiny_checkpoint_repair_audio_review_reject_all",
            expected_overall_decision="reject_all",
            expected_primary_failure="plunk_and_stop",
            expected_file_count=2,
            require_no_keep_claim=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_decision"], "reject")
        self.assertEqual(summary["primary_failure"], "plunk_and_stop")
        self.assertFalse(summary["human_audio_keep_claimed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertTrue(summary["auto_progress_allowed"])

    def test_rejects_source_audio_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairUserListeningReviewError):
            build_user_listening_review(
                audio_render_report(quality_claimed=True),
                output_dir=Path("outputs/review"),
                reviewer="user",
                overall_decision="reject_all",
                candidate_decision="reject",
                primary_failure="plunk_and_stop",
                timing="too_short_or_stiff",
                phrase="fragmented",
                vocabulary="not_musical",
                assessment="all candidates only plunk briefly and end",
                notes="",
            )

    def test_rejects_invalid_failure_value(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairUserListeningReviewError):
            build_user_listening_review(
                audio_render_report(),
                output_dir=Path("outputs/review"),
                reviewer="user",
                overall_decision="reject_all",
                candidate_decision="reject",
                primary_failure="invalid",
                timing="too_short_or_stiff",
                phrase="fragmented",
                vocabulary="not_musical",
                assessment="all candidates only plunk briefly and end",
                notes="",
            )


if __name__ == "__main__":
    unittest.main()
