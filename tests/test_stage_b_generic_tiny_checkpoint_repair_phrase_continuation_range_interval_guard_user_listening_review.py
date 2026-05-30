from __future__ import annotations

import unittest
from pathlib import Path

from scripts.fill_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review import (
    AUDIO_BOUNDARY,
    NEXT_BOUNDARY,
    REVIEW_BOUNDARY_REJECT_ALL,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardUserListeningReviewError,
    build_user_listening_review,
    validate_user_listening_review,
)


def rendered_item(rank: int) -> dict:
    return {
        "review_rank": rank,
        "interval_cap": 9,
        "sample_seed": 70 + rank,
        "sample_index": rank,
        "source_midi_path": f"sample_{rank}.mid",
        "wav_file": {
            "path": f"sample_{rank}.wav",
            "exists": True,
            "duration_seconds": 6.8 + rank / 10,
            "sample_rate": 44100,
            "sha256": str(rank) * 64,
        },
    }


def audio_render_report(*, quality_claimed: bool = False) -> dict:
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_local_audio_render_attempt_v1"
        ),
        "audio_render_boundary": {
            "boundary": AUDIO_BOUNDARY,
            "render_attempted": True,
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "audio_rendered_quality_claimed": quality_claimed,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
        },
        "rendered_audio_files": [
            rendered_item(1),
            rendered_item(2),
            rendered_item(3),
        ],
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardUserListeningReviewTest(
    unittest.TestCase
):
    def test_records_reject_all_without_keep_or_quality_claim(self) -> None:
        report = build_user_listening_review(
            audio_render_report(),
            output_dir=Path("outputs/review"),
            reviewer="user",
            overall_decision="reject_all",
            candidate_decision="reject",
            primary_failure="subjective_not_musical",
            timing="outside_or_unclear",
            phrase="not_musical",
            vocabulary="not_musical",
            assessment="all rendered candidates rejected by single-user listening review",
            notes="range/interval objective guard was insufficient for listening acceptance",
        )
        summary = validate_user_listening_review(
            report,
            expected_boundary=REVIEW_BOUNDARY_REJECT_ALL,
            expected_overall_decision="reject_all",
            expected_primary_failure="subjective_not_musical",
            expected_file_count=3,
            require_no_keep_claim=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["reviewed_audio_file_count"], 3)
        self.assertEqual(summary["candidate_decision"], "reject")
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertFalse(summary["human_audio_keep_claimed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertTrue(summary["auto_progress_allowed"])

    def test_rejects_source_quality_claim(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardUserListeningReviewError
        ):
            build_user_listening_review(
                audio_render_report(quality_claimed=True),
                output_dir=Path("outputs/review"),
                reviewer="user",
                overall_decision="reject_all",
                candidate_decision="reject",
                primary_failure="subjective_not_musical",
                timing="outside_or_unclear",
                phrase="not_musical",
                vocabulary="not_musical",
                assessment="all candidates rejected",
                notes="",
            )

    def test_rejects_invalid_primary_failure(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardUserListeningReviewError
        ):
            build_user_listening_review(
                audio_render_report(),
                output_dir=Path("outputs/review"),
                reviewer="user",
                overall_decision="reject_all",
                candidate_decision="reject",
                primary_failure="unsupported_failure",
                timing="outside_or_unclear",
                phrase="not_musical",
                vocabulary="not_musical",
                assessment="all candidates rejected",
                notes="",
            )


if __name__ == "__main__":
    unittest.main()
