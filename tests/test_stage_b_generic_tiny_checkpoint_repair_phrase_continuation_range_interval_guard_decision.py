from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard import (
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError,
    build_guard_decision,
    validate_guard_decision,
)


def failure_review(*, keep_claimed: bool = False, max_interval: int = 60, pitch_span: int = 60) -> dict:
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review_v1",
        "user_listening_review": {
            "overall_decision": "reject_all",
            "primary_failure": "midi_note_random_large_leaps",
        },
        "midi_note_failure": {
            "all_reviewed_candidates_failed": True,
        },
        "claim_boundary": {
            "boundary": "generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all",
            "human_audio_keep_claimed": keep_claimed,
            "musical_quality_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
        },
        "reviewed_audio_files": [
            {
                "midi_note_audit": {
                    "note_count": 9,
                    "pitch_min": 29,
                    "pitch_max": 89,
                    "pitch_span": pitch_span,
                    "max_abs_interval": max_interval,
                    "large_interval_ratio": 0.875,
                    "severe_interval_count": 6,
                    "intervals": [15, -24, 60, -60, 34, -3, 27, -34],
                    "pitch_name_sequence": ["D2", "F3", "F1", "F6"],
                }
            }
        ],
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionTest(unittest.TestCase):
    def test_builds_guard_decision_without_quality_claim(self) -> None:
        report = build_guard_decision(
            failure_review(),
            output_dir=Path("outputs/decision"),
            target_max_pitch_span=24,
            target_max_abs_interval=12,
            target_max_large_interval_ratio=0.35,
            target_max_severe_interval_count=0,
        )
        summary = validate_guard_decision(
            report,
            expected_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
            expected_next_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep",
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["observed_max_abs_interval"], 60)
        self.assertEqual(summary["target_max_abs_interval"], 12)
        self.assertEqual(summary["repair_target_count"], 5)
        self.assertTrue(summary["auto_progress_allowed"])
        self.assertFalse(summary["musical_quality_claimed"])

    def test_rejects_keep_claimed_source(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError):
            build_guard_decision(
                failure_review(keep_claimed=True),
                output_dir=Path("outputs/decision"),
                target_max_pitch_span=24,
                target_max_abs_interval=12,
                target_max_large_interval_ratio=0.35,
                target_max_severe_interval_count=0,
            )

    def test_validation_requires_observed_interval_above_target(self) -> None:
        report = build_guard_decision(
            failure_review(max_interval=8, pitch_span=10),
            output_dir=Path("outputs/decision"),
            target_max_pitch_span=24,
            target_max_abs_interval=12,
            target_max_large_interval_ratio=0.35,
            target_max_severe_interval_count=0,
        )
        with self.assertRaises(StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardDecisionError):
            validate_guard_decision(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
                expected_next_boundary="stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep",
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
