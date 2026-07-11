from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair import (
    BOUNDARY,
    NEXT_BOUNDARY,
    PRIMARY_TARGET,
    SOURCE_BOUNDARY,
    StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError,
    build_sparse_phrase_repair_decision,
    validate_sparse_phrase_repair_decision,
)


def candidate(rank: int, *, gap_ratio: float, max_gap: float, max_interval: int, adjacent: int) -> dict:
    return {
        "review_rank": rank,
        "metrics": {
            "gap_ratio_to_window": gap_ratio,
            "max_internal_gap_beats": max_gap,
            "note_count": 9 + rank,
            "max_abs_interval": max_interval,
            "adjacent_repeat_count": adjacent,
        },
        "evidence_flags": ["high_dead_air_or_sparse_phrase"],
    }


def rejection_analysis(*, quality_cause_claimed: bool = False, primary_target: str = PRIMARY_TARGET) -> dict:
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis_v1"
        ),
        "analysis_boundary": {
            "boundary": SOURCE_BOUNDARY,
            "input_reject_all_verified": True,
            "human_audio_keep_claimed": False,
            "musical_quality_claimed": False,
            "quality_cause_claimed": quality_cause_claimed,
        },
        "rejection_analysis": {
            "candidate_count": 3,
            "common_evidence_flags": ["high_dead_air_or_sparse_phrase"],
            "evidence_flag_counts": {
                "high_dead_air_or_sparse_phrase": 3,
                "long_internal_gap_present": 2,
                "octave_or_larger_interval_present": 2,
            },
            "primary_next_repair_target": primary_target,
        },
        "candidates": [
            candidate(1, gap_ratio=0.4688, max_gap=1.5, max_interval=9, adjacent=1),
            candidate(2, gap_ratio=0.4688, max_gap=0.75, max_interval=12, adjacent=1),
            candidate(3, gap_ratio=0.5312, max_gap=1.25, max_interval=12, adjacent=0),
        ],
        "decision": {
            "next_boundary": BOUNDARY,
        },
    }


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionTest(
    unittest.TestCase
):
    def test_builds_sparse_phrase_repair_decision_without_quality_claim(self) -> None:
        report = build_sparse_phrase_repair_decision(
            rejection_analysis(),
            output_dir=Path("outputs/decision"),
            target_max_gap_ratio_to_window=0.40,
            target_max_internal_gap_beats=0.75,
            target_min_note_count=10,
            target_min_phrase_coverage_ratio=0.90,
            target_max_tail_empty_steps=0,
            target_max_abs_interval=12,
        )
        summary = validate_sparse_phrase_repair_decision(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_primary_target=PRIMARY_TARGET,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["candidate_count"], 3)
        self.assertEqual(summary["primary_repair_target"], PRIMARY_TARGET)
        self.assertEqual(summary["target_max_gap_ratio_to_window"], 0.40)
        self.assertEqual(summary["target_max_internal_gap_beats"], 0.75)
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertFalse(summary["quality_cause_claimed"])
        self.assertTrue(summary["auto_progress_allowed"])

    def test_rejects_quality_cause_claimed_source(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError
        ):
            build_sparse_phrase_repair_decision(
                rejection_analysis(quality_cause_claimed=True),
                output_dir=Path("outputs/decision"),
                target_max_gap_ratio_to_window=0.40,
                target_max_internal_gap_beats=0.75,
                target_min_note_count=10,
                target_min_phrase_coverage_ratio=0.90,
                target_max_tail_empty_steps=0,
                target_max_abs_interval=12,
            )

    def test_rejects_non_sparse_primary_target(self) -> None:
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError
        ):
            build_sparse_phrase_repair_decision(
                rejection_analysis(primary_target="manual_review"),
                output_dir=Path("outputs/decision"),
                target_max_gap_ratio_to_window=0.40,
                target_max_internal_gap_beats=0.75,
                target_min_note_count=10,
                target_min_phrase_coverage_ratio=0.90,
                target_max_tail_empty_steps=0,
                target_max_abs_interval=12,
            )

    def test_validation_requires_gap_target_below_observed_max(self) -> None:
        report = build_sparse_phrase_repair_decision(
            rejection_analysis(),
            output_dir=Path("outputs/decision"),
            target_max_gap_ratio_to_window=0.60,
            target_max_internal_gap_beats=0.75,
            target_min_note_count=10,
            target_min_phrase_coverage_ratio=0.90,
            target_max_tail_empty_steps=0,
            target_max_abs_interval=12,
        )
        with self.assertRaises(
            StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError
        ):
            validate_sparse_phrase_repair_decision(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_primary_target=PRIMARY_TARGET,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
