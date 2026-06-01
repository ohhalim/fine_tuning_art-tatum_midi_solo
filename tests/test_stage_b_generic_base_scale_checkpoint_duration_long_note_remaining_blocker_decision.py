from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointDurationLongNoteRemainingBlockerDecisionError,
    build_decision_report,
    validate_decision_report,
)


def duration_repair_report(
    *,
    target_qualified: bool = True,
    long_note_removed: bool = True,
    dead_air_failure: bool = True,
    coverage_regression: bool = True,
) -> dict:
    failure_reasons = {"dead-air ratio too high: 0.800 >= 0.800": 1} if dead_air_failure else {}
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe_v1",
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "duration_long_note_target_qualified": target_qualified,
            "raw_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "duration_repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 2,
            "strict_valid_sample_count": 2,
            "grammar_gate_sample_count": 3,
            "long_note_failure_count": 0 if long_note_removed else 1,
            "diagnostic_failure_reasons": failure_reasons,
            "avg_onset_coverage_ratio": 0.1875,
            "avg_sustained_coverage_ratio": 0.3645833333333333,
            "max_longest_sustained_empty_run_steps": 8,
        },
        "comparison": {
            "long_note_failure_delta": 2,
            "valid_sample_delta": 1,
            "strict_valid_sample_delta": 1,
            "onset_coverage_delta": 0.020833333333333343,
            "sustained_coverage_delta": -0.2708333333333333 if coverage_regression else 0.0,
            "coverage_regression_observed": coverage_regression,
        },
    }


class StageBGenericBaseScaleCheckpointDurationLongNoteRemainingBlockerDecisionTest(
    unittest.TestCase
):
    def test_selects_sustained_coverage_dead_air_repair_target(self) -> None:
        report = build_decision_report(
            duration_repair_report(),
            output_dir=Path("outputs/remaining_blocker"),
        )
        summary = validate_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_dead_air_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["decision"], "select_sustained_coverage_dead_air_repair_probe")
        self.assertEqual(summary["selected_target"], "sustained_coverage_dead_air_repair")
        self.assertEqual(summary["long_note_failure_count"], 0)
        self.assertEqual(summary["dead_air_failure_count"], 1)
        self.assertTrue(summary["coverage_regression_observed"])
        self.assertEqual(summary["remaining_blocker"], "sustained_coverage_dead_air")
        self.assertFalse(summary["audio_review_selected"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unqualified_duration_target(self) -> None:
        with self.assertRaises(
            StageBGenericBaseScaleCheckpointDurationLongNoteRemainingBlockerDecisionError
        ):
            build_decision_report(
                duration_repair_report(target_qualified=False),
                output_dir=Path("outputs/remaining_blocker"),
            )

    def test_rejects_remaining_long_note_failure(self) -> None:
        with self.assertRaises(
            StageBGenericBaseScaleCheckpointDurationLongNoteRemainingBlockerDecisionError
        ):
            build_decision_report(
                duration_repair_report(long_note_removed=False),
                output_dir=Path("outputs/remaining_blocker"),
            )

    def test_rejects_missing_remaining_blocker_evidence(self) -> None:
        with self.assertRaises(
            StageBGenericBaseScaleCheckpointDurationLongNoteRemainingBlockerDecisionError
        ):
            build_decision_report(
                duration_repair_report(dead_air_failure=False, coverage_regression=False),
                output_dir=Path("outputs/remaining_blocker"),
            )


if __name__ == "__main__":
    unittest.main()
