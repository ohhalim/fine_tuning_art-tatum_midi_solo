from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointDensityCoverageRemainingBlockerDecisionError,
    build_decision_report,
    validate_decision_report,
)


def repair_report(*, target_qualified: bool = True, long_note_failure: bool = True) -> dict:
    failure_reasons = {"too many long notes: 0.333 > 0.250": 2} if long_note_failure else {}
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe_v1",
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "density_coverage_target_qualified": target_qualified,
            "raw_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 1,
            "strict_valid_sample_count": 1,
            "grammar_gate_sample_count": 3,
            "diagnostic_failure_reasons": failure_reasons,
            "max_longest_sustained_empty_run_steps": 7,
        },
        "comparison": {
            "note_count_failure_delta": 3,
            "onset_coverage_delta": 0.10416666666666666,
            "sustained_coverage_delta": 0.5416666666666666,
        },
    }


class StageBGenericBaseScaleCheckpointDensityCoverageRemainingBlockerDecisionTest(unittest.TestCase):
    def test_selects_duration_long_note_repair_target(self) -> None:
        report = build_decision_report(
            repair_report(),
            output_dir=Path("outputs/remaining_blocker"),
        )
        summary = validate_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_duration_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["decision"], "select_duration_long_note_repair_probe")
        self.assertEqual(summary["selected_target"], "duration_long_note_ratio_repair")
        self.assertEqual(summary["long_note_failure_count"], 2)
        self.assertEqual(summary["remaining_blocker"], "duration_long_note_ratio")
        self.assertFalse(summary["audio_review_selected"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unqualified_density_coverage_target(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDensityCoverageRemainingBlockerDecisionError):
            build_decision_report(
                repair_report(target_qualified=False),
                output_dir=Path("outputs/remaining_blocker"),
            )

    def test_rejects_missing_long_note_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDensityCoverageRemainingBlockerDecisionError):
            build_decision_report(
                repair_report(long_note_failure=False),
                output_dir=Path("outputs/remaining_blocker"),
            )


if __name__ == "__main__":
    unittest.main()
