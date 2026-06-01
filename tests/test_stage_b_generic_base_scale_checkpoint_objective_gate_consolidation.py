from __future__ import annotations

import unittest
from pathlib import Path

from scripts.consolidate_stage_b_generic_base_scale_checkpoint_objective_gate import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError,
    build_consolidation_report,
    validate_consolidation_report,
)


def repair_probe(
    *,
    target_qualified: bool = True,
    all_pass: bool = True,
    quality_claim: bool = False,
    dead_air_failure_count: int = 0,
    long_note_failure_count: int = 0,
) -> dict:
    sample_count = 3
    pass_count = sample_count if all_pass else 2
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe_v1",
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "sustained_coverage_dead_air_target_qualified": target_qualified,
            "raw_generation_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "input": {
            "constrained_note_groups_per_bar": 8,
            "jazz_duration_tokens": True,
            "coverage_aware_positions": True,
        },
        "repair_summary": {
            "sample_count": sample_count,
            "valid_sample_count": pass_count,
            "strict_valid_sample_count": pass_count,
            "grammar_gate_sample_count": pass_count,
            "dead_air_failure_count": dead_air_failure_count,
            "long_note_failure_count": long_note_failure_count,
            "diagnostic_failure_reasons": {},
            "avg_onset_coverage_ratio": 0.3854166666666667,
            "avg_sustained_coverage_ratio": 0.6354166666666666,
            "max_longest_sustained_empty_run_steps": 4,
        },
        "comparison": {
            "dead_air_failure_delta": 1,
            "valid_sample_delta": 1,
            "strict_valid_sample_delta": 1,
            "onset_coverage_delta": 0.19791666666666669,
            "sustained_coverage_delta": 0.2708333333333333,
            "long_note_failure_reintroduced": long_note_failure_count > 0,
        },
    }


class StageBGenericBaseScaleCheckpointObjectiveGateConsolidationTest(unittest.TestCase):
    def test_selects_objective_gate_repeatability_sweep(self) -> None:
        report = build_consolidation_report(
            repair_probe(),
            output_dir=Path("outputs/objective_gate"),
        )
        summary = validate_consolidation_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_repeatability_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["decision"], "select_objective_gate_repeatability_sweep")
        self.assertEqual(summary["selected_target"], "objective_gate_repeatability_sweep")
        self.assertTrue(summary["objective_gate_support"])
        self.assertTrue(summary["single_seed_set_only"])
        self.assertEqual(summary["valid_sample_count"], 3)
        self.assertEqual(summary["strict_valid_sample_count"], 3)
        self.assertEqual(summary["grammar_gate_sample_count"], 3)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertEqual(summary["long_note_failure_count"], 0)
        self.assertFalse(summary["repeatability_claimed"])
        self.assertFalse(summary["musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unqualified_repair(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError):
            build_consolidation_report(
                repair_probe(target_qualified=False),
                output_dir=Path("outputs/objective_gate"),
            )

    def test_rejects_partial_objective_gate_pass(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError):
            build_consolidation_report(
                repair_probe(all_pass=False),
                output_dir=Path("outputs/objective_gate"),
            )

    def test_rejects_remaining_dead_air_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError):
            build_consolidation_report(
                repair_probe(dead_air_failure_count=1),
                output_dir=Path("outputs/objective_gate"),
            )

    def test_rejects_long_note_regression(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError):
            build_consolidation_report(
                repair_probe(long_note_failure_count=1),
                output_dir=Path("outputs/objective_gate"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError):
            build_consolidation_report(
                repair_probe(quality_claim=True),
                output_dir=Path("outputs/objective_gate"),
            )


if __name__ == "__main__":
    unittest.main()
