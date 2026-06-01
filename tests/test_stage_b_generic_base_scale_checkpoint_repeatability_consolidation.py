from __future__ import annotations

import unittest
from pathlib import Path

from scripts.consolidate_stage_b_generic_base_scale_checkpoint_repeatability import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError,
    build_repeatability_consolidation_report,
    validate_repeatability_consolidation_report,
)


def repeatability_sweep_report(
    *,
    target_qualified: bool = True,
    strict_count: int = 9,
    failures: dict | None = None,
    quality_claim: bool = False,
) -> dict:
    failure_reasons = failures or {}
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep_v1",
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "objective_gate_repeatability_sweep_completed": True,
            "objective_gate_repeatability_target_qualified": target_qualified,
            "repeatability_claimed": target_qualified,
            "raw_generation_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "input": {
            "generation_mode": "constrained",
            "constrained_note_groups_per_bar": 8,
            "coverage_aware_positions": True,
            "coverage_position_window": 1,
            "jazz_duration_tokens": True,
            "postprocess_overlap": True,
            "max_simultaneous_notes": 2,
        },
        "aggregate": {
            "seeds": [44, 52, 60],
            "seed_count": 3,
            "sample_count": 9,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 9,
            "valid_sample_rate": strict_count / 9,
            "strict_valid_sample_rate": strict_count / 9,
            "grammar_gate_sample_rate": 1.0,
            "failure_reasons": failure_reasons,
            "diagnostic_failure_reasons": failure_reasons,
            "strict_failure_reasons": failure_reasons,
            "avg_onset_coverage_ratio": 0.4236111111111111,
            "avg_sustained_coverage_ratio": 0.6805555555555556,
            "max_longest_sustained_empty_run_steps": 4,
        },
        "comparison": {
            "strict_valid_sample_delta": 6,
            "sustained_coverage_delta": 0.04513888888888895,
        },
    }


class StageBGenericBaseScaleCheckpointRepeatabilityConsolidationTest(unittest.TestCase):
    def test_consolidates_objective_gate_repeatability_without_quality_claim(self) -> None:
        report = build_repeatability_consolidation_report(
            repeatability_sweep_report(),
            output_dir=Path("outputs/repeatability_consolidation"),
        )
        summary = validate_repeatability_consolidation_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["objective_midi_gate_repeatability_claimed"])
        self.assertTrue(summary["configured_seed_sweep_repeatability_claimed"])
        self.assertEqual(summary["seed_count"], 3)
        self.assertEqual(summary["sample_count"], 9)
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertIn("musical_quality", report["not_proven"])

    def test_rejects_unqualified_repeatability_sweep(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                repeatability_sweep_report(target_qualified=False),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_rejects_partial_strict_pass(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                repeatability_sweep_report(strict_count=8),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_rejects_failure_reasons(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                repeatability_sweep_report(failures={"dead-air ratio too high: 0.889 >= 0.800": 1}),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError):
            build_repeatability_consolidation_report(
                repeatability_sweep_report(quality_claim=True),
                output_dir=Path("outputs/repeatability_consolidation"),
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            SOURCE_BOUNDARY,
            "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep",
        )
        self.assertEqual(BOUNDARY, "stage_b_generic_base_scale_checkpoint_repeatability_consolidation")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_model_core_evidence_readme_refresh")


if __name__ == "__main__":
    unittest.main()
