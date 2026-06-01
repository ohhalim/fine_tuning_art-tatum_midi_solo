from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError,
    aggregate_seed_rows,
    build_sweep_report,
    validate_sweep_report,
)


def seed_row(seed: int, *, strict_count: int = 3, failures: dict | None = None) -> dict:
    failure_reasons = failures or {}
    return {
        "seed": seed,
        "generation_report_path": f"report_{seed}.json",
        "generation_command": {"returncode": 0},
        "sample_count": 3,
        "valid_sample_count": strict_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": 3,
        "failure_reasons": failure_reasons,
        "diagnostic_failure_reasons": failure_reasons,
        "strict_failure_reasons": failure_reasons,
        "collapse_warning_sample_count": 0,
        "avg_onset_coverage_ratio": 0.4,
        "avg_sustained_coverage_ratio": 0.7,
        "max_longest_sustained_empty_run_steps": 4,
        "passed_strict_review_gate": strict_count == 3 and not failure_reasons,
        "passed_grammar_gate": True,
    }


def sweep_report(
    *,
    target_qualified: bool = True,
    quality_claim: bool = False,
    strict_count: int = 9,
    failures: dict | None = None,
) -> dict:
    failure_reasons = failures or {}
    return {
        "readiness": {
            "boundary": BOUNDARY,
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
            "next_boundary": NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
        "aggregate": {
            "seed_count": 3,
            "sample_count": 9,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 9,
            "diagnostic_failure_reasons": failure_reasons,
            "avg_sustained_coverage_ratio": 0.68,
        },
        "comparison": {
            "strict_valid_sample_delta": 6,
        },
        "next_recommended_issue": "Stage B generic base scale checkpoint repeatability consolidation",
    }


class StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepTest(unittest.TestCase):
    def test_aggregate_counts_across_seed_rows(self) -> None:
        aggregate = aggregate_seed_rows([seed_row(44), seed_row(52), seed_row(60)])

        self.assertEqual(aggregate["seed_count"], 3)
        self.assertEqual(aggregate["sample_count"], 9)
        self.assertEqual(aggregate["strict_valid_sample_count"], 9)
        self.assertEqual(aggregate["grammar_gate_sample_count"], 9)
        self.assertEqual(aggregate["diagnostic_failure_reasons"], {})
        self.assertTrue(aggregate["all_seed_strict_review_gate_passed"])

    def test_accepts_repeatability_sweep_without_quality_claim(self) -> None:
        summary = validate_sweep_report(
            sweep_report(),
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_target_qualified=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["objective_gate_repeatability_target_qualified"])
        self.assertTrue(summary["repeatability_claimed"])
        self.assertEqual(summary["seed_count"], 3)
        self.assertEqual(summary["sample_count"], 9)
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertFalse(summary["raw_generation_quality_claimed"])

    def test_rejects_unqualified_repeatability_target(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError):
            validate_sweep_report(
                sweep_report(target_qualified=False),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_partial_strict_pass(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError):
            validate_sweep_report(
                sweep_report(strict_count=8),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_diagnostic_failures(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError):
            validate_sweep_report(
                sweep_report(failures={"dead-air ratio too high: 0.889 >= 0.800": 1}),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError):
            validate_sweep_report(
                sweep_report(quality_claim=True),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_build_report_records_repeatability_delta(self) -> None:
        args = argparse.Namespace(
            issue_number=469,
            seeds="44,52,60",
            num_samples=3,
            max_sequence=96,
            temperature=0.9,
            top_k=4,
            constrained_note_groups_per_bar=8,
            coverage_position_window=1,
            max_simultaneous_notes=2,
        )
        report = build_sweep_report(
            run_dir=Path("outputs/repeatability"),
            consolidation_summary={
                "source_sample_count": 3,
                "source_strict_valid_sample_count": 3,
                "source_avg_sustained_coverage_ratio": 0.6354166666666666,
            },
            repair_config={"checkpoint_dir": "checkpoints"},
            seed_rows=[seed_row(44), seed_row(52), seed_row(60)],
            args=args,
        )

        self.assertTrue(report["comparison"]["target_qualified"])
        self.assertEqual(report["aggregate"]["sample_count"], 9)
        self.assertEqual(report["aggregate"]["strict_valid_sample_count"], 9)
        self.assertEqual(report["comparison"]["strict_valid_sample_delta"], 6)
        self.assertEqual(report["decision"]["next_boundary"], NEXT_BOUNDARY)

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(SOURCE_BOUNDARY, "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation")
        self.assertEqual(BOUNDARY, "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep")


if __name__ == "__main__":
    unittest.main()
