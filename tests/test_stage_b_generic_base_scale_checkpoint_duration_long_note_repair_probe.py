from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe import (
    BASELINE_REPAIR_BOUNDARY,
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError,
    build_repair_report,
    validate_repair_report,
)


def repair_report(
    *,
    target_qualified: bool = True,
    quality_claim: bool = False,
    long_note_delta: int = 2,
    long_note_failure_count: int = 0,
) -> dict:
    return {
        "readiness": {
            "boundary": BOUNDARY,
            "duration_long_note_repair_probe_completed": True,
            "duration_long_note_target_qualified": target_qualified,
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
        "generation_command": {
            "returncode": 0,
        },
        "duration_repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 2,
            "strict_valid_sample_count": 2,
            "grammar_gate_sample_count": 3,
            "long_note_failure_count": long_note_failure_count,
        },
        "comparison": {
            "long_note_failure_delta": long_note_delta,
            "valid_sample_delta": 1,
            "strict_valid_sample_delta": 1,
            "onset_coverage_delta": 0.020833333333333343,
            "sustained_coverage_delta": -0.2708333333333333,
            "coverage_regression_observed": True,
        },
        "next_recommended_issue": "Stage B generic base scale checkpoint duration long-note remaining blocker decision",
    }


class StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeTest(unittest.TestCase):
    def test_accepts_duration_repair_without_quality_claim(self) -> None:
        summary = validate_repair_report(
            repair_report(),
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_target_qualified=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["duration_long_note_target_qualified"])
        self.assertEqual(summary["valid_sample_count"], 2)
        self.assertEqual(summary["strict_valid_sample_count"], 2)
        self.assertEqual(summary["grammar_gate_sample_count"], 3)
        self.assertEqual(summary["long_note_failure_count"], 0)
        self.assertEqual(summary["long_note_failure_delta"], 2)
        self.assertEqual(summary["valid_sample_delta"], 1)
        self.assertEqual(summary["strict_valid_sample_delta"], 1)
        self.assertTrue(summary["coverage_regression_observed"])
        self.assertFalse(summary["raw_generation_quality_claimed"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_missing_target_qualification(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError):
            validate_repair_report(
                repair_report(target_qualified=False),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_no_long_note_improvement(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError):
            validate_repair_report(
                repair_report(long_note_delta=0),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_remaining_long_note_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError):
            validate_repair_report(
                repair_report(long_note_failure_count=1),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError):
            validate_repair_report(
                repair_report(quality_claim=True),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_build_report_records_long_note_delta_and_remaining_boundary(self) -> None:
        args = argparse.Namespace(
            issue_number=461,
            num_samples=3,
            seed=44,
            max_sequence=96,
            temperature=0.9,
            top_k=4,
            min_valid_samples=1,
            min_strict_valid_samples=1,
            constrained_note_groups_per_bar=4,
            coverage_position_window=1,
            max_simultaneous_notes=2,
        )
        report = build_repair_report(
            run_dir=Path("outputs/duration_repair"),
            decision_summary={"selected_target": "duration_long_note_ratio_repair"},
            source_repair_summary={
                "sample_count": 3,
                "valid_sample_count": 1,
                "strict_valid_sample_count": 1,
                "grammar_gate_sample_count": 3,
                "long_note_failure_count": 2,
                "avg_onset_coverage_ratio": 0.16666666666666666,
                "avg_sustained_coverage_ratio": 0.6354166666666666,
                "max_longest_sustained_empty_run_steps": 7,
            },
            generation_report_path=Path("report.json"),
            generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
            generation_report={
                "passed_generation_gate": True,
                "passed_grammar_gate": True,
                "passed_strict_review_gate": True,
                "summary": {
                    "sample_count": 3,
                    "valid_sample_count": 2,
                    "strict_valid_sample_count": 2,
                    "grammar_gate_sample_count": 3,
                    "avg_onset_coverage_ratio": 0.1875,
                    "avg_sustained_coverage_ratio": 0.3645833333333333,
                    "max_longest_sustained_empty_run_steps": 8,
                    "diagnostic_failure_reasons": {
                        "dead-air ratio too high: 0.800 >= 0.800": 1
                    },
                    "avg_duration_diversity_ratio": 0.4583333333333333,
                    "avg_most_common_duration_ratio": 0.625,
                },
            },
            args=args,
        )

        self.assertTrue(report["comparison"]["target_qualified"])
        self.assertEqual(report["comparison"]["long_note_failure_delta"], 2)
        self.assertEqual(report["duration_repair_summary"]["long_note_failure_count"], 0)
        self.assertTrue(report["comparison"]["coverage_regression_observed"])
        self.assertEqual(report["decision"]["next_boundary"], NEXT_BOUNDARY)

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(
            SOURCE_BOUNDARY,
            "stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision",
        )
        self.assertEqual(
            BASELINE_REPAIR_BOUNDARY,
            "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe",
        )
        self.assertEqual(BOUNDARY, "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe")


if __name__ == "__main__":
    unittest.main()
