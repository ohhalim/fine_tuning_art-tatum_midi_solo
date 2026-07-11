from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe import (
    BASELINE_BOUNDARY,
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError,
    build_repair_report,
    validate_repair_report,
)


def repair_report(
    *,
    target_qualified: bool = True,
    quality_claim: bool = False,
    onset_delta: float = 0.1,
    sustained_delta: float = 0.5,
    note_delta: int = 3,
) -> dict:
    return {
        "readiness": {
            "boundary": BOUNDARY,
            "density_coverage_repair_probe_completed": True,
            "density_coverage_target_qualified": target_qualified,
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
        "repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 1,
            "strict_valid_sample_count": 1,
            "grammar_gate_sample_count": 3,
        },
        "comparison": {
            "note_count_failure_delta": note_delta,
            "onset_coverage_delta": onset_delta,
            "sustained_coverage_delta": sustained_delta,
        },
        "next_recommended_issue": "Stage B generic base scale checkpoint density coverage remaining blocker decision",
    }


class StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeTest(unittest.TestCase):
    def test_accepts_target_qualified_repair_without_quality_claim(self) -> None:
        summary = validate_repair_report(
            repair_report(),
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_target_qualified=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["density_coverage_target_qualified"])
        self.assertEqual(summary["valid_sample_count"], 1)
        self.assertEqual(summary["strict_valid_sample_count"], 1)
        self.assertEqual(summary["grammar_gate_sample_count"], 3)
        self.assertEqual(summary["note_count_failure_delta"], 3)
        self.assertGreater(summary["onset_coverage_delta"], 0.0)
        self.assertGreater(summary["sustained_coverage_delta"], 0.0)
        self.assertFalse(summary["raw_generation_quality_claimed"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_missing_target_qualification(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError):
            validate_repair_report(
                repair_report(target_qualified=False),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_no_note_count_improvement(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError):
            validate_repair_report(
                repair_report(note_delta=0),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_no_coverage_improvement(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError):
            validate_repair_report(
                repair_report(onset_delta=0.0),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError):
            validate_repair_report(
                repair_report(quality_claim=True),
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )

    def test_build_report_records_baseline_to_repair_delta(self) -> None:
        args = argparse.Namespace(
            issue_number=457,
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
            run_dir=Path("outputs/repair"),
            decision_summary={"selected_target": "target_density_coverage_repair"},
            baseline_summary={
                "sample_count": 3,
                "valid_sample_count": 0,
                "strict_valid_sample_count": 0,
                "grammar_gate_sample_count": 0,
                "note_count_failure_count": 3,
                "avg_onset_coverage_ratio": 0.0625,
                "avg_sustained_coverage_ratio": 0.09375,
            },
            generation_report_path=Path("report.json"),
            generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
            generation_report={
                "passed_generation_gate": True,
                "passed_grammar_gate": True,
                "passed_strict_review_gate": True,
                "summary": {
                    "sample_count": 3,
                    "valid_sample_count": 1,
                    "strict_valid_sample_count": 1,
                    "grammar_gate_sample_count": 3,
                    "avg_onset_coverage_ratio": 0.16666666666666666,
                    "avg_sustained_coverage_ratio": 0.6354166666666666,
                    "diagnostic_failure_reasons": {"too many long notes: 0.333 > 0.250": 2},
                },
            },
            args=args,
        )

        self.assertTrue(report["comparison"]["target_qualified"])
        self.assertEqual(report["comparison"]["note_count_failure_delta"], 3)
        self.assertGreater(report["comparison"]["onset_coverage_delta"], 0.0)
        self.assertGreater(report["comparison"]["sustained_coverage_delta"], 0.0)

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(SOURCE_BOUNDARY, "stage_b_generic_base_scale_checkpoint_grammar_representation_decision")
        self.assertEqual(BASELINE_BOUNDARY, "stage_b_generic_base_scale_checkpoint_generation_probe")
        self.assertEqual(BOUNDARY, "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe")


if __name__ == "__main__":
    unittest.main()
