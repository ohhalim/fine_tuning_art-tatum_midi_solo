from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (
    StageBGenericTinyCheckpointGenerationProbeError,
    build_probe_report,
    validate_probe_report,
)


def generation_probe_report(
    *,
    returncode: int = 0,
    sample_count: int = 2,
    strict_valid_count: int = 0,
    broad_claim: bool = False,
) -> dict:
    report = {
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_generation_probe",
            "generation_path_executable": returncode == 0 and sample_count > 0,
            "midi_outputs_written": sample_count > 0,
            "raw_generation_quality_ready": strict_valid_count > 0,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_grammar_repair",
            "critical_user_input_required": False,
        },
        "generation_command": {
            "returncode": returncode,
        },
        "generation_summary": {
            "sample_count": sample_count,
            "valid_sample_count": 0,
            "strict_valid_sample_count": strict_valid_count,
            "grammar_gate_sample_count": 0,
            "passed_generation_gate": False,
            "passed_grammar_gate": False,
            "passed_strict_review_gate": strict_valid_count > 0,
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint grammar repair",
    }
    return report


class StageBGenericTinyCheckpointGenerationProbeTest(unittest.TestCase):
    def test_accepts_probe_completion_without_quality_claim(self) -> None:
        summary = validate_probe_report(
            generation_probe_report(),
            expected_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
            require_probe_completed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertTrue(summary["generation_path_executable"])
        self.assertEqual(summary["sample_count"], 2)
        self.assertEqual(summary["strict_valid_sample_count"], 0)
        self.assertFalse(summary["raw_generation_quality_ready"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_generation_command_failure(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointGenerationProbeError):
            validate_probe_report(
                generation_probe_report(returncode=1),
                expected_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
                require_probe_completed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_zero_generated_samples(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointGenerationProbeError):
            validate_probe_report(
                generation_probe_report(sample_count=0),
                expected_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
                require_probe_completed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointGenerationProbeError):
            validate_probe_report(
                generation_probe_report(broad_claim=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_generation_probe",
                require_probe_completed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_failed_strict_gate_to_grammar_repair(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
            args = argparse.Namespace(
                issue_number=393,
                num_samples=2,
                seed=42,
                max_sequence=96,
                temperature=0.9,
                top_k=4,
                min_valid_samples=1,
                min_strict_valid_samples=1,
                max_simultaneous_notes=2,
            )
            report = build_probe_report(
                run_dir=root,
                checkpoint_dir=checkpoint_dir,
                generation_report_path=root / "report.json",
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report={
                    "passed_generation_gate": False,
                    "passed_grammar_gate": False,
                    "passed_strict_review_gate": False,
                    "summary": {
                        "sample_count": 2,
                        "valid_sample_count": 0,
                        "strict_valid_sample_count": 0,
                        "grammar_gate_sample_count": 0,
                        "diagnostic_failure_reasons": {"Stage B generated samples did not satisfy gate": 2},
                    },
                },
                args=args,
            )

        self.assertEqual(
            report["decision"]["next_boundary"],
            "stage_b_generic_tiny_checkpoint_grammar_repair",
        )
        self.assertFalse(report["readiness"]["raw_generation_quality_ready"])
        self.assertFalse(report["readiness"]["broad_trained_model_quality_claimed"])


if __name__ == "__main__":
    unittest.main()
