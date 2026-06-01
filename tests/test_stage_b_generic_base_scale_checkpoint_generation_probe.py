from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_base_scale_checkpoint_generation_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    SCALE_TRAINING_BOUNDARY,
    StageBGenericBaseScaleCheckpointGenerationProbeError,
    build_probe_report,
    validate_probe_report,
)


def generation_probe_report(
    *,
    returncode: int = 0,
    sample_count: int = 3,
    strict_valid_count: int = 0,
    broad_claim: bool = False,
) -> dict:
    return {
        "readiness": {
            "boundary": BOUNDARY,
            "scale_checkpoint_loaded": returncode == 0 and sample_count > 0,
            "generation_path_executable": returncode == 0 and sample_count > 0,
            "midi_outputs_written": sample_count > 0,
            "raw_generation_quality_ready": strict_valid_count > 0,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": FAIL_NEXT_BOUNDARY,
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
        "next_recommended_issue": "Stage B generic base scale checkpoint grammar representation decision",
    }


class StageBGenericBaseScaleCheckpointGenerationProbeTest(unittest.TestCase):
    def test_accepts_probe_completion_without_quality_claim(self) -> None:
        summary = validate_probe_report(
            generation_probe_report(),
            expected_boundary=BOUNDARY,
            require_probe_completed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertTrue(summary["generation_path_executable"])
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["strict_valid_sample_count"], 0)
        self.assertFalse(summary["raw_generation_quality_ready"])
        self.assertFalse(summary["broad_trained_model_quality_claimed"])

    def test_rejects_generation_command_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointGenerationProbeError):
            validate_probe_report(
                generation_probe_report(returncode=1),
                expected_boundary=BOUNDARY,
                require_probe_completed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_zero_generated_samples(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointGenerationProbeError):
            validate_probe_report(
                generation_probe_report(sample_count=0),
                expected_boundary=BOUNDARY,
                require_probe_completed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointGenerationProbeError):
            validate_probe_report(
                generation_probe_report(broad_claim=True),
                expected_boundary=BOUNDARY,
                require_probe_completed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_failed_strict_gate_to_representation_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
            args = argparse.Namespace(
                issue_number=453,
                num_samples=3,
                seed=43,
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
                training_scale_summary={
                    "source_tokenized_train_files": 154136,
                    "source_tokenized_val_files": 21845,
                    "selected_train_records": 128,
                    "selected_val_records": 32,
                    "best_validation_loss": 5.9031,
                    "checkpoint_count": 1,
                },
                generation_report_path=root / "report.json",
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report={
                    "passed_generation_gate": False,
                    "passed_grammar_gate": False,
                    "passed_strict_review_gate": False,
                    "summary": {
                        "sample_count": 3,
                        "valid_sample_count": 0,
                        "strict_valid_sample_count": 0,
                        "grammar_gate_sample_count": 0,
                        "diagnostic_failure_reasons": {"Stage B generated samples did not satisfy gate": 3},
                    },
                },
                args=args,
            )

        self.assertEqual(report["decision"]["next_boundary"], FAIL_NEXT_BOUNDARY)
        self.assertFalse(report["readiness"]["raw_generation_quality_ready"])
        self.assertFalse(report["readiness"]["broad_trained_model_quality_claimed"])

    def test_build_report_routes_strict_gate_pass_to_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
            args = argparse.Namespace(
                issue_number=453,
                num_samples=3,
                seed=43,
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
                training_scale_summary={
                    "source_tokenized_train_files": 154136,
                    "source_tokenized_val_files": 21845,
                    "selected_train_records": 128,
                    "selected_val_records": 32,
                    "best_validation_loss": 5.9031,
                    "checkpoint_count": 1,
                },
                generation_report_path=root / "report.json",
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report={
                    "passed_generation_gate": True,
                    "passed_grammar_gate": True,
                    "passed_strict_review_gate": True,
                    "summary": {
                        "sample_count": 3,
                        "valid_sample_count": 1,
                        "strict_valid_sample_count": 1,
                        "grammar_gate_sample_count": 1,
                    },
                },
                args=args,
            )

        self.assertEqual(report["decision"]["next_boundary"], PASS_NEXT_BOUNDARY)
        self.assertTrue(report["readiness"]["raw_generation_quality_ready"])

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(SCALE_TRAINING_BOUNDARY, "stage_b_generic_base_training_scale_smoke")
        self.assertEqual(BOUNDARY, "stage_b_generic_base_scale_checkpoint_generation_probe")
        self.assertEqual(FAIL_NEXT_BOUNDARY, "stage_b_generic_base_scale_checkpoint_grammar_representation_decision")


if __name__ == "__main__":
    unittest.main()
