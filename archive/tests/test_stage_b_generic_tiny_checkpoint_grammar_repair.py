from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_tiny_checkpoint_grammar_repair import (
    StageBGenericTinyCheckpointGrammarRepairError,
    build_repair_report,
    validate_repair_report,
)


def generation_summary(*, sample_count: int = 2, valid: int = 0, strict: int = 0, grammar: int = 0) -> dict:
    return {
        "sample_count": sample_count,
        "valid_sample_count": valid,
        "strict_valid_sample_count": strict,
        "grammar_gate_sample_count": grammar,
        "valid_sample_rate": valid / sample_count if sample_count else 0.0,
        "strict_valid_sample_rate": strict / sample_count if sample_count else 0.0,
        "grammar_gate_sample_rate": grammar / sample_count if sample_count else 0.0,
        "passed_generation_gate": valid > 0,
        "passed_grammar_gate": grammar > 0,
        "passed_strict_review_gate": strict > 0,
        "diagnostic_failure_reasons": {},
    }


def repair_report(
    *,
    baseline: dict | None = None,
    repair: dict | None = None,
    repair_passed: bool = True,
    broad_claim: bool = False,
) -> dict:
    baseline = baseline or generation_summary()
    repair = repair or generation_summary(valid=2, strict=2, grammar=2)
    return {
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_grammar_repair",
            "grammar_repair_passed": repair_passed,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
            "critical_user_input_required": False,
        },
        "comparison": {
            "grammar_gate_delta": repair["grammar_gate_sample_count"] - baseline["grammar_gate_sample_count"],
            "valid_sample_delta": repair["valid_sample_count"] - baseline["valid_sample_count"],
            "strict_valid_sample_delta": repair["strict_valid_sample_count"] - baseline["strict_valid_sample_count"],
        },
        "baseline": {
            "command": {"returncode": 0},
            "summary": baseline,
        },
        "repair": {
            "command": {"returncode": 0},
            "summary": repair,
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint repair repeatability probe",
    }


class StageBGenericTinyCheckpointGrammarRepairTest(unittest.TestCase):
    def test_accepts_repair_gate_recovery_without_quality_claim(self) -> None:
        summary = validate_repair_report(
            repair_report(),
            expected_boundary="stage_b_generic_tiny_checkpoint_grammar_repair",
            require_repair_passed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertEqual(summary["baseline_valid_sample_count"], 0)
        self.assertEqual(summary["repair_valid_sample_count"], 2)
        self.assertEqual(summary["repair_strict_valid_sample_count"], 2)
        self.assertEqual(summary["grammar_gate_delta"], 2)
        self.assertTrue(summary["grammar_repair_passed"])
        self.assertFalse(summary["raw_generation_quality_claimed"])
        self.assertFalse(summary["constrained_generation_quality_claimed"])

    def test_rejects_repair_command_failure(self) -> None:
        report = repair_report()
        report["repair"]["command"]["returncode"] = 5
        with self.assertRaises(StageBGenericTinyCheckpointGrammarRepairError):
            validate_repair_report(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_grammar_repair",
                require_repair_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_unpassed_repair_gate(self) -> None:
        failed_repair = generation_summary(valid=0, strict=0, grammar=2)
        with self.assertRaises(StageBGenericTinyCheckpointGrammarRepairError):
            validate_repair_report(
                repair_report(repair=failed_repair, repair_passed=False),
                expected_boundary="stage_b_generic_tiny_checkpoint_grammar_repair",
                require_repair_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointGrammarRepairError):
            validate_repair_report(
                repair_report(broad_claim=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_grammar_repair",
                require_repair_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_passed_repair_to_repeatability(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
            args = argparse.Namespace(
                issue_number=395,
                num_samples=2,
                seed=42,
                max_sequence=96,
                temperature=0.9,
                top_k=4,
                min_valid_samples=1,
                min_strict_valid_samples=1,
                max_simultaneous_notes=2,
                repair_note_groups_per_bar=4,
            )
            report = build_repair_report(
                run_dir=root,
                checkpoint_dir=checkpoint_dir,
                baseline_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                baseline_report_path=root / "baseline.json",
                baseline_report={
                    "passed_generation_gate": False,
                    "passed_grammar_gate": False,
                    "passed_strict_review_gate": False,
                    "summary": generation_summary(),
                },
                repair_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                repair_report_path=root / "repair.json",
                repair_report={
                    "passed_generation_gate": True,
                    "passed_grammar_gate": True,
                    "passed_strict_review_gate": True,
                    "summary": generation_summary(valid=2, strict=2, grammar=2),
                },
                args=args,
            )

        self.assertTrue(report["readiness"]["grammar_repair_passed"])
        self.assertEqual(
            report["decision"]["next_boundary"],
            "stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
        )
        self.assertFalse(report["readiness"]["broad_trained_model_quality_claimed"])


if __name__ == "__main__":
    unittest.main()
