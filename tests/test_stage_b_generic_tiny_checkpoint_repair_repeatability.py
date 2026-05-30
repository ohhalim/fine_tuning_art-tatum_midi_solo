from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

from scripts.run_stage_b_generic_tiny_checkpoint_repair_repeatability import (
    StageBGenericTinyCheckpointRepairRepeatabilityError,
    build_repeatability_report,
    validate_repeatability_report,
)


def repeatability_summary(
    *,
    sample_count: int = 6,
    valid: int = 5,
    strict: int = 5,
    grammar: int = 6,
    collapse_rate: float = 0.0,
) -> dict:
    return {
        "sample_count": sample_count,
        "valid_sample_count": valid,
        "strict_valid_sample_count": strict,
        "grammar_gate_sample_count": grammar,
        "valid_sample_rate": valid / sample_count if sample_count else 0.0,
        "strict_valid_sample_rate": strict / sample_count if sample_count else 0.0,
        "grammar_gate_sample_rate": grammar / sample_count if sample_count else 0.0,
        "valid_sample_indices": list(range(1, valid + 1)),
        "strict_valid_sample_indices": list(range(1, strict + 1)),
        "grammar_gate_sample_indices": list(range(1, grammar + 1)),
        "collapse_warning_sample_count": 0,
        "collapse_warning_sample_rate": collapse_rate,
        "diagnostic_failure_reasons": {"dead-air ratio too high: 1.000 >= 0.800": 1}
        if valid < sample_count
        else {},
    }


def repeatability_report(
    *,
    summary: dict | None = None,
    repeatability_passed: bool = True,
    broad_claim: bool = False,
) -> dict:
    summary = summary or repeatability_summary()
    return {
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
            "repair_repeatability_passed": repeatability_passed,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_review_package",
            "critical_user_input_required": False,
        },
        "repeatability": {
            "repeatability_passed": repeatability_passed,
            "all_grammar_samples": summary["grammar_gate_sample_count"] == summary["sample_count"],
            "min_valid_met": summary["valid_sample_count"] >= 5,
            "min_strict_met": summary["strict_valid_sample_count"] >= 5,
            "collapse_rate_met": summary["collapse_warning_sample_rate"] <= 0.34,
        },
        "generation": {
            "command": {"returncode": 0},
            "summary": summary,
        },
        "next_recommended_issue": "Stage B generic tiny checkpoint repair review package",
    }


class StageBGenericTinyCheckpointRepairRepeatabilityTest(unittest.TestCase):
    def test_accepts_repeatability_pass_without_quality_claim(self) -> None:
        summary = validate_repeatability_report(
            repeatability_report(),
            expected_boundary="stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
            require_repeatability_passed=True,
            require_no_broad_quality_claim=True,
            require_no_brad_style_claim=True,
        )

        self.assertEqual(summary["sample_count"], 6)
        self.assertEqual(summary["strict_valid_sample_count"], 5)
        self.assertEqual(summary["grammar_gate_sample_count"], 6)
        self.assertTrue(summary["repeatability_passed"])
        self.assertFalse(summary["raw_generation_quality_claimed"])
        self.assertFalse(summary["constrained_generation_quality_claimed"])

    def test_rejects_generation_command_failure(self) -> None:
        report = repeatability_report()
        report["generation"]["command"]["returncode"] = 5
        with self.assertRaises(StageBGenericTinyCheckpointRepairRepeatabilityError):
            validate_repeatability_report(
                report,
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
                require_repeatability_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_failed_repeatability_gate(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairRepeatabilityError):
            validate_repeatability_report(
                repeatability_report(
                    summary=repeatability_summary(valid=4, strict=4, grammar=6),
                    repeatability_passed=False,
                ),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
                require_repeatability_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericTinyCheckpointRepairRepeatabilityError):
            validate_repeatability_report(
                repeatability_report(broad_claim=True),
                expected_boundary="stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
                require_repeatability_passed=True,
                require_no_broad_quality_claim=True,
                require_no_brad_style_claim=True,
            )

    def test_build_report_routes_passed_repeatability_to_review_package(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_dir = root / "checkpoints"
            checkpoint_dir.mkdir()
            (checkpoint_dir / "checkpoint_epoch1.pt").write_bytes(b"stub")
            args = argparse.Namespace(
                issue_number=397,
                num_samples=6,
                seed=42,
                max_sequence=96,
                temperature=0.9,
                top_k=4,
                min_valid_samples=5,
                min_strict_valid_samples=5,
                max_collapse_warning_sample_rate=0.34,
                max_simultaneous_notes=2,
                repair_note_groups_per_bar=4,
            )
            report = build_repeatability_report(
                run_dir=root,
                checkpoint_dir=checkpoint_dir,
                generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
                generation_report_path=root / "report.json",
                generation_report={
                    "passed_generation_gate": True,
                    "passed_grammar_gate": True,
                    "passed_strict_review_gate": True,
                    "summary": repeatability_summary(),
                },
                args=args,
            )

        self.assertTrue(report["readiness"]["repair_repeatability_passed"])
        self.assertEqual(
            report["decision"]["next_boundary"],
            "stage_b_generic_tiny_checkpoint_repair_review_package",
        )
        self.assertFalse(report["readiness"]["broad_trained_model_quality_claimed"])


if __name__ == "__main__":
    unittest.main()
