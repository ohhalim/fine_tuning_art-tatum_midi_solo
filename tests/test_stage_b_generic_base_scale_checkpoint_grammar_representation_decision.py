from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_generic_base_scale_checkpoint_grammar_representation import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError,
    build_decision_report,
    validate_decision_report,
)


def generation_probe(
    *,
    strict_passed: bool = False,
    broad_claim: bool = False,
    note_failures: bool = True,
) -> dict:
    diagnostic = {
        "note count too low: 4 < 6": 1,
        "note count too low: 3 < 6": 1,
        "note count too low: 2 < 6": 1,
    }
    if not note_failures:
        diagnostic = {"some other failure": 3}
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_generation_probe_v1",
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "generation_path_executable": True,
            "raw_generation_quality_ready": strict_passed,
            "broad_trained_model_quality_claimed": broad_claim,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "training_scale_summary": {
            "source_tokenized_train_files": 154136,
            "source_tokenized_val_files": 21845,
            "selected_train_records": 128,
            "selected_val_records": 32,
            "best_validation_loss": 5.9031,
            "checkpoint_count": 1,
        },
        "generation_summary": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 0,
            "passed_strict_review_gate": strict_passed,
            "collapse_warning_sample_rate": 0.0,
            "avg_onset_coverage_ratio": 0.0625,
            "avg_sustained_coverage_ratio": 0.09375,
            "max_longest_sustained_empty_run_steps": 25,
            "diagnostic_failure_reasons": diagnostic,
        },
    }


class StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionTest(unittest.TestCase):
    def test_selects_density_coverage_repair_without_quality_claim(self) -> None:
        report = build_decision_report(
            generation_probe(),
            output_dir=Path("outputs/decision"),
        )
        summary = validate_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_density_coverage_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["decision"], "select_density_coverage_repair_probe")
        self.assertEqual(summary["selected_target"], "target_density_coverage_repair")
        self.assertEqual(summary["sample_count"], 3)
        self.assertEqual(summary["note_count_failure_count"], 3)
        self.assertTrue(summary["all_samples_note_count_failed"])
        self.assertTrue(summary["low_coverage_observed"])
        self.assertTrue(summary["collapse_warning_not_primary"])
        self.assertFalse(summary["postprocess_only_repair_selected"])
        self.assertFalse(summary["quality_root_cause_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_strict_gate_pass(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError):
            build_decision_report(
                generation_probe(strict_passed=True),
                output_dir=Path("outputs/decision"),
            )

    def test_rejects_broad_quality_claim(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError):
            build_decision_report(
                generation_probe(broad_claim=True),
                output_dir=Path("outputs/decision"),
            )

    def test_rejects_absent_note_count_failure(self) -> None:
        with self.assertRaises(StageBGenericBaseScaleCheckpointGrammarRepresentationDecisionError):
            build_decision_report(
                generation_probe(note_failures=False),
                output_dir=Path("outputs/decision"),
            )


if __name__ == "__main__":
    unittest.main()
