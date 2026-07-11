from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair import (
    BOUNDARY as DECISION_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repair_probe import (
    BASELINE_BOUNDARY,
    BOUNDARY,
    PASS_NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepairProbeError,
    build_repair_report,
    validate_baseline,
    validate_decision,
    validate_repair_report,
)


def decision_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": DECISION_BOUNDARY,
        "evidence": {
            "sample_count": 3,
            "note_count_failure_count": 3,
            "grammar_failure_count": 1,
            "collapse_warning_sample_count": 3,
            "avg_postprocess_removal_ratio": 0.7909,
            "avg_onset_coverage_ratio": 0.1146,
            "avg_sustained_coverage_ratio": 0.1458,
            "collapse_across_all_samples": True,
            "partial_grammar_failure": True,
        },
        "repair_decision": {
            "selected_target": "target_density_grammar_collapse_postprocess_repair",
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
        "claim_boundary": {
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
    }


def baseline_report() -> dict:
    return {
        "boundary": BASELINE_BOUNDARY,
        "readiness": {
            "generation_path_executable": True,
        },
        "generation_summary": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 2,
            "collapse_warning_sample_count": 3,
            "collapse_warning_sample_rate": 1.0,
            "avg_postprocess_removal_ratio": 0.7909,
            "max_postprocess_removal_ratio": 0.8,
            "avg_onset_coverage_ratio": 0.1146,
            "avg_sustained_coverage_ratio": 0.1458,
            "max_longest_sustained_empty_run_steps": 21,
            "diagnostic_failure_reasons": {
                "note count too low: 4 < 6; collapse=postprocess_removed_majority": 1,
                "note count too low: 5 < 6; collapse=postprocess_removed_majority": 2,
            },
            "strict_failure_reasons": {
                "grammar_gate_failed": 1,
            },
        },
    }


def args() -> argparse.Namespace:
    return argparse.Namespace(
        issue_number=586,
        num_samples=3,
        seed=47,
        max_sequence=160,
        temperature=0.9,
        top_k=4,
        constrained_note_groups_per_bar=8,
        coverage_position_window=1,
        chord_pitch_mode="approach_tensions",
        chord_pitch_repeat_window=2,
        max_simultaneous_notes=1,
        min_valid_samples=1,
        min_strict_valid_samples=1,
    )


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepairProbeTest(
    unittest.TestCase
):
    def test_validates_decision_and_baseline(self) -> None:
        decision = validate_decision(decision_report())
        baseline = validate_baseline(baseline_report())

        self.assertEqual(
            decision["selected_target"],
            "target_density_grammar_collapse_postprocess_repair",
        )
        self.assertEqual(baseline["note_count_failure_count"], 3)
        self.assertEqual(baseline["grammar_failure_count"], 1)
        self.assertEqual(baseline["collapse_warning_sample_count"], 3)

    def test_builds_repair_support_with_repeatability_next_boundary(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/repair"),
            decision_summary=validate_decision(decision_report()),
            baseline_summary=validate_baseline(baseline_report()),
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
                    "collapse_warning_sample_count": 0,
                    "collapse_warning_sample_rate": 0.0,
                    "avg_postprocess_removal_ratio": 0.1875,
                    "max_postprocess_removal_ratio": 0.25,
                    "avg_onset_coverage_ratio": 0.46875,
                    "avg_sustained_coverage_ratio": 0.6146,
                    "max_longest_sustained_empty_run_steps": 2,
                    "diagnostic_failure_reasons": {
                        "dead-air ratio too high: 0.833 >= 0.800": 1,
                        "dead-air ratio too high: 0.818 >= 0.800": 1,
                    },
                    "strict_failure_reasons": {
                        "midi_review_gate_failed: dead-air ratio too high: 0.833 >= 0.800": 1,
                        "midi_review_gate_failed: dead-air ratio too high: 0.818 >= 0.800": 1,
                    },
                },
            },
            args=args(),
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=PASS_NEXT_BOUNDARY,
            require_target_supported=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["note_count_failure_delta"], 3)
        self.assertEqual(summary["grammar_failure_delta"], 1)
        self.assertEqual(summary["collapse_warning_delta"], 3)
        self.assertGreater(summary["postprocess_removal_delta"], 0.0)
        self.assertTrue(summary["density_grammar_collapse_target_supported"])
        self.assertTrue(summary["strict_gate_recovered"])
        self.assertEqual(summary["dead_air_failure_count"], 2)
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_quality_claim_in_decision(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepairProbeError
        ):
            validate_decision(decision_report(quality_claim=True))

    def test_validation_rejects_missing_target_support(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/repair"),
            decision_summary=validate_decision(decision_report()),
            baseline_summary=validate_baseline(baseline_report()),
            generation_report_path=Path("report.json"),
            generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
            generation_report={
                "passed_generation_gate": False,
                "passed_grammar_gate": True,
                "passed_strict_review_gate": False,
                "summary": {
                    "sample_count": 3,
                    "grammar_gate_sample_count": 3,
                    "collapse_warning_sample_count": 3,
                    "avg_postprocess_removal_ratio": 0.809,
                    "avg_onset_coverage_ratio": 0.0833,
                    "avg_sustained_coverage_ratio": 0.1667,
                    "diagnostic_failure_reasons": {
                        "note count too low: 3 < 6; collapse=postprocess_removed_majority": 3,
                    },
                },
            },
            args=args(),
        )
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDensityGrammarCollapseRepairProbeError
        ):
            validate_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=None,
                require_target_supported=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
