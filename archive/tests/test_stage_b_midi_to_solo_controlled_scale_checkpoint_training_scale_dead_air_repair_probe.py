from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker import (
    BOUNDARY as DECISION_BOUNDARY,
    SOURCE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe import (
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError,
    build_repair_report,
    validate_decision,
    validate_repair_report,
)


def decision_report(*, quality_claim: bool = False) -> dict:
    return {
        "input_boundary": SOURCE_BOUNDARY,
        "evidence": {
            "seed_count": 3,
            "sample_count": 9,
            "valid_sample_count": 2,
            "strict_valid_sample_count": 2,
            "grammar_gate_sample_count": 9,
            "note_count_failure_count": 0,
            "grammar_failure_count": 0,
            "dead_air_failure_count": 7,
            "collapse_warning_sample_count": 0,
            "avg_postprocess_removal_ratio": 0.1944,
            "avg_onset_coverage_ratio": 0.4549,
            "avg_sustained_coverage_ratio": 0.625,
            "failure_reasons": {
                "dead-air ratio too high: 0.833 >= 0.800": 4,
                "dead-air ratio too high: 0.917 >= 0.800": 3,
            },
        },
        "decision": {
            "current_boundary": DECISION_BOUNDARY,
            "decision": "select_dead_air_repair_probe",
            "selected_target": "selected_scale_dead_air_sustained_coverage_repair",
            "density_grammar_collapse_followup_selected": False,
            "audio_review_selected": False,
            "additional_training_scale_selected": False,
            "next_boundary": BOUNDARY,
        },
        "claim_boundary": {
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
    }


def args() -> argparse.Namespace:
    return argparse.Namespace(
        issue_number=592,
        num_samples=3,
        seed=47,
        max_sequence=160,
        temperature=0.9,
        top_k=4,
        constrained_note_groups_per_bar=12,
        coverage_position_window=1,
        chord_pitch_mode="approach_tensions",
        chord_pitch_repeat_window=2,
        jazz_rhythm_profile="swing_motif",
        max_simultaneous_notes=1,
        min_valid_samples=1,
        min_strict_valid_samples=1,
    )


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeTest(
    unittest.TestCase
):
    def test_validates_selected_scale_dead_air_decision(self) -> None:
        source = validate_decision(decision_report())

        self.assertEqual(source["seed_count"], 3)
        self.assertEqual(source["sample_count"], 9)
        self.assertEqual(source["dead_air_failure_count"], 7)
        self.assertEqual(source["note_count_failure_count"], 0)
        self.assertEqual(source["grammar_failure_count"], 0)
        self.assertEqual(source["collapse_warning_sample_count"], 0)

    def test_builds_dead_air_repair_success(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/dead_air_repair"),
            source_summary=validate_decision(decision_report()),
            generation_report_path=Path("report.json"),
            generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
            generation_report={
                "passed_strict_review_gate": True,
                "summary": {
                    "sample_count": 3,
                    "valid_sample_count": 3,
                    "strict_valid_sample_count": 3,
                    "grammar_gate_sample_count": 3,
                    "collapse_warning_sample_count": 0,
                    "avg_postprocess_removal_ratio": 0.3889,
                    "max_postprocess_removal_ratio": 0.4167,
                    "avg_onset_coverage_ratio": 0.5729,
                    "avg_sustained_coverage_ratio": 0.7083,
                    "max_longest_sustained_empty_run_steps": 2,
                    "avg_adjacent_repeated_pitch_ratio": 0.0145,
                    "avg_direction_change_ratio": 0.6768,
                    "diagnostic_failure_reasons": {},
                    "strict_failure_reasons": {},
                },
            },
            args=args(),
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_target_qualified=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["dead_air_failure_delta"], 7)
        self.assertGreater(summary["valid_sample_rate_delta"], 0.0)
        self.assertGreater(summary["strict_valid_sample_rate_delta"], 0.0)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertTrue(summary["selected_scale_dead_air_target_qualified"])
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_quality_claim_in_decision(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError
        ):
            validate_decision(decision_report(quality_claim=True))

    def test_rejects_missing_source_dead_air(self) -> None:
        report = decision_report()
        report["evidence"]["dead_air_failure_count"] = 0
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError
        ):
            validate_decision(report)

    def test_rejects_unqualified_repair_when_dead_air_remains(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/dead_air_repair"),
            source_summary=validate_decision(decision_report()),
            generation_report_path=Path("report.json"),
            generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
            generation_report={
                "passed_strict_review_gate": False,
                "summary": {
                    "sample_count": 3,
                    "valid_sample_count": 1,
                    "strict_valid_sample_count": 0,
                    "grammar_gate_sample_count": 3,
                    "collapse_warning_sample_count": 0,
                    "diagnostic_failure_reasons": {
                        "dead-air ratio too high: 0.917 >= 0.800": 1,
                    },
                    "strict_failure_reasons": {},
                },
            },
            args=args(),
        )
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError
        ):
            validate_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
