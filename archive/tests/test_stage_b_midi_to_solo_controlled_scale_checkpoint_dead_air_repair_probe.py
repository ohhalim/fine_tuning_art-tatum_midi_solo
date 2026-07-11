from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker import (
    BOUNDARY as DECISION_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe import (
    BASELINE_BOUNDARY,
    BOUNDARY,
    NEXT_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointDeadAirRepairProbeError,
    build_repair_report,
    validate_baseline,
    validate_decision,
    validate_repair_report,
)


def decision_report(*, quality_claim: bool = False) -> dict:
    return {
        "decision": {
            "current_boundary": DECISION_BOUNDARY,
            "selected_target": "dead_air_sustained_coverage_repair",
            "next_boundary": BOUNDARY,
            "audio_review_selected": False,
            "training_scale_change_selected": False,
        },
        "evidence": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 3,
            "dead_air_failure_count": 3,
            "avg_onset_coverage_ratio": 0.4583,
            "avg_sustained_coverage_ratio": 0.7188,
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
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe_v1",
        "boundary": BASELINE_BOUNDARY,
        "readiness": {
            "boundary": BASELINE_BOUNDARY,
            "density_collapse_target_supported": True,
            "strict_gate_recovered": False,
        },
        "decision": {
            "next_boundary": DECISION_BOUNDARY,
        },
        "repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 3,
            "note_count_failure_count": 0,
            "dead_air_failure_count": 3,
            "collapse_warning_sample_count": 0,
            "avg_postprocess_removal_ratio": 0.229,
            "max_postprocess_removal_ratio": 0.3125,
            "avg_onset_coverage_ratio": 0.4583,
            "avg_sustained_coverage_ratio": 0.7188,
            "max_longest_sustained_empty_run_steps": 2,
            "diagnostic_failure_reasons": {
                "dead-air ratio too high: 0.917 >= 0.800": 3,
            },
        },
        "comparison": {
            "note_count_failure_delta": 3,
            "collapse_warning_delta": 3,
            "postprocess_removal_delta": 0.5798,
        },
    }


def args() -> argparse.Namespace:
    return argparse.Namespace(
        issue_number=562,
        num_samples=3,
        seed=44,
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


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepairProbeTest(unittest.TestCase):
    def test_validates_decision_and_baseline(self) -> None:
        decision = validate_decision(decision_report())
        baseline = validate_baseline(baseline_report())

        self.assertEqual(decision["selected_target"], "dead_air_sustained_coverage_repair")
        self.assertEqual(baseline["dead_air_failure_count"], 3)
        self.assertEqual(baseline["note_count_failure_count"], 0)
        self.assertEqual(baseline["collapse_warning_sample_count"], 0)

    def test_builds_dead_air_repair_success(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/dead_air_repair"),
            decision_summary=validate_decision(decision_report()),
            baseline_summary=validate_baseline(baseline_report()),
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
                    "avg_postprocess_removal_ratio": 0.3333,
                    "max_postprocess_removal_ratio": 0.375,
                    "avg_onset_coverage_ratio": 0.5729,
                    "avg_sustained_coverage_ratio": 0.7292,
                    "max_longest_sustained_empty_run_steps": 2,
                    "avg_repeated_position_pitch_pair_ratio": 0.05,
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

        self.assertEqual(summary["dead_air_failure_delta"], 3)
        self.assertEqual(summary["valid_sample_delta"], 3)
        self.assertEqual(summary["strict_valid_sample_delta"], 3)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertTrue(summary["dead_air_target_qualified"])
        self.assertFalse(summary["critical_user_input_required"])

    def test_rejects_quality_claim_in_decision(self) -> None:
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointDeadAirRepairProbeError):
            validate_decision(decision_report(quality_claim=True))

    def test_rejects_missing_baseline_dead_air(self) -> None:
        baseline = baseline_report()
        baseline["repair_summary"]["dead_air_failure_count"] = 0
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointDeadAirRepairProbeError):
            validate_baseline(baseline)

    def test_rejects_unqualified_repair_when_dead_air_remains(self) -> None:
        report = build_repair_report(
            run_dir=Path("outputs/dead_air_repair"),
            decision_summary=validate_decision(decision_report()),
            baseline_summary=validate_baseline(baseline_report()),
            generation_report_path=Path("report.json"),
            generation_result={"returncode": 0, "cmd": [], "stdout_tail": "", "stderr_tail": ""},
            generation_report={
                "passed_strict_review_gate": False,
                "summary": {
                    "sample_count": 3,
                    "valid_sample_count": 0,
                    "strict_valid_sample_count": 0,
                    "grammar_gate_sample_count": 3,
                    "collapse_warning_sample_count": 0,
                    "diagnostic_failure_reasons": {
                        "dead-air ratio too high: 0.917 >= 0.800": 3,
                    },
                },
            },
            args=args(),
        )
        with self.assertRaises(StageBMidiToSoloControlledScaleCheckpointDeadAirRepairProbeError):
            validate_repair_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_target_qualified=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
