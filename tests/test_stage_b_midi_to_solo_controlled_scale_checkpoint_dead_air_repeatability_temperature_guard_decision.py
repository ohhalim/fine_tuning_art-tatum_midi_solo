from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError,
    build_decision_report,
    validate_decision_report,
)


def repeatability_report(
    *,
    target_qualified: bool = False,
    strict_count: int = 7,
    grammar_count: int = 9,
    all_seed_gate: bool = True,
    failures: dict | None = None,
    collapse_count: int = 1,
    quality_claim: bool = False,
) -> dict:
    diagnostic_failures = (
        {
            "dead-air ratio too high: 0.800 >= 0.800; collapse=postprocess_removed_majority": 1,
            "dead-air ratio too high: 0.846 >= 0.800": 1,
        }
        if failures is None
        else failures
    )
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe_v1",
        "boundary": SOURCE_BOUNDARY,
        "source_summary": {
            "temperature": 0.9,
            "top_k": 4,
            "max_sequence": 160,
            "constrained_note_groups_per_bar": 12,
            "coverage_position_window": 1,
            "chord_pitch_mode": "approach_tensions",
            "jazz_rhythm_profile": "swing_motif",
            "max_simultaneous_notes": 1,
        },
        "input": {
            "seeds": [44, 52, 60],
            "num_samples": 3,
        },
        "seed_rows": [
            {
                "seed": 44,
                "sample_count": 3,
                "strict_valid_sample_count": 3,
                "diagnostic_failure_reasons": {},
                "collapse_warning_sample_count": 0,
            },
            {
                "seed": 52,
                "sample_count": 3,
                "strict_valid_sample_count": 3,
                "diagnostic_failure_reasons": {},
                "collapse_warning_sample_count": 0,
            },
            {
                "seed": 60,
                "sample_count": 3,
                "strict_valid_sample_count": max(0, strict_count - 6),
                "diagnostic_failure_reasons": diagnostic_failures,
                "collapse_warning_sample_count": collapse_count,
            },
        ],
        "aggregate": {
            "seed_count": 3,
            "seeds": [44, 52, 60],
            "sample_count": 9,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": grammar_count,
            "diagnostic_failure_reasons": diagnostic_failures,
            "strict_failure_reasons": diagnostic_failures,
            "collapse_warning_sample_count": collapse_count,
            "avg_postprocess_removal_ratio": 0.375,
            "avg_onset_coverage_ratio": 0.5486111111111112,
            "avg_sustained_coverage_ratio": 0.7222222222222222,
            "all_seed_commands_succeeded": True,
            "all_seed_gate_passed": all_seed_gate,
            "all_samples_strict_valid": strict_count == 9,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "dead_air_repair_repeatability_probe_completed": True,
            "dead_air_repair_repeatability_target_qualified": target_qualified,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionTest(
    unittest.TestCase
):
    def test_selects_lower_temperature_guard(self) -> None:
        report = build_decision_report(
            repeatability_report(),
            output_dir=Path("outputs/temperature_guard_decision"),
            issue_number=566,
            selected_temperature=0.75,
            selected_top_k=None,
        )
        summary = validate_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_guard_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_target"], "lower_temperature_repeatability_guard_repair")
        self.assertEqual(summary["strict_sample_shortfall"], 2)
        self.assertEqual(summary["failed_seeds"], [60])
        self.assertEqual(summary["source_temperature"], 0.9)
        self.assertEqual(summary["selected_temperature"], 0.75)
        self.assertEqual(summary["selected_top_k"], 4)
        self.assertTrue(summary["temperature_change_selected"])
        self.assertFalse(summary["top_k_change_selected"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_qualified_repeatability(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError
        ):
            build_decision_report(
                repeatability_report(target_qualified=True, strict_count=9, collapse_count=0),
                output_dir=Path("outputs/temperature_guard_decision"),
                issue_number=566,
                selected_temperature=0.75,
                selected_top_k=None,
            )

    def test_rejects_missing_grammar_coverage(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError
        ):
            build_decision_report(
                repeatability_report(grammar_count=8),
                output_dir=Path("outputs/temperature_guard_decision"),
                issue_number=566,
                selected_temperature=0.75,
                selected_top_k=None,
            )

    def test_rejects_missing_failure_reason(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError
        ):
            build_decision_report(
                repeatability_report(failures={}, collapse_count=0),
                output_dir=Path("outputs/temperature_guard_decision"),
                issue_number=566,
                selected_temperature=0.75,
                selected_top_k=None,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError
        ):
            build_decision_report(
                repeatability_report(quality_claim=True),
                output_dir=Path("outputs/temperature_guard_decision"),
                issue_number=566,
                selected_temperature=0.75,
                selected_top_k=None,
            )

    def test_rejects_non_lower_temperature(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError
        ):
            build_decision_report(
                repeatability_report(),
                output_dir=Path("outputs/temperature_guard_decision"),
                issue_number=566,
                selected_temperature=0.9,
                selected_top_k=None,
            )

    def test_validation_rejects_top_k_change_for_isolated_guard(self) -> None:
        report = build_decision_report(
            repeatability_report(),
            output_dir=Path("outputs/temperature_guard_decision"),
            issue_number=566,
            selected_temperature=0.75,
            selected_top_k=3,
        )
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError
        ):
            validate_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_guard_target=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
