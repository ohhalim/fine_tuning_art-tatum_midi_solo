from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError,
    build_decision_report,
    validate_decision_report,
)


def temperature_guard_repair_report(
    *,
    target_qualified: bool = False,
    strict_count: int = 8,
    grammar_count: int = 9,
    note_count_failures: int = 0,
    grammar_failures: int = 0,
    dead_air_failures: int = 1,
    collapse_count: int = 0,
    avg_postprocess_removal_ratio: float = 0.3611,
    quality_claim: bool = False,
) -> dict:
    diagnostic_failures = (
        {"dead-air ratio too high: 0.846 >= 0.800": dead_air_failures}
        if dead_air_failures
        else {}
    )
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe_v1",
        "boundary": SOURCE_BOUNDARY,
        "input": {
            "source_temperature": 0.9,
            "temperature": 0.75,
            "top_k": 4,
            "seeds": [47, 52, 60],
            "num_samples": 3,
            "max_sequence": 160,
            "constrained_note_groups_per_bar": 12,
            "coverage_position_window": 1,
            "chord_pitch_mode": "approach_tensions",
            "jazz_rhythm_profile": "swing_motif",
            "max_simultaneous_notes": 1,
        },
        "seed_rows": [
            {
                "seed": 47,
                "sample_count": 3,
                "strict_valid_sample_count": 3,
                "diagnostic_failure_reasons": {},
                "collapse_warning_sample_count": 0,
                "avg_postprocess_removal_ratio": 0.375,
            },
            {
                "seed": 52,
                "sample_count": 3,
                "strict_valid_sample_count": max(0, strict_count - 6),
                "diagnostic_failure_reasons": diagnostic_failures,
                "collapse_warning_sample_count": collapse_count,
                "avg_postprocess_removal_ratio": 0.3889,
            },
            {
                "seed": 60,
                "sample_count": 3,
                "strict_valid_sample_count": 3,
                "diagnostic_failure_reasons": {},
                "collapse_warning_sample_count": 0,
                "avg_postprocess_removal_ratio": 0.3194,
            },
        ],
        "aggregate": {
            "seed_count": 3,
            "seeds": [47, 52, 60],
            "sample_count": 9,
            "valid_sample_count": strict_count,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": grammar_count,
            "note_count_failure_count": note_count_failures,
            "grammar_failure_count": grammar_failures,
            "dead_air_failure_count": dead_air_failures,
            "diagnostic_failure_reasons": diagnostic_failures,
            "strict_failure_reasons": {
                f"midi_review_gate_failed: {reason}": count
                for reason, count in diagnostic_failures.items()
            },
            "collapse_warning_sample_count": collapse_count,
            "avg_postprocess_removal_ratio": avg_postprocess_removal_ratio,
            "avg_onset_coverage_ratio": 0.5764,
            "avg_sustained_coverage_ratio": 0.7222,
            "all_seed_commands_succeeded": True,
            "all_seed_gate_passed": True,
            "all_samples_strict_valid": strict_count == 9,
        },
        "failure_summary": {
            "dead_air_failure_count": dead_air_failures,
            "postprocess_collapse_failure_count": 0,
            "postprocess_removal_failure_count": 0,
        },
        "comparison": {
            "source_strict_sample_shortfall": 2,
            "repair_strict_sample_shortfall": 9 - strict_count,
            "source_dead_air_failure_count": 2,
            "repair_dead_air_failure_count": dead_air_failures,
            "source_collapse_warning_sample_count": 1,
            "repair_collapse_warning_sample_count": collapse_count,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "temperature_guard_repair_probe_completed": True,
            "selected_scale_temperature_guard_repair_target_qualified": target_qualified,
            "model_direct_generation_quality_claimed": False,
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


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionTest(
    unittest.TestCase
):
    def test_selects_postprocess_removal_dead_air_repair(self) -> None:
        report = build_decision_report(
            temperature_guard_repair_report(),
            output_dir=Path("outputs/temperature_guard_followup"),
            issue_number=600,
            target_avg_postprocess_removal_ratio=0.3,
            target_dead_air_failure_count=0,
        )
        summary = validate_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_repair_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["selected_target"], "postprocess_removal_dead_air_repair")
        self.assertEqual(summary["strict_sample_shortfall"], 1)
        self.assertEqual(summary["failed_seeds"], [52])
        self.assertEqual(summary["dead_air_failure_count"], 1)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertEqual(summary["avg_postprocess_removal_ratio"], 0.3611)
        self.assertEqual(summary["target_dead_air_failure_count"], 0)
        self.assertFalse(summary["temperature_followup_selected"])
        self.assertTrue(summary["postprocess_removal_repair_selected"])
        self.assertFalse(summary["critical_user_input_required"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_qualified_temperature_guard(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError
        ):
            build_decision_report(
                temperature_guard_repair_report(
                    target_qualified=True,
                    strict_count=9,
                    dead_air_failures=0,
                ),
                output_dir=Path("outputs/temperature_guard_followup"),
                issue_number=600,
                target_avg_postprocess_removal_ratio=0.3,
                target_dead_air_failure_count=0,
            )

    def test_rejects_remaining_collapse(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError
        ):
            build_decision_report(
                temperature_guard_repair_report(collapse_count=1),
                output_dir=Path("outputs/temperature_guard_followup"),
                issue_number=600,
                target_avg_postprocess_removal_ratio=0.3,
                target_dead_air_failure_count=0,
            )

    def test_rejects_note_count_or_grammar_failures(self) -> None:
        for note_count_failures, grammar_failures in [(1, 0), (0, 1)]:
            with self.subTest(
                note_count_failures=note_count_failures,
                grammar_failures=grammar_failures,
            ):
                with self.assertRaises(
                    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError
                ):
                    build_decision_report(
                        temperature_guard_repair_report(
                            note_count_failures=note_count_failures,
                            grammar_failures=grammar_failures,
                        ),
                        output_dir=Path("outputs/temperature_guard_followup"),
                        issue_number=600,
                        target_avg_postprocess_removal_ratio=0.3,
                        target_dead_air_failure_count=0,
                    )

    def test_rejects_missing_dead_air_evidence(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError
        ):
            build_decision_report(
                temperature_guard_repair_report(dead_air_failures=0),
                output_dir=Path("outputs/temperature_guard_followup"),
                issue_number=600,
                target_avg_postprocess_removal_ratio=0.3,
                target_dead_air_failure_count=0,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError
        ):
            build_decision_report(
                temperature_guard_repair_report(quality_claim=True),
                output_dir=Path("outputs/temperature_guard_followup"),
                issue_number=600,
                target_avg_postprocess_removal_ratio=0.3,
                target_dead_air_failure_count=0,
            )

    def test_validation_rejects_nonzero_dead_air_target(self) -> None:
        report = build_decision_report(
            temperature_guard_repair_report(),
            output_dir=Path("outputs/temperature_guard_followup"),
            issue_number=600,
            target_avg_postprocess_removal_ratio=0.3,
            target_dead_air_failure_count=1,
        )
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError
        ):
            validate_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_repair_target=True,
                require_no_quality_claim=True,
            )


if __name__ == "__main__":
    unittest.main()
