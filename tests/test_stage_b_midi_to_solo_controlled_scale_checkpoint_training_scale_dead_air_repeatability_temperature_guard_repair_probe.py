from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe import (
    BOUNDARY,
    FAIL_NEXT_BOUNDARY,
    PASS_NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError,
    build_repair_report,
    validate_guard_decision,
    validate_repair_report,
)


def guard_decision_report(
    *,
    selected_temperature: float = 0.75,
    top_k_change: bool = False,
    quality_claim: bool = False,
) -> dict:
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision_v1",
        "boundary": SOURCE_BOUNDARY,
        "evidence": {
            "sample_count": 9,
            "strict_valid_sample_count": 7,
            "grammar_gate_sample_count": 9,
            "strict_sample_shortfall": 2,
            "dead_air_failure_count": 2,
            "collapse_warning_sample_count": 1,
            "avg_postprocess_removal_ratio": 0.412,
            "avg_onset_coverage_ratio": 0.552,
            "avg_sustained_coverage_ratio": 0.722,
            "source_temperature": 0.9,
            "source_top_k": 4,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "selected_target": "lower_temperature_repeatability_guard_repair",
            "temperature_change_selected": True,
            "top_k_change_selected": top_k_change,
            "guard_config": {
                "temperature": selected_temperature,
                "top_k": 3 if top_k_change else 4,
                "seeds": [47, 52, 60],
                "num_samples": 3,
                "max_sequence": 160,
                "constrained_note_groups_per_bar": 12,
                "coverage_position_window": 1,
                "chord_pitch_mode": "approach_tensions",
                "jazz_rhythm_profile": "swing_motif",
                "max_simultaneous_notes": 1,
            },
        },
        "claim_boundary": {
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
    }


def seed_row(
    seed: int,
    *,
    strict_count: int = 3,
    failures: dict | None = None,
    collapse: int = 0,
) -> dict:
    diagnostic = failures or {}
    return {
        "seed": seed,
        "generation_command": {"returncode": 0},
        "sample_count": 3,
        "valid_sample_count": strict_count,
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": 3,
        "note_count_failure_count": 0,
        "grammar_failure_count": 0,
        "dead_air_failure_count": sum(diagnostic.values()),
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": {
            f"midi_review_gate_failed: {reason}": count
            for reason, count in diagnostic.items()
        },
        "collapse_warning_sample_count": collapse,
        "avg_postprocess_removal_ratio": 0.25,
        "avg_onset_coverage_ratio": 0.58,
        "avg_sustained_coverage_ratio": 0.74,
        "passed_strict_review_gate": strict_count >= 1,
    }


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeTest(
    unittest.TestCase
):
    def test_validates_guard_decision(self) -> None:
        summary = validate_guard_decision(guard_decision_report())

        self.assertEqual(summary["source_temperature"], 0.9)
        self.assertEqual(summary["temperature"], 0.75)
        self.assertEqual(summary["top_k"], 4)
        self.assertEqual(summary["seeds"], [47, 52, 60])
        self.assertEqual(summary["source_strict_sample_shortfall"], 2)

    def test_builds_qualified_repair_report(self) -> None:
        decision = validate_guard_decision(guard_decision_report())
        report = build_repair_report(
            run_dir=Path("outputs/temperature_guard_repair"),
            decision_summary=decision,
            seed_rows=[seed_row(47), seed_row(52), seed_row(60)],
            issue_number=598,
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=PASS_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["selected_scale_temperature_guard_repair_target_qualified"])
        self.assertEqual(summary["strict_valid_sample_count"], 9)
        self.assertEqual(summary["strict_valid_sample_delta"], 2)
        self.assertEqual(summary["repair_strict_sample_shortfall"], 0)
        self.assertEqual(summary["dead_air_failure_count"], 0)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertEqual(summary["temperature"], 0.75)
        self.assertEqual(summary["next_boundary"], PASS_NEXT_BOUNDARY)

    def test_builds_partial_repair_report(self) -> None:
        decision = validate_guard_decision(guard_decision_report())
        report = build_repair_report(
            run_dir=Path("outputs/temperature_guard_repair"),
            decision_summary=decision,
            seed_rows=[
                seed_row(47),
                seed_row(52, strict_count=2, failures={"dead-air ratio too high: 0.846 >= 0.800": 1}),
                seed_row(60),
            ],
            issue_number=598,
        )
        summary = validate_repair_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=FAIL_NEXT_BOUNDARY,
            require_completed=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["selected_scale_temperature_guard_repair_target_qualified"])
        self.assertEqual(summary["strict_valid_sample_count"], 8)
        self.assertEqual(summary["repair_strict_sample_shortfall"], 1)
        self.assertEqual(summary["dead_air_failure_count"], 1)
        self.assertEqual(summary["next_boundary"], FAIL_NEXT_BOUNDARY)

    def test_rejects_non_lower_temperature(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError
        ):
            validate_guard_decision(guard_decision_report(selected_temperature=0.9))

    def test_rejects_top_k_change(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError
        ):
            validate_guard_decision(guard_decision_report(top_k_change=True))

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError
        ):
            validate_guard_decision(guard_decision_report(quality_claim=True))


if __name__ == "__main__":
    unittest.main()
