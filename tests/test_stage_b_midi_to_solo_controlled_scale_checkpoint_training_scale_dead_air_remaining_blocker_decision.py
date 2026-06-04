from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError,
    build_decision_report,
    validate_decision_report,
)


def repeatability_report(
    *,
    target_supported: bool = True,
    strict_gate_stable: bool = False,
    note_count_failure_count: int = 0,
    grammar_failure_count: int = 0,
    collapse_warning_count: int = 0,
    dead_air_failure_count: int = 7,
    quality_claim: bool = False,
) -> dict:
    failure_reasons = (
        {
            "dead-air ratio too high: 0.833 >= 0.800": 2,
            "dead-air ratio too high: 0.917 >= 0.800": 1,
        }
        if dead_air_failure_count
        else {}
    )
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_density_grammar_collapse_repeatability_probe_v1",
        "boundary": SOURCE_BOUNDARY,
        "aggregate": {
            "seed_count": 3,
            "sample_count": 9,
            "valid_sample_count": 2,
            "strict_valid_sample_count": 2,
            "grammar_gate_sample_count": 9,
            "note_count_failure_count": note_count_failure_count,
            "grammar_failure_count": grammar_failure_count,
            "dead_air_failure_count": dead_air_failure_count,
            "collapse_warning_sample_count": collapse_warning_count,
            "avg_postprocess_removal_ratio": 0.1944,
            "avg_onset_coverage_ratio": 0.4549,
            "avg_sustained_coverage_ratio": 0.625,
            "diagnostic_failure_reasons": failure_reasons,
        },
        "comparison": {
            "strict_valid_sample_delta": 1,
            "postprocess_removal_delta": 0.0069,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "density_grammar_collapse_repeatability_target_supported": target_supported,
            "strict_gate_stable": strict_gate_stable,
            "dead_air_remaining": bool(dead_air_failure_count),
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionTest(
    unittest.TestCase
):
    def test_selects_selected_scale_dead_air_repair_target(self) -> None:
        report = build_decision_report(
            repeatability_report(),
            output_dir=Path("outputs/dead_air_decision"),
        )
        summary = validate_decision_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_dead_air_target=True,
            require_no_quality_claim=True,
        )

        self.assertEqual(summary["decision"], "select_dead_air_repair_probe")
        self.assertEqual(
            summary["selected_target"],
            "selected_scale_dead_air_sustained_coverage_repair",
        )
        self.assertEqual(summary["seed_count"], 3)
        self.assertEqual(summary["sample_count"], 9)
        self.assertEqual(summary["dead_air_failure_count"], 7)
        self.assertEqual(summary["note_count_failure_count"], 0)
        self.assertEqual(summary["grammar_failure_count"], 0)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertEqual(summary["remaining_blocker"], "dead_air_sustained_coverage")
        self.assertFalse(summary["density_grammar_collapse_followup_selected"])
        self.assertFalse(summary["audio_review_selected"])
        self.assertFalse(summary["additional_training_scale_selected"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unqualified_density_grammar_collapse_target(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(target_supported=False),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_stable_strict_gate(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(strict_gate_stable=True, dead_air_failure_count=0),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_remaining_note_count_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(note_count_failure_count=1),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_remaining_grammar_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(grammar_failure_count=1),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_remaining_collapse_warning(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(collapse_warning_count=1),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_missing_dead_air_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(dead_air_failure_count=0),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repeatability_report(quality_claim=True),
                output_dir=Path("outputs/dead_air_decision"),
            )


if __name__ == "__main__":
    unittest.main()
