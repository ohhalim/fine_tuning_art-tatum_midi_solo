from __future__ import annotations

import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SOURCE_BOUNDARY,
    StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError,
    build_decision_report,
    validate_decision_report,
)


def repair_report(
    *,
    target_supported: bool = True,
    strict_recovered: bool = False,
    note_count_removed: bool = True,
    collapse_removed: bool = True,
    dead_air_failure: bool = True,
    quality_claim: bool = False,
) -> dict:
    failure_reasons = (
        {
            "dead-air ratio too high: 0.800 >= 0.800": 1,
            "dead-air ratio too high: 0.917 >= 0.800": 2,
        }
        if dead_air_failure
        else {}
    )
    return {
        "schema_version": "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe_v1",
        "boundary": SOURCE_BOUNDARY,
        "repair_summary": {
            "sample_count": 3,
            "valid_sample_count": 0,
            "strict_valid_sample_count": 0,
            "grammar_gate_sample_count": 3,
            "note_count_failure_count": 0 if note_count_removed else 1,
            "dead_air_failure_count": 3 if dead_air_failure else 0,
            "collapse_warning_sample_count": 0 if collapse_removed else 1,
            "avg_postprocess_removal_ratio": 0.22916666666666666,
            "avg_onset_coverage_ratio": 0.4583333333333333,
            "avg_sustained_coverage_ratio": 0.71875,
            "max_longest_sustained_empty_run_steps": 2,
            "diagnostic_failure_reasons": failure_reasons,
        },
        "comparison": {
            "note_count_failure_delta": 3,
            "collapse_warning_delta": 3,
            "postprocess_removal_delta": 0.5798761423761424,
            "onset_coverage_delta": 0.375,
            "sustained_coverage_delta": 0.5520833333333334,
            "density_collapse_target_supported": target_supported,
            "strict_gate_recovered": strict_recovered,
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "density_collapse_target_supported": target_supported,
            "strict_gate_recovered": strict_recovered,
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


class StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionTest(
    unittest.TestCase
):
    def test_selects_dead_air_repair_target(self) -> None:
        report = build_decision_report(
            repair_report(),
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
        self.assertEqual(summary["selected_target"], "dead_air_sustained_coverage_repair")
        self.assertEqual(summary["dead_air_failure_count"], 3)
        self.assertEqual(summary["note_count_failure_count"], 0)
        self.assertEqual(summary["collapse_warning_sample_count"], 0)
        self.assertEqual(summary["remaining_blocker"], "dead_air_sustained_coverage")
        self.assertFalse(summary["audio_review_selected"])
        self.assertFalse(summary["training_scale_change_selected"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_unqualified_density_collapse_target(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repair_report(target_supported=False),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_recovered_strict_gate(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repair_report(strict_recovered=True, dead_air_failure=False),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_remaining_note_count_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repair_report(note_count_removed=False),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_remaining_collapse_warning(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repair_report(collapse_removed=False),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_missing_dead_air_failure(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repair_report(dead_air_failure=False),
                output_dir=Path("outputs/dead_air_decision"),
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(
            StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError
        ):
            build_decision_report(
                repair_report(quality_claim=True),
                output_dir=Path("outputs/dead_air_decision"),
            )


if __name__ == "__main__":
    unittest.main()
