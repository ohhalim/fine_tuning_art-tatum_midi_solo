from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_targeted_quality_repair_followup import (
    BOUNDARY,
    DOMINANT_TARGET_LABEL,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    StageBMidiToSoloTargetedQualityRepairFollowupDecisionError,
    build_followup_decision_report,
    validate_followup_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_targeted_quality_repair_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_targeted_quality_repair_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
)


def objective_next_report(*, quality_claim: bool = False) -> dict:
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "objective_summary": {
            "review_item_count": 6,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 6,
            "failure_label_delta": 4,
            "current_quality_claim_ready": False,
        },
        "readiness": {
            "objective_next_decision_completed": True,
            "targeted_quality_followup_required": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "next_boundary": BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def repair_sweep_report(*, technical_regression_count: int = 0) -> dict:
    return {
        "boundary": SWEEP_BOUNDARY,
        "candidate_repairs": [
            {
                "repaired_labeling": {
                    "failure_labels": ["songlike_melody_not_soloing"],
                    "not_evaluable_labels": [
                        "outside_soloing_without_context",
                        "weak_chord_tone_landing",
                    ],
                }
            }
            for _ in range(6)
        ],
        "aggregate": {
            "candidate_count": 6,
            "source_total_failure_label_count": 12,
            "repaired_total_failure_label_count": 8,
            "failure_label_delta": 4,
            "improved_candidate_count": 4,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": {
                "dead_air_or_density_gap": 1,
                "phrase_shape_missing_tension_release": 2,
                DOMINANT_TARGET_LABEL: 5,
            },
        },
        "readiness": {
            "targeted_quality_repair_sweep_completed": True,
            "targeted_quality_repair_target_supported": True,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloTargetedQualityRepairFollowupDecisionTest(unittest.TestCase):
    def test_selects_dominant_songlike_repair_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_followup_decision_report(
                objective_next_report=objective_next_report(),
                repair_sweep_report=repair_sweep_report(),
                output_dir=Path(tmp) / "followup",
                issue_number=760,
            )
            summary = validate_followup_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_followup_decision=True,
                require_dominant_songlike_target=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["followup_decision_completed"])
            self.assertTrue(summary["dominant_songlike_target_selected"])
            self.assertEqual(
                summary["dominant_remaining_failure_label"],
                DOMINANT_TARGET_LABEL,
            )
            self.assertEqual(summary["dominant_remaining_failure_count"], 5)
            self.assertEqual(summary["failure_label_delta"], 4)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_objective_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloTargetedQualityRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(quality_claim=True),
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=760,
                )

    def test_rejects_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloTargetedQualityRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=repair_sweep_report(technical_regression_count=1),
                    output_dir=Path(tmp) / "followup",
                    issue_number=760,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_targeted_quality_repair_followup_decision")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_repair_sweep")
        self.assertEqual(SELECTED_TARGET, "songlike_melody_contour_repair_sweep")


if __name__ == "__main__":
    unittest.main()
