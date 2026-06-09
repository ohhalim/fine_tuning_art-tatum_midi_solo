from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup import (
    BOUNDARY,
    CONTEXT_TARGET_LABELS,
    NEXT_BOUNDARY,
    SELECTED_TARGET,
    StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError,
    build_followup_decision_report,
    validate_followup_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (
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
            "failure_label_delta": 3,
            "phrase_rhythm_failure_delta": 3,
            "current_quality_claim_ready": False,
        },
        "readiness": {
            "objective_next_decision_completed": True,
            "phrase_rhythm_followup_required": True,
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
                "phrase_rhythm_repaired_labeling": {
                    "failure_labels": [],
                    "not_evaluable_labels": list(CONTEXT_TARGET_LABELS),
                }
            }
            for _ in range(6)
        ],
        "aggregate": {
            "candidate_count": 6,
            "source_total_failure_label_count": 4,
            "repaired_total_failure_label_count": 1,
            "failure_label_delta": 3,
            "source_phrase_rhythm_failure_count": 4,
            "repaired_phrase_rhythm_failure_count": 1,
            "phrase_rhythm_failure_delta": 3,
            "improved_candidate_count": 2,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": {
                "rhythmic_monotony": 1,
            },
        },
        "readiness": {
            "songlike_melody_contour_phrase_rhythm_repair_sweep_completed": True,
            "songlike_melody_contour_phrase_rhythm_repair_target_supported": True,
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


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionTest(unittest.TestCase):
    def test_selects_chord_context_pitch_role_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_followup_decision_report(
                objective_next_report=objective_next_report(),
                repair_sweep_report=repair_sweep_report(),
                output_dir=Path(tmp) / "followup",
                issue_number=784,
            )
            summary = validate_followup_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_followup_decision=True,
                require_context_pitch_role_bridge=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["followup_decision_completed"])
            self.assertTrue(summary["chord_context_pitch_role_bridge_selected"])
            self.assertEqual(
                summary["context_target_labels"],
                list(CONTEXT_TARGET_LABELS),
            )
            self.assertEqual(summary["primary_remaining_failure_labels"], ["rhythmic_monotony"])
            self.assertEqual(summary["primary_remaining_failure_count"], 1)
            self.assertEqual(summary["failure_label_delta"], 3)
            self.assertEqual(summary["phrase_rhythm_failure_delta"], 3)
            self.assertEqual(summary["context_not_evaluable_min_count"], 6)
            self.assertEqual(summary["technical_regression_count"], 0)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_objective_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(quality_claim=True),
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=772,
                )

    def test_rejects_technical_regression(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=repair_sweep_report(technical_regression_count=1),
                    output_dir=Path(tmp) / "followup",
                    issue_number=772,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge")
        self.assertEqual(SELECTED_TARGET, "songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge")


if __name__ == "__main__":
    unittest.main()
