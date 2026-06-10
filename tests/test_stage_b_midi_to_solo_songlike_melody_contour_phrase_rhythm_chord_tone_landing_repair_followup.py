from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup import (
    BOUNDARY,
    NEXT_BOUNDARY,
    PRIMARY_RISK_LABEL,
    SELECTED_TARGET,
    StageBMidiToSoloChordToneLandingRepairFollowupDecisionError,
    build_followup_decision_report,
    validate_followup_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    SELECTED_TARGET as OBJECTIVE_NEXT_SELECTED_TARGET,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
)


def objective_next_report(*, quality_claim: bool = False, outside_after: int = 2) -> dict:
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input_guard",
        "objective_summary": {
            "review_item_count": 6,
            "required_input_field_count": 4,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 6,
            "sample_rate": 44100,
            "duration_min_seconds": 18.871,
            "duration_max_seconds": 19.0,
            "changed_note_total": 40,
            "weak_chord_tone_landing_risk_delta": 6,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": outside_after,
            "outside_soloing_pitch_role_risk_delta": 5 - outside_after,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
            "final_landing_chord_tone_count_after": 6,
            "audio_review_required": True,
            "chord_tone_landing_followup_required": True,
            "current_quality_claim_ready": False,
        },
        "selected_next_target": {
            "target": OBJECTIVE_NEXT_SELECTED_TARGET,
            "next_boundary": FOLLOWUP_DECISION_NEXT_BOUNDARY,
        },
        "readiness": {
            "objective_next_decision_completed": True,
            "technical_wav_validation": True,
            "chord_tone_landing_followup_required": True,
            "current_quality_claim_ready": False,
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
            "next_boundary": FOLLOWUP_DECISION_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


def repair_sweep_report(*, target_supported: bool = True) -> dict:
    return {
        "boundary": SWEEP_BOUNDARY,
        "aggregate": {
            "candidate_count": 6,
            "repaired_midi_count": 6,
            "changed_note_total": 40,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "weak_chord_tone_landing_risk_count_before": 6,
            "weak_chord_tone_landing_risk_count_after": 0,
            "weak_chord_tone_landing_risk_delta": 6,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": 2,
            "outside_soloing_pitch_role_risk_delta": 3,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
            "final_landing_chord_tone_count_before": 1,
            "final_landing_chord_tone_count_after": 6,
            "target_supported": target_supported,
        },
        "readiness": {
            "chord_tone_landing_repair_sweep_completed": True,
            "candidate_count": 6,
            "repaired_midi_count": 6,
            "target_supported": target_supported,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": 2,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
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


class StageBMidiToSoloChordToneLandingRepairFollowupDecisionTest(unittest.TestCase):
    def test_selects_outside_soloing_repair_sweep(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_followup_decision_report(
                objective_next_report=objective_next_report(),
                repair_sweep_report=repair_sweep_report(),
                output_dir=Path(tmp) / "followup",
                issue_number=884,
            )
            summary = validate_followup_decision_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                expected_target=SELECTED_TARGET,
                require_followup_decision=True,
                require_outside_soloing_repair=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["followup_decision_completed"])
            self.assertTrue(summary["outside_soloing_repair_selected"])
            self.assertEqual(summary["primary_remaining_risk_label"], PRIMARY_RISK_LABEL)
            self.assertEqual(summary["primary_remaining_risk_count"], 2)
            self.assertTrue(summary["weak_chord_tone_landing_resolved"])
            self.assertEqual(summary["changed_note_total"], 40)
            self.assertEqual(summary["weak_chord_tone_landing_risk_delta"], 6)
            self.assertEqual(summary["objective_outside_soloing_pitch_role_risk_count"], 5)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_before"], 5)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 2)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["outside_soloing_repair_targeted"])
            self.assertTrue(summary["outside_soloing_residual_risk_preserved"])
            self.assertEqual(summary["final_landing_chord_tone_count_after"], 6)
            self.assertTrue(summary["technical_wav_validation"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_objective_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(quality_claim=True),
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=884,
                )

    def test_rejects_missing_repair_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=repair_sweep_report(target_supported=False),
                    output_dir=Path(tmp) / "followup",
                    issue_number=884,
                )

    def test_rejects_missing_outside_soloing_risk(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(outside_after=0),
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=884,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision",
        )
        self.assertEqual(
            NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep",
        )
        self.assertEqual(
            SELECTED_TARGET,
            "songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep",
        )


if __name__ == "__main__":
    unittest.main()
