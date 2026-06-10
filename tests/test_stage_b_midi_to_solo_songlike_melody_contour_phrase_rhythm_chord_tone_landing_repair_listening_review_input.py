from __future__ import annotations

import unittest

from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package import (
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input import (
    BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    StageBMidiToSoloChordToneLandingRepairListeningInputGuardError,
    build_listening_review_input_guard_report,
    validate_listening_review_input_guard_report,
)

SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def source_package(*, quality_claim: bool = False, validated_input: bool = False) -> dict:
    review_items = [
        {
            "candidate_index": index,
            "rank": index,
            "wav_path": f"candidate_{index}.wav",
            "midi_path": f"candidate_{index}.mid",
            "review_status": "pending",
        }
        for index in range(1, 7)
    ]
    return {
        "boundary": SOURCE_BOUNDARY,
        "source_summary": {
            "technical_wav_validation": True,
            "rendered_audio_file_count": 6,
            "sample_rate": 44100,
            "duration_min_seconds": 18.871,
            "duration_max_seconds": 19.000,
            "changed_note_total": 40,
            "objective_outside_soloing_pitch_role_risk_count": 5,
            "weak_chord_tone_landing_risk_delta": 6,
            "outside_soloing_pitch_role_risk_count_before": 5,
            "outside_soloing_pitch_role_risk_count_after": 2,
            "outside_soloing_pitch_role_risk_delta": 3,
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": True,
            "final_landing_chord_tone_count_after": 6,
            "audio_review_required": True,
            **SOURCE_CONTEXT,
        },
        "review_package": {
            "package_ready": True,
            "review_item_count": 6,
            "validated_review_input": validated_input,
            "required_input_fields": [
                "candidate_index",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": review_items,
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "listening_review_package_ready": True,
            "review_item_count": 6,
            "validated_review_input": validated_input,
            "human_review_required_now": False,
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
            "current_boundary": SOURCE_BOUNDARY,
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloChordToneLandingRepairListeningInputGuardTest(unittest.TestCase):
    def test_blocks_preference_fill_without_validated_input(self) -> None:
        report = build_listening_review_input_guard_report(
            source_package(),
            output_dir="unused",
            issue_number=796,
        )
        summary = validate_listening_review_input_guard_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=OBJECTIVE_NEXT_BOUNDARY,
            require_guard_completed=True,
            require_preference_blocked=True,
            require_pending_input=True,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertEqual(summary["review_item_count"], 6)
        self.assertEqual(summary["required_input_field_count"], 4)
        self.assertEqual(summary["weak_chord_tone_landing_risk_delta"], 6)
        self.assertEqual(summary["objective_outside_soloing_pitch_role_risk_count"], 5)
        self.assertEqual(summary["outside_soloing_pitch_role_risk_count_before"], 5)
        self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 2)
        self.assertEqual(summary["outside_soloing_pitch_role_risk_delta"], 3)
        self.assertFalse(summary["outside_soloing_repair_targeted"])
        self.assertTrue(summary["outside_soloing_residual_risk_preserved"])
        self.assertEqual(summary["final_landing_chord_tone_count_after"], 6)
        self.assertEqual(
            summary[
                "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before"
            ],
            5,
        )
        self.assertEqual(
            summary[
                "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after"
            ],
            2,
        )
        self.assertEqual(
            summary[
                "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after"
            ],
            0,
        )
        self.assertEqual(
            summary[
                "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"
            ],
            3,
        )
        self.assertEqual(
            summary[
                "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta"
            ],
            2,
        )
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_source_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloChordToneLandingRepairListeningInputGuardError):
            build_listening_review_input_guard_report(
                source_package(quality_claim=True),
                output_dir="unused",
                issue_number=796,
            )

    def test_routes_to_fill_when_validated_input_exists(self) -> None:
        report = build_listening_review_input_guard_report(
            source_package(validated_input=True),
            output_dir="unused",
            issue_number=796,
        )
        self.assertTrue(report["guard_result"]["preference_fill_allowed"])
        self.assertTrue(report["guard_result"]["validated_review_input_present"])

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input_guard",
        )
        self.assertEqual(
            OBJECTIVE_NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_only_next_decision",
        )


if __name__ == "__main__":
    unittest.main()
