from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next import (
    BOUNDARY,
    CURRENT_EVIDENCE_NEXT_BOUNDARY,
    REPAIR_RETRY_NEXT_BOUNDARY,
    StageBMidiToSoloOutsideSoloingRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_input import (
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


SOURCE_SUMMARY = {
    "technical_wav_validation": True,
    "rendered_audio_file_count": 6,
    "sample_rate": 44100,
    "duration_min_seconds": 18.871,
    "duration_max_seconds": 19.000,
    "changed_note_total": 2,
    "source_objective_outside_soloing_pitch_role_risk_count": 5,
    "source_outside_soloing_pitch_role_risk_count_before": 5,
    "source_outside_soloing_pitch_role_risk_count_after": 2,
    "source_outside_soloing_pitch_role_risk_delta": 3,
    "source_outside_soloing_repair_targeted": False,
    "source_outside_soloing_residual_risk_preserved": True,
    "outside_soloing_pitch_role_risk_count_after": 0,
    "outside_soloing_pitch_role_risk_delta": 2,
    "outside_soloing_repair_targeted": True,
    "weak_chord_tone_landing_risk_count_after": 0,
    "final_landing_chord_tone_count_after": 6,
    "max_non_chord_tone_run_after": 3,
    "audio_review_required": True,
}


def input_guard(
    *,
    preference_fill_allowed: bool = False,
    quality_claim: bool = False,
    source_summary: dict | None = None,
) -> dict:
    summary = source_summary if source_summary is not None else dict(SOURCE_SUMMARY)
    return {
        "boundary": SOURCE_BOUNDARY,
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": preference_fill_allowed,
            "review_item_count": 6,
            "required_input_field_count": 4,
            "source_summary": dict(summary),
        },
        "readiness": {
            "listening_review_input_guard_completed": True,
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
            "next_boundary": SOURCE_NEXT_BOUNDARY,
            "critical_user_input_required": False,
        },
    }


class StageBMidiToSoloOutsideSoloingRepairObjectiveNextTest(unittest.TestCase):
    def test_routes_to_current_evidence_when_objective_targets_pass(self) -> None:
        report = build_objective_next_report(
            input_guard_report=input_guard(),
            output_dir="out",
            issue_number=894,
            max_non_chord_tone_run_threshold=3,
            min_final_landing_chord_tone_count=6,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=CURRENT_EVIDENCE_NEXT_BOUNDARY,
            require_objective_support=True,
            require_current_evidence_ready=True,
            require_no_quality_claim=True,
        )

        self.assertTrue(summary["objective_next_completed"])
        self.assertEqual(summary["selected_target"], "current_evidence_consolidation")
        self.assertFalse(summary["validated_review_input_present"])
        self.assertFalse(summary["preference_fill_allowed"])
        self.assertTrue(summary["technical_wav_validation"])
        self.assertEqual(summary["rendered_audio_file_count"], 6)
        self.assertEqual(summary["changed_note_total"], 2)
        self.assertEqual(
            summary["source_objective_outside_soloing_pitch_role_risk_count"], 5
        )
        self.assertEqual(summary["source_outside_soloing_pitch_role_risk_count_before"], 5)
        self.assertEqual(summary["source_outside_soloing_pitch_role_risk_count_after"], 2)
        self.assertEqual(summary["source_outside_soloing_pitch_role_risk_delta"], 3)
        self.assertFalse(summary["source_outside_soloing_repair_targeted"])
        self.assertTrue(summary["source_outside_soloing_residual_risk_preserved"])
        self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_pitch_role_risk_delta"], 2)
        self.assertTrue(summary["outside_soloing_repair_targeted"])
        self.assertTrue(summary["outside_soloing_target_supported"])
        self.assertEqual(summary["weak_chord_tone_landing_risk_count_after"], 0)
        self.assertTrue(summary["weak_landing_target_supported"])
        self.assertEqual(summary["final_landing_chord_tone_count_after"], 6)
        self.assertTrue(summary["final_landing_target_supported"])
        self.assertEqual(summary["max_non_chord_tone_run_after"], 3)
        self.assertTrue(summary["non_chord_run_target_supported"])
        self.assertTrue(summary["outside_soloing_repair_objective_path_supported"])
        self.assertTrue(summary["current_evidence_consolidation_ready"])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_routes_to_retry_when_outside_risk_remains(self) -> None:
        broken_summary = dict(SOURCE_SUMMARY)
        broken_summary["outside_soloing_pitch_role_risk_count_after"] = 1
        report = build_objective_next_report(
            input_guard_report=input_guard(source_summary=broken_summary),
            output_dir="out",
            issue_number=894,
            max_non_chord_tone_run_threshold=3,
            min_final_landing_chord_tone_count=6,
        )
        summary = validate_objective_next_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=REPAIR_RETRY_NEXT_BOUNDARY,
            require_objective_support=False,
            require_current_evidence_ready=False,
            require_no_quality_claim=True,
        )

        self.assertFalse(summary["outside_soloing_target_supported"])
        self.assertFalse(summary["outside_soloing_repair_objective_path_supported"])
        self.assertFalse(summary["current_evidence_consolidation_ready"])

    def test_rejects_preference_fill_allowed(self) -> None:
        with self.assertRaises(StageBMidiToSoloOutsideSoloingRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(preference_fill_allowed=True),
                output_dir="out",
                issue_number=894,
                max_non_chord_tone_run_threshold=3,
                min_final_landing_chord_tone_count=6,
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloOutsideSoloingRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(quality_claim=True),
                output_dir="out",
                issue_number=894,
                max_non_chord_tone_run_threshold=3,
                min_final_landing_chord_tone_count=6,
            )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_only_next_decision",
        )
        self.assertEqual(
            CURRENT_EVIDENCE_NEXT_BOUNDARY,
            "stage_b_midi_to_solo_mvp_current_evidence_consolidation",
        )


if __name__ == "__main__":
    unittest.main()
