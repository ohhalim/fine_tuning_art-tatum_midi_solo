from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next import (
    BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    StageBMidiToSoloChordToneLandingRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package import (
    SCHEMA_VERSION as SOURCE_PACKAGE_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio import (
    SCHEMA_VERSION as SOURCE_AUDIO_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input import (
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_INPUT_GUARD_SCHEMA_VERSION,
)

SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
}


def input_guard_report(*, quality_claim: bool = False, outside_after: int = 2) -> dict:
    return {
        "schema_version": SOURCE_INPUT_GUARD_SCHEMA_VERSION,
        "boundary": SOURCE_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package",
        "source_schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
        "source_audio_schema_version": SOURCE_AUDIO_SCHEMA_VERSION,
        "guard_result": {
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "review_item_count": 6,
            "required_input_field_count": 4,
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
                "outside_soloing_pitch_role_risk_count_after": outside_after,
                "outside_soloing_pitch_role_risk_delta": 5 - outside_after,
                "outside_soloing_repair_targeted": False,
                "outside_soloing_residual_risk_preserved": True,
                "final_landing_chord_tone_count_after": 6,
                "audio_review_required": True,
                **SOURCE_CONTEXT,
            },
        },
        "readiness": {
            "boundary": SOURCE_BOUNDARY,
            "listening_review_input_guard_completed": True,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
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


class StageBMidiToSoloChordToneLandingRepairObjectiveNextTest(unittest.TestCase):
    def test_routes_pending_review_to_followup_decision_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_objective_next_report(
                input_guard_report=input_guard_report(),
                output_dir=root / "objective_next",
                issue_number=1136,
            )
            summary = validate_objective_next_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=FOLLOWUP_DECISION_NEXT_BOUNDARY,
                require_objective_decision=True,
                require_followup_required=True,
                require_no_quality_claim=True,
            )

            self.assertTrue(summary["objective_next_decision_completed"])
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(summary["source_schema_version"], SOURCE_INPUT_GUARD_SCHEMA_VERSION)
            self.assertEqual(summary["source_package_schema_version"], SOURCE_PACKAGE_SCHEMA_VERSION)
            self.assertEqual(summary["source_audio_schema_version"], SOURCE_AUDIO_SCHEMA_VERSION)
            self.assertEqual(summary["selected_target"], SELECTED_TARGET)
            self.assertFalse(summary["validated_review_input_present"])
            self.assertFalse(summary["preference_fill_allowed"])
            self.assertTrue(summary["technical_wav_validation"])
            self.assertEqual(summary["rendered_audio_file_count"], 6)
            self.assertEqual(summary["weak_chord_tone_landing_risk_delta"], 6)
            self.assertEqual(summary["objective_outside_soloing_pitch_role_risk_count"], 5)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_before"], 5)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_count_after"], 2)
            self.assertEqual(summary["outside_soloing_pitch_role_risk_delta"], 3)
            self.assertFalse(summary["outside_soloing_repair_targeted"])
            self.assertTrue(summary["outside_soloing_residual_risk_preserved"])
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
            self.assertTrue(
                summary["followup_objective_source_outside_soloing_source_context_preserved"]
            )
            self.assertEqual(
                summary[
                    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"
                ],
                3,
            )
            self.assertTrue(
                summary["followup_repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertEqual(
                summary[
                    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta"
                ],
                2,
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertTrue(summary["chord_tone_landing_followup_required"])
            self.assertFalse(summary["current_quality_claim_ready"])
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_rejects_source_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=input_guard_report(quality_claim=True),
                    output_dir=root / "objective_next",
                    issue_number=1136,
                )

    def test_rejects_source_input_guard_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            source["schema_version"] = (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input_guard_v2"
            )
            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1136,
                )

    def test_rejects_source_context_preservation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = input_guard_report()
            key = "followup_objective_source_outside_soloing_source_context_preserved"
            source["guard_result"]["source_summary"][key] = False
            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairObjectiveNextError):
                build_objective_next_report(
                    input_guard_report=source,
                    output_dir=root / "objective_next",
                    issue_number=1136,
                )

    def test_rejects_report_preserved_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = build_objective_next_report(
                input_guard_report=input_guard_report(),
                output_dir=root / "objective_next",
                issue_number=1136,
            )
            report["objective_summary"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False

            with self.assertRaises(StageBMidiToSoloChordToneLandingRepairObjectiveNextError):
                validate_objective_next_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=FOLLOWUP_DECISION_NEXT_BOUNDARY,
                    require_objective_decision=True,
                    require_followup_required=True,
                    require_no_quality_claim=True,
                )

    def test_constants_are_stable(self) -> None:
        self.assertEqual(
            BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_only_next_decision",
        )
        self.assertEqual(
            FOLLOWUP_DECISION_NEXT_BOUNDARY,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision",
        )
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next_v3",
        )


if __name__ == "__main__":
    unittest.main()
