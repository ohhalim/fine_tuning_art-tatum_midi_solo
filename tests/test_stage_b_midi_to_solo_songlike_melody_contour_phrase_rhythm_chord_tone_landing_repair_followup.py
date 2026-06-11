from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup import (
    BOUNDARY,
    NEXT_BOUNDARY,
    PRIMARY_RISK_LABEL,
    SCHEMA_VERSION,
    SELECTED_TARGET,
    StageBMidiToSoloChordToneLandingRepairFollowupDecisionError,
    build_followup_decision_report,
    validate_followup_decision_report,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next import (
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    SCHEMA_VERSION as OBJECTIVE_NEXT_SCHEMA_VERSION,
    SELECTED_TARGET as OBJECTIVE_NEXT_SELECTED_TARGET,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input import (
    SCHEMA_VERSION as SOURCE_INPUT_GUARD_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package import (
    SCHEMA_VERSION as SOURCE_PACKAGE_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio import (
    SCHEMA_VERSION as SOURCE_AUDIO_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective import (
    SCHEMA_VERSION as SWEEP_SOURCE_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    SCHEMA_VERSION as SWEEP_BRIDGE_SCHEMA_VERSION,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    BOUNDARY as SWEEP_BOUNDARY,
    SCHEMA_VERSION as SWEEP_SCHEMA_VERSION,
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


def objective_next_report(*, quality_claim: bool = False, outside_after: int = 2) -> dict:
    return {
        "schema_version": OBJECTIVE_NEXT_SCHEMA_VERSION,
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "source_boundary": "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input_guard",
        "source_schema_version": SOURCE_INPUT_GUARD_SCHEMA_VERSION,
        "source_package_schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
        "source_audio_schema_version": SOURCE_AUDIO_SCHEMA_VERSION,
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
            **SOURCE_CONTEXT,
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
        "schema_version": SWEEP_SCHEMA_VERSION,
        "boundary": SWEEP_BOUNDARY,
        "source_schema_version": SWEEP_SOURCE_SCHEMA_VERSION,
        "bridge_schema_version": SWEEP_BRIDGE_SCHEMA_VERSION,
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
            **SOURCE_CONTEXT,
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
                issue_number=1138,
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
            self.assertEqual(report["schema_version"], SCHEMA_VERSION)
            self.assertEqual(summary["source_schema_version"], OBJECTIVE_NEXT_SCHEMA_VERSION)
            self.assertEqual(
                summary["source_input_guard_schema_version"],
                SOURCE_INPUT_GUARD_SCHEMA_VERSION,
            )
            self.assertEqual(summary["source_package_schema_version"], SOURCE_PACKAGE_SCHEMA_VERSION)
            self.assertEqual(summary["source_audio_schema_version"], SOURCE_AUDIO_SCHEMA_VERSION)
            self.assertEqual(summary["repair_sweep_schema_version"], SWEEP_SCHEMA_VERSION)
            self.assertEqual(summary["repair_sweep_source_schema_version"], SWEEP_SOURCE_SCHEMA_VERSION)
            self.assertEqual(summary["repair_sweep_bridge_schema_version"], SWEEP_BRIDGE_SCHEMA_VERSION)
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
                summary[
                    "followup_objective_source_outside_soloing_source_context_preserved"
                ]
            )
            self.assertEqual(
                summary[
                    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta"
                ],
                3,
            )
            self.assertTrue(
                summary[
                    "followup_repair_sweep_source_outside_soloing_source_context_preserved"
                ]
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
                    issue_number=1138,
                )

    def test_rejects_objective_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = objective_next_report()
            report["schema_version"] = (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next_v2"
            )
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=report,
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1138,
                )

    def test_rejects_repair_sweep_schema_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            sweep = repair_sweep_report()
            sweep["schema_version"] = (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep_v3"
            )
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=objective_next_report(),
                    repair_sweep_report=sweep,
                    output_dir=Path(tmp) / "followup",
                    issue_number=1138,
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
                    issue_number=1138,
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
                    issue_number=1138,
                )

    def test_rejects_source_context_preservation_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = objective_next_report()
            report["objective_summary"][
                "followup_objective_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                build_followup_decision_report(
                    objective_next_report=report,
                    repair_sweep_report=repair_sweep_report(),
                    output_dir=Path(tmp) / "followup",
                    issue_number=1138,
                )

    def test_rejects_report_preserved_flag_false(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = build_followup_decision_report(
                objective_next_report=objective_next_report(),
                repair_sweep_report=repair_sweep_report(),
                output_dir=Path(tmp) / "followup",
                issue_number=1138,
            )
            report["readiness"][BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS[0]] = False

            with self.assertRaises(
                StageBMidiToSoloChordToneLandingRepairFollowupDecisionError
            ):
                validate_followup_decision_report(
                    report,
                    expected_boundary=BOUNDARY,
                    expected_next_boundary=NEXT_BOUNDARY,
                    expected_target=SELECTED_TARGET,
                    require_followup_decision=True,
                    require_outside_soloing_repair=True,
                    require_no_quality_claim=True,
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
        self.assertEqual(
            SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision_v3",
        )


if __name__ == "__main__":
    unittest.main()
