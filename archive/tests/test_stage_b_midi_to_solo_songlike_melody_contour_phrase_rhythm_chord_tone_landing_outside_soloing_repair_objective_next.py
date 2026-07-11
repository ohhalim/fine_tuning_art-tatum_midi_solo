from __future__ import annotations

import unittest

from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next import (
    BOUNDARY,
    CURRENT_EVIDENCE_NEXT_BOUNDARY,
    REPAIR_RETRY_NEXT_BOUNDARY,
    SCHEMA_VERSION as OBJECTIVE_SCHEMA_VERSION,
    SOURCE_OBJECTIVE_SCHEMA_CONTEXT_KEYS,
    StageBMidiToSoloOutsideSoloingRepairObjectiveNextError,
    build_objective_next_report,
    validate_objective_next_report,
)
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_input import (
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    SOURCE_GUARD_SCHEMA_CONTEXT_KEYS,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package import (
    SCHEMA_VERSION as SOURCE_LISTENING_PACKAGE_SCHEMA_VERSION,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio import (
    SCHEMA_VERSION as SOURCE_AUDIO_PACKAGE_SCHEMA_VERSION,
    SOURCE_SWEEP_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup import (
    SCHEMA_VERSION as SOURCE_FOLLOWUP_SCHEMA_VERSION,
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
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (
    SCHEMA_VERSION as CHORD_TONE_REPAIR_SWEEP_SCHEMA_VERSION,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective import (
    SCHEMA_VERSION as CHORD_CONTEXT_OBJECTIVE_SCHEMA_VERSION,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    SCHEMA_VERSION as BRIDGE_SCHEMA_VERSION,
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
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
}


SOURCE_SUMMARY = {
    "source_schema_version": SOURCE_LISTENING_PACKAGE_SCHEMA_VERSION,
    "source_audio_package_schema_version": SOURCE_AUDIO_PACKAGE_SCHEMA_VERSION,
    "source_repair_sweep_schema_version": SOURCE_SWEEP_SCHEMA_VERSION,
    "source_followup_schema_version": SOURCE_FOLLOWUP_SCHEMA_VERSION,
    "source_objective_input_guard_schema_version": SOURCE_INPUT_GUARD_SCHEMA_VERSION,
    "source_package_schema_version": SOURCE_PACKAGE_SCHEMA_VERSION,
    "source_audio_schema_version": SOURCE_AUDIO_SCHEMA_VERSION,
    "chord_tone_repair_sweep_schema_version": CHORD_TONE_REPAIR_SWEEP_SCHEMA_VERSION,
    "chord_tone_repair_sweep_source_schema_version": CHORD_CONTEXT_OBJECTIVE_SCHEMA_VERSION,
    "chord_tone_repair_sweep_bridge_schema_version": BRIDGE_SCHEMA_VERSION,
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
    **SOURCE_CONTEXT,
}


def input_guard(
    *,
    preference_fill_allowed: bool = False,
    quality_claim: bool = False,
    source_summary: dict | None = None,
) -> dict:
    summary = source_summary if source_summary is not None else dict(SOURCE_SUMMARY)
    return {
        "schema_version": SOURCE_SCHEMA_VERSION,
        "boundary": SOURCE_BOUNDARY,
        "source_schema_version": SOURCE_LISTENING_PACKAGE_SCHEMA_VERSION,
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
            **{key: summary.get(key) for key in SOURCE_GUARD_SCHEMA_CONTEXT_KEYS},
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
            issue_number=1148,
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
        self.assertEqual(report["schema_version"], OBJECTIVE_SCHEMA_VERSION)
        self.assertEqual(report["issue_number"], 1148)
        self.assertEqual(report["source_schema_version"], SOURCE_SCHEMA_VERSION)
        self.assertEqual(summary["source_schema_version"], SOURCE_SCHEMA_VERSION)
        self.assertEqual(
            summary["source_listening_package_schema_version"],
            SOURCE_LISTENING_PACKAGE_SCHEMA_VERSION,
        )
        self.assertEqual(
            summary["source_audio_package_schema_version"],
            SOURCE_AUDIO_PACKAGE_SCHEMA_VERSION,
        )
        for key in SOURCE_OBJECTIVE_SCHEMA_CONTEXT_KEYS:
            self.assertEqual(report["objective_summary"][key], summary[key])
            self.assertEqual(report["readiness"][key], summary[key])
        for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
            self.assertIn(key, report["objective_summary"])
            self.assertIn(key, report["readiness"])
            self.assertEqual(summary[key], SOURCE_CONTEXT[key])
        self.assertFalse(summary["human_audio_preference_claimed"])
        self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])

    def test_routes_to_retry_when_outside_risk_remains(self) -> None:
        broken_summary = dict(SOURCE_SUMMARY)
        broken_summary["outside_soloing_pitch_role_risk_count_after"] = 1
        report = build_objective_next_report(
            input_guard_report=input_guard(source_summary=broken_summary),
            output_dir="out",
            issue_number=1148,
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
                issue_number=1148,
                max_non_chord_tone_run_threshold=3,
                min_final_landing_chord_tone_count=6,
            )

    def test_rejects_upstream_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloOutsideSoloingRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(quality_claim=True),
                output_dir="out",
                issue_number=1148,
                max_non_chord_tone_run_threshold=3,
                min_final_landing_chord_tone_count=6,
            )

    def test_rejects_missing_source_context_preserved_flag(self) -> None:
        broken_summary = dict(SOURCE_SUMMARY)
        broken_summary[
            "repair_sweep_source_outside_soloing_source_context_preserved"
        ] = False
        with self.assertRaises(StageBMidiToSoloOutsideSoloingRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=input_guard(source_summary=broken_summary),
                output_dir="out",
                issue_number=1148,
                max_non_chord_tone_run_threshold=3,
                min_final_landing_chord_tone_count=6,
            )

    def test_rejects_input_guard_schema_mismatch(self) -> None:
        source = input_guard()
        source["schema_version"] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloOutsideSoloingRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=source,
                output_dir="out",
                issue_number=1148,
                max_non_chord_tone_run_threshold=3,
                min_final_landing_chord_tone_count=6,
            )

    def test_rejects_input_guard_schema_context_mismatch(self) -> None:
        source = input_guard()
        source["readiness"][SOURCE_GUARD_SCHEMA_CONTEXT_KEYS[1]] = "stale_schema"
        with self.assertRaises(StageBMidiToSoloOutsideSoloingRepairObjectiveNextError):
            build_objective_next_report(
                input_guard_report=source,
                output_dir="out",
                issue_number=1148,
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
        self.assertEqual(
            OBJECTIVE_SCHEMA_VERSION,
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_objective_next_v4",
        )


if __name__ == "__main__":
    unittest.main()
