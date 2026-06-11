from __future__ import annotations

import unittest
from pathlib import Path

from scripts.audit_stage_b_midi_to_solo_mvp_completion import (
    BOUNDARY,
    NEXT_BOUNDARY,
    SCHEMA_VERSION,
    StageBMidiToSoloMvpCompletionAuditError,
    build_mvp_completion_audit_report,
    validate_mvp_completion_audit_report,
)
from scripts.consolidate_stage_b_midi_to_solo_mvp_current_evidence import (
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BOUNDARY as CURRENT_EVIDENCE_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    SOURCE_OBJECTIVE_SCHEMA_CONTEXT_KEYS,
)


SOURCE_CONTEXT = {
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_objective_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_objective_source_outside_soloing_source_targeted": False,
    "followup_objective_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_objective_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_objective_source_outside_soloing_source_context_preserved": True,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "followup_repair_sweep_source_outside_soloing_source_targeted": False,
    "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "followup_repair_sweep_source_outside_soloing_source_context_preserved": True,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before": 5,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after": 2,
    "repair_sweep_source_outside_soloing_source_pitch_role_risk_delta": 3,
    "repair_sweep_source_outside_soloing_source_targeted": False,
    "repair_sweep_source_outside_soloing_source_residual_risk_preserved": True,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after": 0,
    "repair_sweep_source_outside_soloing_current_pitch_role_risk_delta": 2,
    "repair_sweep_source_outside_soloing_source_context_preserved": True,
}


SOURCE_SCHEMA_CONTEXT = {
    "source_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "outside_soloing_repair_listening_review_input_guard_v4"
    ),
    "source_listening_package_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "outside_soloing_repair_listening_review_package_v4"
    ),
    "source_audio_package_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "outside_soloing_repair_audio_package_v4"
    ),
    "source_repair_sweep_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "outside_soloing_repair_sweep_v3"
    ),
    "source_followup_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "repair_followup_decision_v3"
    ),
    "source_objective_input_guard_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "repair_listening_review_input_guard_v3"
    ),
    "source_package_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "repair_listening_review_package_v3"
    ),
    "source_audio_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "repair_audio_package_v4"
    ),
    "chord_tone_repair_sweep_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
        "repair_sweep_v4"
    ),
    "chord_tone_repair_sweep_source_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
        "chord_context_pitch_role_objective_decision_v4"
    ),
    "chord_tone_repair_sweep_bridge_schema_version": (
        "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
        "chord_context_pitch_role_bridge_v4"
    ),
}


def current_evidence(
    *,
    quality_claim: bool = False,
    strict_count: int = 9,
    pitch_contour_supported: bool = True,
    changed_ratio_repair_supported: bool = True,
    outside_soloing_repair_supported: bool = True,
) -> dict:
    current_evidence_supported = bool(
        pitch_contour_supported
        and changed_ratio_repair_supported
        and outside_soloing_repair_supported
    )
    return {
        "boundary": CURRENT_EVIDENCE_BOUNDARY,
        "readiness": {
            "mvp_current_evidence_consolidated": True,
            "input_contract_ready": True,
            "context_extraction_ready": True,
            "training_resource_ready": True,
            "ranked_midi_candidates_exported": True,
            "technical_wav_path_ready": True,
            "selected_scale_objective_path_complete": True,
            "phrase_bank_cli_technical_path_ready": True,
            "model_conditioned_pitch_contour_objective_path_ready": pitch_contour_supported,
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready": changed_ratio_repair_supported,
            "outside_soloing_repair_objective_path_ready": outside_soloing_repair_supported,
            "outside_soloing_repair_source_context_preserved": outside_soloing_repair_supported,
            "outside_soloing_repair_schema_context_preserved": outside_soloing_repair_supported,
            "current_mvp_technical_execution_evidence_supported": True,
            "current_mvp_objective_repair_evidence_supported": True,
            "midi_to_solo_mvp_current_evidence_supported": current_evidence_supported,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": quality_claim,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "ranked_midi_generation": {
            "generation_source": "context_conditioned_fallback",
            "exported_candidate_count": 3,
            "exported_qualified_candidate_count": 3,
        },
        "technical_audio_render": {
            "rendered_audio_file_count": 3,
            "technical_wav_validation": True,
            "duration_min_seconds": 18.617,
            "duration_max_seconds": 18.991,
        },
        "selected_scale_objective_path": {
            "objective_path_supported": True,
            "sample_count": 9,
            "strict_valid_sample_count": strict_count,
            "grammar_gate_sample_count": 9,
            "dead_air_failure_count": 0,
            "collapse_warning_sample_count": 0,
            "avg_postprocess_removal_ratio": 0.2176,
            "target_avg_postprocess_removal_ratio": 0.3,
        },
        "phrase_bank_cli_technical_path": {
            "technical_midi_to_solo_cli_path_ready": True,
            "candidate_count": 3,
            "rendered_audio_file_count": 3,
            "input_context_bars": 228,
            "preference_fill_allowed": False,
        },
        "model_conditioned_pitch_contour_objective_path": {
            "boundary": (
                "stage_b_midi_to_solo_model_conditioned_input_path_"
                "dead_air_timing_repair_pitch_contour_objective_only_next_decision"
            ),
            "objective_next_decision_completed": True,
            "current_evidence_consolidation_ready": pitch_contour_supported,
            "review_item_count": 3,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 3,
            "max_repaired_interval": 11 if pitch_contour_supported else 13,
            "max_interval_threshold": 12,
            "pitch_contour_target_supported": pitch_contour_supported,
            "max_pitch_changed_ratio": 0.7174,
            "pitch_changed_ratio_review_required": True,
            "audio_review_required": True,
        },
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_path": {
            "objective_next_completed": True,
            "current_evidence_consolidation_ready": changed_ratio_repair_supported,
            "changed_ratio_repair_objective_path_supported": changed_ratio_repair_supported,
            "review_item_count": 3,
            "required_input_field_count": 4,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 3,
            "max_repaired_interval": 12,
            "max_interval_threshold": 12,
            "interval_target_supported": True,
            "max_repaired_pitch_changed_ratio": 0.4348
            if changed_ratio_repair_supported
            else 0.6522,
            "target_max_pitch_changed_ratio": 0.5,
            "changed_ratio_target_supported": changed_ratio_repair_supported,
            "audio_review_required": True,
        },
        "outside_soloing_repair_objective_path": {
            "boundary": (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
                "chord_tone_landing_outside_soloing_repair_objective_only_next_decision"
            ),
            "outside_soloing_repair_objective_schema_version": (
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
            ),
            **SOURCE_SCHEMA_CONTEXT,
            "objective_next_completed": True,
            "objective_next_decision_completed": True,
            "current_evidence_consolidation_ready": outside_soloing_repair_supported,
            "outside_soloing_repair_objective_path_supported": outside_soloing_repair_supported,
            "review_item_count": 6,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "technical_wav_validation": True,
            "rendered_audio_file_count": 6,
            "changed_note_total": 2,
            "source_objective_outside_soloing_pitch_role_risk_count": 5,
            "source_outside_soloing_pitch_role_risk_count_before": 5,
            "source_outside_soloing_pitch_role_risk_count_after": 2,
            "source_outside_soloing_pitch_role_risk_delta": 3,
            "source_outside_soloing_repair_targeted": False,
            "source_outside_soloing_residual_risk_preserved": True,
            "outside_soloing_pitch_role_risk_count_after": (
                0 if outside_soloing_repair_supported else 1
            ),
            "outside_soloing_pitch_role_risk_delta": 2,
            "outside_soloing_target_supported": outside_soloing_repair_supported,
            "weak_chord_tone_landing_risk_count_after": 0,
            "weak_landing_target_supported": True,
            "final_landing_chord_tone_count_after": 6,
            "final_landing_target_supported": True,
            "max_non_chord_tone_run_after": 3,
            "non_chord_run_target_supported": True,
            **SOURCE_CONTEXT,
        },
        "decision": {
            "critical_user_input_required": False,
        },
    }


def readme_text(*, missing_boundary: bool = False) -> str:
    boundary = "stale_boundary" if missing_boundary else CURRENT_EVIDENCE_BOUNDARY
    return "\n".join(
        [
            "- latest evidence boundary: `stage_b_midi_to_solo_mvp_completion_audit`",
            f"- current evidence boundary: `{boundary}`",
            "- current MVP evidence support: `true`",
            "- input MIDI -> context -> ranked MIDI -> WAV technical path: `true`",
            "- selected-scale objective repair path complete: `true`",
            "- phrase-bank CLI technical path included in current evidence: `true`",
            "- model-conditioned pitch-contour objective path ready: `true`",
            "- model-conditioned pitch-contour changed-ratio review required: `true`",
            "- model-conditioned pitch-contour changed-ratio repair objective path ready: `true`",
            "- current evidence changed-ratio repair objective path included: `true`",
            "- outside-soloing repair objective path included in current evidence: `true`",
            "- current evidence outside-soloing repair objective path included: `true`",
            "- outside-soloing source-context evidence reflected: `true`",
            "- outside-soloing schema-context evidence reflected: `true`",
            "- outside-soloing repair objective schema version: "
            f"`{OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION}`",
            "- outside-soloing source pitch-role risk count: `5 -> 2`",
            "- outside-soloing source residual risk preserved: `true`",
            "- outside-soloing current repair pitch-role risk count after: `0`",
            "- follow-up objective source outside-soloing source context preserved: `true`",
            "- follow-up repair sweep source outside-soloing source context preserved: `true`",
            "- bridge repair sweep source outside-soloing source context preserved: `true`",
            "- README evidence refreshed: `true`",
            "- human/audio preference claim: `false`",
            "- MIDI-to-solo musical quality claim: `false`",
            "- broad trained-model quality claim: `false`",
            "- Brad style adaptation claim: `false`",
        ]
    )


class StageBMidiToSoloMvpCompletionAuditTest(unittest.TestCase):
    def test_audits_technical_mvp_completion_without_quality_claim(self) -> None:
        report = build_mvp_completion_audit_report(
            current_evidence=current_evidence(),
            readme_text=readme_text(),
            output_dir=Path("outputs/audit"),
            issue_number=1070,
        )
        summary = validate_mvp_completion_audit_report(
            report,
            expected_boundary=BOUNDARY,
            expected_next_boundary=NEXT_BOUNDARY,
            require_technical_mvp_completion=True,
            require_no_quality_claim=True,
            require_model_conditioned_pitch_contour_objective=True,
            require_model_conditioned_pitch_contour_changed_ratio_repair_objective=True,
            require_outside_soloing_repair_objective=True,
        )

        self.assertTrue(summary["technical_model_core_mvp_completed"])
        self.assertTrue(summary["input_to_ranked_midi_completed"])
        self.assertTrue(summary["input_to_rendered_wav_completed"])
        self.assertTrue(summary["selected_scale_objective_repair_completed"])
        self.assertTrue(summary["phrase_bank_cli_technical_path_completed"])
        self.assertFalse(summary["musical_quality_mvp_completed"])
        self.assertFalse(summary["human_audio_preference_completed"])
        self.assertFalse(summary["product_mvp_completed"])
        self.assertEqual(summary["generation_source"], "context_conditioned_fallback")
        self.assertEqual(summary["exported_candidate_count"], 3)
        self.assertEqual(summary["rendered_audio_file_count"], 3)
        self.assertEqual(summary["objective_sample_count"], 9)
        self.assertEqual(summary["objective_strict_valid_sample_count"], 9)
        self.assertEqual(summary["objective_dead_air_failure_count"], 0)
        self.assertTrue(summary["phrase_bank_cli_technical_path_ready"])
        self.assertEqual(summary["cli_candidate_count"], 3)
        self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
        self.assertEqual(summary["cli_input_context_bars"], 228)
        self.assertFalse(summary["cli_preference_fill_allowed"])
        self.assertTrue(summary["model_conditioned_pitch_contour_objective_completed"])
        self.assertTrue(summary["model_conditioned_pitch_contour_objective_path_ready"])
        self.assertEqual(summary["model_conditioned_pitch_contour_max_interval"], 11)
        self.assertEqual(summary["model_conditioned_pitch_contour_max_interval_threshold"], 12)
        self.assertTrue(summary["model_conditioned_pitch_contour_target_supported"])
        self.assertTrue(
            summary["model_conditioned_pitch_contour_pitch_changed_ratio_review_required"]
        )
        self.assertTrue(summary["model_conditioned_pitch_contour_audio_review_required"])
        self.assertTrue(
            summary[
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed"
            ]
        )
        self.assertTrue(
            summary[
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready"
            ]
        )
        self.assertEqual(
            summary[
                "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count"
            ],
            3,
        )
        self.assertEqual(
            summary[
                "model_conditioned_pitch_contour_changed_ratio_repair_max_interval"
            ],
            12,
        )
        self.assertAlmostEqual(
            summary[
                "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio"
            ],
            0.4348,
        )
        self.assertTrue(
            summary[
                "model_conditioned_pitch_contour_changed_ratio_repair_target_supported"
            ]
        )
        self.assertTrue(summary["outside_soloing_repair_objective_completed"])
        self.assertTrue(summary["outside_soloing_repair_objective_path_ready"])
        self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
        self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
        self.assertEqual(
            summary["outside_soloing_repair_objective_schema_version"],
            OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
        )
        for key in SOURCE_OBJECTIVE_SCHEMA_CONTEXT_KEYS:
            self.assertEqual(
                summary[f"outside_soloing_repair_{key}"],
                SOURCE_SCHEMA_CONTEXT[key],
            )
        self.assertTrue(summary["outside_soloing_repair_current_evidence_ready"])
        self.assertEqual(summary["outside_soloing_repair_rendered_audio_file_count"], 6)
        self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
        self.assertEqual(
            summary["outside_soloing_repair_source_objective_pitch_role_risk_count"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_before"],
            5,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_count_after"],
            2,
        )
        self.assertEqual(
            summary["outside_soloing_repair_source_pitch_role_risk_delta"],
            3,
        )
        self.assertFalse(summary["outside_soloing_repair_source_targeted"])
        self.assertTrue(summary["outside_soloing_repair_source_residual_risk_preserved"])
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
        self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
        self.assertTrue(summary["outside_soloing_repair_objective_path_supported"])
        self.assertTrue(summary["outside_soloing_repair_target_supported"])
        self.assertTrue(summary["outside_soloing_repair_weak_landing_target_supported"])
        self.assertTrue(summary["outside_soloing_repair_final_landing_target_supported"])
        self.assertTrue(summary["outside_soloing_repair_non_chord_run_target_supported"])
        self.assertFalse(summary["outside_soloing_repair_preference_fill_allowed"])
        for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
            self.assertEqual(summary[key], SOURCE_CONTEXT[key])
        self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)
        self.assertEqual(report["schema_version"], SCHEMA_VERSION)
        self.assertEqual(report["issue_number"], 1070)

    def test_rejects_missing_outside_soloing_source_context_field(self) -> None:
        evidence = current_evidence()
        del evidence["outside_soloing_repair_objective_path"][
            "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
        ]
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=evidence,
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=1070,
            )

    def test_rejects_false_outside_soloing_source_context_preserved_flag(self) -> None:
        evidence = current_evidence()
        evidence["outside_soloing_repair_objective_path"][
            "followup_repair_sweep_source_outside_soloing_source_context_preserved"
        ] = False
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=evidence,
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=1070,
            )

    def test_rejects_missing_outside_soloing_schema_context(self) -> None:
        evidence = current_evidence()
        evidence["readiness"]["outside_soloing_repair_schema_context_preserved"] = False
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=evidence,
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=1152,
            )

        evidence = current_evidence()
        del evidence["outside_soloing_repair_objective_path"]["source_schema_version"]
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=evidence,
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=1152,
            )

    def test_rejects_missing_readme_evidence_boundary(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(),
                readme_text=readme_text(missing_boundary=True),
                output_dir=Path("outputs/audit"),
                issue_number=616,
            )

    def test_rejects_quality_claim(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(quality_claim=True),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=616,
            )

    def test_rejects_objective_strict_shortfall(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(strict_count=8),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=616,
            )

    def test_rejects_missing_pitch_contour_support(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(pitch_contour_supported=False),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=712,
            )

    def test_rejects_missing_changed_ratio_repair_support(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(changed_ratio_repair_supported=False),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=732,
            )

    def test_rejects_missing_outside_soloing_repair_support(self) -> None:
        with self.assertRaises(StageBMidiToSoloMvpCompletionAuditError):
            build_mvp_completion_audit_report(
                current_evidence=current_evidence(outside_soloing_repair_supported=False),
                readme_text=readme_text(),
                output_dir=Path("outputs/audit"),
                issue_number=816,
            )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_completion_audit")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_quality_gap_decision")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_mvp_completion_audit_v3")


if __name__ == "__main__":
    unittest.main()
