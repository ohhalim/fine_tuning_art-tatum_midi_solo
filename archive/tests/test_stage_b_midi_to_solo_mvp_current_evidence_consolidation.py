from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.consolidate_stage_b_midi_to_solo_mvp_current_evidence import (
    BOUNDARY,
    BRIDGE_SOURCE_CONTEXT_KEYS,
    NEXT_BOUNDARY,
    OBJECTIVE_FINAL_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    SCHEMA_VERSION,
    SOURCE_OBJECTIVE_SCHEMA_CONTEXT_KEYS,
    StageBMidiToSoloMvpCurrentEvidenceConsolidationError,
    build_current_evidence_consolidation_report,
    validate_current_evidence_consolidation_report,
)


def touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")
    return str(path)


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


def reports(
    root: Path,
    *,
    strict_count: int = 9,
    dead_air_failure_count: int = 0,
    pitch_contour_supported: bool = True,
    changed_ratio_repair_supported: bool = True,
    outside_soloing_repair_supported: bool = True,
    collapse_count: int = 0,
    quality_claim: bool = False,
) -> dict[str, dict]:
    midi_paths = [touch(root / f"rank_{index:02d}.mid") for index in range(1, 4)]
    wav_paths = [touch(root / f"rank_{index:02d}.wav") for index in range(1, 4)]
    objective_support = strict_count == 9 and dead_air_failure_count == 0 and collapse_count == 0
    return {
        "contract": {
            "boundary": "stage_b_midi_to_solo_mvp_input_contract",
            "output_contract": {
                "candidate_count": 32,
                "export_top_midi_count": 3,
                "target_solo_bars": 8,
            },
            "objective_gate": {
                "min_note_count": 24,
                "min_unique_pitch_count": 8,
                "max_dead_air_ratio": 0.5,
                "max_long_note_ratio": 0.5,
                "max_simultaneous_notes": 1,
                "min_phrase_coverage_ratio": 0.75,
            },
            "generation_stack": {
                "primary_path": "generic_base_checkpoint_conditioned_generation",
                "fallback_path": "phrase_retrieval_data_motif_hybrid",
            },
            "decision": {"critical_user_input_required": False},
        },
        "context": {
            "boundary": "stage_b_midi_to_solo_context_extraction_mvp",
            "summary": {
                "context_bars": 8,
                "positions_per_bar": 16,
                "context_event_count": 128,
                "inferred_chord_bar_count": 4,
                "carry_forward_chord_bar_count": 4,
                "unknown_chord_bar_count": 0,
                "low_confidence_bar_count": 4,
                "bass_note_bar_count": 4,
            },
            "readiness": {
                "context_extraction_completed": True,
                "required_context_fields_present": True,
                "midi_to_solo_mvp_claimed": False,
                "harmony_analysis_quality_claimed": False,
                "brad_style_fine_tuning_completed": False,
            },
        },
        "resource": {
            "boundary": "stage_b_midi_to_solo_training_resource_probe",
            "readiness": {
                "training_resource_probe_completed": True,
                "midi_to_solo_training_resource_ready": True,
                "conditioned_generation_probe_ready": True,
                "midi_to_solo_mvp_claimed": False,
                "broad_training_executed": False,
                "broad_trained_model_quality_claimed": False,
                "brad_style_adaptation_claimed": False,
                "musical_quality_claimed": False,
            },
        },
        "generation": {
            "boundary": "stage_b_midi_to_solo_conditioned_generation_probe",
            "generation_config": {
                "generation_source": "context_conditioned_fallback",
                "model_checkpoint_generation_used": False,
                "checkpoint_direct_generation_skip_reason": "sequence budget shortfall",
            },
            "summary": {
                "candidate_count": 8,
                "qualified_candidate_count": 8,
                "exported_candidate_count": 3,
                "exported_qualified_candidate_count": 3,
                "best_score": 1.89,
                "best_note_count": 60,
                "best_unique_pitch_count": 14,
                "best_max_simultaneous_notes": 1,
                "best_chord_tone_ratio": 1.0,
            },
            "readiness": {
                "conditioned_generation_probe_completed": True,
                "ranked_midi_candidates_exported": True,
                "midi_to_solo_mvp_claimed": False,
                "model_checkpoint_generation_quality_claimed": False,
                "broad_trained_model_quality_claimed": False,
                "brad_style_adaptation_claimed": False,
                "human_audio_preference_claimed": False,
            },
            "top_candidates": [{"export_midi_path": path} for path in midi_paths],
        },
        "audio": {
            "audio_render_boundary": {
                "boundary": "stage_b_midi_to_solo_candidate_audio_render_package",
                "render_attempted": True,
                "rendered_audio_file_count": 3,
                "technical_wav_validation": True,
                "audio_rendered_quality_claimed": False,
                "human_audio_preference_claimed": False,
                "musical_quality_claimed": quality_claim,
                "midi_to_solo_mvp_claimed": False,
                "broad_trained_model_quality_claimed": False,
                "brad_style_adaptation_claimed": False,
            },
            "rendered_audio_files": [
                {
                    "wav_file": {
                        "path": path,
                        "sample_rate": 44100,
                        "duration_seconds": 18.0 + index,
                    }
                }
                for index, path in enumerate(wav_paths)
            ],
        },
        "objective": {
            "boundary": OBJECTIVE_NEXT_BOUNDARY,
            "final_boundary": OBJECTIVE_FINAL_BOUNDARY,
            "postprocess_removal_dead_air_repair_summary": {
                "sample_count": 9,
                "seed_count": 3,
                "valid_sample_count": strict_count,
                "strict_valid_sample_count": strict_count,
                "grammar_gate_sample_count": 9,
                "dead_air_failure_count": dead_air_failure_count,
                "collapse_warning_sample_count": collapse_count,
                "strict_valid_sample_delta": 1,
                "dead_air_failure_delta": -1,
                "temperature": 0.75,
                "top_k": 4,
                "avoid_reused_positions": True,
                "avg_postprocess_removal_ratio": 0.2176,
                "max_postprocess_removal_ratio": 0.2917,
                "target_avg_postprocess_removal_ratio": 0.3,
                "postprocess_removal_delta": -0.1435,
                "avg_onset_coverage_ratio": 0.7326,
                "avg_sustained_coverage_ratio": 0.7708,
            },
            "review_boundary_summary": {
                "candidate_count": 3,
                "rendered_audio_file_count": 3,
                "review_input_template_written": True,
                "validated_review_input_present": False,
                "preference_fill_allowed": False,
                "pending_status_field_count": 4,
                "pending_candidate_decision_count": 3,
                "pending_candidate_field_count": 9,
            },
            "readiness": {
                "objective_only_decision_completed": True,
                "objective_postprocess_removal_dead_air_repair_path_supported": objective_support,
                "human_audio_preference_claimed": False,
                "midi_to_solo_musical_quality_claimed": quality_claim,
                "broad_trained_model_quality_claimed": False,
                "brad_style_adaptation_claimed": False,
            },
            "decision": {
                "next_boundary": "stage_b_midi_to_solo_mvp_current_evidence_consolidation",
                "critical_user_input_required": False,
            },
        },
        "cli_objective": {
            "boundary": "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision",
            "objective_summary": {
                "technical_midi_to_solo_cli_path_ready": True,
                "mvp_current_evidence_consolidation_ready": True,
                "explicit_input_used": True,
                "candidate_count": 3,
                "objective_supported_candidate_count": 3,
                "repaired_midi_file_count": 3,
                "rendered_audio_file_count": 3,
                "technical_wav_validation": True,
                "input_context_bars": 228,
                "dead_air_range": [0.1895, 0.2211],
                "validated_review_input_present": False,
                "preference_fill_allowed": False,
            },
            "readiness": {
                "cli_objective_only_next_decision_completed": True,
                "technical_midi_to_solo_cli_path_ready": True,
                "mvp_current_evidence_consolidation_ready": True,
                "human_audio_preference_claimed": False,
                "midi_to_solo_musical_quality_claimed": quality_claim,
                "audio_rendered_quality_claimed": False,
                "phrase_bank_musical_quality_claimed": False,
                "broad_trained_model_quality_claimed": False,
                "brad_style_adaptation_claimed": False,
                "production_ready_claimed": False,
            },
            "decision": {
                "next_boundary": "stage_b_midi_to_solo_mvp_current_evidence_consolidation",
                "critical_user_input_required": False,
            },
        },
        "model_conditioned_pitch_contour_objective": {
            "boundary": (
                "stage_b_midi_to_solo_model_conditioned_input_path_"
                "dead_air_timing_repair_pitch_contour_objective_only_next_decision"
            ),
            "objective_summary": {
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
                "current_evidence_consolidation_ready": pitch_contour_supported,
            },
            "readiness": {
                "objective_next_decision_completed": True,
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
                "next_boundary": "stage_b_midi_to_solo_mvp_current_evidence_consolidation",
                "critical_user_input_required": False,
            },
        },
        "model_conditioned_pitch_contour_changed_ratio_repair_objective": {
            "boundary": (
                "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_"
                "objective_only_next_decision"
            ),
            "objective_summary": {
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
                "changed_ratio_repair_objective_path_supported": changed_ratio_repair_supported,
                "current_evidence_consolidation_ready": changed_ratio_repair_supported,
            },
            "readiness": {
                "objective_next_completed": True,
                "objective_next_decision_completed": True,
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
                "next_boundary": "stage_b_midi_to_solo_mvp_current_evidence_consolidation",
                "critical_user_input_required": False,
            },
        },
        "outside_soloing_repair_objective": {
            "schema_version": OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            "boundary": (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
                "chord_tone_landing_outside_soloing_repair_objective_only_next_decision"
            ),
            "objective_summary": {
                **SOURCE_SCHEMA_CONTEXT,
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
                "outside_soloing_pitch_role_risk_count_after": 0
                if outside_soloing_repair_supported
                else 1,
                "outside_soloing_pitch_role_risk_delta": 2,
                "outside_soloing_target_supported": outside_soloing_repair_supported,
                "weak_chord_tone_landing_risk_count_after": 0,
                "weak_landing_target_supported": True,
                "final_landing_chord_tone_count_after": 6,
                "final_landing_target_supported": True,
                "max_non_chord_tone_run_after": 3,
                "non_chord_run_target_supported": True,
                "outside_soloing_repair_objective_path_supported": (
                    outside_soloing_repair_supported
                ),
                "current_evidence_consolidation_ready": outside_soloing_repair_supported,
                **SOURCE_CONTEXT,
            },
            "readiness": {
                **SOURCE_SCHEMA_CONTEXT,
                "objective_next_completed": True,
                "objective_next_decision_completed": True,
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
                "next_boundary": "stage_b_midi_to_solo_mvp_current_evidence_consolidation",
                "critical_user_input_required": False,
            },
        },
    }


class StageBMidiToSoloMvpCurrentEvidenceConsolidationTest(unittest.TestCase):
    def test_consolidates_current_evidence_without_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            report = build_current_evidence_consolidation_report(
                contract_report=data["contract"],
                context_report=data["context"],
                resource_probe=data["resource"],
                generation_probe=data["generation"],
                audio_render=data["audio"],
                objective_next=data["objective"],
                cli_objective_next=data["cli_objective"],
                model_conditioned_pitch_contour_objective_next=data[
                    "model_conditioned_pitch_contour_objective"
                ],
                model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                    "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                ],
                outside_soloing_repair_objective_next=data[
                    "outside_soloing_repair_objective"
                ],
                output_dir=Path(tmp) / "out",
                issue_number=1150,
            )
            summary = validate_current_evidence_consolidation_report(
                report,
                expected_boundary=BOUNDARY,
                expected_next_boundary=NEXT_BOUNDARY,
                require_current_evidence_support=True,
                require_no_quality_claim=True,
                min_exported_candidates=3,
                min_rendered_wav_files=3,
                min_objective_sample_count=9,
                require_model_conditioned_pitch_contour_objective=True,
                require_model_conditioned_pitch_contour_changed_ratio_repair_objective=True,
                require_outside_soloing_repair_objective=True,
            )

            self.assertTrue(summary["midi_to_solo_mvp_current_evidence_supported"])
            self.assertTrue(summary["technical_execution_evidence_supported"])
            self.assertTrue(summary["selected_scale_objective_path_complete"])
            self.assertTrue(summary["phrase_bank_cli_technical_path_ready"])
            self.assertTrue(summary["model_conditioned_pitch_contour_objective_path_ready"])
            self.assertTrue(
                summary[
                    "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready"
                ]
            )
            self.assertTrue(summary["outside_soloing_repair_objective_path_ready"])
            self.assertEqual(summary["generation_source"], "context_conditioned_fallback")
            self.assertEqual(summary["exported_candidate_count"], 3)
            self.assertEqual(summary["rendered_audio_file_count"], 3)
            self.assertEqual(summary["cli_candidate_count"], 3)
            self.assertEqual(summary["cli_rendered_audio_file_count"], 3)
            self.assertEqual(summary["cli_input_context_bars"], 228)
            self.assertFalse(summary["cli_preference_fill_allowed"])
            self.assertEqual(summary["objective_sample_count"], 9)
            self.assertEqual(summary["objective_strict_valid_sample_count"], 9)
            self.assertEqual(summary["objective_grammar_gate_sample_count"], 9)
            self.assertEqual(summary["objective_dead_air_failure_count"], 0)
            self.assertEqual(summary["objective_collapse_warning_sample_count"], 0)
            self.assertEqual(summary["model_conditioned_pitch_contour_max_interval"], 11)
            self.assertTrue(summary["model_conditioned_pitch_contour_target_supported"])
            self.assertTrue(
                summary[
                    "model_conditioned_pitch_contour_pitch_changed_ratio_review_required"
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
                    "model_conditioned_pitch_contour_changed_ratio_repair_changed_ratio_target_supported"
                ]
            )
            self.assertEqual(summary["outside_soloing_repair_rendered_audio_file_count"], 6)
            self.assertEqual(summary["outside_soloing_repair_changed_note_total"], 2)
            self.assertEqual(
                summary[
                    "outside_soloing_repair_source_objective_pitch_role_risk_count"
                ],
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
            self.assertTrue(
                summary["outside_soloing_repair_source_residual_risk_preserved"]
            )
            self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_count_after"], 0)
            self.assertEqual(summary["outside_soloing_repair_pitch_role_risk_delta"], 2)
            self.assertTrue(summary["outside_soloing_repair_target_supported"])
            self.assertTrue(summary["outside_soloing_repair_weak_landing_target_supported"])
            self.assertTrue(summary["outside_soloing_repair_final_landing_target_supported"])
            self.assertTrue(summary["outside_soloing_repair_non_chord_run_target_supported"])
            self.assertTrue(summary["outside_soloing_repair_objective_path_supported"])
            self.assertTrue(summary["outside_soloing_repair_source_context_preserved"])
            self.assertTrue(summary["outside_soloing_repair_schema_context_preserved"])
            self.assertEqual(
                summary["outside_soloing_repair_objective_schema_version"],
                OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
            )
            for key in SOURCE_OBJECTIVE_SCHEMA_CONTEXT_KEYS:
                self.assertIn(key, report["outside_soloing_repair_objective_path"])
                self.assertEqual(
                    summary[f"outside_soloing_repair_{key}"],
                    SOURCE_SCHEMA_CONTEXT[key],
                )
            for key in BRIDGE_SOURCE_CONTEXT_KEYS:
                self.assertIn(key, report["outside_soloing_repair_objective_path"])
                self.assertEqual(summary[key], SOURCE_CONTEXT[key])
            self.assertTrue(
                summary["followup_objective_source_outside_soloing_source_context_preserved"]
            )
            self.assertTrue(
                summary["followup_repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertTrue(
                summary["repair_sweep_source_outside_soloing_source_context_preserved"]
            )
            self.assertFalse(summary["human_audio_preference_claimed"])
            self.assertFalse(summary["midi_to_solo_musical_quality_claimed"])
            self.assertEqual(summary["next_boundary"], NEXT_BOUNDARY)

    def test_rejects_missing_ranked_midi_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            Path(data["generation"]["top_candidates"][0]["export_midi_path"]).unlink()
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    output_dir=Path(tmp) / "out",
                    issue_number=612,
                )

    def test_rejects_objective_strict_shortfall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp), strict_count=8)
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    output_dir=Path(tmp) / "out",
                    issue_number=612,
                )

    def test_rejects_dead_air_or_quality_claim(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp), dead_air_failure_count=1)
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    output_dir=Path(tmp) / "out",
                    issue_number=612,
                )

        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp), quality_claim=True)
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    output_dir=Path(tmp) / "out",
                    issue_number=612,
                )

    def test_rejects_missing_model_conditioned_pitch_contour_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp), pitch_contour_supported=False)
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=612,
                )

    def test_rejects_missing_changed_ratio_repair_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp), changed_ratio_repair_supported=False)
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=728,
                )

    def test_rejects_missing_outside_soloing_repair_support(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp), outside_soloing_repair_supported=False)
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=812,
                )

    def test_rejects_missing_outside_soloing_source_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            data["outside_soloing_repair_objective"]["schema_version"] = "legacy_schema"
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=1150,
                )

        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            data["outside_soloing_repair_objective"]["readiness"][
                "source_schema_version"
            ] = "mismatched_schema"
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=1150,
                )

        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            data["outside_soloing_repair_objective"]["objective_summary"][
                "source_outside_soloing_residual_risk_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=898,
                )

        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            del data["outside_soloing_repair_objective"]["objective_summary"][
                "followup_objective_source_outside_soloing_source_pitch_role_risk_delta"
            ]
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=1066,
                )

        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            data["outside_soloing_repair_objective"]["objective_summary"][
                "repair_sweep_source_outside_soloing_source_context_preserved"
            ] = False
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=1066,
                )

        with tempfile.TemporaryDirectory() as tmp:
            data = reports(Path(tmp))
            data["outside_soloing_repair_objective"]["objective_summary"][
                "source_outside_soloing_pitch_role_risk_count_after"
            ] = 6
            with self.assertRaises(StageBMidiToSoloMvpCurrentEvidenceConsolidationError):
                build_current_evidence_consolidation_report(
                    contract_report=data["contract"],
                    context_report=data["context"],
                    resource_probe=data["resource"],
                    generation_probe=data["generation"],
                    audio_render=data["audio"],
                    objective_next=data["objective"],
                    cli_objective_next=data["cli_objective"],
                    model_conditioned_pitch_contour_objective_next=data[
                        "model_conditioned_pitch_contour_objective"
                    ],
                    model_conditioned_pitch_contour_changed_ratio_repair_objective_next=data[
                        "model_conditioned_pitch_contour_changed_ratio_repair_objective"
                    ],
                    outside_soloing_repair_objective_next=data[
                        "outside_soloing_repair_objective"
                    ],
                    output_dir=Path(tmp) / "out",
                    issue_number=898,
                )

    def test_boundary_constants_are_stable(self) -> None:
        self.assertEqual(BOUNDARY, "stage_b_midi_to_solo_mvp_current_evidence_consolidation")
        self.assertEqual(NEXT_BOUNDARY, "stage_b_midi_to_solo_readme_evidence_refresh")
        self.assertEqual(SCHEMA_VERSION, "stage_b_midi_to_solo_mvp_current_evidence_consolidation_v4")


if __name__ == "__main__":
    unittest.main()
