"""Consolidate the current Stage B MIDI-to-solo MVP evidence boundary."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402


class StageBMidiToSoloMvpCurrentEvidenceConsolidationError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_mvp_current_evidence_consolidation"
NEXT_BOUNDARY = "stage_b_midi_to_solo_readme_evidence_refresh"
SCHEMA_VERSION = "stage_b_midi_to_solo_mvp_current_evidence_consolidation_v1"

CONTRACT_BOUNDARY = "stage_b_midi_to_solo_mvp_input_contract"
CONTEXT_BOUNDARY = "stage_b_midi_to_solo_context_extraction_mvp"
RESOURCE_BOUNDARY = "stage_b_midi_to_solo_training_resource_probe"
GENERATION_BOUNDARY = "stage_b_midi_to_solo_conditioned_generation_probe"
AUDIO_BOUNDARY = "stage_b_midi_to_solo_candidate_audio_render_package"
OBJECTIVE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_objective_only_next_decision"
)
OBJECTIVE_FINAL_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_objective_path_complete"
)
CLI_OBJECTIVE_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_cli_objective_only_next_decision"
MODEL_CONDITIONED_PITCH_CONTOUR_OBJECTIVE_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_objective_only_next_decision"
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _artifact_exists(path: str) -> bool:
    return bool(path and Path(path).exists())


def _duration_range(items: list[dict[str, Any]]) -> dict[str, float]:
    durations = [
        _float(_dict(_dict(item).get("wav_file")).get("duration_seconds"))
        for item in items
    ]
    durations = [item for item in durations if item > 0]
    return {
        "min_seconds": min(durations) if durations else 0.0,
        "max_seconds": max(durations) if durations else 0.0,
    }


def validate_contract(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != CONTRACT_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "MIDI-to-solo MVP input contract boundary required"
        )
    output = _dict(report.get("output_contract"))
    gate = _dict(report.get("objective_gate"))
    stack = _dict(report.get("generation_stack"))
    decision = _dict(report.get("decision"))
    if _int(output.get("candidate_count")) < 32:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract candidate count below 32")
    if _int(output.get("export_top_midi_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract export count below 3")
    if _int(output.get("target_solo_bars")) < 8:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract target solo bars below 8")
    if _int(gate.get("min_note_count")) < 24:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract min note count below 24")
    if _int(gate.get("min_unique_pitch_count")) < 8:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract min unique pitch count below 8")
    if _int(gate.get("max_simultaneous_notes")) != 1:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract must require monophonic output")
    if not str(stack.get("fallback_path") or ""):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract fallback path required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("contract should not require critical user input")
    return {
        "boundary": CONTRACT_BOUNDARY,
        "candidate_count": _int(output.get("candidate_count")),
        "export_top_midi_count": _int(output.get("export_top_midi_count")),
        "target_solo_bars": _int(output.get("target_solo_bars")),
        "min_note_count": _int(gate.get("min_note_count")),
        "min_unique_pitch_count": _int(gate.get("min_unique_pitch_count")),
        "max_dead_air_ratio": _float(gate.get("max_dead_air_ratio")),
        "max_long_note_ratio": _float(gate.get("max_long_note_ratio")),
        "max_simultaneous_notes": _int(gate.get("max_simultaneous_notes")),
        "min_phrase_coverage_ratio": _float(gate.get("min_phrase_coverage_ratio")),
        "primary_path": str(stack.get("primary_path") or ""),
        "fallback_path": str(stack.get("fallback_path") or ""),
    }


def validate_context(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != CONTEXT_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("context extraction boundary required")
    summary = _dict(report.get("summary"))
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("context_extraction_completed", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("context extraction completion required")
    if not bool(readiness.get("required_context_fields_present", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("required context fields missing")
    if _int(summary.get("context_bars")) < 4:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("context bars below minimum")
    if _int(summary.get("context_event_count")) <= 0:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("context events required")
    blocked_claims = [
        "midi_to_solo_mvp_claimed",
        "harmony_analysis_quality_claimed",
        "brad_style_fine_tuning_completed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected context claim: {claimed}"
        )
    return {
        "boundary": CONTEXT_BOUNDARY,
        "context_bars": _int(summary.get("context_bars")),
        "positions_per_bar": _int(summary.get("positions_per_bar")),
        "context_event_count": _int(summary.get("context_event_count")),
        "inferred_chord_bar_count": _int(summary.get("inferred_chord_bar_count")),
        "carry_forward_chord_bar_count": _int(summary.get("carry_forward_chord_bar_count")),
        "unknown_chord_bar_count": _int(summary.get("unknown_chord_bar_count")),
        "low_confidence_bar_count": _int(summary.get("low_confidence_bar_count")),
        "bass_note_bar_count": _int(summary.get("bass_note_bar_count")),
    }


def validate_resource(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != RESOURCE_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("training resource boundary required")
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("training_resource_probe_completed", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("training resource probe completion required")
    if not bool(readiness.get("midi_to_solo_training_resource_ready", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("MIDI-to-solo training resource readiness required")
    if not bool(readiness.get("conditioned_generation_probe_ready", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("conditioned generation readiness required")
    blocked_claims = [
        "midi_to_solo_mvp_claimed",
        "broad_training_executed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "musical_quality_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected resource claim: {claimed}"
        )
    return {
        "boundary": RESOURCE_BOUNDARY,
        "midi_to_solo_training_resource_ready": True,
        "conditioned_generation_probe_ready": True,
        "broad_training_executed": False,
    }


def validate_generation(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != GENERATION_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("conditioned generation boundary required")
    summary = _dict(report.get("summary"))
    readiness = _dict(report.get("readiness"))
    config = _dict(report.get("generation_config"))
    candidates = [_dict(item) for item in _list(report.get("top_candidates")) if isinstance(item, dict)]
    midi_paths = [str(item.get("export_midi_path") or "") for item in candidates]
    if not bool(readiness.get("conditioned_generation_probe_completed", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("conditioned generation completion required")
    if not bool(readiness.get("ranked_midi_candidates_exported", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("ranked MIDI candidate export required")
    if _int(summary.get("exported_candidate_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("exported candidate count below 3")
    if _int(summary.get("exported_qualified_candidate_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("exported qualified candidate count below 3")
    if len(midi_paths) < 3 or not all(_artifact_exists(path) for path in midi_paths[:3]):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("top ranked MIDI artifacts required")
    blocked_claims = [
        "midi_to_solo_mvp_claimed",
        "model_checkpoint_generation_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "human_audio_preference_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected generation claim: {claimed}"
        )
    return {
        "boundary": GENERATION_BOUNDARY,
        "generation_source": str(config.get("generation_source") or ""),
        "model_checkpoint_generation_used": bool(config.get("model_checkpoint_generation_used", False)),
        "checkpoint_direct_generation_skip_reason": str(
            config.get("checkpoint_direct_generation_skip_reason") or ""
        ),
        "candidate_count": _int(summary.get("candidate_count")),
        "qualified_candidate_count": _int(summary.get("qualified_candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "exported_qualified_candidate_count": _int(summary.get("exported_qualified_candidate_count")),
        "best_score": _float(summary.get("best_score")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "best_max_simultaneous_notes": _int(summary.get("best_max_simultaneous_notes")),
        "best_chord_tone_ratio": _float(summary.get("best_chord_tone_ratio")),
        "midi_paths": midi_paths[:3],
    }


def validate_audio(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    if str(boundary.get("boundary") or "") != AUDIO_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("candidate audio render boundary required")
    rendered = [_dict(item) for item in _list(report.get("rendered_audio_files")) if isinstance(item, dict)]
    wav_paths = [str(_dict(item.get("wav_file")).get("path") or "") for item in rendered]
    if _int(boundary.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("rendered WAV count below 3")
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("technical WAV validation required")
    if len(wav_paths) < 3 or not all(_artifact_exists(path) for path in wav_paths[:3]):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("rendered WAV artifacts required")
    blocked_claims = [
        "audio_rendered_quality_claimed",
        "human_audio_preference_claimed",
        "musical_quality_claimed",
        "midi_to_solo_mvp_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(boundary.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected audio claim: {claimed}"
        )
    durations = _duration_range(rendered)
    return {
        "boundary": AUDIO_BOUNDARY,
        "render_attempted": bool(boundary.get("render_attempted", False)),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "sample_rate": _int(_dict(_dict(rendered[0]).get("wav_file")).get("sample_rate")) if rendered else 0,
        "duration_min_seconds": durations["min_seconds"],
        "duration_max_seconds": durations["max_seconds"],
        "wav_paths": wav_paths[:3],
    }


def validate_objective_next(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != OBJECTIVE_NEXT_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective-only next boundary required")
    if str(report.get("final_boundary") or "") != OBJECTIVE_FINAL_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective final boundary required")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("postprocess_removal_dead_air_repair_summary"))
    review = _dict(report.get("review_boundary_summary"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective next boundary should route to current evidence")
    if not bool(readiness.get("objective_only_decision_completed", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective-only decision completion required")
    if not bool(readiness.get("objective_postprocess_removal_dead_air_repair_path_supported", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective repair path support required")
    sample_count = _int(repair.get("sample_count"))
    if sample_count <= 0:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective sample count required")
    if _int(repair.get("strict_valid_sample_count")) != sample_count:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective strict valid count must match sample count")
    if _int(repair.get("grammar_gate_sample_count")) != sample_count:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective grammar count must match sample count")
    if _int(repair.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective dead-air failures should be zero")
    if _int(repair.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective collapse warnings should be zero")
    target_ratio = _float(repair.get("target_avg_postprocess_removal_ratio"))
    if target_ratio and _float(repair.get("avg_postprocess_removal_ratio")) > target_ratio:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("postprocess removal average above target")
    if _int(review.get("candidate_count")) < 3 or _int(review.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective review package candidate/render count below 3")
    if bool(review.get("validated_review_input_present", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("validated review input should remain absent")
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected objective claim: {claimed}"
        )
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "final_boundary": OBJECTIVE_FINAL_BOUNDARY,
        "sample_count": sample_count,
        "seed_count": _int(repair.get("seed_count")),
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "dead_air_failure_count": _int(repair.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(repair.get("collapse_warning_sample_count")),
        "strict_valid_sample_delta": _int(repair.get("strict_valid_sample_delta")),
        "dead_air_failure_delta": _int(repair.get("dead_air_failure_delta")),
        "temperature": _float(repair.get("temperature")),
        "top_k": _int(repair.get("top_k")),
        "avoid_reused_positions": bool(repair.get("avoid_reused_positions", False)),
        "avg_postprocess_removal_ratio": _float(repair.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(repair.get("max_postprocess_removal_ratio")),
        "target_avg_postprocess_removal_ratio": target_ratio,
        "postprocess_removal_delta": _float(repair.get("postprocess_removal_delta")),
        "avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(repair.get("avg_sustained_coverage_ratio")),
        "candidate_count": _int(review.get("candidate_count")),
        "rendered_audio_file_count": _int(review.get("rendered_audio_file_count")),
        "review_input_template_written": bool(review.get("review_input_template_written", False)),
        "validated_review_input_present": bool(review.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(review.get("preference_fill_allowed", True)),
        "pending_status_field_count": _int(review.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(review.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(review.get("pending_candidate_field_count")),
        "objective_path_supported": True,
    }


def validate_cli_objective_next(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != CLI_OBJECTIVE_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI objective boundary required")
    summary = _dict(report.get("objective_summary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "CLI objective next boundary should route to current evidence"
        )
    if not bool(readiness.get("cli_objective_only_next_decision_completed", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI objective decision completion required")
    if not bool(summary.get("technical_midi_to_solo_cli_path_ready", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI technical path readiness required")
    if not bool(summary.get("mvp_current_evidence_consolidation_ready", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "CLI current evidence readiness required"
        )
    if not bool(summary.get("explicit_input_used", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI explicit input evidence required")
    if _int(summary.get("candidate_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI candidate count below 3")
    if _int(summary.get("objective_supported_candidate_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI objective support below 3")
    if _int(summary.get("repaired_midi_file_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI repaired MIDI count below 3")
    if _int(summary.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI rendered audio count below 3")
    if not bool(summary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI technical WAV validation required")
    if bool(summary.get("validated_review_input_present", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "CLI validated review input should remain absent"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI preference fill should remain blocked")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI critical user input should not be required")
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "audio_rendered_quality_claimed",
        "phrase_bank_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected CLI objective claim: {claimed}"
        )
    dead_air = _list(summary.get("dead_air_range"))
    return {
        "boundary": CLI_OBJECTIVE_BOUNDARY,
        "technical_midi_to_solo_cli_path_ready": bool(
            summary.get("technical_midi_to_solo_cli_path_ready", False)
        ),
        "mvp_current_evidence_consolidation_ready": bool(
            summary.get("mvp_current_evidence_consolidation_ready", False)
        ),
        "explicit_input_used": bool(summary.get("explicit_input_used", False)),
        "candidate_count": _int(summary.get("candidate_count")),
        "objective_supported_candidate_count": _int(summary.get("objective_supported_candidate_count")),
        "repaired_midi_file_count": _int(summary.get("repaired_midi_file_count")),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "input_context_bars": _int(summary.get("input_context_bars")),
        "min_dead_air_ratio": _float(dead_air[0]) if len(dead_air) >= 1 else 0.0,
        "max_dead_air_ratio": _float(dead_air[1]) if len(dead_air) >= 2 else 0.0,
        "validated_review_input_present": bool(summary.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
    }


def validate_model_conditioned_pitch_contour_objective_next(
    report: dict[str, Any],
) -> dict[str, Any]:
    if str(report.get("boundary") or "") != MODEL_CONDITIONED_PITCH_CONTOUR_OBJECTIVE_BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour objective boundary required"
        )
    summary = _dict(report.get("objective_summary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour objective should route to current evidence"
        )
    if not bool(readiness.get("objective_next_decision_completed", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour objective completion required"
        )
    if not bool(summary.get("pitch_contour_target_supported", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour target support required"
        )
    if not bool(summary.get("current_evidence_consolidation_ready", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour current evidence readiness required"
        )
    if not bool(summary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour technical WAV validation required"
        )
    if _int(summary.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour rendered WAV count below 3"
        )
    if bool(summary.get("validated_review_input_present", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour review input should remain absent"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour preference fill should remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour critical user input should not be required"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "audio_rendered_quality_claimed",
        "model_checkpoint_generation_quality_claimed",
        "model_direct_generation_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"unexpected model-conditioned pitch-contour claim: {claimed}"
        )
    return {
        "boundary": MODEL_CONDITIONED_PITCH_CONTOUR_OBJECTIVE_BOUNDARY,
        "objective_next_decision_completed": bool(
            readiness.get("objective_next_decision_completed", False)
        ),
        "current_evidence_consolidation_ready": bool(
            summary.get("current_evidence_consolidation_ready", False)
        ),
        "review_item_count": _int(summary.get("review_item_count")),
        "validated_review_input_present": bool(
            summary.get("validated_review_input_present", True)
        ),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
        "max_interval_threshold": _int(summary.get("max_interval_threshold")),
        "pitch_contour_target_supported": bool(summary.get("pitch_contour_target_supported", False)),
        "max_pitch_changed_ratio": _float(summary.get("max_pitch_changed_ratio")),
        "pitch_changed_ratio_review_required": bool(
            summary.get("pitch_changed_ratio_review_required", False)
        ),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
    }


def build_current_evidence_consolidation_report(
    *,
    contract_report: dict[str, Any],
    context_report: dict[str, Any],
    resource_probe: dict[str, Any],
    generation_probe: dict[str, Any],
    audio_render: dict[str, Any],
    objective_next: dict[str, Any],
    cli_objective_next: dict[str, Any],
    model_conditioned_pitch_contour_objective_next: dict[str, Any] | None = None,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    contract = validate_contract(contract_report)
    context = validate_context(context_report)
    resource = validate_resource(resource_probe)
    generation = validate_generation(generation_probe)
    audio = validate_audio(audio_render)
    objective = validate_objective_next(objective_next)
    cli_objective = validate_cli_objective_next(cli_objective_next)
    model_conditioned_pitch_contour_objective = (
        validate_model_conditioned_pitch_contour_objective_next(
            model_conditioned_pitch_contour_objective_next
        )
        if model_conditioned_pitch_contour_objective_next
        else {}
    )
    technical_path_supported = (
        bool(resource["midi_to_solo_training_resource_ready"])
        and generation["exported_candidate_count"] >= 3
        and generation["exported_qualified_candidate_count"] >= 3
        and audio["rendered_audio_file_count"] >= 3
        and bool(audio["technical_wav_validation"])
    )
    selected_scale_objective_path_complete = bool(objective["objective_path_supported"])
    phrase_bank_cli_technical_path_ready = bool(cli_objective["technical_midi_to_solo_cli_path_ready"])
    model_conditioned_pitch_contour_objective_path_ready = bool(
        model_conditioned_pitch_contour_objective.get(
            "current_evidence_consolidation_ready",
            not bool(model_conditioned_pitch_contour_objective_next),
        )
    )
    current_evidence_supported = bool(
        technical_path_supported
        and selected_scale_objective_path_complete
        and phrase_bank_cli_technical_path_ready
        and model_conditioned_pitch_contour_objective_path_ready
    )
    source_boundaries = {
        "contract": contract["boundary"],
        "context": context["boundary"],
        "resource": resource["boundary"],
        "generation": generation["boundary"],
        "audio": audio["boundary"],
        "objective_next": objective["boundary"],
        "objective_final": objective["final_boundary"],
        "phrase_bank_cli_objective_next": cli_objective["boundary"],
    }
    if model_conditioned_pitch_contour_objective:
        source_boundaries["model_conditioned_pitch_contour_objective_next"] = (
            model_conditioned_pitch_contour_objective["boundary"]
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": source_boundaries,
        "mvp_contract": contract,
        "context_extraction": context,
        "training_resource": resource,
        "ranked_midi_generation": generation,
        "technical_audio_render": audio,
        "selected_scale_objective_path": objective,
        "phrase_bank_cli_technical_path": cli_objective,
        "model_conditioned_pitch_contour_objective_path": model_conditioned_pitch_contour_objective,
        "readiness": {
            "boundary": BOUNDARY,
            "mvp_current_evidence_consolidated": True,
            "input_contract_ready": True,
            "context_extraction_ready": True,
            "training_resource_ready": True,
            "ranked_midi_candidates_exported": True,
            "technical_wav_path_ready": True,
            "selected_scale_objective_path_complete": selected_scale_objective_path_complete,
            "phrase_bank_cli_technical_path_ready": phrase_bank_cli_technical_path_ready,
            "model_conditioned_pitch_contour_objective_path_ready": bool(
                model_conditioned_pitch_contour_objective_path_ready
            ),
            "current_mvp_technical_execution_evidence_supported": technical_path_supported,
            "current_mvp_objective_repair_evidence_supported": selected_scale_objective_path_complete,
            "midi_to_solo_mvp_current_evidence_supported": current_evidence_supported,
            "validated_review_input_present": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "input contract, context extraction, ranked MIDI export, technical WAV render, "
                "and selected-scale objective repair evidence are consolidated; README evidence can be refreshed "
                "without musical quality or human preference claim"
            ),
        },
        "proven": [
            "input_contract_defined",
            "context_rows_extracted",
            "ranked_midi_candidates_exported",
            "technical_wav_render_path_validated",
            "selected_scale_objective_dead_air_repair_path_complete",
            "explicit_input_cli_ranked_midi_wav_path_validated",
            *(
                ["model_conditioned_pitch_contour_objective_path_ready"]
                if model_conditioned_pitch_contour_objective
                else []
            ),
            "quality_claim_guard_preserved",
        ],
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo README evidence refresh",
    }


def validate_current_evidence_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_current_evidence_support: bool,
    require_no_quality_claim: bool,
    min_exported_candidates: int,
    min_rendered_wav_files: int,
    min_objective_sample_count: int,
    require_model_conditioned_pitch_contour_objective: bool = False,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("ranked_midi_generation"))
    audio = _dict(report.get("technical_audio_render"))
    objective = _dict(report.get("selected_scale_objective_path"))
    cli_objective = _dict(report.get("phrase_bank_cli_technical_path"))
    model_conditioned_pitch_contour = _dict(
        report.get("model_conditioned_pitch_contour_objective_path")
    )
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("unexpected next boundary")
    if require_current_evidence_support and not bool(
        readiness.get("midi_to_solo_mvp_current_evidence_supported", False)
    ):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("current MVP evidence support required")
    if _int(generation.get("exported_candidate_count")) < int(min_exported_candidates):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("exported candidate count below threshold")
    if _int(audio.get("rendered_audio_file_count")) < int(min_rendered_wav_files):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("rendered WAV count below threshold")
    if _int(objective.get("sample_count")) < int(min_objective_sample_count):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective sample count below threshold")
    if not bool(objective.get("objective_path_supported", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("objective path support required")
    if not bool(cli_objective.get("technical_midi_to_solo_cli_path_ready", False)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI technical path support required")
    if bool(cli_objective.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("CLI preference fill should remain blocked")
    if require_model_conditioned_pitch_contour_objective and not bool(
        readiness.get("model_conditioned_pitch_contour_objective_path_ready", False)
    ):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour objective path support required"
        )
    if model_conditioned_pitch_contour and bool(
        model_conditioned_pitch_contour.get("preference_fill_allowed", True)
    ):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
            "model-conditioned pitch-contour preference fill should remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError("critical user input should not be required")
    for item in _list(generation.get("midi_paths")) + _list(audio.get("wav_paths")):
        if not _artifact_exists(str(item)):
            raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(f"artifact missing: {item}")
    if require_no_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloMvpCurrentEvidenceConsolidationError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "midi_to_solo_mvp_current_evidence_supported": bool(
            readiness.get("midi_to_solo_mvp_current_evidence_supported", False)
        ),
        "technical_execution_evidence_supported": bool(
            readiness.get("current_mvp_technical_execution_evidence_supported", False)
        ),
        "selected_scale_objective_path_complete": bool(
            readiness.get("selected_scale_objective_path_complete", False)
        ),
        "phrase_bank_cli_technical_path_ready": bool(
            readiness.get("phrase_bank_cli_technical_path_ready", False)
        ),
        "model_conditioned_pitch_contour_objective_path_ready": bool(
            readiness.get("model_conditioned_pitch_contour_objective_path_ready", False)
        ),
        "model_conditioned_pitch_contour_max_interval": _int(
            model_conditioned_pitch_contour.get("max_repaired_interval")
        ),
        "model_conditioned_pitch_contour_target_supported": bool(
            model_conditioned_pitch_contour.get("pitch_contour_target_supported", False)
        ),
        "model_conditioned_pitch_contour_pitch_changed_ratio_review_required": bool(
            model_conditioned_pitch_contour.get("pitch_changed_ratio_review_required", False)
        ),
        "model_conditioned_pitch_contour_audio_review_required": bool(
            model_conditioned_pitch_contour.get("audio_review_required", False)
        ),
        "cli_candidate_count": _int(cli_objective.get("candidate_count")),
        "cli_rendered_audio_file_count": _int(cli_objective.get("rendered_audio_file_count")),
        "cli_input_context_bars": _int(cli_objective.get("input_context_bars")),
        "cli_preference_fill_allowed": bool(cli_objective.get("preference_fill_allowed", True)),
        "generation_source": str(generation.get("generation_source") or ""),
        "exported_candidate_count": _int(generation.get("exported_candidate_count")),
        "exported_qualified_candidate_count": _int(generation.get("exported_qualified_candidate_count")),
        "rendered_audio_file_count": _int(audio.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(audio.get("technical_wav_validation", False)),
        "objective_sample_count": _int(objective.get("sample_count")),
        "objective_strict_valid_sample_count": _int(objective.get("strict_valid_sample_count")),
        "objective_grammar_gate_sample_count": _int(objective.get("grammar_gate_sample_count")),
        "objective_dead_air_failure_count": _int(objective.get("dead_air_failure_count")),
        "objective_collapse_warning_sample_count": _int(objective.get("collapse_warning_sample_count")),
        "objective_avg_postprocess_removal_ratio": _float(
            objective.get("avg_postprocess_removal_ratio")
        ),
        "objective_target_avg_postprocess_removal_ratio": _float(
            objective.get("target_avg_postprocess_removal_ratio")
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    contract = report["mvp_contract"]
    context = report["context_extraction"]
    generation = report["ranked_midi_generation"]
    audio = report["technical_audio_render"]
    objective = report["selected_scale_objective_path"]
    cli_objective = report["phrase_bank_cli_technical_path"]
    model_conditioned_pitch_contour = _dict(
        report.get("model_conditioned_pitch_contour_objective_path")
    )
    lines = [
        "# Stage B MIDI-to-Solo MVP Current Evidence Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- current MVP evidence supported: `{_bool_token(readiness['midi_to_solo_mvp_current_evidence_supported'])}`",
        f"- technical execution evidence supported: `{_bool_token(readiness['current_mvp_technical_execution_evidence_supported'])}`",
        f"- selected-scale objective path complete: `{_bool_token(readiness['selected_scale_objective_path_complete'])}`",
        f"- phrase-bank CLI technical path ready: `{_bool_token(readiness['phrase_bank_cli_technical_path_ready'])}`",
        f"- model-conditioned pitch-contour objective path ready: `{_bool_token(readiness['model_conditioned_pitch_contour_objective_path_ready'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Input Contract",
        "",
        f"- candidate count: `{contract['candidate_count']}`",
        f"- exported MIDI candidates: `{contract['export_top_midi_count']}`",
        f"- target solo bars: `{contract['target_solo_bars']}`",
        f"- min note count: `{contract['min_note_count']}`",
        f"- min unique pitch count: `{contract['min_unique_pitch_count']}`",
        f"- max simultaneous notes: `{contract['max_simultaneous_notes']}`",
        f"- fallback path: `{contract['fallback_path']}`",
        "",
        "## Context and Ranked MIDI",
        "",
        f"- context bars / events: `{context['context_bars']}` / `{context['context_event_count']}`",
        f"- low-confidence chord bars: `{context['low_confidence_bar_count']}`",
        f"- generation source: `{generation['generation_source']}`",
        f"- exported / qualified candidates: `{generation['exported_candidate_count']}` / `{generation['exported_qualified_candidate_count']}`",
        f"- best note / unique pitch / max simultaneous notes: `{generation['best_note_count']}` / `{generation['best_unique_pitch_count']}` / `{generation['best_max_simultaneous_notes']}`",
        "",
        "## Technical Audio Path",
        "",
        f"- rendered WAV files: `{audio['rendered_audio_file_count']}`",
        f"- sample rate: `{audio['sample_rate']}`",
        f"- duration range: `{audio['duration_min_seconds']:.3f}s-{audio['duration_max_seconds']:.3f}s`",
        f"- technical WAV validation: `{_bool_token(audio['technical_wav_validation'])}`",
        "",
        "## Selected-Scale Objective Path",
        "",
        f"- final boundary: `{objective['final_boundary']}`",
        f"- sample / seed count: `{objective['sample_count']}` / `{objective['seed_count']}`",
        f"- valid / strict / grammar: `{objective['valid_sample_count']}` / `{objective['strict_valid_sample_count']}` / `{objective['grammar_gate_sample_count']}`",
        f"- dead-air / collapse failure count: `{objective['dead_air_failure_count']}` / `{objective['collapse_warning_sample_count']}`",
        f"- avg / max postprocess removal ratio: `{objective['avg_postprocess_removal_ratio']}` / `{objective['max_postprocess_removal_ratio']}`",
        f"- target avg postprocess removal ratio: `{objective['target_avg_postprocess_removal_ratio']}`",
        f"- validated review input present: `{_bool_token(objective['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(objective['preference_fill_allowed'])}`",
        "",
        "## Phrase-Bank CLI Technical Path",
        "",
        f"- technical MIDI-to-solo CLI path ready: `{_bool_token(cli_objective['technical_midi_to_solo_cli_path_ready'])}`",
        f"- explicit input used: `{_bool_token(cli_objective['explicit_input_used'])}`",
        f"- candidate / objective supported: `{cli_objective['candidate_count']}` / `{cli_objective['objective_supported_candidate_count']}`",
        f"- repaired MIDI / rendered WAV: `{cli_objective['repaired_midi_file_count']}` / `{cli_objective['rendered_audio_file_count']}`",
        f"- input context bars: `{cli_objective['input_context_bars']}`",
        f"- dead-air range: `{cli_objective['min_dead_air_ratio']:.4f}-{cli_objective['max_dead_air_ratio']:.4f}`",
        f"- preference fill allowed: `{_bool_token(cli_objective['preference_fill_allowed'])}`",
        "",
    ]
    if model_conditioned_pitch_contour:
        lines.extend(
            [
                "## Model-Conditioned Pitch-Contour Objective Path",
                "",
                f"- current evidence consolidation ready: `{_bool_token(model_conditioned_pitch_contour['current_evidence_consolidation_ready'])}`",
                f"- rendered WAV files: `{model_conditioned_pitch_contour['rendered_audio_file_count']}`",
                f"- technical WAV validation: `{_bool_token(model_conditioned_pitch_contour['technical_wav_validation'])}`",
                f"- max interval / threshold: `{model_conditioned_pitch_contour['max_repaired_interval']}` / `{model_conditioned_pitch_contour['max_interval_threshold']}`",
                f"- pitch-contour target supported: `{_bool_token(model_conditioned_pitch_contour['pitch_contour_target_supported'])}`",
                f"- max pitch changed ratio: `{model_conditioned_pitch_contour['max_pitch_changed_ratio']:.4f}`",
                f"- pitch changed ratio review required: `{_bool_token(model_conditioned_pitch_contour['pitch_changed_ratio_review_required'])}`",
                f"- audio review required: `{_bool_token(model_conditioned_pitch_contour['audio_review_required'])}`",
                f"- preference fill allowed: `{_bool_token(model_conditioned_pitch_contour['preference_fill_allowed'])}`",
                "",
            ]
        )
    lines.extend(["## MIDI Paths", ""])
    for item in generation["midi_paths"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## WAV Paths", ""])
    for item in audio["wav_paths"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`"])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate current MIDI-to-solo MVP evidence")
    parser.add_argument("--contract_report", type=str, required=True)
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--resource_probe", type=str, required=True)
    parser.add_argument("--generation_probe", type=str, required=True)
    parser.add_argument("--audio_render", type=str, required=True)
    parser.add_argument("--objective_next", type=str, required=True)
    parser.add_argument("--cli_objective_next", type=str, required=True)
    parser.add_argument("--model_conditioned_pitch_contour_objective_next", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_mvp_current_evidence_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=612)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_current_evidence_support", action="store_true")
    parser.add_argument("--require_model_conditioned_pitch_contour_objective", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    parser.add_argument("--min_exported_candidates", type=int, default=3)
    parser.add_argument("--min_rendered_wav_files", type=int, default=3)
    parser.add_argument("--min_objective_sample_count", type=int, default=9)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_current_evidence_consolidation_report(
        contract_report=read_json(Path(args.contract_report)),
        context_report=read_json(Path(args.context_report)),
        resource_probe=read_json(Path(args.resource_probe)),
        generation_probe=read_json(Path(args.generation_probe)),
        audio_render=read_json(Path(args.audio_render)),
        objective_next=read_json(Path(args.objective_next)),
        cli_objective_next=read_json(Path(args.cli_objective_next)),
        model_conditioned_pitch_contour_objective_next=(
            read_json(Path(args.model_conditioned_pitch_contour_objective_next))
            if args.model_conditioned_pitch_contour_objective_next
            else None
        ),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_current_evidence_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_current_evidence_support=bool(args.require_current_evidence_support),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_exported_candidates=int(args.min_exported_candidates),
        min_rendered_wav_files=int(args.min_rendered_wav_files),
        min_objective_sample_count=int(args.min_objective_sample_count),
        require_model_conditioned_pitch_contour_objective=bool(
            args.require_model_conditioned_pitch_contour_objective
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_current_evidence_consolidation.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_mvp_current_evidence_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_mvp_current_evidence_consolidation.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
