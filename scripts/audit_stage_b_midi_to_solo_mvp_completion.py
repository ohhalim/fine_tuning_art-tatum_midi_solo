"""Audit the Stage B MIDI-to-solo MVP completion boundary."""

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
from scripts.consolidate_stage_b_midi_to_solo_mvp_current_evidence import (  # noqa: E402
    BOUNDARY as CURRENT_EVIDENCE_BOUNDARY,
)


class StageBMidiToSoloMvpCompletionAuditError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_mvp_completion_audit"
NEXT_BOUNDARY = "stage_b_midi_to_solo_quality_gap_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_mvp_completion_audit_v1"


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


def validate_current_evidence(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != CURRENT_EVIDENCE_BOUNDARY:
        raise StageBMidiToSoloMvpCompletionAuditError("current evidence boundary required")
    readiness = _dict(report.get("readiness"))
    generation = _dict(report.get("ranked_midi_generation"))
    audio = _dict(report.get("technical_audio_render"))
    objective = _dict(report.get("selected_scale_objective_path"))
    cli_objective = _dict(report.get("phrase_bank_cli_technical_path"))
    model_conditioned_pitch_contour = _dict(
        report.get("model_conditioned_pitch_contour_objective_path")
    )
    model_conditioned_pitch_contour_changed_ratio_repair = _dict(
        report.get("model_conditioned_pitch_contour_changed_ratio_repair_objective_path")
    )
    outside_soloing_repair = _dict(report.get("outside_soloing_repair_objective_path"))
    decision = _dict(report.get("decision"))
    required_true = [
        "mvp_current_evidence_consolidated",
        "input_contract_ready",
        "context_extraction_ready",
        "training_resource_ready",
        "ranked_midi_candidates_exported",
        "technical_wav_path_ready",
        "selected_scale_objective_path_complete",
        "phrase_bank_cli_technical_path_ready",
        "model_conditioned_pitch_contour_objective_path_ready",
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready",
        "outside_soloing_repair_objective_path_ready",
        "current_mvp_technical_execution_evidence_supported",
        "current_mvp_objective_repair_evidence_supported",
        "midi_to_solo_mvp_current_evidence_supported",
    ]
    missing = [name for name in required_true if not bool(readiness.get(name, False))]
    if missing:
        raise StageBMidiToSoloMvpCompletionAuditError(f"missing current evidence readiness: {missing}")
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloMvpCompletionAuditError(f"unexpected quality claim: {claimed}")
    if _int(generation.get("exported_candidate_count")) < 3:
        raise StageBMidiToSoloMvpCompletionAuditError("ranked MIDI export count below 3")
    if _int(audio.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCompletionAuditError("rendered WAV count below 3")
    if not bool(audio.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpCompletionAuditError("technical WAV validation required")
    if not bool(objective.get("objective_path_supported", False)):
        raise StageBMidiToSoloMvpCompletionAuditError("selected-scale objective path support required")
    if _int(objective.get("strict_valid_sample_count")) != _int(objective.get("sample_count")):
        raise StageBMidiToSoloMvpCompletionAuditError("objective strict valid count must match sample count")
    if _int(objective.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloMvpCompletionAuditError("objective dead-air failure count must be zero")
    if not bool(cli_objective.get("technical_midi_to_solo_cli_path_ready", False)):
        raise StageBMidiToSoloMvpCompletionAuditError("CLI technical path support required")
    if _int(cli_objective.get("candidate_count")) < 3:
        raise StageBMidiToSoloMvpCompletionAuditError("CLI candidate count below 3")
    if _int(cli_objective.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCompletionAuditError("CLI rendered WAV count below 3")
    if bool(cli_objective.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloMvpCompletionAuditError("CLI preference fill should remain blocked")
    if not bool(model_conditioned_pitch_contour):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour objective path required"
        )
    if not bool(model_conditioned_pitch_contour.get("current_evidence_consolidation_ready", False)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour current evidence readiness required"
        )
    if not bool(model_conditioned_pitch_contour.get("pitch_contour_target_supported", False)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour target support required"
        )
    if _int(model_conditioned_pitch_contour.get("max_repaired_interval")) > _int(
        model_conditioned_pitch_contour.get("max_interval_threshold")
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour interval threshold exceeded"
        )
    if _int(model_conditioned_pitch_contour.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour rendered WAV count below 3"
        )
    if not bool(model_conditioned_pitch_contour.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour technical WAV validation required"
        )
    if bool(model_conditioned_pitch_contour.get("validated_review_input_present", True)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour review input should remain absent"
        )
    if bool(model_conditioned_pitch_contour.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour preference fill should remain blocked"
        )
    if not bool(model_conditioned_pitch_contour_changed_ratio_repair):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair objective path required"
        )
    if not bool(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "current_evidence_consolidation_ready",
            False,
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair current evidence readiness required"
        )
    if not bool(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "changed_ratio_repair_objective_path_supported",
            False,
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair support required"
        )
    if _int(
        model_conditioned_pitch_contour_changed_ratio_repair.get("max_repaired_interval")
    ) > _int(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "max_interval_threshold"
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair interval threshold exceeded"
        )
    target_ratio = _float(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "target_max_pitch_changed_ratio"
        )
    )
    if target_ratio and _float(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "max_repaired_pitch_changed_ratio"
        )
    ) > target_ratio:
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair ratio threshold exceeded"
        )
    if _int(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "rendered_audio_file_count"
        )
    ) < 3:
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair rendered WAV count below 3"
        )
    if not bool(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "technical_wav_validation",
            False,
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair technical WAV validation required"
        )
    if bool(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "validated_review_input_present",
            True,
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair review input should remain absent"
        )
    if bool(
        model_conditioned_pitch_contour_changed_ratio_repair.get(
            "preference_fill_allowed",
            True,
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair preference fill should remain blocked"
        )
    if not bool(outside_soloing_repair):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair objective path required"
        )
    if not bool(outside_soloing_repair.get("current_evidence_consolidation_ready", False)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair current evidence readiness required"
        )
    if not bool(outside_soloing_repair.get("outside_soloing_repair_objective_path_supported", False)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair objective path support required"
        )
    if _int(outside_soloing_repair.get("rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair rendered WAV count below 6"
        )
    if not bool(outside_soloing_repair.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair technical WAV validation required"
        )
    if _int(outside_soloing_repair.get("outside_soloing_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair residual pitch-role risk should be zero"
        )
    if _int(outside_soloing_repair.get("weak_chord_tone_landing_risk_count_after")) != 0:
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair weak landing risk should be zero"
        )
    required_outside_targets = [
        "outside_soloing_target_supported",
        "weak_landing_target_supported",
        "final_landing_target_supported",
        "non_chord_run_target_supported",
    ]
    missing_outside_targets = [
        name for name in required_outside_targets if not bool(outside_soloing_repair.get(name, False))
    ]
    if missing_outside_targets:
        raise StageBMidiToSoloMvpCompletionAuditError(
            f"outside-soloing repair targets missing: {missing_outside_targets}"
        )
    if bool(outside_soloing_repair.get("validated_review_input_present", True)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair review input should remain absent"
        )
    if bool(outside_soloing_repair.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair preference fill should remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpCompletionAuditError("critical user input should not be required")
    return {
        "current_evidence_boundary": CURRENT_EVIDENCE_BOUNDARY,
        "current_evidence_supported": True,
        "technical_execution_evidence_supported": True,
        "selected_scale_objective_path_complete": True,
        "generation_source": str(generation.get("generation_source") or ""),
        "exported_candidate_count": _int(generation.get("exported_candidate_count")),
        "exported_qualified_candidate_count": _int(generation.get("exported_qualified_candidate_count")),
        "rendered_audio_file_count": _int(audio.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(audio.get("technical_wav_validation", False)),
        "wav_duration_min_seconds": _float(audio.get("duration_min_seconds")),
        "wav_duration_max_seconds": _float(audio.get("duration_max_seconds")),
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
        "phrase_bank_cli_technical_path_ready": bool(
            cli_objective.get("technical_midi_to_solo_cli_path_ready", False)
        ),
        "cli_candidate_count": _int(cli_objective.get("candidate_count")),
        "cli_rendered_audio_file_count": _int(cli_objective.get("rendered_audio_file_count")),
        "cli_input_context_bars": _int(cli_objective.get("input_context_bars")),
        "cli_preference_fill_allowed": bool(cli_objective.get("preference_fill_allowed", True)),
        "model_conditioned_pitch_contour_objective_path_ready": bool(
            readiness.get("model_conditioned_pitch_contour_objective_path_ready", False)
        ),
        "model_conditioned_pitch_contour_current_evidence_ready": bool(
            model_conditioned_pitch_contour.get("current_evidence_consolidation_ready", False)
        ),
        "model_conditioned_pitch_contour_rendered_audio_file_count": _int(
            model_conditioned_pitch_contour.get("rendered_audio_file_count")
        ),
        "model_conditioned_pitch_contour_technical_wav_validation": bool(
            model_conditioned_pitch_contour.get("technical_wav_validation", False)
        ),
        "model_conditioned_pitch_contour_max_interval": _int(
            model_conditioned_pitch_contour.get("max_repaired_interval")
        ),
        "model_conditioned_pitch_contour_max_interval_threshold": _int(
            model_conditioned_pitch_contour.get("max_interval_threshold")
        ),
        "model_conditioned_pitch_contour_target_supported": bool(
            model_conditioned_pitch_contour.get("pitch_contour_target_supported", False)
        ),
        "model_conditioned_pitch_contour_max_pitch_changed_ratio": _float(
            model_conditioned_pitch_contour.get("max_pitch_changed_ratio")
        ),
        "model_conditioned_pitch_contour_pitch_changed_ratio_review_required": bool(
            model_conditioned_pitch_contour.get("pitch_changed_ratio_review_required", False)
        ),
        "model_conditioned_pitch_contour_audio_review_required": bool(
            model_conditioned_pitch_contour.get("audio_review_required", False)
        ),
        "model_conditioned_pitch_contour_validated_review_input_present": bool(
            model_conditioned_pitch_contour.get("validated_review_input_present", True)
        ),
        "model_conditioned_pitch_contour_preference_fill_allowed": bool(
            model_conditioned_pitch_contour.get("preference_fill_allowed", True)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready": bool(
            readiness.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_current_evidence_ready": bool(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "current_evidence_consolidation_ready",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count": _int(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "rendered_audio_file_count"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_technical_wav_validation": bool(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "technical_wav_validation",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval": _int(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "max_repaired_interval"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold": _int(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "max_interval_threshold"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio": _float(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "max_repaired_pitch_changed_ratio"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio": target_ratio,
        "model_conditioned_pitch_contour_changed_ratio_repair_target_supported": bool(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "changed_ratio_target_supported",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required": bool(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "audio_review_required",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_preference_fill_allowed": bool(
            model_conditioned_pitch_contour_changed_ratio_repair.get(
                "preference_fill_allowed",
                True,
            )
        ),
        "outside_soloing_repair_objective_path_ready": bool(
            readiness.get("outside_soloing_repair_objective_path_ready", False)
        ),
        "outside_soloing_repair_current_evidence_ready": bool(
            outside_soloing_repair.get("current_evidence_consolidation_ready", False)
        ),
        "outside_soloing_repair_rendered_audio_file_count": _int(
            outside_soloing_repair.get("rendered_audio_file_count")
        ),
        "outside_soloing_repair_changed_note_total": _int(
            outside_soloing_repair.get("changed_note_total")
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            outside_soloing_repair.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            outside_soloing_repair.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_objective_path_supported": bool(
            outside_soloing_repair.get(
                "outside_soloing_repair_objective_path_supported",
                False,
            )
        ),
        "outside_soloing_repair_target_supported": bool(
            outside_soloing_repair.get("outside_soloing_target_supported", False)
        ),
        "outside_soloing_repair_weak_landing_target_supported": bool(
            outside_soloing_repair.get("weak_landing_target_supported", False)
        ),
        "outside_soloing_repair_final_landing_target_supported": bool(
            outside_soloing_repair.get("final_landing_target_supported", False)
        ),
        "outside_soloing_repair_non_chord_run_target_supported": bool(
            outside_soloing_repair.get("non_chord_run_target_supported", False)
        ),
        "outside_soloing_repair_preference_fill_allowed": bool(
            outside_soloing_repair.get("preference_fill_allowed", True)
        ),
    }


def validate_readme_refresh(readme_text: str) -> dict[str, Any]:
    required_snippets = {
        "current_evidence_boundary": f"current evidence boundary: `{CURRENT_EVIDENCE_BOUNDARY}`",
        "current_evidence_support": "current MVP evidence support: `true`",
        "technical_path": "input MIDI -> context -> ranked MIDI -> WAV technical path: `true`",
        "objective_path": "selected-scale objective repair path complete: `true`",
        "cli_path": "phrase-bank CLI technical path included in current evidence: `true`",
        "model_conditioned_pitch_contour": (
            "model-conditioned pitch-contour objective path ready: `true`"
        ),
        "model_conditioned_pitch_contour_ratio": (
            "model-conditioned pitch-contour changed-ratio review required: `true`"
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair": (
            "model-conditioned pitch-contour changed-ratio repair objective path ready: `true`"
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_in_current_evidence": (
            "current evidence changed-ratio repair objective path included: `true`"
        ),
        "outside_soloing_repair": (
            "outside-soloing repair objective path included in current evidence: `true`"
        ),
        "outside_soloing_repair_in_current_evidence": (
            "current evidence outside-soloing repair objective path included: `true`"
        ),
        "readme_refresh": "README evidence refreshed: `true`",
        "human_preference_false": "human/audio preference claim: `false`",
        "musical_quality_false": "MIDI-to-solo musical quality claim: `false`",
        "broad_quality_false": "broad trained-model quality claim: `false`",
        "brad_false": "Brad style adaptation claim: `false`",
    }
    missing = [name for name, snippet in required_snippets.items() if snippet not in readme_text]
    if missing:
        raise StageBMidiToSoloMvpCompletionAuditError(f"README evidence snippets missing: {missing}")
    return {
        "readme_current_evidence_refreshed": True,
        "required_snippet_count": len(required_snippets),
        "missing_snippet_count": 0,
    }


def build_mvp_completion_audit_report(
    *,
    current_evidence: dict[str, Any],
    readme_text: str,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    evidence = validate_current_evidence(current_evidence)
    readme = validate_readme_refresh(readme_text)
    technical_model_core_mvp_completed = bool(
        evidence["current_evidence_supported"]
        and evidence["technical_execution_evidence_supported"]
        and evidence["selected_scale_objective_path_complete"]
        and evidence["phrase_bank_cli_technical_path_ready"]
        and evidence["model_conditioned_pitch_contour_objective_path_ready"]
        and evidence[
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready"
        ]
        and evidence["outside_soloing_repair_objective_path_ready"]
        and readme["readme_current_evidence_refreshed"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": CURRENT_EVIDENCE_BOUNDARY,
        "current_evidence": evidence,
        "readme_refresh": readme,
        "completion_audit": {
            "technical_model_core_mvp_completed": technical_model_core_mvp_completed,
            "input_to_ranked_midi_completed": True,
            "input_to_rendered_wav_completed": True,
            "selected_scale_objective_repair_completed": True,
            "phrase_bank_cli_technical_path_completed": True,
            "model_conditioned_pitch_contour_objective_completed": True,
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed": True,
            "outside_soloing_repair_objective_completed": True,
            "readme_evidence_boundary_refreshed": True,
            "musical_quality_mvp_completed": False,
            "human_audio_preference_completed": False,
            "broad_trained_model_completed": False,
            "brad_style_adaptation_completed": False,
            "product_mvp_completed": False,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "mvp_completion_audit_completed": True,
            "technical_model_core_mvp_completed": technical_model_core_mvp_completed,
            "model_conditioned_pitch_contour_objective_path_ready": bool(
                evidence["model_conditioned_pitch_contour_objective_path_ready"]
            ),
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready": bool(
                evidence[
                    "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready"
                ]
            ),
            "outside_soloing_repair_objective_path_ready": bool(
                evidence["outside_soloing_repair_objective_path_ready"]
            ),
            "quality_gap_decision_required": True,
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
                "technical model-core MVP completion is supported by current evidence and README refresh; "
                "musical quality, preference, and broad model claims remain excluded"
            ),
        },
        "proven": [
            "input_midi_to_context_path",
            "ranked_midi_candidate_export",
            "technical_wav_render_path",
            "selected_scale_objective_repair_path",
            "phrase_bank_cli_technical_path",
            "model_conditioned_pitch_contour_objective_path",
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_path",
            "outside_soloing_repair_objective_path",
            "readme_current_evidence_refresh",
        ],
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo quality gap decision",
    }


def validate_mvp_completion_audit_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_technical_mvp_completion: bool,
    require_no_quality_claim: bool,
    require_model_conditioned_pitch_contour_objective: bool,
    require_model_conditioned_pitch_contour_changed_ratio_repair_objective: bool = False,
    require_outside_soloing_repair_objective: bool = False,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    audit = _dict(report.get("completion_audit"))
    evidence = _dict(report.get("current_evidence"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloMvpCompletionAuditError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloMvpCompletionAuditError("unexpected next boundary")
    if require_technical_mvp_completion and not bool(audit.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloMvpCompletionAuditError("technical model-core MVP completion required")
    if require_model_conditioned_pitch_contour_objective and not bool(
        audit.get("model_conditioned_pitch_contour_objective_completed", False)
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour objective completion required"
        )
    if require_model_conditioned_pitch_contour_changed_ratio_repair_objective and not bool(
        audit.get(
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed",
            False,
        )
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "model-conditioned pitch-contour changed-ratio repair objective completion required"
        )
    if require_outside_soloing_repair_objective and not bool(
        audit.get("outside_soloing_repair_objective_completed", False)
    ):
        raise StageBMidiToSoloMvpCompletionAuditError(
            "outside-soloing repair objective completion required"
        )
    if not bool(readiness.get("quality_gap_decision_required", False)):
        raise StageBMidiToSoloMvpCompletionAuditError("quality gap decision should be required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpCompletionAuditError("critical user input should not be required")
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
            raise StageBMidiToSoloMvpCompletionAuditError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "technical_model_core_mvp_completed": bool(
            audit.get("technical_model_core_mvp_completed", False)
        ),
        "input_to_ranked_midi_completed": bool(audit.get("input_to_ranked_midi_completed", False)),
        "input_to_rendered_wav_completed": bool(audit.get("input_to_rendered_wav_completed", False)),
        "selected_scale_objective_repair_completed": bool(
            audit.get("selected_scale_objective_repair_completed", False)
        ),
        "phrase_bank_cli_technical_path_completed": bool(
            audit.get("phrase_bank_cli_technical_path_completed", False)
        ),
        "model_conditioned_pitch_contour_objective_completed": bool(
            audit.get("model_conditioned_pitch_contour_objective_completed", False)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed": bool(
            audit.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed",
                False,
            )
        ),
        "outside_soloing_repair_objective_completed": bool(
            audit.get("outside_soloing_repair_objective_completed", False)
        ),
        "musical_quality_mvp_completed": bool(audit.get("musical_quality_mvp_completed", True)),
        "human_audio_preference_completed": bool(audit.get("human_audio_preference_completed", True)),
        "product_mvp_completed": bool(audit.get("product_mvp_completed", True)),
        "generation_source": str(evidence.get("generation_source") or ""),
        "exported_candidate_count": _int(evidence.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(evidence.get("rendered_audio_file_count")),
        "objective_sample_count": _int(evidence.get("objective_sample_count")),
        "objective_strict_valid_sample_count": _int(
            evidence.get("objective_strict_valid_sample_count")
        ),
        "objective_dead_air_failure_count": _int(evidence.get("objective_dead_air_failure_count")),
        "phrase_bank_cli_technical_path_ready": bool(
            evidence.get("phrase_bank_cli_technical_path_ready", False)
        ),
        "cli_candidate_count": _int(evidence.get("cli_candidate_count")),
        "cli_rendered_audio_file_count": _int(evidence.get("cli_rendered_audio_file_count")),
        "cli_input_context_bars": _int(evidence.get("cli_input_context_bars")),
        "cli_preference_fill_allowed": bool(evidence.get("cli_preference_fill_allowed", True)),
        "model_conditioned_pitch_contour_objective_path_ready": bool(
            evidence.get("model_conditioned_pitch_contour_objective_path_ready", False)
        ),
        "model_conditioned_pitch_contour_max_interval": _int(
            evidence.get("model_conditioned_pitch_contour_max_interval")
        ),
        "model_conditioned_pitch_contour_max_interval_threshold": _int(
            evidence.get("model_conditioned_pitch_contour_max_interval_threshold")
        ),
        "model_conditioned_pitch_contour_target_supported": bool(
            evidence.get("model_conditioned_pitch_contour_target_supported", False)
        ),
        "model_conditioned_pitch_contour_pitch_changed_ratio_review_required": bool(
            evidence.get("model_conditioned_pitch_contour_pitch_changed_ratio_review_required", False)
        ),
        "model_conditioned_pitch_contour_audio_review_required": bool(
            evidence.get("model_conditioned_pitch_contour_audio_review_required", False)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready": bool(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count": _int(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval": _int(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_max_interval"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold": _int(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio": _float(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio": _float(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_target_supported": bool(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_target_supported",
                False,
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required": bool(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required",
                False,
            )
        ),
        "outside_soloing_repair_objective_path_ready": bool(
            evidence.get("outside_soloing_repair_objective_path_ready", False)
        ),
        "outside_soloing_repair_current_evidence_ready": bool(
            evidence.get("outside_soloing_repair_current_evidence_ready", False)
        ),
        "outside_soloing_repair_rendered_audio_file_count": _int(
            evidence.get("outside_soloing_repair_rendered_audio_file_count")
        ),
        "outside_soloing_repair_changed_note_total": _int(
            evidence.get("outside_soloing_repair_changed_note_total")
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            evidence.get("outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            evidence.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_objective_path_supported": bool(
            evidence.get("outside_soloing_repair_objective_path_supported", False)
        ),
        "outside_soloing_repair_target_supported": bool(
            evidence.get("outside_soloing_repair_target_supported", False)
        ),
        "outside_soloing_repair_weak_landing_target_supported": bool(
            evidence.get("outside_soloing_repair_weak_landing_target_supported", False)
        ),
        "outside_soloing_repair_final_landing_target_supported": bool(
            evidence.get("outside_soloing_repair_final_landing_target_supported", False)
        ),
        "outside_soloing_repair_non_chord_run_target_supported": bool(
            evidence.get("outside_soloing_repair_non_chord_run_target_supported", False)
        ),
        "outside_soloing_repair_preference_fill_allowed": bool(
            evidence.get("outside_soloing_repair_preference_fill_allowed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    audit = report["completion_audit"]
    evidence = report["current_evidence"]
    lines = [
        "# Stage B MIDI-to-Solo MVP Completion Audit",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- technical model-core MVP completed: `{_bool_token(audit['technical_model_core_mvp_completed'])}`",
        f"- phrase-bank CLI technical path completed: `{_bool_token(audit['phrase_bank_cli_technical_path_completed'])}`",
        f"- model-conditioned pitch-contour objective completed: `{_bool_token(audit['model_conditioned_pitch_contour_objective_completed'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair objective completed: `{_bool_token(audit['model_conditioned_pitch_contour_changed_ratio_repair_objective_completed'])}`",
        f"- outside-soloing repair objective completed: `{_bool_token(audit['outside_soloing_repair_objective_completed'])}`",
        f"- musical quality MVP completed: `{_bool_token(audit['musical_quality_mvp_completed'])}`",
        f"- human/audio preference completed: `{_bool_token(audit['human_audio_preference_completed'])}`",
        f"- product MVP completed: `{_bool_token(audit['product_mvp_completed'])}`",
        "",
        "## Evidence",
        "",
        f"- source boundary: `{report['source_boundary']}`",
        f"- generation source: `{evidence['generation_source']}`",
        f"- exported / qualified candidates: `{evidence['exported_candidate_count']}` / `{evidence['exported_qualified_candidate_count']}`",
        f"- rendered WAV files: `{evidence['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(evidence['technical_wav_validation'])}`",
        f"- WAV duration range: `{evidence['wav_duration_min_seconds']:.3f}s-{evidence['wav_duration_max_seconds']:.3f}s`",
        f"- objective sample / strict / grammar: `{evidence['objective_sample_count']}` / `{evidence['objective_strict_valid_sample_count']}` / `{evidence['objective_grammar_gate_sample_count']}`",
        f"- objective dead-air / collapse failure: `{evidence['objective_dead_air_failure_count']}` / `{evidence['objective_collapse_warning_sample_count']}`",
        f"- objective avg / target postprocess removal: `{evidence['objective_avg_postprocess_removal_ratio']}` / `{evidence['objective_target_avg_postprocess_removal_ratio']}`",
        f"- phrase-bank CLI technical path ready: `{_bool_token(evidence['phrase_bank_cli_technical_path_ready'])}`",
        f"- CLI candidate / rendered WAV: `{evidence['cli_candidate_count']}` / `{evidence['cli_rendered_audio_file_count']}`",
        f"- CLI input context bars: `{evidence['cli_input_context_bars']}`",
        f"- CLI preference fill allowed: `{_bool_token(evidence['cli_preference_fill_allowed'])}`",
        f"- model-conditioned pitch-contour objective path ready: `{_bool_token(evidence['model_conditioned_pitch_contour_objective_path_ready'])}`",
        f"- model-conditioned pitch-contour rendered WAV files: `{evidence['model_conditioned_pitch_contour_rendered_audio_file_count']}`",
        f"- model-conditioned pitch-contour technical WAV validation: `{_bool_token(evidence['model_conditioned_pitch_contour_technical_wav_validation'])}`",
        f"- model-conditioned pitch-contour max interval / threshold: `{evidence['model_conditioned_pitch_contour_max_interval']}` / `{evidence['model_conditioned_pitch_contour_max_interval_threshold']}`",
        f"- model-conditioned pitch-contour target supported: `{_bool_token(evidence['model_conditioned_pitch_contour_target_supported'])}`",
        f"- model-conditioned pitch-contour max pitch changed ratio: `{evidence['model_conditioned_pitch_contour_max_pitch_changed_ratio']:.4f}`",
        f"- model-conditioned pitch-contour changed-ratio review required: `{_bool_token(evidence['model_conditioned_pitch_contour_pitch_changed_ratio_review_required'])}`",
        f"- model-conditioned pitch-contour audio review required: `{_bool_token(evidence['model_conditioned_pitch_contour_audio_review_required'])}`",
        f"- model-conditioned pitch-contour preference fill allowed: `{_bool_token(evidence['model_conditioned_pitch_contour_preference_fill_allowed'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair objective path ready: `{_bool_token(evidence['model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair rendered WAV files: `{evidence['model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count']}`",
        f"- model-conditioned pitch-contour changed-ratio repair technical WAV validation: `{_bool_token(evidence['model_conditioned_pitch_contour_changed_ratio_repair_technical_wav_validation'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair max interval / threshold: `{evidence['model_conditioned_pitch_contour_changed_ratio_repair_max_interval']}` / `{evidence['model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold']}`",
        f"- model-conditioned pitch-contour changed-ratio repair max ratio / target: `{evidence['model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio']:.4f}` / `{evidence['model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio']:.4f}`",
        f"- model-conditioned pitch-contour changed-ratio repair target supported: `{_bool_token(evidence['model_conditioned_pitch_contour_changed_ratio_repair_target_supported'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair audio review required: `{_bool_token(evidence['model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair preference fill allowed: `{_bool_token(evidence['model_conditioned_pitch_contour_changed_ratio_repair_preference_fill_allowed'])}`",
        f"- outside-soloing repair objective path ready: `{_bool_token(evidence['outside_soloing_repair_objective_path_ready'])}`",
        f"- outside-soloing repair current evidence ready: `{_bool_token(evidence['outside_soloing_repair_current_evidence_ready'])}`",
        f"- outside-soloing repair rendered WAV files: `{evidence['outside_soloing_repair_rendered_audio_file_count']}`",
        f"- outside-soloing repair changed note total: `{evidence['outside_soloing_repair_changed_note_total']}`",
        f"- outside-soloing pitch-role risk after / delta: `{evidence['outside_soloing_repair_pitch_role_risk_count_after']}` / `{evidence['outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- outside-soloing repair objective path supported: `{_bool_token(evidence['outside_soloing_repair_objective_path_supported'])}`",
        f"- outside-soloing repair target supported: `{_bool_token(evidence['outside_soloing_repair_target_supported'])}`",
        f"- outside-soloing repair weak landing target supported: `{_bool_token(evidence['outside_soloing_repair_weak_landing_target_supported'])}`",
        f"- outside-soloing repair final landing target supported: `{_bool_token(evidence['outside_soloing_repair_final_landing_target_supported'])}`",
        f"- outside-soloing repair non-chord run target supported: `{_bool_token(evidence['outside_soloing_repair_non_chord_run_target_supported'])}`",
        f"- outside-soloing repair preference fill allowed: `{_bool_token(evidence['outside_soloing_repair_preference_fill_allowed'])}`",
        "",
        "## Claim Boundary",
        "",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        f"- production ready claimed: `{_bool_token(readiness['production_ready_claimed'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in _list(report.get("not_proven")):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next", "", f"- `{report['next_recommended_issue']}`"])
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit MIDI-to-solo MVP completion boundary")
    parser.add_argument("--current_evidence", type=str, required=True)
    parser.add_argument("--readme_path", type=str, default="README.md")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_mvp_completion_audit",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=616)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_technical_mvp_completion", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    parser.add_argument("--require_model_conditioned_pitch_contour_objective", action="store_true")
    parser.add_argument(
        "--require_model_conditioned_pitch_contour_changed_ratio_repair_objective",
        action="store_true",
    )
    parser.add_argument("--require_outside_soloing_repair_objective", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    readme_path = Path(args.readme_path)
    if not readme_path.exists():
        raise StageBMidiToSoloMvpCompletionAuditError(f"README missing: {readme_path}")
    report = build_mvp_completion_audit_report(
        current_evidence=read_json(Path(args.current_evidence)),
        readme_text=readme_path.read_text(encoding="utf-8"),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_mvp_completion_audit_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_technical_mvp_completion=bool(args.require_technical_mvp_completion),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        require_model_conditioned_pitch_contour_objective=bool(
            args.require_model_conditioned_pitch_contour_objective
        ),
        require_model_conditioned_pitch_contour_changed_ratio_repair_objective=bool(
            args.require_model_conditioned_pitch_contour_changed_ratio_repair_objective
        ),
        require_outside_soloing_repair_objective=bool(
            args.require_outside_soloing_repair_objective
        ),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_completion_audit.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_completion_audit_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_mvp_completion_audit.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
