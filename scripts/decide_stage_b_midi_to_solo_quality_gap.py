"""Decide the next quality-gap target after the MIDI-to-solo MVP completion audit."""

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
from scripts.audit_stage_b_midi_to_solo_mvp_completion import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    BOUNDARY as MVP_COMPLETION_AUDIT_BOUNDARY,
)


class StageBMidiToSoloQualityGapDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_quality_gap_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment"
SELECTED_TARGET = "model_conditioned_input_path_quality_alignment"
PITCH_CONTOUR_CHANGED_RATIO_TARGET = "model_conditioned_pitch_contour_changed_ratio_review"
PITCH_CONTOUR_CHANGED_RATIO_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision"
)
LISTENING_REVIEW_TARGET = "listening_review_quality_gap"
LISTENING_REVIEW_NEXT_BOUNDARY = "stage_b_midi_to_solo_listening_review_quality_gap"
SCHEMA_VERSION = "stage_b_midi_to_solo_quality_gap_decision_v3"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloQualityGapDecisionError(
                f"{label} source-context field required: {key}"
            )
    missing_preserved = [
        key for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS if not bool(container.get(key))
    ]
    if missing_preserved:
        raise StageBMidiToSoloQualityGapDecisionError(
            f"{label} source-context preserved field must be true: {missing_preserved}"
        )
    return {key: container[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS}


def validate_mvp_completion_audit(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != MVP_COMPLETION_AUDIT_BOUNDARY:
        raise StageBMidiToSoloQualityGapDecisionError("MVP completion audit boundary required")
    readiness = _dict(report.get("readiness"))
    audit = _dict(report.get("completion_audit"))
    evidence = _dict(report.get("current_evidence"))
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloQualityGapDecisionError("MVP audit should route to quality gap decision")
    if not bool(readiness.get("mvp_completion_audit_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("MVP completion audit completion required")
    if not bool(audit.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("technical model-core MVP completion required")
    if bool(audit.get("musical_quality_mvp_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("musical quality should remain incomplete")
    if bool(audit.get("human_audio_preference_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("human/audio preference should remain incomplete")
    if bool(audit.get("product_mvp_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("product MVP should remain incomplete")
    if not bool(audit.get("phrase_bank_cli_technical_path_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("phrase-bank CLI technical path completion required")
    if not bool(audit.get("model_conditioned_pitch_contour_objective_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "model-conditioned pitch-contour objective completion required"
        )
    if not bool(audit.get("outside_soloing_repair_objective_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair objective completion required"
        )
    if not bool(readiness.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair source context readiness required"
        )
    if not bool(audit.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair source context audit preservation required"
        )
    if _int(evidence.get("exported_candidate_count")) < 3:
        raise StageBMidiToSoloQualityGapDecisionError("exported candidate count below 3")
    if _int(evidence.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloQualityGapDecisionError("rendered WAV count below 3")
    if not bool(evidence.get("phrase_bank_cli_technical_path_ready", False)):
        raise StageBMidiToSoloQualityGapDecisionError("phrase-bank CLI technical path readiness required")
    if _int(evidence.get("cli_candidate_count")) < 3:
        raise StageBMidiToSoloQualityGapDecisionError("CLI candidate count below 3")
    if _int(evidence.get("cli_rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloQualityGapDecisionError("CLI rendered WAV count below 3")
    if bool(evidence.get("cli_preference_fill_allowed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("CLI preference fill should remain blocked")
    if not bool(evidence.get("model_conditioned_pitch_contour_objective_path_ready", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "model-conditioned pitch-contour objective path readiness required"
        )
    if _int(evidence.get("model_conditioned_pitch_contour_max_interval")) > _int(
        evidence.get("model_conditioned_pitch_contour_max_interval_threshold")
    ):
        raise StageBMidiToSoloQualityGapDecisionError(
            "model-conditioned pitch-contour interval threshold exceeded"
        )
    if not bool(evidence.get("model_conditioned_pitch_contour_target_supported", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "model-conditioned pitch-contour target support required"
        )
    if bool(audit.get("model_conditioned_pitch_contour_changed_ratio_repair_objective_completed", False)):
        if not bool(
            evidence.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready",
                False,
            )
        ):
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair path readiness required"
            )
        if _int(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count")
        ) < 3:
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair rendered WAV count below 3"
            )
        if not bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_technical_wav_validation")
        ):
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair technical WAV validation required"
            )
        if _int(evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval")) > _int(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold")
        ):
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair interval threshold exceeded"
            )
        if _float(evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio")) > _float(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio")
        ):
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair ratio threshold exceeded"
            )
        if not bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_target_supported")
        ):
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair target support required"
            )
        if bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_preference_fill_allowed", True)
        ):
            raise StageBMidiToSoloQualityGapDecisionError(
                "model-conditioned pitch-contour changed-ratio repair preference fill should remain blocked"
            )
    if not bool(evidence.get("outside_soloing_repair_objective_path_ready", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair objective path readiness required"
        )
    if not bool(evidence.get("outside_soloing_repair_current_evidence_ready", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair current evidence readiness required"
        )
    if not bool(evidence.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair source context evidence preservation required"
        )
    source_context = _source_context_fields(
        evidence,
        label="MVP completion audit evidence",
    )
    if _int(evidence.get("outside_soloing_repair_rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair rendered WAV count below 6"
        )
    if _int(evidence.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair residual pitch-role risk should be zero"
        )
    source_objective_risk = _int(
        evidence.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        evidence.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(
        evidence.get("outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_risk_delta = _int(
        evidence.get("outside_soloing_repair_source_pitch_role_risk_delta")
    )
    if source_objective_risk <= 0:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(evidence.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(evidence.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source residual risk preservation required"
        )
    outside_targets = [
        "outside_soloing_repair_objective_path_supported",
        "outside_soloing_repair_target_supported",
        "outside_soloing_repair_weak_landing_target_supported",
        "outside_soloing_repair_final_landing_target_supported",
        "outside_soloing_repair_non_chord_run_target_supported",
    ]
    missing_outside_targets = [
        name for name in outside_targets if not bool(evidence.get(name, False))
    ]
    if missing_outside_targets:
        raise StageBMidiToSoloQualityGapDecisionError(
            f"outside-soloing repair targets missing: {missing_outside_targets}"
        )
    if bool(evidence.get("outside_soloing_repair_preference_fill_allowed", True)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair preference fill should remain blocked"
        )
    if _int(evidence.get("objective_strict_valid_sample_count")) != _int(evidence.get("objective_sample_count")):
        raise StageBMidiToSoloQualityGapDecisionError("objective strict valid count must match sample count")
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloQualityGapDecisionError(f"unexpected quality claim: {claimed}")
    return {
        "technical_model_core_mvp_completed": True,
        "phrase_bank_cli_technical_path_completed": True,
        "model_conditioned_pitch_contour_objective_completed": True,
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed": bool(
            audit.get("model_conditioned_pitch_contour_changed_ratio_repair_objective_completed", False)
        ),
        "outside_soloing_repair_objective_completed": bool(
            audit.get("outside_soloing_repair_objective_completed", False)
        ),
        "outside_soloing_repair_source_context_preserved": bool(
            evidence.get("outside_soloing_repair_source_context_preserved", False)
        ),
        "musical_quality_mvp_completed": False,
        "human_audio_preference_completed": False,
        "product_mvp_completed": False,
        "generation_source": str(evidence.get("generation_source") or ""),
        "exported_candidate_count": _int(evidence.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(evidence.get("rendered_audio_file_count")),
        "objective_sample_count": _int(evidence.get("objective_sample_count")),
        "objective_strict_valid_sample_count": _int(evidence.get("objective_strict_valid_sample_count")),
        "objective_dead_air_failure_count": _int(evidence.get("objective_dead_air_failure_count")),
        "objective_avg_postprocess_removal_ratio": _float(
            evidence.get("objective_avg_postprocess_removal_ratio")
        ),
        "objective_target_avg_postprocess_removal_ratio": _float(
            evidence.get("objective_target_avg_postprocess_removal_ratio")
        ),
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
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count")
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_technical_wav_validation": bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_technical_wav_validation", False)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval": _int(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval")
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold": _int(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold")
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio": _float(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio")
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio": _float(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio")
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_target_supported": bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_target_supported", False)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required": bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required", False)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_preference_fill_allowed": bool(
            evidence.get("model_conditioned_pitch_contour_changed_ratio_repair_preference_fill_allowed", True)
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
        "outside_soloing_repair_source_objective_pitch_role_risk_count": source_objective_risk,
        "outside_soloing_repair_source_pitch_role_risk_count_before": source_risk_before,
        "outside_soloing_repair_source_pitch_role_risk_count_after": source_risk_after,
        "outside_soloing_repair_source_pitch_role_risk_delta": source_risk_delta,
        "outside_soloing_repair_source_targeted": bool(
            evidence.get("outside_soloing_repair_source_targeted", True)
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            evidence.get("outside_soloing_repair_source_residual_risk_preserved", False)
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
        **source_context,
    }


def select_quality_gap_target(audit_summary: dict[str, Any]) -> dict[str, Any]:
    generation_source = str(audit_summary.get("generation_source") or "")
    fallback_path_active = generation_source == "context_conditioned_fallback"
    pitch_contour_path_ready = bool(
        audit_summary.get("model_conditioned_pitch_contour_objective_path_ready", False)
    )
    pitch_contour_changed_ratio_review_required = bool(
        audit_summary.get("model_conditioned_pitch_contour_pitch_changed_ratio_review_required", False)
    )
    changed_ratio_repair_ready = bool(
        audit_summary.get(
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready",
            False,
        )
    )
    changed_ratio_repair_supported = bool(
        audit_summary.get("model_conditioned_pitch_contour_changed_ratio_repair_target_supported", False)
    )
    outside_soloing_repair_ready = bool(
        audit_summary.get("outside_soloing_repair_objective_path_ready", False)
    )
    outside_soloing_repair_supported = bool(
        audit_summary.get("outside_soloing_repair_target_supported", False)
    )
    if (
        pitch_contour_path_ready
        and pitch_contour_changed_ratio_review_required
        and not changed_ratio_repair_supported
    ):
        target = PITCH_CONTOUR_CHANGED_RATIO_TARGET
        next_boundary = PITCH_CONTOUR_CHANGED_RATIO_NEXT_BOUNDARY
        reason = (
            "model-conditioned pitch-contour path is objective-complete, but pitch changed ratio still requires "
            "review before any musical quality claim"
        )
    elif (
        changed_ratio_repair_ready
        and changed_ratio_repair_supported
        and outside_soloing_repair_ready
        and outside_soloing_repair_supported
    ):
        target = LISTENING_REVIEW_TARGET
        next_boundary = LISTENING_REVIEW_NEXT_BOUNDARY
        reason = (
            "changed-ratio and outside-soloing repair objective paths meet current targets; remaining gap is "
            "listening review and musical quality evidence, not another objective repair loop"
        )
    elif fallback_path_active:
        target = SELECTED_TARGET
        next_boundary = NEXT_BOUNDARY
        reason = (
            "input-to-WAV path still uses context_conditioned_fallback while selected-scale and CLI paths are "
            "technical evidence, not model-conditioned quality"
        )
    else:
        target = LISTENING_REVIEW_TARGET
        next_boundary = LISTENING_REVIEW_NEXT_BOUNDARY
        reason = "model-conditioned path is available; listening review gap remains"
    return {
        "selected_target": target,
        "selected_next_boundary": next_boundary,
        "fallback_path_active": fallback_path_active,
        "model_conditioned_pitch_contour_objective_path_ready": pitch_contour_path_ready,
        "pitch_contour_changed_ratio_review_required": pitch_contour_changed_ratio_review_required,
        "pitch_contour_changed_ratio_repair_objective_path_ready": changed_ratio_repair_ready,
        "pitch_contour_changed_ratio_repair_target_supported": changed_ratio_repair_supported,
        "outside_soloing_repair_objective_path_ready": outside_soloing_repair_ready,
        "outside_soloing_repair_target_supported": outside_soloing_repair_supported,
        "quality_gap_reason": reason,
        "human_review_required_now": False,
    }


def build_quality_gap_decision_report(
    *,
    mvp_completion_audit: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    audit_summary = validate_mvp_completion_audit(mvp_completion_audit)
    target = select_quality_gap_target(audit_summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": MVP_COMPLETION_AUDIT_BOUNDARY,
        "mvp_completion_summary": audit_summary,
        "quality_gap": {
            "technical_model_core_mvp_completed": True,
            "phrase_bank_cli_technical_path_completed": True,
            "model_conditioned_pitch_contour_objective_completed": bool(
                audit_summary["model_conditioned_pitch_contour_objective_completed"]
            ),
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed": bool(
                audit_summary["model_conditioned_pitch_contour_changed_ratio_repair_objective_completed"]
            ),
            "outside_soloing_repair_objective_completed": bool(
                audit_summary["outside_soloing_repair_objective_completed"]
            ),
            "outside_soloing_repair_source_context_preserved": bool(
                audit_summary["outside_soloing_repair_source_context_preserved"]
            ),
            "musical_quality_mvp_completed": False,
            "human_audio_preference_completed": False,
            "product_mvp_completed": False,
            "fallback_path_active": bool(target["fallback_path_active"]),
            "model_conditioned_input_path_alignment_required": bool(
                target["fallback_path_active"]
                and str(target["selected_target"]) == SELECTED_TARGET
            ),
            "model_conditioned_pitch_contour_objective_path_ready": bool(
                target["model_conditioned_pitch_contour_objective_path_ready"]
            ),
            "pitch_contour_changed_ratio_review_required": bool(
                target["pitch_contour_changed_ratio_review_required"]
            ),
            "pitch_contour_changed_ratio_repair_objective_path_ready": bool(
                target["pitch_contour_changed_ratio_repair_objective_path_ready"]
            ),
            "pitch_contour_changed_ratio_repair_target_supported": bool(
                target["pitch_contour_changed_ratio_repair_target_supported"]
            ),
            "outside_soloing_repair_objective_path_ready": bool(
                target["outside_soloing_repair_objective_path_ready"]
            ),
            "outside_soloing_repair_target_supported": bool(
                target["outside_soloing_repair_target_supported"]
            ),
            "outside_soloing_repair_source_objective_pitch_role_risk_count": int(
                audit_summary["outside_soloing_repair_source_objective_pitch_role_risk_count"]
            ),
            "outside_soloing_repair_source_pitch_role_risk_count_before": int(
                audit_summary["outside_soloing_repair_source_pitch_role_risk_count_before"]
            ),
            "outside_soloing_repair_source_pitch_role_risk_count_after": int(
                audit_summary["outside_soloing_repair_source_pitch_role_risk_count_after"]
            ),
            "outside_soloing_repair_source_pitch_role_risk_delta": int(
                audit_summary["outside_soloing_repair_source_pitch_role_risk_delta"]
            ),
            "outside_soloing_repair_source_targeted": bool(
                audit_summary["outside_soloing_repair_source_targeted"]
            ),
            "outside_soloing_repair_source_residual_risk_preserved": bool(
                audit_summary["outside_soloing_repair_source_residual_risk_preserved"]
            ),
            **{key: audit_summary[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
            "human_review_required_now": False,
        },
        "selected_target": target,
        "readiness": {
            "boundary": BOUNDARY,
            "quality_gap_decision_completed": True,
            "selected_target": str(target["selected_target"]),
            "next_boundary_selected": str(target["selected_next_boundary"]),
            "outside_soloing_repair_source_context_preserved": bool(
                audit_summary["outside_soloing_repair_source_context_preserved"]
            ),
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": str(target["selected_next_boundary"]),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": str(target["quality_gap_reason"]),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio review decision"
            if str(target["selected_target"]) == PITCH_CONTOUR_CHANGED_RATIO_TARGET
            else (
                "Stage B MIDI-to-solo listening review quality gap source-context refresh"
                if str(target["selected_target"]) == LISTENING_REVIEW_TARGET
                else "Stage B MIDI-to-solo model-conditioned input path quality alignment"
            )
        ),
    }


def validate_quality_gap_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    quality_gap = _dict(report.get("quality_gap"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloQualityGapDecisionError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloQualityGapDecisionError("unexpected next boundary")
    if expected_target and str(readiness.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloQualityGapDecisionError("unexpected selected target")
    if not bool(readiness.get("quality_gap_decision_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("quality gap decision completion required")
    if not bool(quality_gap.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("technical model-core MVP completion required")
    if not bool(quality_gap.get("model_conditioned_pitch_contour_objective_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "model-conditioned pitch-contour objective completion required"
        )
    if not bool(quality_gap.get("outside_soloing_repair_objective_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair objective completion required"
        )
    if not bool(readiness.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair source context readiness required"
        )
    if not bool(quality_gap.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing repair source context preservation required"
        )
    source_context = _source_context_fields(
        quality_gap,
        label="quality gap",
    )
    if bool(quality_gap.get("musical_quality_mvp_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("musical quality should remain incomplete")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloQualityGapDecisionError("critical user input should not be required")
    source_objective_risk = _int(
        quality_gap.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        quality_gap.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(
        quality_gap.get("outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_risk_delta = _int(
        quality_gap.get("outside_soloing_repair_source_pitch_role_risk_delta")
    )
    if source_objective_risk <= 0:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(quality_gap.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(quality_gap.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloQualityGapDecisionError(
            "outside-soloing source residual risk preservation required"
        )
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
            raise StageBMidiToSoloQualityGapDecisionError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(readiness.get("selected_target") or ""),
        "fallback_path_active": bool(quality_gap.get("fallback_path_active", False)),
        "model_conditioned_input_path_alignment_required": bool(
            quality_gap.get("model_conditioned_input_path_alignment_required", False)
        ),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "technical_model_core_mvp_completed": bool(
            quality_gap.get("technical_model_core_mvp_completed", False)
        ),
        "phrase_bank_cli_technical_path_completed": bool(
            quality_gap.get("phrase_bank_cli_technical_path_completed", False)
        ),
        "model_conditioned_pitch_contour_objective_completed": bool(
            quality_gap.get("model_conditioned_pitch_contour_objective_completed", False)
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed": bool(
            quality_gap.get(
                "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed",
                False,
            )
        ),
        "outside_soloing_repair_objective_completed": bool(
            quality_gap.get("outside_soloing_repair_objective_completed", False)
        ),
        "outside_soloing_repair_source_context_preserved": bool(
            quality_gap.get("outside_soloing_repair_source_context_preserved", False)
        ),
        "model_conditioned_pitch_contour_objective_path_ready": bool(
            quality_gap.get("model_conditioned_pitch_contour_objective_path_ready", False)
        ),
        "pitch_contour_changed_ratio_review_required": bool(
            quality_gap.get("pitch_contour_changed_ratio_review_required", False)
        ),
        "pitch_contour_changed_ratio_repair_objective_path_ready": bool(
            quality_gap.get("pitch_contour_changed_ratio_repair_objective_path_ready", False)
        ),
        "pitch_contour_changed_ratio_repair_target_supported": bool(
            quality_gap.get("pitch_contour_changed_ratio_repair_target_supported", False)
        ),
        "outside_soloing_repair_objective_path_ready": bool(
            quality_gap.get("outside_soloing_repair_objective_path_ready", False)
        ),
        "outside_soloing_repair_target_supported": bool(
            quality_gap.get("outside_soloing_repair_target_supported", False)
        ),
        "outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source_objective_risk
        ),
        "outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source_risk_before
        ),
        "outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source_risk_after
        ),
        "outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source_risk_delta
        ),
        "outside_soloing_repair_source_targeted": bool(
            quality_gap.get("outside_soloing_repair_source_targeted", True)
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            quality_gap.get("outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        **source_context,
        "musical_quality_mvp_completed": bool(quality_gap.get("musical_quality_mvp_completed", True)),
        "cli_candidate_count": _int(_dict(report.get("mvp_completion_summary")).get("cli_candidate_count")),
        "cli_rendered_audio_file_count": _int(
            _dict(report.get("mvp_completion_summary")).get("cli_rendered_audio_file_count")
        ),
        "cli_input_context_bars": _int(_dict(report.get("mvp_completion_summary")).get("cli_input_context_bars")),
        "cli_preference_fill_allowed": bool(
            _dict(report.get("mvp_completion_summary")).get("cli_preference_fill_allowed", True)
        ),
        "model_conditioned_pitch_contour_max_interval": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_max_interval"
            )
        ),
        "model_conditioned_pitch_contour_max_interval_threshold": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_max_interval_threshold"
            )
        ),
        "model_conditioned_pitch_contour_target_supported": bool(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_target_supported", False
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_changed_ratio_repair_max_interval"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio": _float(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio"
            )
        ),
        "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio": _float(
            _dict(report.get("mvp_completion_summary")).get(
                "model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio"
            )
        ),
        "outside_soloing_repair_rendered_audio_file_count": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_rendered_audio_file_count"
            )
        ),
        "outside_soloing_repair_changed_note_total": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_changed_note_total"
            )
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_pitch_role_risk_count_after"
            )
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_pitch_role_risk_delta"
            )
        ),
        "outside_soloing_repair_objective_path_supported": bool(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_objective_path_supported",
                False,
            )
        ),
        "outside_soloing_repair_weak_landing_target_supported": bool(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_weak_landing_target_supported",
                False,
            )
        ),
        "outside_soloing_repair_final_landing_target_supported": bool(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_final_landing_target_supported",
                False,
            )
        ),
        "outside_soloing_repair_non_chord_run_target_supported": bool(
            _dict(report.get("mvp_completion_summary")).get(
                "outside_soloing_repair_non_chord_run_target_supported",
                False,
            )
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["mvp_completion_summary"]
    quality_gap = report["quality_gap"]
    selected = report["selected_target"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Quality Gap Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- fallback path active: `{_bool_token(quality_gap['fallback_path_active'])}`",
        f"- pitch-contour changed-ratio review required: `{_bool_token(quality_gap['pitch_contour_changed_ratio_review_required'])}`",
        f"- human review required now: `{_bool_token(quality_gap['human_review_required_now'])}`",
        "",
        "## Evidence",
        "",
        f"- technical model-core MVP completed: `{_bool_token(summary['technical_model_core_mvp_completed'])}`",
        f"- phrase-bank CLI technical path completed: `{_bool_token(summary['phrase_bank_cli_technical_path_completed'])}`",
        f"- model-conditioned pitch-contour objective completed: `{_bool_token(summary['model_conditioned_pitch_contour_objective_completed'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair objective completed: `{_bool_token(summary['model_conditioned_pitch_contour_changed_ratio_repair_objective_completed'])}`",
        f"- outside-soloing repair objective completed: `{_bool_token(summary['outside_soloing_repair_objective_completed'])}`",
        f"- outside-soloing repair source context preserved: `{_bool_token(summary['outside_soloing_repair_source_context_preserved'])}`",
        f"- musical quality MVP completed: `{_bool_token(summary['musical_quality_mvp_completed'])}`",
        f"- generation source: `{summary['generation_source']}`",
        f"- exported candidates: `{summary['exported_candidate_count']}`",
        f"- rendered WAV files: `{summary['rendered_audio_file_count']}`",
        f"- objective strict/sample: `{summary['objective_strict_valid_sample_count']}` / `{summary['objective_sample_count']}`",
        f"- objective dead-air failure: `{summary['objective_dead_air_failure_count']}`",
        f"- CLI candidate / rendered WAV: `{summary['cli_candidate_count']}` / `{summary['cli_rendered_audio_file_count']}`",
        f"- CLI input context bars: `{summary['cli_input_context_bars']}`",
        f"- CLI preference fill allowed: `{_bool_token(summary['cli_preference_fill_allowed'])}`",
        f"- model-conditioned pitch-contour max interval / threshold: `{summary['model_conditioned_pitch_contour_max_interval']}` / `{summary['model_conditioned_pitch_contour_max_interval_threshold']}`",
        f"- model-conditioned pitch-contour target supported: `{_bool_token(summary['model_conditioned_pitch_contour_target_supported'])}`",
        f"- model-conditioned pitch-contour changed-ratio review required: `{_bool_token(summary['model_conditioned_pitch_contour_pitch_changed_ratio_review_required'])}`",
        f"- model-conditioned pitch-contour audio review required: `{_bool_token(summary['model_conditioned_pitch_contour_audio_review_required'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair objective path ready: `{_bool_token(summary['model_conditioned_pitch_contour_changed_ratio_repair_objective_path_ready'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair rendered WAV files: `{summary['model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count']}`",
        f"- model-conditioned pitch-contour changed-ratio repair technical WAV validation: `{_bool_token(summary['model_conditioned_pitch_contour_changed_ratio_repair_technical_wav_validation'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair max interval / threshold: `{summary['model_conditioned_pitch_contour_changed_ratio_repair_max_interval']}` / `{summary['model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold']}`",
        f"- model-conditioned pitch-contour changed-ratio repair max ratio / target: `{summary['model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio']:.4f}` / `{summary['model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio']:.4f}`",
        f"- model-conditioned pitch-contour changed-ratio repair target supported: `{_bool_token(summary['model_conditioned_pitch_contour_changed_ratio_repair_target_supported'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair audio review required: `{_bool_token(summary['model_conditioned_pitch_contour_changed_ratio_repair_audio_review_required'])}`",
        f"- model-conditioned pitch-contour changed-ratio repair preference fill allowed: `{_bool_token(summary['model_conditioned_pitch_contour_changed_ratio_repair_preference_fill_allowed'])}`",
        f"- outside-soloing repair objective path ready: `{_bool_token(summary['outside_soloing_repair_objective_path_ready'])}`",
        f"- outside-soloing repair rendered WAV files: `{summary['outside_soloing_repair_rendered_audio_file_count']}`",
        f"- outside-soloing repair changed note total: `{summary['outside_soloing_repair_changed_note_total']}`",
        f"- outside-soloing source objective pitch-role risk: `{summary['outside_soloing_repair_source_objective_pitch_role_risk_count']}`",
        f"- outside-soloing source pitch-role risk before / after / delta: `{summary['outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{summary['outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{summary['outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- outside-soloing source repair targeted: `{_bool_token(summary['outside_soloing_repair_source_targeted'])}`",
        f"- outside-soloing source residual risk preserved: `{_bool_token(summary['outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- outside-soloing current repair pitch-role risk after / delta: `{summary['outside_soloing_repair_pitch_role_risk_count_after']}` / `{summary['outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- outside-soloing repair objective path supported: `{_bool_token(summary['outside_soloing_repair_objective_path_supported'])}`",
        f"- outside-soloing repair target supported: `{_bool_token(summary['outside_soloing_repair_target_supported'])}`",
        f"- outside-soloing repair weak landing target supported: `{_bool_token(summary['outside_soloing_repair_weak_landing_target_supported'])}`",
        f"- outside-soloing repair final landing target supported: `{_bool_token(summary['outside_soloing_repair_final_landing_target_supported'])}`",
        f"- outside-soloing repair non-chord run target supported: `{_bool_token(summary['outside_soloing_repair_non_chord_run_target_supported'])}`",
        "",
        "## Claim Boundary",
        "",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Next",
        "",
        f"- `{report['next_recommended_issue']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide next MIDI-to-solo quality-gap target")
    parser.add_argument("--mvp_completion_audit", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_quality_gap_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=618)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_quality_gap_decision_report(
        mvp_completion_audit=read_json(Path(args.mvp_completion_audit)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_quality_gap_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_quality_gap_decision.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_quality_gap_decision_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_quality_gap_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
