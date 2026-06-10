"""Select the next boundary after outside-soloing repair input guard evidence."""

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
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_input import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (  # noqa: E402
    BRIDGE_SOURCE_CONTEXT_KEYS,
)


class StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_objective_only_next_decision"
)
CURRENT_EVIDENCE_NEXT_BOUNDARY = "stage_b_midi_to_solo_mvp_current_evidence_consolidation"
REPAIR_RETRY_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_sweep"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_objective_next_v2"
)

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
                f"{label} source-context field required: {key}"
            )
    return {key: container[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS}


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _dict(guard.get("source_summary"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "outside-soloing repair listening review input guard boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "input guard must route to outside-soloing repair objective-only next decision"
        )
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "input guard completion required"
        )
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "objective-only decision requires pending review input"
        )
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if not bool(source.get("technical_wav_validation", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "technical WAV validation required"
        )
    if _int(source.get("rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "rendered WAV count below 6"
        )
    if bool(source.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(source.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(source.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "outside-soloing repair should be targeted before objective decision"
        )
    source_context = _source_context_fields(source, label="input guard source summary")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="outside-soloing repair input guard readiness")
    return {
        "boundary": SOURCE_BOUNDARY,
        "review_item_count": _int(guard.get("review_item_count")),
        "required_input_field_count": _int(guard.get("required_input_field_count")),
        "validated_review_input_present": bool(
            guard.get("validated_review_input_present", False)
        ),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
        "technical_wav_validation": bool(source.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(source.get("rendered_audio_file_count")),
        "sample_rate": _int(source.get("sample_rate")),
        "duration_min_seconds": _float(source.get("duration_min_seconds")),
        "duration_max_seconds": _float(source.get("duration_max_seconds")),
        "changed_note_total": _int(source.get("changed_note_total")),
        "source_objective_outside_soloing_pitch_role_risk_count": _int(
            source.get("source_objective_outside_soloing_pitch_role_risk_count")
        ),
        "source_outside_soloing_pitch_role_risk_count_before": _int(
            source.get("source_outside_soloing_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_targeted": bool(
            source.get("source_outside_soloing_repair_targeted", True)
        ),
        "source_outside_soloing_residual_risk_preserved": bool(
            source.get("source_outside_soloing_residual_risk_preserved", False)
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            source.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            source.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            source.get("outside_soloing_repair_targeted", False)
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            source.get("weak_chord_tone_landing_risk_count_after")
        ),
        "final_landing_chord_tone_count_after": _int(
            source.get("final_landing_chord_tone_count_after")
        ),
        "max_non_chord_tone_run_after": _int(source.get("max_non_chord_tone_run_after")),
        "audio_review_required": bool(source.get("audio_review_required", False)),
        **source_context,
    }


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    output_dir: Path | str,
    issue_number: int,
    max_non_chord_tone_run_threshold: int,
    min_final_landing_chord_tone_count: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    source = validate_input_guard_report(input_guard_report)
    outside_target_supported = (
        _int(source["outside_soloing_pitch_role_risk_count_after"]) == 0
    )
    weak_landing_target_supported = (
        _int(source["weak_chord_tone_landing_risk_count_after"]) == 0
    )
    non_chord_run_target_supported = _int(source["max_non_chord_tone_run_after"]) <= int(
        max_non_chord_tone_run_threshold
    )
    final_landing_target_supported = _int(
        source["final_landing_chord_tone_count_after"]
    ) >= int(min_final_landing_chord_tone_count)
    objective_path_supported = bool(
        outside_target_supported
        and weak_landing_target_supported
        and non_chord_run_target_supported
        and final_landing_target_supported
    )
    next_boundary = (
        CURRENT_EVIDENCE_NEXT_BOUNDARY if objective_path_supported else REPAIR_RETRY_NEXT_BOUNDARY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "objective_summary": {
            "review_item_count": _int(source["review_item_count"]),
            "required_input_field_count": _int(source["required_input_field_count"]),
            "validated_review_input_present": bool(
                source["validated_review_input_present"]
            ),
            "preference_fill_allowed": bool(source["preference_fill_allowed"]),
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "rendered_audio_file_count": _int(source["rendered_audio_file_count"]),
            "sample_rate": _int(source["sample_rate"]),
            "duration_min_seconds": _float(source["duration_min_seconds"]),
            "duration_max_seconds": _float(source["duration_max_seconds"]),
            "changed_note_total": _int(source["changed_note_total"]),
            "source_objective_outside_soloing_pitch_role_risk_count": _int(
                source["source_objective_outside_soloing_pitch_role_risk_count"]
            ),
            "source_outside_soloing_pitch_role_risk_count_before": _int(
                source["source_outside_soloing_pitch_role_risk_count_before"]
            ),
            "source_outside_soloing_pitch_role_risk_count_after": _int(
                source["source_outside_soloing_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_pitch_role_risk_delta": _int(
                source["source_outside_soloing_pitch_role_risk_delta"]
            ),
            "source_outside_soloing_repair_targeted": bool(
                source["source_outside_soloing_repair_targeted"]
            ),
            "source_outside_soloing_residual_risk_preserved": bool(
                source["source_outside_soloing_residual_risk_preserved"]
            ),
            "outside_soloing_pitch_role_risk_count_after": _int(
                source["outside_soloing_pitch_role_risk_count_after"]
            ),
            "outside_soloing_pitch_role_risk_delta": _int(
                source["outside_soloing_pitch_role_risk_delta"]
            ),
            "outside_soloing_repair_targeted": bool(
                source["outside_soloing_repair_targeted"]
            ),
            "outside_soloing_target_supported": bool(outside_target_supported),
            "weak_chord_tone_landing_risk_count_after": _int(
                source["weak_chord_tone_landing_risk_count_after"]
            ),
            "weak_landing_target_supported": bool(weak_landing_target_supported),
            "final_landing_chord_tone_count_after": _int(
                source["final_landing_chord_tone_count_after"]
            ),
            "min_final_landing_chord_tone_count": int(
                min_final_landing_chord_tone_count
            ),
            "final_landing_target_supported": bool(final_landing_target_supported),
            "max_non_chord_tone_run_after": _int(source["max_non_chord_tone_run_after"]),
            "max_non_chord_tone_run_threshold": int(max_non_chord_tone_run_threshold),
            "non_chord_run_target_supported": bool(non_chord_run_target_supported),
            "audio_review_required": bool(source["audio_review_required"]),
            "outside_soloing_repair_objective_path_supported": bool(
                objective_path_supported
            ),
            "current_evidence_consolidation_ready": bool(objective_path_supported),
            **{key: source.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        },
        "selected_next_target": {
            "target": (
                "current_evidence_consolidation"
                if objective_path_supported
                else "outside_soloing_repair_retry"
            ),
            "next_boundary": next_boundary,
            "reason": (
                "outside-soloing repair objective targets passed; route to current evidence consolidation without quality claim"
                if objective_path_supported
                else "outside-soloing repair objective targets still fail; retry repair sweep"
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "objective_next_completed": True,
            "objective_next_decision_completed": True,
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "source_objective_outside_soloing_pitch_role_risk_count": _int(
                source["source_objective_outside_soloing_pitch_role_risk_count"]
            ),
            "source_outside_soloing_pitch_role_risk_count_before": _int(
                source["source_outside_soloing_pitch_role_risk_count_before"]
            ),
            "source_outside_soloing_pitch_role_risk_count_after": _int(
                source["source_outside_soloing_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_pitch_role_risk_delta": _int(
                source["source_outside_soloing_pitch_role_risk_delta"]
            ),
            "source_outside_soloing_repair_targeted": bool(
                source["source_outside_soloing_repair_targeted"]
            ),
            "source_outside_soloing_residual_risk_preserved": bool(
                source["source_outside_soloing_residual_risk_preserved"]
            ),
            "outside_soloing_repair_targeted": bool(
                source["outside_soloing_repair_targeted"]
            ),
            "outside_soloing_target_supported": bool(outside_target_supported),
            "weak_landing_target_supported": bool(weak_landing_target_supported),
            "non_chord_run_target_supported": bool(non_chord_run_target_supported),
            "final_landing_target_supported": bool(final_landing_target_supported),
            "outside_soloing_repair_objective_path_supported": bool(
                objective_path_supported
            ),
            "current_evidence_consolidation_ready": bool(objective_path_supported),
            **{key: source.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective outside-soloing repair evidence selected the next boundary without quality claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo MVP current evidence consolidation"
            if objective_path_supported
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair sweep"
        ),
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_support: bool,
    require_current_evidence_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    selected = _dict(report.get("selected_next_target"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "unexpected next boundary"
        )
    if not bool(readiness.get("objective_next_decision_completed", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "objective next decision completion required"
        )
    if require_objective_support and not bool(
        readiness.get("outside_soloing_repair_objective_path_supported", False)
    ):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "outside-soloing repair objective path support required"
        )
    if require_current_evidence_ready and not bool(
        readiness.get("current_evidence_consolidation_ready", False)
    ):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "current evidence consolidation readiness required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "critical user input should not be required"
        )
    if bool(summary.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(summary.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(summary.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
            "outside-soloing repair should remain targeted in objective summary"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="outside-soloing objective next readiness")
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in summary:
            raise StageBMidiToSoloOutsideSoloingRepairObjectiveNextError(
                f"objective summary source-context field required: {key}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(selected.get("target") or ""),
        "objective_next_completed": bool(readiness.get("objective_next_completed", False)),
        "validated_review_input_present": bool(
            summary.get("validated_review_input_present", True)
        ),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "changed_note_total": _int(summary.get("changed_note_total")),
        "source_objective_outside_soloing_pitch_role_risk_count": _int(
            summary.get("source_objective_outside_soloing_pitch_role_risk_count")
        ),
        "source_outside_soloing_pitch_role_risk_count_before": _int(
            summary.get("source_outside_soloing_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_pitch_role_risk_delta": _int(
            summary.get("source_outside_soloing_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_targeted": bool(
            summary.get("source_outside_soloing_repair_targeted", True)
        ),
        "source_outside_soloing_residual_risk_preserved": bool(
            summary.get("source_outside_soloing_residual_risk_preserved", False)
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            summary.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            summary.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            summary.get("outside_soloing_repair_targeted", False)
        ),
        "outside_soloing_target_supported": bool(
            summary.get("outside_soloing_target_supported", False)
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            summary.get("weak_chord_tone_landing_risk_count_after")
        ),
        "weak_landing_target_supported": bool(
            summary.get("weak_landing_target_supported", False)
        ),
        "final_landing_chord_tone_count_after": _int(
            summary.get("final_landing_chord_tone_count_after")
        ),
        "final_landing_target_supported": bool(
            summary.get("final_landing_target_supported", False)
        ),
        "max_non_chord_tone_run_after": _int(summary.get("max_non_chord_tone_run_after")),
        "non_chord_run_target_supported": bool(
            summary.get("non_chord_run_target_supported", False)
        ),
        "outside_soloing_repair_objective_path_supported": bool(
            summary.get("outside_soloing_repair_objective_path_supported", False)
        ),
        "current_evidence_consolidation_ready": bool(
            summary.get("current_evidence_consolidation_ready", False)
        ),
        **{key: summary.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    summary = report["objective_summary"]
    selected = report["selected_next_target"]
    lines = [
        "# Stage B MIDI-to-Solo Outside-Soloing Repair Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['target']}`",
        f"- review item count: `{summary['review_item_count']}`",
        f"- validated review input present: `{_bool_token(summary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(summary['preference_fill_allowed'])}`",
        f"- technical WAV validation: `{_bool_token(summary['technical_wav_validation'])}`",
        f"- rendered audio file count: `{summary['rendered_audio_file_count']}`",
        f"- changed note total: `{summary['changed_note_total']}`",
        f"- source objective outside-soloing pitch-role risk count: `{summary['source_objective_outside_soloing_pitch_role_risk_count']}`",
        f"- source outside-soloing pitch-role risk count: `{summary['source_outside_soloing_pitch_role_risk_count_before']} -> {summary['source_outside_soloing_pitch_role_risk_count_after']}`",
        f"- source outside-soloing pitch-role risk delta: `{summary['source_outside_soloing_pitch_role_risk_delta']}`",
        f"- source outside-soloing repair targeted: `{_bool_token(summary['source_outside_soloing_repair_targeted'])}`",
        f"- source outside-soloing residual risk preserved: `{_bool_token(summary['source_outside_soloing_residual_risk_preserved'])}`",
        f"- outside-soloing pitch-role risk count after: `{summary['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing pitch-role risk delta: `{summary['outside_soloing_pitch_role_risk_delta']}`",
        f"- outside-soloing repair targeted: `{_bool_token(summary['outside_soloing_repair_targeted'])}`",
        f"- outside-soloing target supported: `{_bool_token(summary['outside_soloing_target_supported'])}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk delta: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source targeted: `{_bool_token(summary['followup_objective_source_outside_soloing_source_targeted'])}`",
        f"- follow-up objective source outside-soloing source residual risk preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk delta: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source targeted: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- follow-up repair sweep source outside-soloing source residual risk preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk delta: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source targeted: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- bridge repair sweep source outside-soloing source residual risk preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- weak chord-tone landing risk count after: `{summary['weak_chord_tone_landing_risk_count_after']}`",
        f"- weak landing target supported: `{_bool_token(summary['weak_landing_target_supported'])}`",
        f"- final landing chord-tone count after: `{summary['final_landing_chord_tone_count_after']}`",
        f"- final landing target supported: `{_bool_token(summary['final_landing_target_supported'])}`",
        f"- max non-chord-tone run after: `{summary['max_non_chord_tone_run_after']}`",
        f"- non-chord run target supported: `{_bool_token(summary['non_chord_run_target_supported'])}`",
        f"- current evidence consolidation ready: `{_bool_token(summary['current_evidence_consolidation_ready'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Decision",
        "",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- reason: `{decision['reason']}`",
        f"- next recommended issue: `{report['next_recommended_issue']}`",
        "",
        "## Claim Boundary",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select outside-soloing repair objective-only next boundary"
    )
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_objective_only_next_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=980)
    parser.add_argument("--max_non_chord_tone_run_threshold", type=int, default=3)
    parser.add_argument("--min_final_landing_chord_tone_count", type=int, default=6)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_support", action="store_true")
    parser.add_argument("--require_current_evidence_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_next_report(
        input_guard_report=read_json(Path(args.input_guard_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        max_non_chord_tone_run_threshold=int(args.max_non_chord_tone_run_threshold),
        min_final_landing_chord_tone_count=int(args.min_final_landing_chord_tone_count),
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_support=bool(args.require_objective_support),
        require_current_evidence_ready=bool(args.require_current_evidence_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_objective_only_next_decision.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_objective_only_next_decision_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_objective_only_next_decision.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
