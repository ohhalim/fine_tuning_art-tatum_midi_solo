"""Guard outside-soloing repair listening review preference fill against missing input."""

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
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SCHEMA_VERSION as SOURCE_SCHEMA_VERSION,
    SOURCE_PACKAGE_SCHEMA_CONTEXT_KEYS,
)


class StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_listening_review_input_guard"
)
FILL_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_listening_review_fill"
)
OBJECTIVE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_objective_only_next_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_"
    "outside_soloing_repair_listening_review_input_guard_v4"
)
SOURCE_GUARD_SCHEMA_CONTEXT_KEYS = [
    "source_schema_version",
    "source_audio_package_schema_version",
    "source_repair_sweep_schema_version",
    "source_followup_schema_version",
    "source_objective_input_guard_schema_version",
    "source_package_schema_version",
    "source_audio_schema_version",
    "chord_tone_repair_sweep_schema_version",
    "chord_tone_repair_sweep_source_schema_version",
    "chord_tone_repair_sweep_bridge_schema_version",
]

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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
                f"{label} source-context field required: {key}"
            )
    missing_preserved = [
        key for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS if not bool(container.get(key))
    ]
    if missing_preserved:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            f"{label} source-context preserved field must be true: {missing_preserved}"
        )
    return {key: container[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS}


def _source_schema_context(
    report: dict[str, Any],
    source: dict[str, Any],
    readiness: dict[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    if str(report.get("schema_version") or "") != SOURCE_SCHEMA_VERSION:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "listening review package schema version must match current package report"
        )
    for key in SOURCE_PACKAGE_SCHEMA_CONTEXT_KEYS:
        if not str(source.get(key) or ""):
            raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
                f"{label} schema-context field required: {key}"
            )
        if str(readiness.get(key) or "") != str(source.get(key) or ""):
            raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
                f"{label} readiness schema-context mismatch: {key}"
            )
    return {
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_audio_package_schema_version": str(source.get("source_schema_version") or ""),
        "source_repair_sweep_schema_version": str(
            source.get("source_repair_sweep_schema_version") or ""
        ),
        "source_followup_schema_version": str(source.get("source_followup_schema_version") or ""),
        "source_objective_input_guard_schema_version": str(
            source.get("source_objective_input_guard_schema_version") or ""
        ),
        "source_package_schema_version": str(source.get("source_package_schema_version") or ""),
        "source_audio_schema_version": str(source.get("source_audio_schema_version") or ""),
        "chord_tone_repair_sweep_schema_version": str(
            source.get("chord_tone_repair_sweep_schema_version") or ""
        ),
        "chord_tone_repair_sweep_source_schema_version": str(
            source.get("chord_tone_repair_sweep_source_schema_version") or ""
        ),
        "chord_tone_repair_sweep_bridge_schema_version": str(
            source.get("chord_tone_repair_sweep_bridge_schema_version") or ""
        ),
    }


def validate_source_package(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    package = _dict(report.get("review_package"))
    source = _dict(report.get("source_summary"))
    review_items = [_dict(item) for item in _list(report.get("review_items"))]
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "outside-soloing repair listening review package boundary required"
        )
    source_schema_context = _source_schema_context(
        report, source, readiness, label="listening package source summary"
    )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "listening package must route to input guard"
        )
    if not bool(readiness.get("listening_review_package_ready", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "listening review package readiness required"
        )
    if not bool(package.get("package_ready", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "review package ready flag required"
        )
    review_item_count = _int(readiness.get("review_item_count") or package.get("review_item_count"))
    if review_item_count <= 0 or len(review_items) < review_item_count:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "review items required"
        )
    if not bool(source.get("technical_wav_validation", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "technical WAV validation required"
        )
    if _int(source.get("rendered_audio_file_count")) < review_item_count:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "rendered audio count below review item count"
        )
    if not bool(source.get("audio_review_required", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "audio review requirement should remain recorded"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "critical user input should not be required"
        )
    if bool(source.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(source.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(source.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "outside-soloing repair should remain targeted before input guard"
        )
    source_context = _source_context_fields(source, label="listening package source summary")
    _require_no_quality_claim(readiness, label="outside-soloing repair listening package readiness")
    return {
        "boundary": SOURCE_BOUNDARY,
        "review_item_count": review_item_count,
        "validated_review_input": bool(readiness.get("validated_review_input", False)),
        "human_review_required_now": bool(readiness.get("human_review_required_now", False)),
        "required_input_fields": [
            str(item) for item in _list(package.get("required_input_fields"))
        ],
        "wav_paths": [str(item.get("wav_path") or "") for item in review_items],
        "source_summary": {
            **source_schema_context,
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
        },
    }


def build_listening_review_input_guard_report(
    source_package: dict[str, Any],
    *,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_source_package(source_package)
    validated_input = bool(source["validated_review_input"])
    next_boundary = FILL_BOUNDARY if validated_input else OBJECTIVE_NEXT_BOUNDARY
    reason = (
        "validated listening review input present; preference fill allowed"
        if validated_input
        else "listening review input pending; preference fill blocked"
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
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "source_package_summary": source,
        "guard_result": {
            "validated_review_input_present": bool(validated_input),
            "preference_fill_allowed": bool(validated_input),
            "review_item_count": int(source["review_item_count"]),
            "required_input_field_count": len(source["required_input_fields"]),
            "missing_validated_input_reason": ""
            if validated_input
            else "validated_review_input=false",
            "source_summary": source["source_summary"],
        },
        "readiness": {
            "boundary": BOUNDARY,
            "listening_review_input_guard_completed": True,
            "validated_review_input_present": bool(validated_input),
            "preference_fill_allowed": bool(validated_input),
            "listening_review_completed": False,
            "human_review_required_now": bool(source["human_review_required_now"]),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
            **source["source_summary"],
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": reason,
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
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing "
            "outside-soloing repair listening review fill source-context refresh"
            if validated_input
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing "
            "outside-soloing repair objective-only next decision source-context refresh"
        ),
    }


def validate_listening_review_input_guard_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_guard_completed: bool,
    require_preference_blocked: bool,
    require_pending_input: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _dict(guard.get("source_summary"))
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "input guard schema version must match current report"
        )
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "unexpected next boundary"
        )
    if require_guard_completed and not bool(
        readiness.get("listening_review_input_guard_completed", False)
    ):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "guard completion required"
        )
    if require_preference_blocked and bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "preference fill should remain blocked"
        )
    if require_pending_input and bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "validated input should remain pending"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "critical user input should not be required"
        )
    if bool(source.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(source.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(source.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "outside-soloing repair should remain targeted in guard summary"
        )
    if str(report.get("source_schema_version") or "") != SOURCE_SCHEMA_VERSION:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            "unexpected source schema version"
        )
    for key in SOURCE_GUARD_SCHEMA_CONTEXT_KEYS:
        if not str(source.get(key) or ""):
            raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
                f"input guard schema-context field required: {key}"
            )
        if str(readiness.get(key) or "") != str(source.get(key) or ""):
            raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
                f"input guard readiness schema-context mismatch: {key}"
            )
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in source:
            raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
                f"input guard source-context field required: {key}"
            )
    missing_preserved = [
        key for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS if not bool(source.get(key))
    ]
    if missing_preserved:
        raise StageBMidiToSoloOutsideSoloingRepairListeningInputGuardError(
            f"input guard source-context preserved field must be true: {missing_preserved}"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="outside-soloing repair input guard readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        **{key: str(source.get(key) or "") for key in SOURCE_GUARD_SCHEMA_CONTEXT_KEYS},
        "validated_review_input_present": bool(
            guard.get("validated_review_input_present", True)
        ),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", True)),
        "review_item_count": _int(guard.get("review_item_count")),
        "required_input_field_count": _int(guard.get("required_input_field_count")),
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
        **{key: source.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    guard = report["guard_result"]
    source = guard["source_summary"]
    package = report["source_package_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Chord-Tone Landing Outside-Soloing Repair Listening Review Input Guard",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- source schema version: `{source['source_schema_version']}`",
        f"- source audio package schema version: `{source['source_audio_package_schema_version']}`",
        f"- source repair sweep schema version: `{source['source_repair_sweep_schema_version']}`",
        f"- source follow-up schema version: `{source['source_followup_schema_version']}`",
        f"- source objective input guard schema version: `{source['source_objective_input_guard_schema_version']}`",
        f"- source package schema version: `{source['source_package_schema_version']}`",
        f"- source audio schema version: `{source['source_audio_schema_version']}`",
        f"- chord-tone repair sweep schema version: `{source['chord_tone_repair_sweep_schema_version']}`",
        f"- chord-tone repair sweep source schema version: `{source['chord_tone_repair_sweep_source_schema_version']}`",
        f"- chord-tone repair sweep bridge schema version: `{source['chord_tone_repair_sweep_bridge_schema_version']}`",
        f"- review item count: `{guard['review_item_count']}`",
        f"- required input field count: `{guard['required_input_field_count']}`",
        f"- validated review input present: `{_bool_token(guard['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(guard['preference_fill_allowed'])}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- rendered audio file count: `{source['rendered_audio_file_count']}`",
        f"- duration range: `{source['duration_min_seconds']:.3f}s-{source['duration_max_seconds']:.3f}s`",
        f"- changed note total: `{source['changed_note_total']}`",
        f"- source objective outside-soloing pitch-role risk count: `{source['source_objective_outside_soloing_pitch_role_risk_count']}`",
        f"- source outside-soloing pitch-role risk count: `{source['source_outside_soloing_pitch_role_risk_count_before']} -> {source['source_outside_soloing_pitch_role_risk_count_after']}`",
        f"- source outside-soloing pitch-role risk delta: `{source['source_outside_soloing_pitch_role_risk_delta']}`",
        f"- source outside-soloing repair targeted: `{_bool_token(source['source_outside_soloing_repair_targeted'])}`",
        f"- source outside-soloing residual risk preserved: `{_bool_token(source['source_outside_soloing_residual_risk_preserved'])}`",
        f"- outside-soloing pitch-role risk count after: `{source['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing pitch-role risk delta: `{source['outside_soloing_pitch_role_risk_delta']}`",
        f"- outside-soloing repair targeted: `{_bool_token(source['outside_soloing_repair_targeted'])}`",
        f"- weak chord-tone landing risk count after: `{source['weak_chord_tone_landing_risk_count_after']}`",
        f"- final landing chord-tone count after: `{source['final_landing_chord_tone_count_after']}`",
        f"- max non-chord-tone run after: `{source['max_non_chord_tone_run_after']}`",
        f"- audio review required: `{_bool_token(source['audio_review_required'])}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{source['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{source['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {source['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(source['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{source['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{source['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {source['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(source['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{source['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{source['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {source['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(source['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
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
        "## Required Input Fields",
        "",
    ]
    for field in package["required_input_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Review WAV Paths", ""])
    for path in package["wav_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guard outside-soloing repair listening review input"
    )
    parser.add_argument("--source_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_input_guard"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1146)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_guard_completed", action="store_true")
    parser.add_argument("--require_preference_blocked", action="store_true")
    parser.add_argument("--require_pending_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_listening_review_input_guard_report(
        read_json(Path(args.source_package)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_listening_review_input_guard_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_guard_completed=bool(args.require_guard_completed),
        require_preference_blocked=bool(
            args.require_preference_blocked or args.require_pending_input
        ),
        require_pending_input=bool(args.require_pending_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_input_guard.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_input_guard_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_input_guard.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
