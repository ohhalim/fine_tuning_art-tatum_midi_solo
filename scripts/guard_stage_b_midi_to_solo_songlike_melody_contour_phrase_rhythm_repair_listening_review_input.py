"""Guard songlike melody contour phrase/rhythm repair listening review preference fill against missing input."""

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
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BRIDGE_SOURCE_CONTEXT_KEYS,
)
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_package import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard"
FILL_BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_fill"
OBJECTIVE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision"
)
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard_v3"

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
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(source: dict[str, Any]) -> dict[str, Any]:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        objective_key = f"objective_{key}"
        if objective_key not in source or source[objective_key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
                f"objective source-context field required: {objective_key}"
            )
        if key not in source or source[key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
                f"source-context field required: {key}"
            )
    return {
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            source.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_delta"
            )
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            source.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            source.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            source.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        **{f"objective_{key}": source.get(f"objective_{key}") for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        **{key: source.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
    }


def _validate_source_context_group(source: dict[str, Any], *, base: str, label: str) -> None:
    objective_risk = _int(source.get(f"{base}_source_objective_pitch_role_risk_count"))
    source_before = _int(source.get(f"{base}_source_pitch_role_risk_count_before"))
    source_after = _int(source.get(f"{base}_source_pitch_role_risk_count_after"))
    source_delta = _int(source.get(f"{base}_source_pitch_role_risk_delta"))
    current_after = _int(source.get(f"{base}_pitch_role_risk_count_after"))
    current_delta = _int(source.get(f"{base}_pitch_role_risk_delta"))
    if not bool(source.get(f"{base}_source_context_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} source context preservation required"
        )
    if objective_risk <= 0 or source_before <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} source pitch-role risk context required"
        )
    if objective_risk != source_before:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} objective/source risk mismatch"
        )
    if source_before - source_after != source_delta:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} source pitch-role risk delta mismatch"
        )
    if source_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} positive source pitch-role risk delta required"
        )
    if bool(source.get(f"{base}_source_targeted", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} source-targeted flag should remain false"
        )
    if not bool(source.get(f"{base}_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} source residual risk preservation required"
        )
    if current_after != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} current repair residual pitch-role risk should be zero"
        )
    if current_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"{label} positive current repair pitch-role risk delta required"
        )


def _validate_source_context(source: dict[str, Any], *, label: str) -> None:
    _validate_source_context_group(
        source,
        base="objective_source_outside_soloing_repair",
        label=f"objective {label}",
    )
    _validate_source_context_group(
        source,
        base="source_outside_soloing_repair",
        label=label,
    )


def validate_source_package(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    package = _dict(report.get("review_package"))
    source = _dict(report.get("source_summary"))
    review_items = [_dict(item) for item in _list(report.get("review_items"))]
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "songlike melody contour phrase/rhythm repair listening review package boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "songlike melody contour phrase/rhythm repair listening package must route to input guard"
        )
    if not bool(readiness.get("listening_review_package_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "listening review package readiness required"
        )
    if not bool(package.get("package_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "review package ready flag required"
        )
    review_item_count = _int(readiness.get("review_item_count") or package.get("review_item_count"))
    if review_item_count <= 0 or len(review_items) < review_item_count:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "review items required"
        )
    if not bool(source.get("technical_wav_validation", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "technical WAV validation required"
        )
    if _int(source.get("rendered_audio_file_count")) < review_item_count:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "rendered audio count below review item count"
        )
    if not bool(source.get("audio_review_required", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "audio review requirement should remain recorded"
        )
    if not bool(source.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "outside-soloing repair evidence should be ready"
        )
    if _int(source.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "source outside-soloing repair pitch-role risk should be resolved"
        )
    source_context = _source_context_fields(source)
    _validate_source_context(source_context, label="source outside-soloing repair")
    if _int(source.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "source outside-soloing not-evaluable count should be preserved"
        )
    if _int(source.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "repaired outside-soloing not-evaluable count should be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="phrase/rhythm repair listening package readiness")
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
            "technical_wav_validation": bool(source.get("technical_wav_validation", False)),
            "rendered_audio_file_count": _int(source.get("rendered_audio_file_count")),
            "sample_rate": _int(source.get("sample_rate")),
            "duration_min_seconds": _float(source.get("duration_min_seconds")),
            "duration_max_seconds": _float(source.get("duration_max_seconds")),
            "failure_label_delta": _int(source.get("failure_label_delta")),
            "source_phrase_rhythm_failure_count": _int(
                source.get("source_phrase_rhythm_failure_count")
            ),
            "repaired_phrase_rhythm_failure_count": _int(
                source.get("repaired_phrase_rhythm_failure_count")
            ),
            "phrase_rhythm_failure_delta": _int(source.get("phrase_rhythm_failure_delta")),
            "remaining_failure_counts": _dict(source.get("remaining_failure_counts")),
            "source_outside_soloing_repair_evidence_ready": bool(
                source.get("source_outside_soloing_repair_evidence_ready", False)
            ),
            **source_context,
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                source.get("source_outside_soloing_repair_pitch_role_risk_count_after")
            ),
            "source_outside_soloing_not_evaluable_count": _int(
                source.get("source_outside_soloing_not_evaluable_count")
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                source.get("repaired_outside_soloing_not_evaluable_count")
            ),
            "repaired_not_evaluable_counts": _dict(source.get("repaired_not_evaluable_counts")),
            "audio_review_required": bool(source.get("audio_review_required", False)),
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
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair listening review fill"
            if validated_input
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair objective-only next decision source-context refresh"
        ),
    }


def validate_listening_review_input_guard_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_guard_completed: bool,
    require_preference_blocked: bool,
    require_pending_input: bool = False,
    require_no_quality_claim: bool = False,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _dict(guard.get("source_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "unexpected next boundary"
        )
    if require_guard_completed and not bool(
        readiness.get("listening_review_input_guard_completed", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "guard completion required"
        )
    if require_preference_blocked and bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "preference fill should remain blocked"
        )
    if require_pending_input and bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "validated input should remain pending"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairListeningInputGuardError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="phrase/rhythm repair input guard readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
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
        "failure_label_delta": _int(source.get("failure_label_delta")),
        "phrase_rhythm_failure_delta": _int(source.get("phrase_rhythm_failure_delta")),
        "source_outside_soloing_repair_evidence_ready": bool(
            source.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            source.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get("objective_source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            source.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            source.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            source.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        **{f"objective_{key}": source.get(f"objective_{key}") for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        **{key: source.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        "source_outside_soloing_not_evaluable_count": _int(
            source.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            source.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(source.get("repaired_not_evaluable_counts")),
        "audio_review_required": bool(source.get("audio_review_required", False)),
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
        "# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Listening Review Input Guard",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- review item count: `{guard['review_item_count']}`",
        f"- required input field count: `{guard['required_input_field_count']}`",
        f"- validated review input present: `{_bool_token(guard['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(guard['preference_fill_allowed'])}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- rendered audio file count: `{source['rendered_audio_file_count']}`",
        f"- duration range: `{source['duration_min_seconds']:.3f}s-{source['duration_max_seconds']:.3f}s`",
        f"- failure label delta: `{source['failure_label_delta']}`",
        f"- phrase/rhythm failure count: `{source['source_phrase_rhythm_failure_count']} -> {source['repaired_phrase_rhythm_failure_count']}`",
        f"- phrase/rhythm failure delta: `{source['phrase_rhythm_failure_delta']}`",
        f"- source outside-soloing repair evidence ready: `{_bool_token(source['source_outside_soloing_repair_evidence_ready'])}`",
        f"- objective source outside-soloing source context preserved: `{_bool_token(source['objective_source_outside_soloing_repair_source_context_preserved'])}`",
        f"- objective source outside-soloing source pitch-role risk before / after / delta: `{source['objective_source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{source['objective_source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{source['objective_source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- objective source outside-soloing current repair pitch-role risk after / delta: `{source['objective_source_outside_soloing_repair_pitch_role_risk_count_after']}` / `{source['objective_source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- source outside-soloing source context preserved: `{_bool_token(source['source_outside_soloing_repair_source_context_preserved'])}`",
        f"- source outside-soloing source pitch-role risk before / after / delta: `{source['source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{source['source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{source['source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- source outside-soloing source repair targeted: `{_bool_token(source['source_outside_soloing_repair_source_targeted'])}`",
        f"- source outside-soloing source residual risk preserved: `{_bool_token(source['source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- source outside-soloing repair pitch-role risk count after: `{source['source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- source outside-soloing current repair pitch-role risk delta: `{source['source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- source/repaired outside-soloing not evaluable count: `{source['source_outside_soloing_not_evaluable_count']}/{source['repaired_outside_soloing_not_evaluable_count']}`",
        f"- audio review required: `{_bool_token(source['audio_review_required'])}`",
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
    lines.extend(["", "## Repaired Not Evaluable Counts", ""])
    for label, count in sorted(source["repaired_not_evaluable_counts"].items()):
        lines.append(f"- `{label}`: `{count}`")
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
        description="Guard songlike melody contour phrase/rhythm repair listening review input"
    )
    parser.add_argument("--source_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1034)
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
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input_guard.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
