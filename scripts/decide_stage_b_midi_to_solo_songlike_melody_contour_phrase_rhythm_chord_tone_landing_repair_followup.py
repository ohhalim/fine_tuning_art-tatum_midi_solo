"""Decide the follow-up target after chord-tone landing repair evidence."""

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
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next import (  # noqa: E402
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
    SELECTED_TARGET as OBJECTIVE_NEXT_SELECTED_TARGET,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (  # noqa: E402
    BOUNDARY as REPAIR_SWEEP_BOUNDARY,
)


class StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep"
)
SELECTED_TARGET = (
    "songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision_v1"
)
PRIMARY_RISK_LABEL = "outside_soloing_pitch_role_risk"
RESOLVED_RISK_LABEL = "weak_chord_tone_landing_risk"
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
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_objective_next_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    selected = _dict(report.get("selected_next_target"))
    if str(report.get("boundary") or "") != OBJECTIVE_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "chord-tone landing objective-only next decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != FOLLOWUP_DECISION_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "objective-only next decision must route to follow-up decision"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "follow-up decision boundary mismatch"
        )
    if str(selected.get("target") or "") != OBJECTIVE_NEXT_SELECTED_TARGET:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "objective-only next selected target mismatch"
        )
    if not bool(readiness.get("objective_next_decision_completed", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "objective next decision completion required"
        )
    if not bool(readiness.get("chord_tone_landing_followup_required", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "chord-tone landing follow-up requirement required"
        )
    if bool(summary.get("validated_review_input_present", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "follow-up decision expects pending listening input"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "preference fill must remain blocked"
        )
    if not bool(summary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "technical WAV validation required"
        )
    if _int(summary.get("rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "rendered WAV count below 6"
        )
    if _int(summary.get("outside_soloing_pitch_role_risk_count_after")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "residual outside-soloing pitch-role risk required"
        )
    if _int(summary.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        summary.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "objective outside-soloing count must match source count"
        )
    if bool(summary.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "outside-soloing repair target should remain false before follow-up"
        )
    if not bool(summary.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "outside-soloing residual risk context must be preserved"
        )
    if _int(summary.get("weak_chord_tone_landing_risk_delta")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "weak chord-tone landing risk delta required"
        )
    if bool(summary.get("current_quality_claim_ready", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "quality claim readiness must remain false"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="objective next readiness")
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "review_item_count": _int(summary.get("review_item_count")),
        "required_input_field_count": _int(summary.get("required_input_field_count")),
        "validated_review_input_present": bool(
            summary.get("validated_review_input_present", False)
        ),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", False)),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "sample_rate": _int(summary.get("sample_rate")),
        "duration_min_seconds": _float(summary.get("duration_min_seconds")),
        "duration_max_seconds": _float(summary.get("duration_max_seconds")),
        "changed_note_total": _int(summary.get("changed_note_total")),
        "weak_chord_tone_landing_risk_delta": _int(
            summary.get("weak_chord_tone_landing_risk_delta")
        ),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            summary.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "outside_soloing_pitch_role_risk_count_before": _int(
            summary.get("outside_soloing_pitch_role_risk_count_before")
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            summary.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            summary.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            summary.get("outside_soloing_repair_targeted", True)
        ),
        "outside_soloing_residual_risk_preserved": bool(
            summary.get("outside_soloing_residual_risk_preserved", False)
        ),
        "final_landing_chord_tone_count_after": _int(
            summary.get("final_landing_chord_tone_count_after")
        ),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
        "chord_tone_landing_followup_required": bool(
            summary.get("chord_tone_landing_followup_required", False)
        ),
        "current_quality_claim_ready": False,
    }


def validate_repair_sweep_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if str(report.get("boundary") or "") != REPAIR_SWEEP_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "chord-tone landing repair sweep boundary required"
        )
    if not bool(readiness.get("chord_tone_landing_repair_sweep_completed", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "repair sweep completion required"
        )
    if not bool(readiness.get("target_supported", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "chord-tone landing repair target support required"
        )
    if _int(aggregate.get("candidate_count")) < 6:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "candidate count below 6"
        )
    if _int(aggregate.get("repaired_midi_count")) != _int(aggregate.get("candidate_count")):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "all candidates must have repaired MIDI"
        )
    if _int(aggregate.get("changed_note_total")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "changed note count required"
        )
    if _int(aggregate.get("weak_chord_tone_landing_risk_count_after")) != 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "weak chord-tone landing risk must be cleared"
        )
    if _int(aggregate.get("weak_chord_tone_landing_risk_delta")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "positive weak chord-tone landing risk delta required"
        )
    if _int(aggregate.get("outside_soloing_pitch_role_risk_count_after")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "residual outside-soloing pitch-role risk required"
        )
    if _int(aggregate.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        aggregate.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "repair sweep objective outside-soloing count must match source count"
        )
    if bool(aggregate.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "repair sweep must preserve outside-soloing as untargeted"
        )
    if not bool(aggregate.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "repair sweep must preserve outside-soloing residual risk context"
        )
    if _int(aggregate.get("final_landing_chord_tone_count_after")) <= _int(
        aggregate.get("final_landing_chord_tone_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "final landing chord-tone improvement required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="repair sweep readiness")
    return {
        "boundary": REPAIR_SWEEP_BOUNDARY,
        "candidate_count": _int(aggregate.get("candidate_count")),
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "changed_note_total": _int(aggregate.get("changed_note_total")),
        "weak_chord_tone_landing_risk_count_before": _int(
            aggregate.get("weak_chord_tone_landing_risk_count_before")
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            aggregate.get("weak_chord_tone_landing_risk_count_after")
        ),
        "weak_chord_tone_landing_risk_delta": _int(
            aggregate.get("weak_chord_tone_landing_risk_delta")
        ),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            aggregate.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "outside_soloing_pitch_role_risk_count_before": _int(
            aggregate.get("outside_soloing_pitch_role_risk_count_before")
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            aggregate.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            aggregate.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            aggregate.get("outside_soloing_repair_targeted", True)
        ),
        "outside_soloing_residual_risk_preserved": bool(
            aggregate.get("outside_soloing_residual_risk_preserved", False)
        ),
        "final_landing_chord_tone_count_before": _int(
            aggregate.get("final_landing_chord_tone_count_before")
        ),
        "final_landing_chord_tone_count_after": _int(
            aggregate.get("final_landing_chord_tone_count_after")
        ),
        "target_supported": bool(aggregate.get("target_supported", False)),
    }


def build_followup_decision_report(
    *,
    objective_next_report: dict[str, Any],
    repair_sweep_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    objective = validate_objective_next_source(objective_next_report)
    sweep = validate_repair_sweep_source(repair_sweep_report)
    if _int(objective["outside_soloing_pitch_role_risk_count_before"]) != _int(
        sweep["outside_soloing_pitch_role_risk_count_before"]
    ) or _int(objective["outside_soloing_pitch_role_risk_count_after"]) != _int(
        sweep["outside_soloing_pitch_role_risk_count_after"]
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "objective and repair sweep outside-soloing risk counts must match"
        )
    if bool(objective["outside_soloing_repair_targeted"]) != bool(
        sweep["outside_soloing_repair_targeted"]
    ) or bool(objective["outside_soloing_residual_risk_preserved"]) != bool(
        sweep["outside_soloing_residual_risk_preserved"]
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "objective and repair sweep outside-soloing context flags must match"
        )
    primary_risk_count = _int(objective["outside_soloing_pitch_role_risk_count_after"])
    weak_resolved = _int(sweep["weak_chord_tone_landing_risk_count_after"]) == 0
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": objective["boundary"],
        "repair_sweep_boundary": sweep["boundary"],
        "objective_summary": objective,
        "repair_sweep_summary": sweep,
        "selected_next_target": {
            "selected_target": SELECTED_TARGET,
            "selected_next_boundary": NEXT_BOUNDARY,
            "primary_remaining_risk_label": PRIMARY_RISK_LABEL,
            "primary_remaining_risk_count": primary_risk_count,
            "resolved_risk_label": RESOLVED_RISK_LABEL,
            "resolved_risk_count_after": _int(
                sweep["weak_chord_tone_landing_risk_count_after"]
            ),
            "reason": (
                "weak chord-tone landing risk cleared while residual outside-soloing pitch-role risk remains"
            ),
        },
        "followup_targets": {
            "primary_risk_label": PRIMARY_RISK_LABEL,
            "primary_risk_count": primary_risk_count,
            "preserve_gates": [
                "technical_wav_validation",
                "weak_chord_tone_landing_risk_count_zero",
                "final_landing_chord_tone_count_preserved",
                "monophonic_candidate_path",
                "no_quality_claim",
            ],
            "review_boundary": "objective_only",
        },
        "readiness": {
            "boundary": BOUNDARY,
            "followup_decision_completed": True,
            "outside_soloing_repair_selected": True,
            "weak_chord_tone_landing_resolved": weak_resolved,
            "primary_remaining_risk_label": PRIMARY_RISK_LABEL,
            "primary_remaining_risk_count": primary_risk_count,
            "candidate_count": _int(sweep["candidate_count"]),
            "repaired_midi_count": _int(sweep["repaired_midi_count"]),
            "changed_note_total": _int(sweep["changed_note_total"]),
            "weak_chord_tone_landing_risk_delta": _int(
                sweep["weak_chord_tone_landing_risk_delta"]
            ),
            "objective_outside_soloing_pitch_role_risk_count": _int(
                sweep["objective_outside_soloing_pitch_role_risk_count"]
            ),
            "outside_soloing_pitch_role_risk_count_before": _int(
                sweep["outside_soloing_pitch_role_risk_count_before"]
            ),
            "outside_soloing_pitch_role_risk_count_after": _int(
                sweep["outside_soloing_pitch_role_risk_count_after"]
            ),
            "outside_soloing_pitch_role_risk_delta": _int(
                sweep["outside_soloing_pitch_role_risk_delta"]
            ),
            "outside_soloing_repair_targeted": bool(
                sweep["outside_soloing_repair_targeted"]
            ),
            "outside_soloing_residual_risk_preserved": bool(
                sweep["outside_soloing_residual_risk_preserved"]
            ),
            "final_landing_chord_tone_count_after": _int(
                sweep["final_landing_chord_tone_count_after"]
            ),
            "technical_wav_validation": bool(objective["technical_wav_validation"]),
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
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "residual outside-soloing pitch-role risk selected as next objective repair target",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "outside_soloing_pitch_role_risk_repaired",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair sweep"
        ),
    }


def validate_followup_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_followup_decision: bool,
    require_outside_soloing_repair: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "unexpected next boundary"
        )
    if expected_target and str(selected.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "unexpected selected target"
        )
    if require_followup_decision and not bool(
        readiness.get("followup_decision_completed", False)
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "follow-up decision completion required"
        )
    if require_outside_soloing_repair and not bool(
        readiness.get("outside_soloing_repair_selected", False)
    ):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "outside-soloing repair selection required"
        )
    if require_outside_soloing_repair and _int(
        readiness.get("primary_remaining_risk_count")
    ) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "primary remaining risk count required"
        )
    if not bool(readiness.get("weak_chord_tone_landing_resolved", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "weak chord-tone landing resolution required"
        )
    if not bool(readiness.get("technical_wav_validation", False)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "technical WAV validation required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairFollowupDecisionError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="follow-up readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "repair_sweep_boundary": str(report.get("repair_sweep_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(selected.get("selected_target") or ""),
        "followup_decision_completed": bool(
            readiness.get("followup_decision_completed", False)
        ),
        "outside_soloing_repair_selected": bool(
            readiness.get("outside_soloing_repair_selected", False)
        ),
        "primary_remaining_risk_label": str(
            selected.get("primary_remaining_risk_label") or ""
        ),
        "primary_remaining_risk_count": _int(
            selected.get("primary_remaining_risk_count")
        ),
        "weak_chord_tone_landing_resolved": bool(
            readiness.get("weak_chord_tone_landing_resolved", False)
        ),
        "candidate_count": _int(readiness.get("candidate_count")),
        "repaired_midi_count": _int(readiness.get("repaired_midi_count")),
        "changed_note_total": _int(readiness.get("changed_note_total")),
        "weak_chord_tone_landing_risk_delta": _int(
            readiness.get("weak_chord_tone_landing_risk_delta")
        ),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            readiness.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "outside_soloing_pitch_role_risk_count_before": _int(
            readiness.get("outside_soloing_pitch_role_risk_count_before")
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            readiness.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            readiness.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            readiness.get("outside_soloing_repair_targeted", True)
        ),
        "outside_soloing_residual_risk_preserved": bool(
            readiness.get("outside_soloing_residual_risk_preserved", False)
        ),
        "final_landing_chord_tone_count_after": _int(
            readiness.get("final_landing_chord_tone_count_after")
        ),
        "technical_wav_validation": bool(readiness.get("technical_wav_validation", False)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    selected = report["selected_next_target"]
    lines = [
        "# Stage B MIDI-to-Solo Chord-Tone Landing Repair Follow-Up Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- repair sweep boundary: `{report['repair_sweep_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- primary remaining risk label: `{selected['primary_remaining_risk_label']}`",
        f"- primary remaining risk count: `{selected['primary_remaining_risk_count']}`",
        f"- weak chord-tone landing resolved: `{_bool_token(readiness['weak_chord_tone_landing_resolved'])}`",
        f"- outside-soloing repair selected: `{_bool_token(readiness['outside_soloing_repair_selected'])}`",
        f"- candidate count: `{readiness['candidate_count']}`",
        f"- repaired MIDI count: `{readiness['repaired_midi_count']}`",
        f"- changed note total: `{readiness['changed_note_total']}`",
        f"- weak chord-tone landing risk delta: `{readiness['weak_chord_tone_landing_risk_delta']}`",
        f"- objective outside-soloing pitch-role risk count: `{readiness['objective_outside_soloing_pitch_role_risk_count']}`",
        f"- outside-soloing pitch-role risk count: `{readiness['outside_soloing_pitch_role_risk_count_before']} -> {readiness['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing pitch-role risk delta: `{readiness['outside_soloing_pitch_role_risk_delta']}`",
        f"- outside-soloing repair targeted: `{_bool_token(readiness['outside_soloing_repair_targeted'])}`",
        f"- outside-soloing residual risk preserved: `{_bool_token(readiness['outside_soloing_residual_risk_preserved'])}`",
        f"- final landing chord-tone count after: `{readiness['final_landing_chord_tone_count_after']}`",
        f"- technical WAV validation: `{_bool_token(readiness['technical_wav_validation'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Decision",
        "",
        f"- reason: `{decision['reason']}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- next recommended issue: `{report['next_recommended_issue']}`",
        "",
        "## Preserve Gates",
        "",
    ]
    for item in report["followup_targets"]["preserve_gates"]:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide chord-tone landing repair follow-up target"
    )
    parser.add_argument("--objective_next_report", type=str, required=True)
    parser.add_argument("--repair_sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_followup_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=884)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_followup_decision", action="store_true")
    parser.add_argument("--require_outside_soloing_repair", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_followup_decision_report(
        objective_next_report=read_json(Path(args.objective_next_report)),
        repair_sweep_report=read_json(Path(args.repair_sweep_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_followup_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_followup_decision=bool(args.require_followup_decision),
        require_outside_soloing_repair=bool(args.require_outside_soloing_repair),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_followup_decision.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_followup_decision_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_followup_decision.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
