"""Decide the follow-up target after songlike melody contour phrase/rhythm repair evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_next import (  # noqa: E402
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    FOLLOWUP_DECISION_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_sweep import (  # noqa: E402
    BOUNDARY as REPAIR_SWEEP_BOUNDARY,
)


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision"
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge"
)
SELECTED_TARGET = "songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge"
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision_v2"
CONTEXT_TARGET_LABELS = (
    "outside_soloing_without_context",
    "weak_chord_tone_landing",
)
RESIDUAL_FAILURE_LABEL = "rhythmic_monotony"

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


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
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
    }


def _validate_source_context(source: dict[str, Any], *, label: str) -> None:
    objective_risk = _int(
        source.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_before = _int(
        source.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_after = _int(
        source.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_delta = _int(source.get("source_outside_soloing_repair_source_pitch_role_risk_delta"))
    current_after = _int(source.get("source_outside_soloing_repair_pitch_role_risk_count_after"))
    current_delta = _int(source.get("source_outside_soloing_repair_pitch_role_risk_delta"))
    if objective_risk <= 0 or source_before <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} source pitch-role risk context required"
        )
    if objective_risk != source_before:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} objective/source risk mismatch"
        )
    if source_before - source_after != source_delta:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} source pitch-role risk delta mismatch"
        )
    if source_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} positive source pitch-role risk delta required"
        )
    if bool(source.get("source_outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} source-targeted flag should remain false"
        )
    if not bool(source.get("source_outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} source residual risk preservation required"
        )
    if current_after != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} current repair residual pitch-role risk should be zero"
        )
    if current_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"{label} positive current repair pitch-role risk delta required"
        )


def validate_objective_next_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if str(report.get("boundary") or "") != OBJECTIVE_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "songlike phrase/rhythm objective-only next decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != FOLLOWUP_DECISION_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "objective-only next decision must route to follow-up decision"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "follow-up decision boundary mismatch"
        )
    if not bool(readiness.get("objective_next_decision_completed", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "objective next decision completion required"
        )
    if not bool(readiness.get("phrase_rhythm_followup_required", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "songlike phrase/rhythm follow-up requirement required"
        )
    if bool(summary.get("validated_review_input_present", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "follow-up decision expects pending listening input"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "preference fill must remain blocked"
        )
    if bool(summary.get("current_quality_claim_ready", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "quality claim readiness must remain false"
        )
    if not bool(summary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "technical WAV validation required"
        )
    if not bool(summary.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "objective outside-soloing repair evidence should be ready"
        )
    if _int(summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "objective source outside-soloing repair pitch-role risk should be resolved"
        )
    source_context = _source_context_fields(summary)
    _validate_source_context(source_context, label="objective source outside-soloing repair")
    if _int(summary.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "objective source outside-soloing not-evaluable count should be preserved"
        )
    if _int(summary.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "objective repaired outside-soloing not-evaluable count should be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="objective next readiness")
    return {
        "boundary": OBJECTIVE_NEXT_BOUNDARY,
        "review_item_count": _int(summary.get("review_item_count")),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "failure_label_delta": _int(summary.get("failure_label_delta")),
        "phrase_rhythm_failure_delta": _int(summary.get("phrase_rhythm_failure_delta")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "source_outside_soloing_repair_evidence_ready": bool(
            summary.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        **source_context,
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_not_evaluable_count": _int(
            summary.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            summary.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(summary.get("repaired_not_evaluable_counts")),
        "validated_review_input_present": bool(
            summary.get("validated_review_input_present", False)
        ),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", False)),
        "current_quality_claim_ready": False,
    }


def _candidate_label_counts(rows: list[Any], key: str) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row_value in rows:
        row = _dict(row_value)
        labeling = _dict(
            row.get("phrase_rhythm_repaired_labeling")
            or row.get("contour_repaired_labeling")
            or row.get("repaired_labeling")
        )
        labels = _list(labeling.get(key))
        for label in labels:
            counts[str(label)] += 1
    return dict(sorted(counts.items()))


def validate_repair_sweep_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    rows = _list(report.get("candidate_repairs"))
    if str(report.get("boundary") or "") != REPAIR_SWEEP_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "songlike melody contour phrase/rhythm repair sweep boundary required"
        )
    if not bool(readiness.get("songlike_melody_contour_phrase_rhythm_repair_sweep_completed", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "repair sweep completion required"
        )
    if not bool(readiness.get("songlike_melody_contour_phrase_rhythm_repair_target_supported", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "songlike phrase/rhythm repair support required"
        )
    if _int(aggregate.get("candidate_count")) < 6:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "candidate count below 6"
        )
    if _int(aggregate.get("failure_label_delta")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "positive failure label delta required"
        )
    if _int(aggregate.get("phrase_rhythm_failure_delta")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "positive phrase/rhythm failure delta required"
        )
    if _int(aggregate.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "technical regression must remain zero"
        )
    if not bool(aggregate.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "repair sweep outside-soloing repair evidence should be ready"
        )
    if _int(aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "repair sweep source outside-soloing repair pitch-role risk should be resolved"
        )
    source_context = _source_context_fields(aggregate)
    _validate_source_context(source_context, label="repair sweep source outside-soloing repair")
    if _int(aggregate.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "repair sweep source outside-soloing not-evaluable count should be preserved"
        )
    if _int(aggregate.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "repair sweep repaired outside-soloing not-evaluable count should be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="repair sweep readiness")

    remaining_counts = {
        str(label): _int(count)
        for label, count in _dict(aggregate.get("repaired_failure_counts")).items()
        if _int(count) > 0
    }
    if not remaining_counts:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "remaining failure label counts required"
        )
    primary_count = max(remaining_counts.values())
    primary_labels = [
        label
        for label, count in sorted(remaining_counts.items())
        if int(count) == int(primary_count)
    ]
    if not rows:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "candidate repair rows required"
        )
    not_evaluable_counts = _candidate_label_counts(rows, "not_evaluable_labels")
    context_target_supported = all(
        _int(not_evaluable_counts.get(label)) >= _int(aggregate.get("candidate_count"))
        for label in CONTEXT_TARGET_LABELS
    )
    return {
        "boundary": REPAIR_SWEEP_BOUNDARY,
        "candidate_count": _int(aggregate.get("candidate_count")),
        "source_total_failure_label_count": _int(
            aggregate.get("source_total_failure_label_count")
        ),
        "repaired_total_failure_label_count": _int(
            aggregate.get("repaired_total_failure_label_count")
        ),
        "failure_label_delta": _int(aggregate.get("failure_label_delta")),
        "source_phrase_rhythm_failure_count": _int(
            aggregate.get("source_phrase_rhythm_failure_count")
        ),
        "repaired_phrase_rhythm_failure_count": _int(
            aggregate.get("repaired_phrase_rhythm_failure_count")
        ),
        "phrase_rhythm_failure_delta": _int(aggregate.get("phrase_rhythm_failure_delta")),
        "improved_candidate_count": _int(aggregate.get("improved_candidate_count")),
        "technical_regression_count": _int(aggregate.get("technical_regression_count")),
        "source_outside_soloing_repair_evidence_ready": bool(
            aggregate.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        **source_context,
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_not_evaluable_count": _int(
            aggregate.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            aggregate.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(aggregate.get("repaired_not_evaluable_counts")),
        "remaining_failure_counts": remaining_counts,
        "primary_remaining_failure_labels": primary_labels,
        "primary_remaining_failure_count": primary_count,
        "residual_rhythmic_monotony_count": _int(
            remaining_counts.get(RESIDUAL_FAILURE_LABEL)
        ),
        "context_target_labels": list(CONTEXT_TARGET_LABELS),
        "context_not_evaluable_target_supported": context_target_supported,
        "context_not_evaluable_min_count": min(
            _int(not_evaluable_counts.get(label)) for label in CONTEXT_TARGET_LABELS
        ),
        "not_evaluable_counts": not_evaluable_counts,
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
    primary_labels = [str(label) for label in _list(sweep["primary_remaining_failure_labels"])]
    context_target_supported = bool(sweep["context_not_evaluable_target_supported"])
    selected_target = (
        SELECTED_TARGET
        if context_target_supported
        else "songlike_melody_contour_phrase_rhythm_residual_rhythm_repair_sweep"
    )
    next_boundary = (
        NEXT_BOUNDARY
        if context_target_supported
        else "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_residual_rhythm_repair_sweep"
    )
    reason = (
        "chord-context not-evaluable labels dominate after phrase/rhythm repair"
        if context_target_supported
        else "residual rhythmic failure remains without full chord-context not-evaluable support"
    )
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
            "selected_target": selected_target,
            "selected_next_boundary": next_boundary,
            "primary_remaining_failure_labels": primary_labels,
            "primary_remaining_failure_count": _int(
                sweep["primary_remaining_failure_count"]
            ),
            "context_target_labels": list(CONTEXT_TARGET_LABELS),
            "context_not_evaluable_min_count": _int(
                sweep["context_not_evaluable_min_count"]
            ),
            "reason": reason,
        },
        "followup_targets": {
            "primary_labels": primary_labels,
            "secondary_failure_counts": sweep["remaining_failure_counts"],
            "not_evaluable_counts": sweep["not_evaluable_counts"],
            "context_target_labels": list(CONTEXT_TARGET_LABELS),
            "residual_failure_label": RESIDUAL_FAILURE_LABEL,
            "preserve_gates": [
                "grammar_valid",
                "strict_valid",
                "phrase_rhythm_failure_delta_positive",
                "technical_regression_count_zero",
                "chord_context_available",
                "pitch_role_metrics_defined",
                "no_quality_claim",
            ],
        },
        "readiness": {
            "boundary": BOUNDARY,
            "followup_decision_completed": True,
            "chord_context_pitch_role_bridge_selected": context_target_supported,
            "candidate_count": _int(sweep["candidate_count"]),
            "source_total_failure_label_count": _int(
                sweep["source_total_failure_label_count"]
            ),
            "repaired_total_failure_label_count": _int(
                sweep["repaired_total_failure_label_count"]
            ),
            "failure_label_delta": _int(sweep["failure_label_delta"]),
            "source_phrase_rhythm_failure_count": _int(
                sweep["source_phrase_rhythm_failure_count"]
            ),
            "repaired_phrase_rhythm_failure_count": _int(
                sweep["repaired_phrase_rhythm_failure_count"]
            ),
            "phrase_rhythm_failure_delta": _int(sweep["phrase_rhythm_failure_delta"]),
            "residual_rhythmic_monotony_count": _int(
                sweep["residual_rhythmic_monotony_count"]
            ),
            "context_not_evaluable_min_count": _int(
                sweep["context_not_evaluable_min_count"]
            ),
            "objective_source_outside_soloing_repair_evidence_ready": bool(
                objective["source_outside_soloing_repair_evidence_ready"]
            ),
            "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
                objective["source_outside_soloing_repair_source_objective_pitch_role_risk_count"]
            ),
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
                objective["source_outside_soloing_repair_source_pitch_role_risk_count_before"]
            ),
            "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
                objective["source_outside_soloing_repair_source_pitch_role_risk_count_after"]
            ),
            "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
                objective["source_outside_soloing_repair_source_pitch_role_risk_delta"]
            ),
            "objective_source_outside_soloing_repair_source_targeted": bool(
                objective["source_outside_soloing_repair_source_targeted"]
            ),
            "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
                objective["source_outside_soloing_repair_source_residual_risk_preserved"]
            ),
            "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                objective["source_outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
                objective["source_outside_soloing_repair_pitch_role_risk_delta"]
            ),
            "objective_source_outside_soloing_not_evaluable_count": _int(
                objective["source_outside_soloing_not_evaluable_count"]
            ),
            "objective_repaired_outside_soloing_not_evaluable_count": _int(
                objective["repaired_outside_soloing_not_evaluable_count"]
            ),
            "repair_sweep_source_outside_soloing_repair_evidence_ready": bool(
                sweep["source_outside_soloing_repair_evidence_ready"]
            ),
            "repair_sweep_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
                sweep["source_outside_soloing_repair_source_objective_pitch_role_risk_count"]
            ),
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
                sweep["source_outside_soloing_repair_source_pitch_role_risk_count_before"]
            ),
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
                sweep["source_outside_soloing_repair_source_pitch_role_risk_count_after"]
            ),
            "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
                sweep["source_outside_soloing_repair_source_pitch_role_risk_delta"]
            ),
            "repair_sweep_source_outside_soloing_repair_source_targeted": bool(
                sweep["source_outside_soloing_repair_source_targeted"]
            ),
            "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved": bool(
                sweep["source_outside_soloing_repair_source_residual_risk_preserved"]
            ),
            "repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                sweep["source_outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            "repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta": _int(
                sweep["source_outside_soloing_repair_pitch_role_risk_delta"]
            ),
            "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
                sweep["source_outside_soloing_not_evaluable_count"]
            ),
            "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
                sweep["repaired_outside_soloing_not_evaluable_count"]
            ),
            "technical_regression_count": _int(sweep["technical_regression_count"]),
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
            "reason": reason,
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "outside_soloing_without_context",
            "weak_chord_tone_landing",
            "broad_trained_model_quality",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-context pitch-role bridge source-context refresh"
        if next_boundary == NEXT_BOUNDARY
        else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm residual rhythm repair sweep",
    }


def validate_followup_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_followup_decision: bool,
    require_context_pitch_role_bridge: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    sweep = _dict(report.get("repair_sweep_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "unexpected next boundary"
        )
    if expected_target and str(selected.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "unexpected selected target"
        )
    if require_followup_decision and not bool(
        readiness.get("followup_decision_completed", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "follow-up decision completion required"
        )
    if require_context_pitch_role_bridge and not bool(
        readiness.get("chord_context_pitch_role_bridge_selected", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "chord-context pitch-role bridge selection required"
        )
    if _int(readiness.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "technical regression must remain zero"
        )
    if _int(readiness.get("failure_label_delta")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "positive failure label delta required"
        )
    if _int(readiness.get("phrase_rhythm_failure_delta")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "positive phrase/rhythm failure delta required"
        )
    if require_context_pitch_role_bridge and _int(
        selected.get("context_not_evaluable_min_count")
    ) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
            "context not-evaluable target count required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairFollowupDecisionError(
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
        "chord_context_pitch_role_bridge_selected": bool(
            readiness.get("chord_context_pitch_role_bridge_selected", False)
        ),
        "primary_remaining_failure_labels": [
            str(label) for label in _list(selected.get("primary_remaining_failure_labels"))
        ],
        "primary_remaining_failure_count": _int(
            selected.get("primary_remaining_failure_count")
        ),
        "candidate_count": _int(readiness.get("candidate_count")),
        "source_total_failure_label_count": _int(
            readiness.get("source_total_failure_label_count")
        ),
        "repaired_total_failure_label_count": _int(
            readiness.get("repaired_total_failure_label_count")
        ),
        "failure_label_delta": _int(readiness.get("failure_label_delta")),
        "source_phrase_rhythm_failure_count": _int(
            readiness.get("source_phrase_rhythm_failure_count")
        ),
        "repaired_phrase_rhythm_failure_count": _int(
            readiness.get("repaired_phrase_rhythm_failure_count")
        ),
        "phrase_rhythm_failure_delta": _int(
            readiness.get("phrase_rhythm_failure_delta")
        ),
        "residual_rhythmic_monotony_count": _int(
            readiness.get("residual_rhythmic_monotony_count")
        ),
        "context_target_labels": [
            str(label) for label in _list(selected.get("context_target_labels"))
        ],
        "context_not_evaluable_min_count": _int(
            selected.get("context_not_evaluable_min_count")
        ),
        "technical_regression_count": _int(readiness.get("technical_regression_count")),
        "remaining_failure_counts": _dict(sweep.get("remaining_failure_counts")),
        "not_evaluable_counts": _dict(sweep.get("not_evaluable_counts")),
        "objective_source_outside_soloing_repair_evidence_ready": bool(
            readiness.get("objective_source_outside_soloing_repair_evidence_ready", False)
        ),
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            readiness.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            readiness.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            readiness.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            readiness.get("objective_source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            readiness.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            readiness.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            readiness.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            readiness.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "objective_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("objective_source_outside_soloing_not_evaluable_count")
        ),
        "objective_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("objective_repaired_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_source_outside_soloing_repair_evidence_ready": bool(
            readiness.get("repair_sweep_source_outside_soloing_repair_evidence_ready", False)
        ),
        "repair_sweep_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            readiness.get(
                "repair_sweep_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            readiness.get(
                "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            readiness.get(
                "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            readiness.get("repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "repair_sweep_source_outside_soloing_repair_source_targeted": bool(
            readiness.get("repair_sweep_source_outside_soloing_repair_source_targeted", True)
        ),
        "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            readiness.get(
                "repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            readiness.get("repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            readiness.get("repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "repair_sweep_source_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_source_outside_soloing_not_evaluable_count")
        ),
        "repair_sweep_repaired_outside_soloing_not_evaluable_count": _int(
            readiness.get("repair_sweep_repaired_outside_soloing_not_evaluable_count")
        ),
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
    sweep = report["repair_sweep_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Follow-Up Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- repair sweep boundary: `{report['repair_sweep_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- primary remaining failure labels: `{','.join(selected['primary_remaining_failure_labels'])}`",
        f"- primary remaining failure count: `{selected['primary_remaining_failure_count']}`",
        f"- chord-context pitch-role bridge selected: `{_bool_token(readiness['chord_context_pitch_role_bridge_selected'])}`",
        f"- candidate count: `{readiness['candidate_count']}`",
        f"- source total failure labels: `{readiness['source_total_failure_label_count']}`",
        f"- repaired total failure labels: `{readiness['repaired_total_failure_label_count']}`",
        f"- failure label delta: `{readiness['failure_label_delta']}`",
        f"- phrase/rhythm failure count: `{readiness['source_phrase_rhythm_failure_count']} -> {readiness['repaired_phrase_rhythm_failure_count']}`",
        f"- phrase/rhythm failure delta: `{readiness['phrase_rhythm_failure_delta']}`",
        f"- residual rhythmic monotony count: `{readiness['residual_rhythmic_monotony_count']}`",
        f"- context not-evaluable min count: `{readiness['context_not_evaluable_min_count']}`",
        f"- objective source/repaired outside-soloing not evaluable count: `{readiness['objective_source_outside_soloing_not_evaluable_count']}/{readiness['objective_repaired_outside_soloing_not_evaluable_count']}`",
        f"- objective source outside-soloing source pitch-role risk before / after / delta: `{readiness['objective_source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{readiness['objective_source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{readiness['objective_source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- objective source outside-soloing source repair targeted: `{_bool_token(readiness['objective_source_outside_soloing_repair_source_targeted'])}`",
        f"- objective source outside-soloing source residual risk preserved: `{_bool_token(readiness['objective_source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- repair sweep source/repaired outside-soloing not evaluable count: `{readiness['repair_sweep_source_outside_soloing_not_evaluable_count']}/{readiness['repair_sweep_repaired_outside_soloing_not_evaluable_count']}`",
        f"- repair sweep source outside-soloing source pitch-role risk before / after / delta: `{readiness['repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{readiness['repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{readiness['repair_sweep_source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- repair sweep source outside-soloing source repair targeted: `{_bool_token(readiness['repair_sweep_source_outside_soloing_repair_source_targeted'])}`",
        f"- repair sweep source outside-soloing source residual risk preserved: `{_bool_token(readiness['repair_sweep_source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- objective source outside-soloing repair pitch-role risk count after: `{readiness['objective_source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- objective source outside-soloing current repair pitch-role risk delta: `{readiness['objective_source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- repair sweep source outside-soloing repair pitch-role risk count after: `{readiness['repair_sweep_source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- repair sweep source outside-soloing current repair pitch-role risk delta: `{readiness['repair_sweep_source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- technical regression count: `{readiness['technical_regression_count']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Remaining Failure Counts",
        "",
    ]
    for label, count in sweep["remaining_failure_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(
        [
            "",
            "## Not Evaluable Counts",
            "",
        ]
    )
    for label, count in sweep["not_evaluable_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(
        [
            "",
            "## Context Target Labels",
            "",
        ]
    )
    for label in selected["context_target_labels"]:
        lines.append(f"- `{label}`")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- reason: `{decision['reason']}`",
            f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
            f"- next recommended issue: `{report['next_recommended_issue']}`",
            "",
            "## Claim Boundary",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide songlike melody contour phrase/rhythm repair follow-up target"
    )
    parser.add_argument("--objective_next_report", type=str, required=True)
    parser.add_argument("--repair_sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=954)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_followup_decision", action="store_true")
    parser.add_argument("--require_context_pitch_role_bridge", action="store_true")
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
        require_context_pitch_role_bridge=bool(args.require_context_pitch_role_bridge),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
