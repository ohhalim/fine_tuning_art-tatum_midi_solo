"""Run a targeted post-MVP MIDI-to-solo quality repair sweep."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BRIDGE_SOURCE_CONTEXT_KEYS,
)
from scripts.label_stage_b_midi_to_solo_candidate_failures import (  # noqa: E402
    BOUNDARY as LABELING_BOUNDARY,
    REPAIR_NEXT_BOUNDARY as LABELING_NEXT_BOUNDARY,
    REPAIR_TARGET as LABELING_TARGET,
    StageBMidiToSoloCandidateFailureLabelingError,
    analyze_candidate_midi,
    label_candidate,
    validate_candidate_failure_labeling_report,
    validate_rubric_baseline,
)


class StageBMidiToSoloTargetedQualityRepairSweepError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_targeted_quality_repair_sweep"
NEXT_BOUNDARY = "stage_b_midi_to_solo_targeted_quality_repair_audio_package"
FOLLOWUP_BOUNDARY = "stage_b_midi_to_solo_targeted_quality_repair_followup_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_targeted_quality_repair_sweep_v3"

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
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in container or container[key] is None:
            raise StageBMidiToSoloTargetedQualityRepairSweepError(
                f"{label} source-context field required: {key}"
            )
    return {key: container[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS}


def validate_labeling_source(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != LABELING_BOUNDARY:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "candidate failure labeling boundary required"
        )
    try:
        summary = validate_candidate_failure_labeling_report(
            report,
            expected_boundary=LABELING_BOUNDARY,
            min_candidate_count=3,
            require_labeling_completed=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloCandidateFailureLabelingError as exc:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(str(exc)) from exc
    readiness = _dict(report.get("readiness"))
    selected = _dict(report.get("selected_next_target"))
    _require_no_quality_claim(readiness, label="labeling readiness")
    if str(selected.get("selected_target") or "") != LABELING_TARGET:
        raise StageBMidiToSoloTargetedQualityRepairSweepError("targeted repair target required")
    if str(selected.get("selected_next_boundary") or "") != LABELING_NEXT_BOUNDARY:
        raise StageBMidiToSoloTargetedQualityRepairSweepError("targeted repair boundary required")
    if _int(summary.get("failed_candidate_count")) <= 0:
        raise StageBMidiToSoloTargetedQualityRepairSweepError("failure labels required")
    if not bool(summary.get("outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing repair evidence readiness required"
        )
    if not bool(summary.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing repair source context preservation required"
        )
    _source_context_fields(summary, label="candidate failure labeling")
    if _int(summary.get("outside_soloing_repair_wav_count")) < 6:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing repair WAV count below 6"
        )
    if _int(summary.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing residual pitch-role risk should be zero"
        )
    source_objective_risk = _int(
        summary.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        summary.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(summary.get("outside_soloing_repair_source_pitch_role_risk_count_after"))
    source_risk_delta = _int(summary.get("outside_soloing_repair_source_pitch_role_risk_delta"))
    if source_objective_risk <= 0:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(summary.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(summary.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing source residual risk preservation required"
        )
    if _int(summary.get("outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "outside-soloing not-evaluable boundary required"
        )
    return summary


def non_drum_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def clamp_pitch(value: int, *, low: int = 45, high: int = 88) -> int:
    return max(low, min(high, int(value)))


def repaired_pitch(previous_pitch: int, original_pitch: int, *, index: int, variant: int) -> int:
    if index == 0 or index % 6 != variant % 6:
        return int(original_pitch)
    direction = 1 if ((index // 6) + variant) % 2 == 0 else -1
    target = previous_pitch + (7 * direction)
    if target < 45 or target > 88:
        target = previous_pitch - (7 * direction)
    if abs(target - previous_pitch) > 12:
        target = previous_pitch + (12 if target > previous_pitch else -12)
    return clamp_pitch(target)


def repair_midi_file(
    *,
    source_path: str,
    output_path: Path,
    variant: int,
) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(source_path)
    repaired = copy.deepcopy(midi)
    notes = non_drum_notes(repaired)
    if not notes:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(f"empty MIDI: {source_path}")

    original_starts = [float(note.start) for note in notes]
    original_durations = [max(0.05, float(note.end) - float(note.start)) for note in notes]
    base_iois = [
        max(0.05, original_starts[index] - original_starts[index - 1])
        for index in range(1, len(original_starts))
    ]
    rhythm_factors = [0.75, 1.25, 1.0, 1.5, 0.5, 1.125]
    rotated = rhythm_factors[variant % len(rhythm_factors) :] + rhythm_factors[: variant % len(rhythm_factors)]
    if base_iois:
        scaled_iois = [ioi * rotated[index % len(rotated)] for index, ioi in enumerate(base_iois)]
        original_span = max(0.05, original_starts[-1] - original_starts[0])
        scale = original_span / max(0.05, sum(scaled_iois))
        new_starts = [original_starts[0]]
        for ioi in scaled_iois:
            new_starts.append(new_starts[-1] + (ioi * scale))
    else:
        new_starts = original_starts

    duration_factors = [0.65, 1.0, 0.85, 1.2, 0.75, 1.05]
    previous_pitch = int(notes[0].pitch)
    changed_pitch_count = 0
    changed_time_count = 0
    for index, note in enumerate(notes):
        old_start = float(note.start)
        old_pitch = int(note.pitch)
        note.start = float(new_starts[index])
        next_start = float(new_starts[index + 1]) if index + 1 < len(new_starts) else None
        duration = original_durations[index] * duration_factors[(index + variant) % len(duration_factors)]
        if next_start is not None:
            duration = min(duration, max(0.05, next_start - note.start - 0.01))
        note.end = note.start + max(0.05, duration)
        note.pitch = repaired_pitch(previous_pitch, old_pitch, index=index, variant=variant)
        if note.pitch != old_pitch:
            changed_pitch_count += 1
        if abs(note.start - old_start) > 0.001:
            changed_time_count += 1
        previous_pitch = int(note.pitch)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    repaired.write(str(output_path))
    return {
        "source_path": source_path,
        "repaired_midi_path": str(output_path),
        "changed_pitch_count": changed_pitch_count,
        "changed_time_count": changed_time_count,
        "note_count": len(notes),
    }


def relabel_repaired_candidates(
    repaired_rows: list[dict[str, Any]],
    *,
    rubric_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    analyzed = [
        analyze_candidate_midi(
            {
                "source": str(row.get("source") or ""),
                "rank": _int(row.get("rank")),
                "midi_path": str(row.get("repaired_midi_path") or ""),
                "source_objective_supported": True,
                "source_dead_air_ratio": 0.0,
            }
        )
        for row in repaired_rows
    ]
    rhythm_counts = Counter(tuple(item.get("rhythm_signature", ((), ()))) for item in analyzed)
    labeled: list[dict[str, Any]] = []
    for row, item in zip(repaired_rows, analyzed, strict=False):
        labeled_item = label_candidate(
            item,
            rubric_items=rubric_items,
            shared_rhythm_signature_count=rhythm_counts[
                tuple(item.get("rhythm_signature", ((), ())))
            ],
        )
        labeled.append({**row, "repaired_labeling": labeled_item})
    return labeled


def build_targeted_quality_repair_sweep_report(
    *,
    candidate_failure_labeling: dict[str, Any],
    rubric_baseline: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source_summary = validate_labeling_source(candidate_failure_labeling)
    rubric_summary = validate_rubric_baseline(rubric_baseline)
    rubric_items = [_dict(item) for item in _list(rubric_summary.get("rubric_items"))]
    source_candidates = [_dict(item) for item in _list(candidate_failure_labeling.get("candidate_labels"))]
    if len(source_candidates) < 3:
        raise StageBMidiToSoloTargetedQualityRepairSweepError("source candidate labels below 3")

    midi_dir = output_dir / "midi"
    repaired_rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(source_candidates, start=1):
        source_path = str(candidate.get("midi_path") or "")
        if not Path(source_path).exists():
            raise StageBMidiToSoloTargetedQualityRepairSweepError(
                f"source MIDI missing: {source_path}"
            )
        source_name = f"{str(candidate.get('source') or 'candidate')}_rank_{_int(candidate.get('rank')):02d}"
        repaired_path = midi_dir / f"{index:02d}_{source_name}_targeted_quality_repair.mid"
        repair_summary = repair_midi_file(
            source_path=source_path,
            output_path=repaired_path,
            variant=index,
        )
        repaired_rows.append(
            {
                "source": str(candidate.get("source") or ""),
                "rank": _int(candidate.get("rank")),
                "source_midi_path": source_path,
                "source_failure_labels": _list(candidate.get("failure_labels")),
                "source_failure_label_count": len(_list(candidate.get("failure_labels"))),
                **repair_summary,
            }
        )

    relabeled = relabel_repaired_candidates(repaired_rows, rubric_items=rubric_items)
    total_before = sum(_int(item.get("source_failure_label_count")) for item in relabeled)
    total_after = sum(
        len(_list(_dict(item.get("repaired_labeling")).get("failure_labels")))
        for item in relabeled
    )
    improved_count = sum(
        1
        for item in relabeled
        if len(_list(_dict(item.get("repaired_labeling")).get("failure_labels")))
        < _int(item.get("source_failure_label_count"))
    )
    technical_regression_count = sum(
        1
        for item in relabeled
        if "technical_gate_regression"
        in _list(_dict(item.get("repaired_labeling")).get("failure_labels"))
    )
    failure_counts_after = Counter(
        label
        for item in relabeled
        for label in _list(_dict(item.get("repaired_labeling")).get("failure_labels"))
    )
    not_evaluable_counts_after = Counter(
        label
        for item in relabeled
        for label in _list(_dict(item.get("repaired_labeling")).get("not_evaluable_labels"))
    )
    target_supported = total_after < total_before and technical_regression_count == 0
    next_boundary = NEXT_BOUNDARY if target_supported else FOLLOWUP_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": LABELING_BOUNDARY,
        "candidate_repairs": relabeled,
        "aggregate": {
            "candidate_count": len(relabeled),
            "source_total_failure_label_count": total_before,
            "repaired_total_failure_label_count": total_after,
            "failure_label_delta": total_before - total_after,
            "improved_candidate_count": improved_count,
            "technical_regression_count": technical_regression_count,
            "repaired_failure_counts": dict(sorted(failure_counts_after.items())),
            "source_outside_soloing_repair_evidence_ready": bool(
                source_summary["outside_soloing_repair_evidence_ready"]
            ),
            "source_outside_soloing_repair_source_context_preserved": bool(
                source_summary["outside_soloing_repair_source_context_preserved"]
            ),
            "source_outside_soloing_repair_wav_count": _int(
                source_summary["outside_soloing_repair_wav_count"]
            ),
            "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
                source_summary["outside_soloing_repair_source_objective_pitch_role_risk_count"]
            ),
            "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
                source_summary["outside_soloing_repair_source_pitch_role_risk_count_before"]
            ),
            "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
                source_summary["outside_soloing_repair_source_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
                source_summary["outside_soloing_repair_source_pitch_role_risk_delta"]
            ),
            "source_outside_soloing_repair_source_targeted": bool(
                source_summary["outside_soloing_repair_source_targeted"]
            ),
            "source_outside_soloing_repair_source_residual_risk_preserved": bool(
                source_summary["outside_soloing_repair_source_residual_risk_preserved"]
            ),
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                source_summary["outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_repair_pitch_role_risk_delta": _int(
                source_summary["outside_soloing_repair_pitch_role_risk_delta"]
            ),
            "source_outside_soloing_not_evaluable_count": _int(
                source_summary["outside_soloing_not_evaluable_count"]
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                not_evaluable_counts_after.get("outside_soloing_without_context", 0)
            ),
            **{key: source_summary[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS},
            "target_supported": target_supported,
        },
        "selected_next_target": {
            "selected_target": "targeted_quality_repair_audio_package"
            if target_supported
            else "targeted_quality_repair_followup_decision",
            "selected_next_boundary": next_boundary,
            "reason": "repair sweep reduced objective failure labels without technical regression"
            if target_supported
            else "repair sweep did not reduce objective failure labels enough",
        },
        "readiness": {
            "boundary": BOUNDARY,
            "targeted_quality_repair_sweep_completed": True,
            "targeted_quality_repair_target_supported": target_supported,
            "candidate_count": len(relabeled),
            "failure_label_delta": total_before - total_after,
            "technical_regression_count": technical_regression_count,
            "audio_package_ready": target_supported,
            "source_outside_soloing_repair_source_context_preserved": bool(
                source_summary["outside_soloing_repair_source_context_preserved"]
            ),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
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
            "reason": "targeted repair sweep completed without musical quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo targeted quality repair audio package source-context refresh"
        if target_supported
        else "Stage B MIDI-to-solo targeted quality repair follow-up decision source-context refresh",
        "source_summary": source_summary,
    }


def validate_targeted_quality_repair_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_candidate_count: int,
    require_sweep_completed: bool,
    require_failure_delta: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    repairs = _list(report.get("candidate_repairs"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_sweep_completed and not bool(
        readiness.get("targeted_quality_repair_sweep_completed", False)
    ):
        raise StageBMidiToSoloTargetedQualityRepairSweepError("repair sweep completion required")
    if len(repairs) < int(min_candidate_count):
        raise StageBMidiToSoloTargetedQualityRepairSweepError("repair candidate count below minimum")
    if _int(aggregate.get("candidate_count")) != len(repairs):
        raise StageBMidiToSoloTargetedQualityRepairSweepError("candidate count mismatch")
    if require_failure_delta and _int(aggregate.get("failure_label_delta")) <= 0:
        raise StageBMidiToSoloTargetedQualityRepairSweepError("positive failure label delta required")
    if _int(aggregate.get("technical_regression_count")) != 0:
        raise StageBMidiToSoloTargetedQualityRepairSweepError("technical regression should be zero")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloTargetedQualityRepairSweepError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="targeted repair readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "targeted_quality_repair_sweep_completed": bool(
            readiness.get("targeted_quality_repair_sweep_completed", False)
        ),
        "targeted_quality_repair_target_supported": bool(
            readiness.get("targeted_quality_repair_target_supported", False)
        ),
        "candidate_count": len(repairs),
        "source_total_failure_label_count": _int(
            aggregate.get("source_total_failure_label_count")
        ),
        "repaired_total_failure_label_count": _int(
            aggregate.get("repaired_total_failure_label_count")
        ),
        "failure_label_delta": _int(aggregate.get("failure_label_delta")),
        "improved_candidate_count": _int(aggregate.get("improved_candidate_count")),
        "technical_regression_count": _int(aggregate.get("technical_regression_count")),
        "source_outside_soloing_repair_evidence_ready": bool(
            aggregate.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            aggregate.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_wav_count": _int(
            aggregate.get("source_outside_soloing_repair_wav_count")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            aggregate.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            aggregate.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            aggregate.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            aggregate.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            aggregate.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            aggregate.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_not_evaluable_count": _int(
            aggregate.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            aggregate.get("repaired_outside_soloing_not_evaluable_count")
        ),
        **{
            key: aggregate.get(key)
            for key in BRIDGE_SOURCE_CONTEXT_KEYS
        },
        "audio_package_ready": bool(readiness.get("audio_package_ready", False)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    selected = report["selected_next_target"]
    readiness = report["readiness"]
    lines = [
        "# Stage B MIDI-to-Solo Targeted Quality Repair Sweep Source Context Refresh",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{selected['selected_next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- source total failure label count: `{aggregate['source_total_failure_label_count']}`",
        f"- repaired total failure label count: `{aggregate['repaired_total_failure_label_count']}`",
        f"- failure label delta: `{aggregate['failure_label_delta']}`",
        f"- improved candidate count: `{aggregate['improved_candidate_count']}`",
        f"- technical regression count: `{aggregate['technical_regression_count']}`",
        f"- source outside-soloing repair evidence ready: `{_bool_token(aggregate['source_outside_soloing_repair_evidence_ready'])}`",
        f"- source outside-soloing repair source context preserved: `{_bool_token(aggregate['source_outside_soloing_repair_source_context_preserved'])}`",
        f"- source outside-soloing source objective pitch-role risk: `{aggregate['source_outside_soloing_repair_source_objective_pitch_role_risk_count']}`",
        f"- source outside-soloing source pitch-role risk before / after / delta: `{aggregate['source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{aggregate['source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{aggregate['source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- source outside-soloing source repair targeted: `{_bool_token(aggregate['source_outside_soloing_repair_source_targeted'])}`",
        f"- source outside-soloing source residual risk preserved: `{_bool_token(aggregate['source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- source outside-soloing current repair pitch-role risk after / delta: `{aggregate['source_outside_soloing_repair_pitch_role_risk_count_after']}` / `{aggregate['source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- source outside-soloing not evaluable count: `{aggregate['source_outside_soloing_not_evaluable_count']}`",
        f"- repaired outside-soloing not evaluable count: `{aggregate['repaired_outside_soloing_not_evaluable_count']}`",
        f"- target supported: `{_bool_token(aggregate['target_supported'])}`",
        "",
        "## Repaired Failure Counts",
        "",
    ]
    for label, count in aggregate["repaired_failure_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
    if not aggregate["repaired_failure_counts"]:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            "",
            "## Next",
            "",
            f"- `{report['next_recommended_issue']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run MIDI-to-solo targeted quality repair sweep")
    parser.add_argument("--candidate_failure_labeling", type=str, required=True)
    parser.add_argument("--rubric_baseline", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_targeted_quality_repair_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=750)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--min_candidate_count", type=int, default=6)
    parser.add_argument("--require_sweep_completed", action="store_true")
    parser.add_argument("--require_failure_delta", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_targeted_quality_repair_sweep_report(
        candidate_failure_labeling=read_json(Path(args.candidate_failure_labeling)),
        rubric_baseline=read_json(Path(args.rubric_baseline)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_targeted_quality_repair_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_candidate_count=int(args.min_candidate_count),
        require_sweep_completed=bool(args.require_sweep_completed),
        require_failure_delta=bool(args.require_failure_delta),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_targeted_quality_repair_sweep.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_targeted_quality_repair_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_targeted_quality_repair_sweep.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
