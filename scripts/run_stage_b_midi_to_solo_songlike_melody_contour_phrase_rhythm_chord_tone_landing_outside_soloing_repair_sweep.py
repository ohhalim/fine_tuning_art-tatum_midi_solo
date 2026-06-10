"""Repair residual outside-soloing pitch-role risk after chord-tone landing repair."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    bridge_candidate,
    parse_chords,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup import (  # noqa: E402
    BOUNDARY as FOLLOWUP_BOUNDARY,
    NEXT_BOUNDARY as FOLLOWUP_NEXT_BOUNDARY,
    SELECTED_TARGET as FOLLOWUP_SELECTED_TARGET,
)
from scripts.run_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep import (  # noqa: E402
    BOUNDARY as CHORD_TONE_REPAIR_SWEEP_BOUNDARY,
    all_notes,
    nearest_chord_tone_pitch,
)
from scripts.run_stage_b_reference_stats import pitch_role_for_group  # noqa: E402
from scripts.stage_b_tokens import quantize_note_position  # noqa: E402


class StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio_package"
)
SELECTED_TARGET = (
    "songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio_package"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_sweep_v2"
)
OUTSIDE_RISK_FLAG = "outside_soloing_pitch_role_risk"
WEAK_LANDING_FLAG = "weak_chord_tone_landing_risk"
NON_CHORD_ROLES = {"approach", "outside", "unknown_chord"}
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
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
                f"{label} source-context field required: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_source_context_preserved",
        "followup_repair_sweep_source_outside_soloing_source_context_preserved",
        "repair_sweep_source_outside_soloing_source_context_preserved",
    ):
        if not bool(container.get(key, False)):
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
                f"{label} source context should be preserved: {key}"
            )
    return {key: container[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS}


def _absolute_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT_DIR / value


def validate_followup_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    if str(report.get("boundary") or "") != FOLLOWUP_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone landing repair follow-up decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != FOLLOWUP_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up decision must route to outside-soloing repair sweep"
        )
    if str(selected.get("selected_target") or "") != FOLLOWUP_SELECTED_TARGET:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up selected target mismatch"
        )
    if not bool(readiness.get("followup_decision_completed", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up decision completion required"
        )
    if not bool(readiness.get("outside_soloing_repair_selected", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "outside-soloing repair selection required"
        )
    if not bool(readiness.get("weak_chord_tone_landing_resolved", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "weak chord-tone landing resolution required"
        )
    if _int(readiness.get("primary_remaining_risk_count")) <= 0:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "primary remaining risk count required"
        )
    if _int(readiness.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        readiness.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up objective outside-soloing count must match source count"
        )
    if bool(readiness.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up source must preserve outside-soloing as untargeted"
        )
    if not bool(readiness.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up source must preserve outside-soloing residual risk context"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="follow-up readiness")
    source_context = _source_context_fields(readiness, label="follow-up readiness")
    return {
        "boundary": FOLLOWUP_BOUNDARY,
        "primary_remaining_risk_label": str(
            readiness.get("primary_remaining_risk_label") or OUTSIDE_RISK_FLAG
        ),
        "primary_remaining_risk_count": _int(
            readiness.get("primary_remaining_risk_count")
        ),
        "candidate_count": _int(readiness.get("candidate_count")),
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
        **source_context,
    }


def validate_chord_tone_repair_sweep_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    context = _dict(report.get("context"))
    rows = [_dict(row) for row in _list(report.get("candidate_repairs"))]
    if str(report.get("boundary") or "") != CHORD_TONE_REPAIR_SWEEP_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone landing repair sweep boundary required"
        )
    if not bool(readiness.get("chord_tone_landing_repair_sweep_completed", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone landing repair sweep completion required"
        )
    if not bool(readiness.get("target_supported", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone landing repair support required"
        )
    if _int(aggregate.get("candidate_count")) < 6 or len(rows) < 6:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "at least 6 chord-tone landing repair candidates required"
        )
    if _int(aggregate.get("weak_chord_tone_landing_risk_count_after")) != 0:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "weak chord-tone landing risk must remain cleared"
        )
    if _int(aggregate.get("outside_soloing_pitch_role_risk_count_after")) <= 0:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "residual outside-soloing pitch-role risk required"
        )
    if _int(aggregate.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        aggregate.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone sweep objective outside-soloing count must match source count"
        )
    if bool(aggregate.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone sweep source must preserve outside-soloing as untargeted"
        )
    if not bool(aggregate.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "chord-tone sweep source must preserve outside-soloing residual risk context"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="chord-tone landing repair sweep readiness")
    source_context = _source_context_fields(aggregate, label="chord-tone repair sweep aggregate")
    return {
        "boundary": CHORD_TONE_REPAIR_SWEEP_BOUNDARY,
        "rows": rows,
        "chord_progression": [
            str(chord) for chord in _list(context.get("chord_progression"))
        ] or parse_chords(""),
        "bpm": _float(context.get("bpm")) or 124.0,
        "repair_policy": str(context.get("repair_policy") or ""),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "changed_note_total": _int(aggregate.get("changed_note_total")),
        "weak_chord_tone_landing_risk_count_after": _int(
            aggregate.get("weak_chord_tone_landing_risk_count_after")
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
        "final_landing_chord_tone_count_after": _int(
            aggregate.get("final_landing_chord_tone_count_after")
        ),
        **source_context,
    }


def count_flag(rows: list[dict[str, Any]], side: str, flag: str) -> int:
    return sum(1 for row in rows if flag in _list(_dict(row.get(side)).get("bridge_flags")))


def repair_outside_soloing_midi(
    *,
    source_path: Path,
    output_path: Path,
    chords: list[str],
    bpm: float,
    bar_count: int,
    max_allowed_non_chord_run: int = 3,
) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(source_path))
    notes = all_notes(midi)
    current_run: list[dict[str, Any]] = []
    changed_positions: list[dict[str, Any]] = []
    for note in notes:
        bar, position = quantize_note_position(float(note.start), bpm)
        if int(bar) < 0 or int(bar) >= int(bar_count):
            current_run = []
            continue
        chord = chords[int(bar) % len(chords)] if chords else ""
        role = pitch_role_for_group(
            {"bar": int(bar), "position": int(position), "pitch": int(note.pitch)},
            chord,
        )
        if role in NON_CHORD_ROLES:
            current_run.append(
                {
                    "note": note,
                    "bar": int(bar),
                    "position": int(position),
                    "chord": chord,
                    "role": role,
                }
            )
        else:
            current_run = []
            continue
        if len(current_run) <= int(max_allowed_non_chord_run):
            continue
        target = current_run[-1]
        target_note = target["note"]
        original_pitch = int(target_note.pitch)
        repaired_pitch = nearest_chord_tone_pitch(original_pitch, str(target["chord"]))
        if repaired_pitch != original_pitch:
            target_note.pitch = int(repaired_pitch)
            changed_positions.append(
                {
                    "bar": int(target["bar"]),
                    "position": int(target["position"]),
                    "chord": str(target["chord"]),
                    "role_before": str(target["role"]),
                    "from": original_pitch,
                    "to": int(repaired_pitch),
                    "reason": "break_non_chord_tone_run",
                }
            )
        current_run = []
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return {
        "source_midi_path": str(source_path),
        "repaired_midi_path": str(output_path),
        "changed_note_count": len(changed_positions),
        "changed_positions": changed_positions,
        "max_allowed_non_chord_run": int(max_allowed_non_chord_run),
    }


def repair_candidate(
    *,
    row: dict[str, Any],
    output_dir: Path,
    chords: list[str],
    bpm: float,
    manifest_path: Path,
) -> dict[str, Any]:
    rank = _int(row.get("rank"))
    before_metrics = _dict(row.get("after"))
    source_path = _absolute_path(str(row.get("repaired_midi_path") or ""))
    if not source_path.exists():
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            f"candidate MIDI missing: {source_path}"
        )
    bar_count = 8
    output_path = output_dir / "midi" / f"{rank:02d}_outside_soloing_repair.mid"
    repair = repair_outside_soloing_midi(
        source_path=source_path,
        output_path=output_path,
        chords=chords,
        bpm=bpm,
        bar_count=bar_count,
    )
    bridge_row = {
        "rank": rank,
        "phrase_rhythm_repaired_midi_path": str(output_path),
        "phrase_rhythm_repaired_labeling": {
            "metrics": {"bar_count": bar_count},
            "failure_labels": [],
            "not_evaluable_labels": [],
        },
    }
    repaired = bridge_candidate(
        row=bridge_row,
        rank=rank,
        chords=chords,
        bpm=bpm,
        manifest_path=manifest_path,
    )
    after_metrics = _dict(repaired.get("bridge_metrics"))
    before_flags = [str(flag) for flag in _list(before_metrics.get("bridge_flags"))]
    after_flags = [str(flag) for flag in _list(repaired.get("bridge_flags"))]
    return {
        "rank": rank,
        "source_midi_path": str(source_path),
        "repaired_midi_path": str(output_path),
        "repair": repair,
        "before": {
            "chord_tone_ratio": _float(before_metrics.get("chord_tone_ratio")),
            "strong_beat_chord_tone_ratio": _float(
                before_metrics.get("strong_beat_chord_tone_ratio")
            ),
            "cadence_landing_chord_tone": bool(
                before_metrics.get("cadence_landing_chord_tone", False)
            ),
            "cadence_landing_role": str(before_metrics.get("cadence_landing_role") or ""),
            "max_non_chord_tone_run": _int(before_metrics.get("max_non_chord_tone_run")),
            "bridge_flags": before_flags,
        },
        "after": {
            "chord_tone_ratio": _float(after_metrics.get("chord_tone_ratio")),
            "strong_beat_chord_tone_ratio": _float(
                after_metrics.get("strong_beat_chord_tone_ratio")
            ),
            "cadence_landing_chord_tone": bool(
                after_metrics.get("cadence_landing_chord_tone", False)
            ),
            "cadence_landing_role": str(after_metrics.get("cadence_landing_role") or ""),
            "max_non_chord_tone_run": _int(after_metrics.get("max_non_chord_tone_run")),
            "bridge_flags": after_flags,
        },
    }


def build_outside_soloing_repair_sweep_report(
    *,
    followup_report: dict[str, Any],
    chord_tone_repair_sweep_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    followup = validate_followup_source(followup_report)
    source = validate_chord_tone_repair_sweep_source(chord_tone_repair_sweep_report)
    source_context_keys = [
        "objective_outside_soloing_pitch_role_risk_count",
        "outside_soloing_pitch_role_risk_count_before",
        "outside_soloing_pitch_role_risk_count_after",
        "outside_soloing_pitch_role_risk_delta",
    ]
    if any(_int(followup[key]) != _int(source[key]) for key in source_context_keys):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up and chord-tone sweep outside-soloing source counts must match"
        )
    if bool(followup["outside_soloing_repair_targeted"]) != bool(
        source["outside_soloing_repair_targeted"]
    ) or bool(followup["outside_soloing_residual_risk_preserved"]) != bool(
        source["outside_soloing_residual_risk_preserved"]
    ):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "follow-up and chord-tone sweep outside-soloing source flags must match"
        )
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if followup.get(key) != source.get(key):
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
                f"follow-up and chord-tone sweep source-context field mismatch: {key}"
            )
    chords = source["chord_progression"] or parse_chords("")
    bpm = _float(source["bpm"]) or 124.0
    manifest_path = ROOT_DIR / "stage_b_midi_to_solo_outside_soloing_repair_manifest.json"
    rows = [
        repair_candidate(
            row=row,
            output_dir=output_dir,
            chords=chords,
            bpm=bpm,
            manifest_path=manifest_path,
        )
        for row in source["rows"]
    ]
    outside_before = count_flag(rows, "before", OUTSIDE_RISK_FLAG)
    outside_after = count_flag(rows, "after", OUTSIDE_RISK_FLAG)
    weak_before = count_flag(rows, "before", WEAK_LANDING_FLAG)
    weak_after = count_flag(rows, "after", WEAK_LANDING_FLAG)
    final_before = sum(1 for row in rows if bool(_dict(row.get("before")).get("cadence_landing_chord_tone", False)))
    final_after = sum(1 for row in rows if bool(_dict(row.get("after")).get("cadence_landing_chord_tone", False)))
    changed_note_total = sum(
        _int(_dict(row.get("repair")).get("changed_note_count")) for row in rows
    )
    max_non_chord_run_before = max(
        _int(_dict(row.get("before")).get("max_non_chord_tone_run")) for row in rows
    )
    max_non_chord_run_after = max(
        _int(_dict(row.get("after")).get("max_non_chord_tone_run")) for row in rows
    )
    target_supported = outside_after < outside_before and weak_after == 0
    next_boundary = NEXT_BOUNDARY if target_supported else FOLLOWUP_BOUNDARY
    selected_target = SELECTED_TARGET if target_supported else FOLLOWUP_SELECTED_TARGET
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": followup["boundary"],
        "chord_tone_repair_sweep_boundary": source["boundary"],
        "context": {
            "chord_progression": chords,
            "bpm": bpm,
            "repair_policy": "break_four_note_non_chord_tone_run_with_nearest_chord_tone",
            "source_repair_policy": source["repair_policy"],
        },
        "candidate_repairs": rows,
        "aggregate": {
            "candidate_count": len(rows),
            "repaired_midi_count": len(rows),
            "changed_note_total": int(changed_note_total),
            "source_objective_outside_soloing_pitch_role_risk_count": _int(
                followup["objective_outside_soloing_pitch_role_risk_count"]
            ),
            "source_outside_soloing_pitch_role_risk_count_before": _int(
                followup["outside_soloing_pitch_role_risk_count_before"]
            ),
            "source_outside_soloing_pitch_role_risk_count_after": _int(
                followup["outside_soloing_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_pitch_role_risk_delta": _int(
                followup["outside_soloing_pitch_role_risk_delta"]
            ),
            "source_outside_soloing_repair_targeted": bool(
                followup["outside_soloing_repair_targeted"]
            ),
            "source_outside_soloing_residual_risk_preserved": bool(
                followup["outside_soloing_residual_risk_preserved"]
            ),
            "outside_soloing_pitch_role_risk_count_before": int(outside_before),
            "outside_soloing_pitch_role_risk_count_after": int(outside_after),
            "outside_soloing_pitch_role_risk_delta": int(outside_before - outside_after),
            "outside_soloing_repair_targeted": True,
            "weak_chord_tone_landing_risk_count_before": int(weak_before),
            "weak_chord_tone_landing_risk_count_after": int(weak_after),
            "final_landing_chord_tone_count_before": int(final_before),
            "final_landing_chord_tone_count_after": int(final_after),
            "max_non_chord_tone_run_before": int(max_non_chord_run_before),
            "max_non_chord_tone_run_after": int(max_non_chord_run_after),
            "target_supported": bool(target_supported),
            "source_primary_remaining_risk_count": int(
                followup["primary_remaining_risk_count"]
            ),
            **{key: followup.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
        },
        "readiness": {
            "boundary": BOUNDARY,
            "outside_soloing_repair_sweep_completed": True,
            "candidate_count": len(rows),
            "repaired_midi_count": len(rows),
            "target_supported": bool(target_supported),
            "source_objective_outside_soloing_pitch_role_risk_count": _int(
                followup["objective_outside_soloing_pitch_role_risk_count"]
            ),
            "source_outside_soloing_pitch_role_risk_count_before": _int(
                followup["outside_soloing_pitch_role_risk_count_before"]
            ),
            "source_outside_soloing_pitch_role_risk_count_after": _int(
                followup["outside_soloing_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_pitch_role_risk_delta": _int(
                followup["outside_soloing_pitch_role_risk_delta"]
            ),
            "source_outside_soloing_repair_targeted": bool(
                followup["outside_soloing_repair_targeted"]
            ),
            "source_outside_soloing_residual_risk_preserved": bool(
                followup["outside_soloing_residual_risk_preserved"]
            ),
            "outside_soloing_repair_targeted": True,
            "outside_soloing_pitch_role_risk_delta": int(outside_before - outside_after),
            "weak_chord_tone_landing_risk_count_after": int(weak_after),
            **{key: followup.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
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
            "selected_target": selected_target,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "outside-soloing repair sweep completed without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair audio package source-context refresh"
            if target_supported
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair follow-up decision source-context refresh"
        ),
    }


def validate_outside_soloing_repair_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_completed: bool,
    require_target_supported: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "unexpected next boundary"
        )
    if require_repair_completed and not bool(
        readiness.get("outside_soloing_repair_sweep_completed", False)
    ):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "outside-soloing repair sweep completion required"
        )
    if require_target_supported and not bool(readiness.get("target_supported", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "target support required"
        )
    if _int(aggregate.get("changed_note_total")) <= 0:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "changed note count required"
        )
    if _int(aggregate.get("outside_soloing_pitch_role_risk_delta")) <= 0:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "positive outside-soloing risk delta required"
        )
    if bool(aggregate.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(aggregate.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(aggregate.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "outside-soloing repair must be targeted in current sweep"
        )
    if _int(aggregate.get("weak_chord_tone_landing_risk_count_after")) != 0:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "weak chord-tone landing risk must remain zero"
        )
    if _int(aggregate.get("final_landing_chord_tone_count_after")) < _int(
        aggregate.get("final_landing_chord_tone_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "final landing chord-tone count must not regress"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="outside-soloing repair readiness")
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in aggregate:
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairSweepError(
                f"outside-soloing repair aggregate source-context field required: {key}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "chord_tone_repair_sweep_boundary": str(
            report.get("chord_tone_repair_sweep_boundary") or ""
        ),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "outside_soloing_repair_sweep_completed": bool(
            readiness.get("outside_soloing_repair_sweep_completed", False)
        ),
        "candidate_count": _int(aggregate.get("candidate_count")),
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "changed_note_total": _int(aggregate.get("changed_note_total")),
        "source_objective_outside_soloing_pitch_role_risk_count": _int(
            aggregate.get("source_objective_outside_soloing_pitch_role_risk_count")
        ),
        "source_outside_soloing_pitch_role_risk_count_before": _int(
            aggregate.get("source_outside_soloing_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_pitch_role_risk_count_after": _int(
            aggregate.get("source_outside_soloing_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_pitch_role_risk_delta": _int(
            aggregate.get("source_outside_soloing_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_targeted": bool(
            aggregate.get("source_outside_soloing_repair_targeted", True)
        ),
        "source_outside_soloing_residual_risk_preserved": bool(
            aggregate.get("source_outside_soloing_residual_risk_preserved", False)
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
            aggregate.get("outside_soloing_repair_targeted", False)
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            aggregate.get("weak_chord_tone_landing_risk_count_after")
        ),
        "final_landing_chord_tone_count_after": _int(
            aggregate.get("final_landing_chord_tone_count_after")
        ),
        "max_non_chord_tone_run_before": _int(
            aggregate.get("max_non_chord_tone_run_before")
        ),
        "max_non_chord_tone_run_after": _int(
            aggregate.get("max_non_chord_tone_run_after")
        ),
        "target_supported": bool(aggregate.get("target_supported", False)),
        **{key: aggregate.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
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
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    context = report["context"]
    lines = [
        "# Stage B MIDI-to-Solo Chord-Tone Landing Outside-Soloing Repair Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- chord-tone repair sweep boundary: `{report['chord_tone_repair_sweep_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- repair policy: `{context['repair_policy']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- repaired MIDI count: `{aggregate['repaired_midi_count']}`",
        f"- changed note total: `{aggregate['changed_note_total']}`",
        f"- source objective outside-soloing pitch-role risk count: `{aggregate['source_objective_outside_soloing_pitch_role_risk_count']}`",
        f"- source outside-soloing pitch-role risk count: `{aggregate['source_outside_soloing_pitch_role_risk_count_before']} -> {aggregate['source_outside_soloing_pitch_role_risk_count_after']}`",
        f"- source outside-soloing pitch-role risk delta: `{aggregate['source_outside_soloing_pitch_role_risk_delta']}`",
        f"- source outside-soloing repair targeted: `{_bool_token(aggregate['source_outside_soloing_repair_targeted'])}`",
        f"- source outside-soloing residual risk preserved: `{_bool_token(aggregate['source_outside_soloing_residual_risk_preserved'])}`",
        f"- outside-soloing pitch-role risk count: `{aggregate['outside_soloing_pitch_role_risk_count_before']} -> {aggregate['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing pitch-role risk delta: `{aggregate['outside_soloing_pitch_role_risk_delta']}`",
        f"- outside-soloing repair targeted: `{_bool_token(aggregate['outside_soloing_repair_targeted'])}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk delta: `{aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(aggregate['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up objective source outside-soloing source targeted: `{_bool_token(aggregate['followup_objective_source_outside_soloing_source_targeted'])}`",
        f"- follow-up objective source outside-soloing source residual risk preserved: `{_bool_token(aggregate['followup_objective_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk delta: `{aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(aggregate['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source targeted: `{_bool_token(aggregate['followup_repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- follow-up repair sweep source outside-soloing source residual risk preserved: `{_bool_token(aggregate['followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk delta: `{aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(aggregate['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source targeted: `{_bool_token(aggregate['repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- bridge repair sweep source outside-soloing source residual risk preserved: `{_bool_token(aggregate['repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- weak chord-tone landing risk count after: `{aggregate['weak_chord_tone_landing_risk_count_after']}`",
        f"- final landing chord-tone count after: `{aggregate['final_landing_chord_tone_count_after']}`",
        f"- max non-chord-tone run: `{aggregate['max_non_chord_tone_run_before']} -> {aggregate['max_non_chord_tone_run_after']}`",
        f"- target supported: `{_bool_token(aggregate['target_supported'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidate Summary",
        "",
    ]
    for row in report["candidate_repairs"]:
        before = row["before"]
        after = row["after"]
        repair = row["repair"]
        lines.append(
            "- rank `{rank}`: changed `{changed}`, max non-chord run `{before_run}` -> `{after_run}`, flags `{before_flags}` -> `{after_flags}`".format(
                rank=row["rank"],
                changed=repair["changed_note_count"],
                before_run=before["max_non_chord_tone_run"],
                after_run=after["max_non_chord_tone_run"],
                before_flags=",".join(before["bridge_flags"]) or "none",
                after_flags=",".join(after["bridge_flags"]) or "none",
            )
        )
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
        description="Run chord-tone landing outside-soloing repair sweep"
    )
    parser.add_argument("--followup_report", type=str, required=True)
    parser.add_argument("--chord_tone_repair_sweep_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_sweep"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1056)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_completed", action="store_true")
    parser.add_argument("--require_target_supported", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_outside_soloing_repair_sweep_report(
        followup_report=read_json(Path(args.followup_report)),
        chord_tone_repair_sweep_report=read_json(
            Path(args.chord_tone_repair_sweep_report)
        ),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_outside_soloing_repair_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repair_completed=bool(args.require_repair_completed),
        require_target_supported=bool(args.require_target_supported),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_sweep.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_sweep_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_sweep.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
