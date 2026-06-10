"""Repair weak chord-tone landing in phrase/rhythm candidates."""

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
    BOUNDARY as BRIDGE_BOUNDARY,
    BRIDGE_SOURCE_CONTEXT_KEYS,
    bridge_candidate,
    parse_chords,
)
from scripts.decide_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_objective import (  # noqa: E402
    BOUNDARY as OBJECTIVE_DECISION_BOUNDARY,
    NEXT_BOUNDARY as OBJECTIVE_DECISION_NEXT_BOUNDARY,
    SELECTED_TARGET as OBJECTIVE_DECISION_SELECTED_TARGET,
)
from scripts.run_stage_b_generation_probe import chord_pitch_classes  # noqa: E402
from scripts.run_stage_b_reference_stats import position_bucket  # noqa: E402
from scripts.stage_b_tokens import PIANO_PITCH_MAX, PIANO_PITCH_MIN, quantize_note_position  # noqa: E402


class StageBMidiToSoloChordToneLandingRepairSweepError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package"
)
SELECTED_TARGET = "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_audio_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep_v2"
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
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"{label} source-context field required: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_source_targeted",
        "followup_repair_sweep_source_outside_soloing_source_targeted",
        "repair_sweep_source_outside_soloing_source_targeted",
    ):
        if bool(container.get(key, True)):
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"{label} source target should remain false: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_source_residual_risk_preserved",
        "followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved",
        "repair_sweep_source_outside_soloing_source_residual_risk_preserved",
    ):
        if not bool(container.get(key, False)):
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"{label} source residual risk should be preserved: {key}"
            )
    for key in (
        "followup_objective_source_outside_soloing_current_pitch_role_risk_count_after",
        "followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after",
        "repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after",
    ):
        if _int(container.get(key)) != 0:
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"{label} current outside-soloing risk should remain resolved: {key}"
            )
    return {key: container[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS}


def _validate_source_context_consistency(
    objective: dict[str, Any], bridge: dict[str, Any]
) -> None:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if objective.get(key) != bridge.get(key):
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"objective and bridge source-context mismatch: {key}"
            )


def validate_objective_decision(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    if str(report.get("boundary") or "") != OBJECTIVE_DECISION_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "pitch-role objective decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != OBJECTIVE_DECISION_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "objective decision must route to chord-tone landing repair sweep"
        )
    if str(selected.get("selected_target") or "") != OBJECTIVE_DECISION_SELECTED_TARGET:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "objective decision selected target mismatch"
        )
    if not bool(readiness.get("pitch_role_objective_decision_completed", False)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "objective decision completion required"
        )
    if _int(readiness.get("weak_chord_tone_landing_risk_count")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "weak chord-tone landing risk required"
        )
    if _int(readiness.get("outside_soloing_pitch_role_risk_count")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "outside-soloing pitch-role risk context required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="objective decision readiness")
    source_context = _source_context_fields(readiness, label="objective decision")
    return {
        "boundary": OBJECTIVE_DECISION_BOUNDARY,
        "candidate_count": _int(readiness.get("candidate_count")),
        "weak_chord_tone_landing_risk_count": _int(
            readiness.get("weak_chord_tone_landing_risk_count")
        ),
        "outside_soloing_pitch_role_risk_count": _int(
            readiness.get("outside_soloing_pitch_role_risk_count")
        ),
        **source_context,
    }


def validate_bridge_source(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    context = _dict(report.get("context"))
    candidates = [_dict(candidate) for candidate in _list(report.get("contextualized_candidates"))]
    if str(report.get("boundary") or "") != BRIDGE_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "chord-context pitch-role bridge boundary required"
        )
    if str(decision.get("next_boundary") or "") != OBJECTIVE_DECISION_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "bridge must route to pitch-role objective decision"
        )
    if _int(readiness.get("candidate_count")) != len(candidates) or len(candidates) < 6:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "bridge candidate count mismatch"
        )
    if _int(readiness.get("not_evaluable_after_count")) != 0:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "not-evaluable labels must be cleared before repair"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="bridge readiness")
    source_context = _source_context_fields(readiness, label="bridge")
    return {
        "boundary": BRIDGE_BOUNDARY,
        "candidates": candidates,
        "chord_progression": [str(chord) for chord in _list(context.get("chord_progression"))],
        "bpm": _float(context.get("bpm")) or 124.0,
        **source_context,
    }


def nearest_chord_tone_pitch(pitch: int, chord: str) -> int:
    pitch_classes = chord_pitch_classes(chord, pitch_mode="tones")
    if not pitch_classes:
        return int(pitch)
    candidates = [
        octave * 12 + pitch_class
        for octave in range(0, 11)
        for pitch_class in pitch_classes
        if PIANO_PITCH_MIN <= octave * 12 + pitch_class <= PIANO_PITCH_MAX
    ]
    if not candidates:
        return int(pitch)
    return min(candidates, key=lambda value: (abs(value - int(pitch)), value))


def all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def repair_midi(
    *,
    source_path: Path,
    output_path: Path,
    chords: list[str],
    bpm: float,
    bar_count: int,
) -> dict[str, Any]:
    midi = pretty_midi.PrettyMIDI(str(source_path))
    notes = all_notes(midi)
    changed = 0
    quantized_notes = [
        (note, *quantize_note_position(float(note.start), bpm))
        for note in notes
    ]
    eligible_notes = [
        note for note, bar, _position in quantized_notes if 0 <= int(bar) < int(bar_count)
    ]
    final_note = eligible_notes[-1] if eligible_notes else None
    changed_positions: list[dict[str, Any]] = []
    for note, bar, position in quantized_notes:
        if bar < 0 or bar >= int(bar_count):
            continue
        chord = chords[bar % len(chords)] if chords else ""
        should_repair = note is final_note or position_bucket(position) == "strong"
        if not should_repair:
            continue
        original_pitch = int(note.pitch)
        repaired_pitch = nearest_chord_tone_pitch(original_pitch, chord)
        if repaired_pitch != original_pitch:
            note.pitch = int(repaired_pitch)
            changed += 1
            changed_positions.append(
                {
                    "bar": int(bar),
                    "position": int(position),
                    "chord": chord,
                    "from": original_pitch,
                    "to": int(repaired_pitch),
                    "final_note": bool(note is final_note),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(output_path))
    return {
        "source_midi_path": str(source_path),
        "repaired_midi_path": str(output_path),
        "changed_note_count": int(changed),
        "changed_positions": changed_positions,
    }


def repair_candidate(
    *,
    candidate: dict[str, Any],
    output_dir: Path,
    chords: list[str],
    bpm: float,
    manifest_path: Path,
) -> dict[str, Any]:
    rank = _int(candidate.get("rank"))
    before_metrics = _dict(candidate.get("bridge_metrics"))
    bar_count = _int(before_metrics.get("bar_count")) or 8
    source_path = Path(str(candidate.get("midi_path") or ""))
    if not source_path.is_absolute():
        source_path = ROOT_DIR / source_path
    if not source_path.exists():
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            f"candidate MIDI missing: {source_path}"
        )
    output_path = output_dir / "midi" / f"{rank:02d}_chord_tone_landing_repair.mid"
    repair = repair_midi(
        source_path=source_path,
        output_path=output_path,
        chords=chords,
        bpm=bpm,
        bar_count=bar_count,
    )
    row = {
        "rank": rank,
        "phrase_rhythm_repaired_midi_path": str(output_path),
        "phrase_rhythm_repaired_labeling": {
            "metrics": _dict(candidate.get("bridge_metrics")),
            "failure_labels": [],
            "not_evaluable_labels": [],
        },
    }
    repaired = bridge_candidate(
        row=row,
        rank=rank,
        chords=chords,
        bpm=bpm,
        manifest_path=manifest_path,
    )
    after_metrics = _dict(repaired.get("bridge_metrics"))
    before_flags = [str(flag) for flag in _list(candidate.get("bridge_flags"))]
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


def count_flag(rows: list[dict[str, Any]], side: str, flag: str) -> int:
    return sum(1 for row in rows if flag in _list(_dict(row.get(side)).get("bridge_flags")))


def build_repair_sweep_report(
    *,
    objective_decision_report: dict[str, Any],
    bridge_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    objective = validate_objective_decision(objective_decision_report)
    bridge = validate_bridge_source(bridge_report)
    _validate_source_context_consistency(objective, bridge)
    chords = bridge["chord_progression"] or parse_chords("")
    bpm = _float(bridge["bpm"]) or 124.0
    manifest_path = ROOT_DIR / "stage_b_midi_to_solo_chord_tone_landing_repair_manifest.json"
    rows = [
        repair_candidate(
            candidate=candidate,
            output_dir=output_dir,
            chords=chords,
            bpm=bpm,
            manifest_path=manifest_path,
        )
        for candidate in bridge["candidates"]
    ]
    weak_before = count_flag(rows, "before", "weak_chord_tone_landing_risk")
    weak_after = count_flag(rows, "after", "weak_chord_tone_landing_risk")
    outside_before = count_flag(rows, "before", "outside_soloing_pitch_role_risk")
    outside_after = count_flag(rows, "after", "outside_soloing_pitch_role_risk")
    objective_outside_count = _int(objective["outside_soloing_pitch_role_risk_count"])
    if outside_before != objective_outside_count:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "objective outside-soloing risk count must match bridge input"
        )
    final_before = sum(1 for row in rows if bool(_dict(row.get("before")).get("cadence_landing_chord_tone", False)))
    final_after = sum(1 for row in rows if bool(_dict(row.get("after")).get("cadence_landing_chord_tone", False)))
    changed_note_total = sum(_int(_dict(row.get("repair")).get("changed_note_count")) for row in rows)
    target_supported = weak_after < weak_before and final_after > final_before
    outside_context_preserved = outside_before == objective_outside_count and objective_outside_count > 0
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": objective["boundary"],
        "bridge_boundary": bridge["boundary"],
        "context": {
            "chord_progression": chords,
            "bpm": bpm,
            "repair_policy": "strong_beat_and_final_note_nearest_chord_tone",
        },
        "candidate_repairs": rows,
        "aggregate": {
            "candidate_count": len(rows),
            "repaired_midi_count": len(rows),
            "changed_note_total": int(changed_note_total),
            "objective_outside_soloing_pitch_role_risk_count": int(
                objective_outside_count
            ),
            "weak_chord_tone_landing_risk_count_before": int(weak_before),
            "weak_chord_tone_landing_risk_count_after": int(weak_after),
            "weak_chord_tone_landing_risk_delta": int(weak_before - weak_after),
            "outside_soloing_pitch_role_risk_count_before": int(outside_before),
            "outside_soloing_pitch_role_risk_count_after": int(outside_after),
            "outside_soloing_pitch_role_risk_delta": int(outside_before - outside_after),
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": bool(outside_context_preserved),
            **{key: objective[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS},
            "final_landing_chord_tone_count_before": int(final_before),
            "final_landing_chord_tone_count_after": int(final_after),
            "target_supported": bool(target_supported),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "chord_tone_landing_repair_sweep_completed": True,
            "candidate_count": len(rows),
            "repaired_midi_count": len(rows),
            "target_supported": bool(target_supported),
            "objective_outside_soloing_pitch_role_risk_count": int(
                objective_outside_count
            ),
            "outside_soloing_pitch_role_risk_count_before": int(outside_before),
            "outside_soloing_pitch_role_risk_count_after": int(outside_after),
            "outside_soloing_repair_targeted": False,
            "outside_soloing_residual_risk_preserved": bool(outside_context_preserved),
            **{key: objective[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS},
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
            "next_boundary": NEXT_BOUNDARY if target_supported else (
                "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision"
            ),
            "selected_target": SELECTED_TARGET if target_supported else (
                "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "chord-tone landing repair sweep completed without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair audio package source-context refresh"
            if target_supported
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair follow-up decision source-context refresh"
        ),
    }


def validate_repair_sweep_report(
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
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "unexpected next boundary"
        )
    if require_repair_completed and not bool(
        readiness.get("chord_tone_landing_repair_sweep_completed", False)
    ):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "repair sweep completion required"
        )
    if require_target_supported and not bool(readiness.get("target_supported", False)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "target support required"
        )
    if _int(readiness.get("repaired_midi_count")) != _int(readiness.get("candidate_count")):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "all candidates must have repaired MIDI"
        )
    if _int(aggregate.get("changed_note_total")) <= 0:
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "changed note count required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "critical user input should not be required"
        )
    if _int(aggregate.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        aggregate.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "outside-soloing objective and bridge counts must match"
        )
    if not bool(aggregate.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "outside-soloing residual risk context must be preserved"
        )
    if bool(aggregate.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairSweepError(
            "outside-soloing repair target should remain false in landing repair"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="repair readiness")
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in readiness or key not in aggregate:
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"repair sweep source-context field required: {key}"
            )
        if readiness.get(key) != aggregate.get(key):
            raise StageBMidiToSoloChordToneLandingRepairSweepError(
                f"repair sweep source-context readiness/aggregate mismatch: {key}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "bridge_boundary": str(report.get("bridge_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "chord_tone_landing_repair_sweep_completed": bool(
            readiness.get("chord_tone_landing_repair_sweep_completed", False)
        ),
        "candidate_count": _int(readiness.get("candidate_count")),
        "repaired_midi_count": _int(readiness.get("repaired_midi_count")),
        "changed_note_total": _int(aggregate.get("changed_note_total")),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            aggregate.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "weak_chord_tone_landing_risk_count_before": _int(
            aggregate.get("weak_chord_tone_landing_risk_count_before")
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            aggregate.get("weak_chord_tone_landing_risk_count_after")
        ),
        "weak_chord_tone_landing_risk_delta": _int(
            aggregate.get("weak_chord_tone_landing_risk_delta")
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
        "target_supported": bool(readiness.get("target_supported", False)),
        **{key: readiness.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
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
        "# Stage B MIDI-to-Solo Chord-Tone Landing Repair Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- bridge boundary: `{report['bridge_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- repair policy: `{context['repair_policy']}`",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- repaired MIDI count: `{aggregate['repaired_midi_count']}`",
        f"- changed note total: `{aggregate['changed_note_total']}`",
        f"- objective outside-soloing pitch-role risk count: `{aggregate['objective_outside_soloing_pitch_role_risk_count']}`",
        f"- weak chord-tone landing risk count: `{aggregate['weak_chord_tone_landing_risk_count_before']} -> {aggregate['weak_chord_tone_landing_risk_count_after']}`",
        f"- outside-soloing pitch-role risk count: `{aggregate['outside_soloing_pitch_role_risk_count_before']} -> {aggregate['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing repair targeted: `{_bool_token(aggregate['outside_soloing_repair_targeted'])}`",
        f"- outside-soloing residual risk preserved: `{_bool_token(aggregate['outside_soloing_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk delta: `{aggregate['followup_objective_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source targeted: `{_bool_token(aggregate['followup_objective_source_outside_soloing_source_targeted'])}`",
        f"- follow-up objective source outside-soloing source residual risk preserved: `{_bool_token(aggregate['followup_objective_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk delta: `{aggregate['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source targeted: `{_bool_token(aggregate['followup_repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- follow-up repair sweep source outside-soloing source residual risk preserved: `{_bool_token(aggregate['followup_repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk delta: `{aggregate['repair_sweep_source_outside_soloing_source_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source targeted: `{_bool_token(aggregate['repair_sweep_source_outside_soloing_source_targeted'])}`",
        f"- bridge repair sweep source outside-soloing source residual risk preserved: `{_bool_token(aggregate['repair_sweep_source_outside_soloing_source_residual_risk_preserved'])}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {aggregate['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- final landing chord-tone count: `{aggregate['final_landing_chord_tone_count_before']} -> {aggregate['final_landing_chord_tone_count_after']}`",
        f"- target supported: `{_bool_token(aggregate['target_supported'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidates",
        "",
        "| rank | changed | weak before | weak after | final before | final after | MIDI |",
        "|---:|---:|---|---|---|---|---|",
    ]
    for row in report["candidate_repairs"]:
        before = row["before"]
        after = row["after"]
        repair = row["repair"]
        lines.append(
            "| {rank} | {changed} | `{before_flags}` | `{after_flags}` | `{before_final}` | `{after_final}` | `{midi}` |".format(
                rank=row["rank"],
                changed=repair["changed_note_count"],
                before_flags=",".join(before["bridge_flags"]) or "none",
                after_flags=",".join(after["bridge_flags"]) or "none",
                before_final=before["cadence_landing_role"],
                after_final=after["cadence_landing_role"],
                midi=row["repaired_midi_path"],
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
        description="Run chord-tone landing repair sweep for phrase/rhythm candidates"
    )
    parser.add_argument("--objective_decision_report", type=str, required=True)
    parser.add_argument("--bridge_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=960)
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
    report = build_repair_sweep_report(
        objective_decision_report=read_json(Path(args.objective_decision_report)),
        bridge_report=read_json(Path(args.bridge_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_repair_sweep_report(
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
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_sweep.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
