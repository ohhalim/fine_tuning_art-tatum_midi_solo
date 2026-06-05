"""Repair phrase-bank MIDI-to-solo candidates for dead-air and density variation."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import (  # noqa: E402
    chord_for_time,
    chord_pitches_in_range,
    parse_chord,
    phrase_duration_sec,
)
from inference.app.metrics import compute_midi_metrics, max_simultaneous_notes  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_phrase_bank_objective_next import (  # noqa: E402
    BOUNDARY as OBJECTIVE_NEXT_BOUNDARY,
    NEXT_BOUNDARY as OBJECTIVE_NEXT_EXPECTED_BOUNDARY,
)


class StageBMidiToSoloPhraseBankDeadAirDensityRepairError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe"
NEXT_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_audio_package"
REMAINING_BLOCKER_BOUNDARY = "stage_b_midi_to_solo_phrase_bank_dead_air_density_remaining_blocker_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe_v1"

PREFERRED_PITCH_MIN = 55
PREFERRED_PITCH_MAX = 84
QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "phrase_bank_musical_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
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
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_objective_next_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if str(report.get("boundary") or "") != OBJECTIVE_NEXT_BOUNDARY:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("objective-only next boundary required")
    if str(decision.get("next_boundary") or "") != OBJECTIVE_NEXT_EXPECTED_BOUNDARY:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("objective report must route to repair probe")
    if not bool(readiness.get("phrase_bank_repair_required", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("repair-required source required")
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("preference fill must remain blocked")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="objective-next readiness")
    candidates = [_dict(item) for item in _list(report.get("candidate_reviews"))]
    if not candidates:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("candidate reviews required")
    return candidates


def request_from_objective_report(report: dict[str, Any], *, bpm: int, bars: int) -> GenerationRequest:
    return GenerationRequest(
        bpm=int(bpm),
        chord_progression=["Cmaj7", "F7", "G7", "Cmaj7", "Cmaj7", "Cmaj7", "Cmaj7", "Cmaj7"][: int(bars)],
        bars=int(bars),
        density="medium",
        energy="mid",
    )


def load_notes(midi_path: Path) -> list[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def copy_note(note: pretty_midi.Note) -> pretty_midi.Note:
    return pretty_midi.Note(
        velocity=int(note.velocity),
        pitch=int(note.pitch),
        start=float(note.start),
        end=float(note.end),
    )


def bar_duration_sec(request: GenerationRequest) -> float:
    return phrase_duration_sec(request) / max(1, int(request.bars))


def per_bar_note_counts(notes: Sequence[pretty_midi.Note], request: GenerationRequest) -> dict[str, int]:
    bar_sec = bar_duration_sec(request)
    counts = {str(index): 0 for index in range(int(request.bars))}
    for note in notes:
        bar_index = int(max(0.0, float(note.start)) // max(1e-6, bar_sec))
        bar_index = max(0, min(int(request.bars) - 1, bar_index))
        counts[str(bar_index)] += 1
    return counts


def rhythm_diversity(notes: Sequence[pretty_midi.Note]) -> dict[str, Any]:
    ordered = sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    if not ordered:
        return {
            "duration_diversity_ratio": 0.0,
            "ioi_diversity_ratio": 0.0,
            "unique_duration_count": 0,
            "unique_ioi_count": 0,
        }
    durations = [round(max(0.0, float(note.end) - float(note.start)), 4) for note in ordered]
    iois = [
        round(max(0.0, float(ordered[index].start) - float(ordered[index - 1].start)), 4)
        for index in range(1, len(ordered))
    ]
    return {
        "duration_diversity_ratio": len(set(durations)) / max(1, len(durations)),
        "ioi_diversity_ratio": len(set(iois)) / max(1, len(iois)),
        "unique_duration_count": len(set(durations)),
        "unique_ioi_count": len(set(iois)),
    }


def pitch_candidates_for_time(request: GenerationRequest, start_sec: float) -> list[int]:
    chord = chord_for_time(request, start_sec)
    root_pc, intervals = parse_chord(chord)
    return chord_pitches_in_range(root_pc, intervals, PREFERRED_PITCH_MIN, PREFERRED_PITCH_MAX)


def choose_fill_pitch(
    *,
    request: GenerationRequest,
    start_sec: float,
    previous_pitch: int,
    next_pitch: int,
    target_pitch: float,
    recent_pitches: Sequence[int],
) -> int:
    chord_tones = set(pitch_candidates_for_time(request, start_sec))
    lower = max(PREFERRED_PITCH_MIN, min(int(previous_pitch), int(next_pitch)) - 3)
    upper = min(PREFERRED_PITCH_MAX, max(int(previous_pitch), int(next_pitch)) + 3)
    passing_tones = set(range(lower, upper + 1))
    recent = set(int(pitch) for pitch in recent_pitches[-4:])
    scored: list[tuple[int, int, float, int, int]] = []
    for pitch in sorted(chord_tones | passing_tones):
        if int(pitch) == int(previous_pitch):
            continue
        if abs(int(pitch) - int(previous_pitch)) > 9:
            continue
        if abs(int(next_pitch) - int(pitch)) > 12:
            continue
        scored.append(
            (
                int(pitch) in recent,
                int(pitch) not in chord_tones,
                abs(float(pitch) - float(target_pitch)),
                abs(int(next_pitch) - int(pitch)),
                int(pitch),
            )
        )
    if scored:
        scored.sort()
        return int(scored[0][-1])
    step = 1 if int(next_pitch) >= int(previous_pitch) else -1
    return max(PREFERRED_PITCH_MIN, min(PREFERRED_PITCH_MAX, int(next_pitch) - step))


def density_quota_pattern(bars: int, additions_per_bar: Sequence[int]) -> list[int]:
    raw = list(additions_per_bar) or [3, 5, 2, 6]
    return [int(raw[index % len(raw)]) for index in range(int(bars))]


def enforce_monophonic_note_ends(
    notes: Sequence[pretty_midi.Note],
    *,
    max_duration_sec: float,
    min_duration_sec: float = 0.04,
    end_epsilon_sec: float = 0.001,
) -> list[pretty_midi.Note]:
    ordered = sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    repaired: list[pretty_midi.Note] = []
    for index, note in enumerate(ordered):
        start = float(note.start)
        next_start = float(ordered[index + 1].start) if index + 1 < len(ordered) else float(max_duration_sec)
        latest_end = min(float(max_duration_sec), next_start - float(end_epsilon_sec))
        if latest_end <= start:
            continue
        end = min(max(float(note.end), start + float(min_duration_sec)), latest_end)
        if end <= start:
            continue
        repaired.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=start,
                end=end,
            )
        )
    return repaired


def build_dead_air_density_repaired_notes(
    source_notes: Sequence[pretty_midi.Note],
    *,
    request: GenerationRequest,
    additions_per_bar: Sequence[int],
    dead_air_threshold_sec: float,
    min_start_separation_sec: float,
) -> tuple[list[pretty_midi.Note], list[dict[str, Any]]]:
    if not source_notes:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("source MIDI has no notes")
    phrase_end = phrase_duration_sec(request)
    sixteenth_sec = 60.0 / float(request.bpm) / 4.0
    bar_sec = bar_duration_sec(request)
    quotas = density_quota_pattern(int(request.bars), additions_per_bar)
    used_starts = {round(float(note.start), 6) for note in source_notes}
    notes = [copy_note(note) for note in source_notes]
    additions: list[dict[str, Any]] = []

    ordered = sorted(source_notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    for previous_note, next_note in zip(ordered, ordered[1:]):
        gap_sec = float(next_note.start) - float(previous_note.start)
        if gap_sec < float(dead_air_threshold_sec):
            continue
        bar_index = int(max(0.0, float(previous_note.start)) // max(1e-6, bar_sec))
        bar_index = max(0, min(int(request.bars) - 1, bar_index))
        if quotas[bar_index] <= 0:
            continue
        slot_count = max(0, math.ceil((gap_sec - 0.001) / sixteenth_sec) - 1)
        if slot_count <= 0:
            continue
        local_pitches = [int(previous_note.pitch)]
        for slot_index in range(slot_count):
            if quotas[bar_index] <= 0:
                break
            start_sec = round(float(previous_note.start) + sixteenth_sec * (slot_index + 1), 6)
            if start_sec >= float(next_note.start) - float(min_start_separation_sec):
                continue
            if any(abs(start_sec - existing) < float(min_start_separation_sec) for existing in used_starts):
                continue
            target_pitch = int(previous_note.pitch) + (
                (int(next_note.pitch) - int(previous_note.pitch)) * ((slot_index + 1) / (slot_count + 1))
            )
            pitch = choose_fill_pitch(
                request=request,
                start_sec=start_sec,
                previous_pitch=local_pitches[-1],
                next_pitch=int(next_note.pitch),
                target_pitch=float(target_pitch),
                recent_pitches=local_pitches,
            )
            duration_sec = max(0.06, sixteenth_sec * (0.80 if (len(additions) % 2 == 0) else 0.60))
            notes.append(
                pretty_midi.Note(
                    velocity=max(55, min(100, int(previous_note.velocity) - 4)),
                    pitch=int(pitch),
                    start=float(start_sec),
                    end=min(float(phrase_end), float(start_sec) + float(duration_sec)),
                )
            )
            used_starts.add(start_sec)
            quotas[bar_index] -= 1
            local_pitches.append(int(pitch))
            additions.append(
                {
                    "bar_index": int(bar_index),
                    "start_sec": float(start_sec),
                    "pitch": int(pitch),
                    "between": [int(previous_note.pitch), int(next_note.pitch)],
                }
            )
    repaired = enforce_monophonic_note_ends(notes, max_duration_sec=float(phrase_end))
    return repaired, additions


def write_repaired_midi(notes: Sequence[pretty_midi.Note], output_path: Path, *, bpm: int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="phrase_bank_dead_air_density_repair")
    instrument.notes = [copy_note(note) for note in notes]
    midi.instruments.append(instrument)
    midi.write(str(output_path))
    return output_path


def repair_gate_flags(
    *,
    source_metrics: dict[str, Any],
    repaired_metrics: dict[str, Any],
    repaired_notes: Sequence[pretty_midi.Note],
    request: GenerationRequest,
    min_dead_air_gain: float,
    max_dead_air_ratio: float,
    min_unique_density_patterns: int,
    min_note_count_gain: int,
) -> list[str]:
    flags: list[str] = []
    dead_air_gain = _float(source_metrics.get("dead_air_ratio")) - _float(repaired_metrics.get("dead_air_ratio"))
    if dead_air_gain < float(min_dead_air_gain):
        flags.append("dead_air_gain_below_target")
    if _float(repaired_metrics.get("dead_air_ratio")) > float(max_dead_air_ratio):
        flags.append("dead_air_ratio_above_target")
    per_bar = per_bar_note_counts(repaired_notes, request)
    if len(set(per_bar.values())) < int(min_unique_density_patterns):
        flags.append("density_pattern_still_uniform")
    if _int(repaired_metrics.get("note_count")) < _int(source_metrics.get("note_count")) + int(min_note_count_gain):
        flags.append("note_count_gain_below_target")
    if _int(repaired_metrics.get("max_simultaneous_notes")) > 1 or max_simultaneous_notes(list(repaired_notes)) > 1:
        flags.append("overlap_or_polyphony_detected")
    if _float(repaired_metrics.get("phrase_coverage_ratio")) < 0.95:
        flags.append("phrase_coverage_below_target")
    return flags


def build_repair_candidate(
    *,
    source_candidate: dict[str, Any],
    output_dir: Path,
    request: GenerationRequest,
    additions_per_bar: Sequence[int],
    dead_air_threshold_sec: float,
    min_start_separation_sec: float,
    min_dead_air_gain: float,
    max_dead_air_ratio: float,
    min_unique_density_patterns: int,
    min_note_count_gain: int,
) -> dict[str, Any]:
    midi_path = Path(str(source_candidate.get("midi_path") or ""))
    if not midi_path.exists():
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError(f"source MIDI missing: {midi_path}")
    source_notes = load_notes(midi_path)
    repaired_notes, additions = build_dead_air_density_repaired_notes(
        source_notes,
        request=request,
        additions_per_bar=additions_per_bar,
        dead_air_threshold_sec=dead_air_threshold_sec,
        min_start_separation_sec=min_start_separation_sec,
    )
    output_path = output_dir / "midi" / f"rank_{_int(source_candidate.get('rank')):02d}_seed_{_int(source_candidate.get('sample_seed'))}_dead_air_density_repair.mid"
    write_repaired_midi(repaired_notes, output_path, bpm=int(request.bpm))
    source_metrics = _dict(source_candidate.get("objective_metrics"))
    repaired_metrics = compute_midi_metrics(output_path, 0, fallback_used=True, request=request).to_dict()
    rhythm = rhythm_diversity(repaired_notes)
    per_bar = per_bar_note_counts(repaired_notes, request)
    flags = repair_gate_flags(
        source_metrics=source_metrics,
        repaired_metrics=repaired_metrics,
        repaired_notes=repaired_notes,
        request=request,
        min_dead_air_gain=min_dead_air_gain,
        max_dead_air_ratio=max_dead_air_ratio,
        min_unique_density_patterns=min_unique_density_patterns,
        min_note_count_gain=min_note_count_gain,
    )
    dead_air_gain = _float(source_metrics.get("dead_air_ratio")) - _float(repaired_metrics.get("dead_air_ratio"))
    return {
        "rank": _int(source_candidate.get("rank")),
        "sample_seed": _int(source_candidate.get("sample_seed")),
        "source_midi_path": str(midi_path),
        "repaired_midi_path": str(output_path),
        "source_metrics": source_metrics,
        "repaired_metrics": repaired_metrics,
        "dead_air_gain": dead_air_gain,
        "note_count_gain": _int(repaired_metrics.get("note_count")) - _int(source_metrics.get("note_count")),
        "per_bar_note_counts": per_bar,
        "unique_bar_note_count_values": len(set(per_bar.values())),
        "rhythm_diversity": rhythm,
        "repair": {
            "addition_count": len(additions),
            "additions_per_bar_target": list(additions_per_bar),
            "additions": additions,
        },
        "repair_gate": {
            "qualified": not flags,
            "flags": flags,
        },
    }


def build_repair_probe_report(
    *,
    objective_next_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    bpm: int,
    bars: int,
    additions_per_bar: Sequence[int],
    dead_air_threshold_sec: float,
    min_start_separation_sec: float,
    min_dead_air_gain: float,
    max_dead_air_ratio: float,
    min_unique_density_patterns: int,
    min_note_count_gain: int,
) -> dict[str, Any]:
    source_candidates = validate_objective_next_report(objective_next_report)
    request = request_from_objective_report(objective_next_report, bpm=bpm, bars=bars)
    repaired_candidates = [
        build_repair_candidate(
            source_candidate=item,
            output_dir=output_dir,
            request=request,
            additions_per_bar=additions_per_bar,
            dead_air_threshold_sec=dead_air_threshold_sec,
            min_start_separation_sec=min_start_separation_sec,
            min_dead_air_gain=min_dead_air_gain,
            max_dead_air_ratio=max_dead_air_ratio,
            min_unique_density_patterns=min_unique_density_patterns,
            min_note_count_gain=min_note_count_gain,
        )
        for item in source_candidates
    ]
    qualified_count = sum(1 for item in repaired_candidates if bool(_dict(item.get("repair_gate")).get("qualified")))
    dead_air_values = [_float(_dict(item.get("repaired_metrics")).get("dead_air_ratio")) for item in repaired_candidates]
    gains = [_float(item.get("dead_air_gain")) for item in repaired_candidates]
    target_passed = qualified_count == len(repaired_candidates)
    next_boundary = NEXT_BOUNDARY if target_passed else REMAINING_BLOCKER_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": OBJECTIVE_NEXT_BOUNDARY,
        "repair_policy": {
            "basis": "objective_midi_repair_probe",
            "additions_per_bar": list(additions_per_bar),
            "dead_air_threshold_sec": float(dead_air_threshold_sec),
            "min_start_separation_sec": float(min_start_separation_sec),
            "min_dead_air_gain": float(min_dead_air_gain),
            "max_dead_air_ratio": float(max_dead_air_ratio),
            "min_unique_density_patterns": int(min_unique_density_patterns),
            "min_note_count_gain": int(min_note_count_gain),
        },
        "summary": {
            "source_candidate_count": len(source_candidates),
            "repaired_candidate_count": len(repaired_candidates),
            "qualified_repaired_candidate_count": qualified_count,
            "repair_probe_target_passed": target_passed,
            "max_repaired_dead_air_ratio": max(dead_air_values) if dead_air_values else 0.0,
            "min_repaired_dead_air_ratio": min(dead_air_values) if dead_air_values else 0.0,
            "min_dead_air_gain": min(gains) if gains else 0.0,
            "max_dead_air_gain": max(gains) if gains else 0.0,
        },
        "repaired_candidates": repaired_candidates,
        "readiness": {
            "boundary": BOUNDARY,
            "dead_air_density_repair_probe_completed": True,
            "repair_probe_target_passed": target_passed,
            "objective_supported_repaired_candidates": qualified_count,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "dead-air and density repair target passed; route repaired MIDI to audio package"
                if target_passed
                else "dead-air and density repair target not fully passed; route to remaining blocker decision"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "phrase_bank_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo phrase-bank dead-air density repair audio package"
            if target_passed
            else "Stage B MIDI-to-solo phrase-bank dead-air density remaining blocker decision"
        ),
    }


def validate_repair_probe_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_probe_completed: bool,
    require_target_passed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("unexpected next boundary")
    if require_repair_probe_completed and not bool(readiness.get("dead_air_density_repair_probe_completed")):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("repair probe completion required")
    if require_target_passed and not bool(summary.get("repair_probe_target_passed", False)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("repair probe target should pass")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPhraseBankDeadAirDensityRepairError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="repair probe readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "repaired_candidate_count": _int(summary.get("repaired_candidate_count")),
        "qualified_repaired_candidate_count": _int(summary.get("qualified_repaired_candidate_count")),
        "repair_probe_target_passed": bool(summary.get("repair_probe_target_passed", False)),
        "min_repaired_dead_air_ratio": _float(summary.get("min_repaired_dead_air_ratio")),
        "max_repaired_dead_air_ratio": _float(summary.get("max_repaired_dead_air_ratio")),
        "min_dead_air_gain": _float(summary.get("min_dead_air_gain")),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    policy = report["repair_policy"]
    lines = [
        "# Stage B MIDI-to-Solo Phrase-Bank Dead-Air Density Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- repaired candidate count: `{summary['repaired_candidate_count']}`",
        f"- qualified repaired candidate count: `{summary['qualified_repaired_candidate_count']}`",
        f"- repair probe target passed: `{_bool_token(summary['repair_probe_target_passed'])}`",
        f"- repaired dead-air range: `{summary['min_repaired_dead_air_ratio']:.4f} - {summary['max_repaired_dead_air_ratio']:.4f}`",
        f"- dead-air gain range: `{summary['min_dead_air_gain']:.4f} - {summary['max_dead_air_gain']:.4f}`",
        f"- additions per bar target: `{policy['additions_per_bar']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Candidate Repair Review",
        "",
    ]
    for item in report["repaired_candidates"]:
        source_metrics = item["source_metrics"]
        repaired_metrics = item["repaired_metrics"]
        rhythm = item["rhythm_diversity"]
        gate = item["repair_gate"]
        lines.extend(
            [
                f"### Rank {item['rank']}",
                "",
                f"- seed: `{item['sample_seed']}`",
                f"- source dead-air: `{_float(source_metrics.get('dead_air_ratio')):.4f}`",
                f"- repaired dead-air: `{_float(repaired_metrics.get('dead_air_ratio')):.4f}`",
                f"- dead-air gain: `{item['dead_air_gain']:.4f}`",
                f"- note count gain: `{item['note_count_gain']}`",
                f"- unique density values: `{item['unique_bar_note_count_values']}`",
                f"- duration diversity / IOI diversity: `{_float(rhythm.get('duration_diversity_ratio')):.4f} / {_float(rhythm.get('ioi_diversity_ratio')):.4f}`",
                f"- qualified: `{_bool_token(gate['qualified'])}`",
                f"- flags: `{', '.join(gate['flags'])}`",
                f"- repaired MIDI: `{item['repaired_midi_path']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision",
            "",
            f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
            f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
            f"- reason: `{decision['reason']}`",
            f"- next recommended issue: `{report['next_recommended_issue']}`",
            "",
            "## Not Proven",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def parse_additions_per_bar(raw: str) -> list[int]:
    values = [int(item.strip()) for item in str(raw or "").split(",") if item.strip()]
    if not values:
        return [3, 5, 2, 6, 3, 5, 2, 6]
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="harness_stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe")
    parser.add_argument("--output_root", type=Path, default=Path("outputs/stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe"))
    parser.add_argument("--objective_next_report", type=Path, required=True)
    parser.add_argument("--doc_path", type=Path)
    parser.add_argument("--issue_number", type=int, default=642)
    parser.add_argument("--bpm", type=int, default=120)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--additions_per_bar", default="3,5,2,6,3,5,2,6")
    parser.add_argument("--dead_air_threshold_sec", type=float, default=0.18)
    parser.add_argument("--min_start_separation_sec", type=float, default=0.04)
    parser.add_argument("--min_dead_air_gain", type=float, default=0.15)
    parser.add_argument("--max_dead_air_ratio", type=float, default=0.45)
    parser.add_argument("--min_unique_density_patterns", type=int, default=3)
    parser.add_argument("--min_note_count_gain", type=int, default=16)
    parser.add_argument("--expected_boundary")
    parser.add_argument("--expected_next_boundary")
    parser.add_argument("--require_repair_probe_completed", action="store_true")
    parser.add_argument("--require_target_passed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_root / args.run_id
    report = build_repair_probe_report(
        objective_next_report=read_json(args.objective_next_report),
        output_dir=output_dir,
        issue_number=args.issue_number,
        bpm=args.bpm,
        bars=args.bars,
        additions_per_bar=parse_additions_per_bar(args.additions_per_bar),
        dead_air_threshold_sec=args.dead_air_threshold_sec,
        min_start_separation_sec=args.min_start_separation_sec,
        min_dead_air_gain=args.min_dead_air_gain,
        max_dead_air_ratio=args.max_dead_air_ratio,
        min_unique_density_patterns=args.min_unique_density_patterns,
        min_note_count_gain=args.min_note_count_gain,
    )
    summary = validate_repair_probe_report(
        report,
        expected_boundary=args.expected_boundary,
        expected_next_boundary=args.expected_next_boundary,
        require_repair_probe_completed=args.require_repair_probe_completed,
        require_target_passed=args.require_target_passed,
        require_no_quality_claim=args.require_no_quality_claim,
    )
    json_path = output_dir / "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe.json"
    md_path = output_dir / "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe.md"
    validation_path = output_dir / "stage_b_midi_to_solo_phrase_bank_dead_air_density_repair_probe_validation_summary.json"
    write_json(json_path, report)
    write_text(md_path, markdown_report(report))
    write_json(validation_path, summary)
    if args.doc_path:
        write_text(args.doc_path, markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
