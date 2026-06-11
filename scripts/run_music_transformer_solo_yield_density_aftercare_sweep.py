"""Repair low note-count density after chord-role-balance repair."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import parse_chord  # noqa: E402
from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.decide_music_transformer_solo_yield_chord_role_balance_objective_next import (  # noqa: E402
    SCHEMA_VERSION as OBJECTIVE_DECISION_SCHEMA_VERSION,
)
from scripts.review_music_transformer_solo_yield_repaired_top8_objective_failures import (  # noqa: E402
    midi_profile,
)
from scripts.run_music_transformer_solo_yield_chord_role_balance_repair_sweep import (  # noqa: E402
    SCHEMA_VERSION as SOURCE_REPAIR_SWEEP_SCHEMA_VERSION,
)


SCHEMA_VERSION = "music_transformer_solo_yield_density_aftercare_sweep_v1"
BOUNDARY = "music_transformer_solo_yield_density_aftercare_sweep"
NEXT_BOUNDARY = "music_transformer_solo_yield_density_aftercare_audio_package"
SELECTED_TARGET = "density_aftercare_audio_package"


class SoloYieldDensityAftercareSweepError(ValueError):
    pass


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SoloYieldDensityAftercareSweepError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _require_no_quality_claim(report: dict[str, Any]) -> None:
    readiness = _dict(report.get("readiness"))
    claimed = [
        key
        for key in (
            "audio_rendered_quality_claimed",
            "musical_quality_claimed",
            "artist_style_claimed",
            "production_ready_claimed",
        )
        if bool(readiness.get(key, False))
    ]
    if claimed:
        raise SoloYieldDensityAftercareSweepError(f"unexpected quality claim: {claimed}")


def _all_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def direction_change_ratio(pitches: list[int]) -> float:
    if len(pitches) < 3:
        return 0.0
    signs: list[int] = []
    for left, right in zip(pitches, pitches[1:]):
        delta = int(right) - int(left)
        if delta > 0:
            signs.append(1)
        elif delta < 0:
            signs.append(-1)
    if len(signs) < 2:
        return 0.0
    changes = sum(1 for left, right in zip(signs, signs[1:]) if left != right)
    return changes / max(1, len(signs) - 1)


def max_abs_interval(pitches: list[int]) -> int:
    return max((abs(int(right) - int(left)) for left, right in zip(pitches, pitches[1:])), default=0)


def _chord_tone_pcs(chord: str) -> set[int]:
    root_pc, intervals = parse_chord(chord)
    return {(root_pc + interval) % 12 for interval in intervals}


def _chord_index(time_sec: float, *, phrase_start: float, phrase_duration: float, chord_count: int) -> int:
    if chord_count <= 1 or phrase_duration <= 0:
        return 0
    ratio = max(0.0, min(0.999999, (float(time_sec) - phrase_start) / phrase_duration))
    return min(chord_count - 1, int(ratio * chord_count))


def chord_contexts(times: list[float], *, phrase_start: float, phrase_duration: float, chords: list[str]) -> list[str]:
    return [
        chords[
            _chord_index(
                time_sec,
                phrase_start=phrase_start,
                phrase_duration=phrase_duration,
                chord_count=len(chords),
            )
        ]
        for time_sec in times
    ]


def chord_tone_ratio(pitches: list[int], note_chords: list[str]) -> float:
    if not pitches:
        return 0.0
    chord_tone_count = 0
    for pitch, chord in zip(pitches, note_chords):
        if int(pitch) % 12 in _chord_tone_pcs(chord):
            chord_tone_count += 1
    return chord_tone_count / max(1, len(pitches))


def chord_tone_pitch_options(
    *,
    center_pitch: float,
    chord: str,
    pitch_min: int,
    pitch_max: int,
) -> list[int]:
    chord_pcs = _chord_tone_pcs(chord)
    options = [pitch for pitch in range(int(pitch_min), int(pitch_max) + 1) if pitch % 12 in chord_pcs]
    return sorted(options, key=lambda pitch: (abs(float(pitch) - float(center_pitch)), pitch))


def note_arrays(notes: list[pretty_midi.Note]) -> tuple[list[int], list[float], list[float]]:
    return (
        [int(note.pitch) for note in notes],
        [float(note.start) for note in notes],
        [float(note.end) for note in notes],
    )


def choose_density_insert(
    notes: list[pretty_midi.Note],
    chords: list[str],
    *,
    min_direction_change: float,
    min_chord_tone_ratio: float,
    max_interval: int,
    pitch_min: int,
    pitch_max: int,
    min_gap_seconds: float,
    max_note_duration_seconds: float,
) -> dict[str, Any] | None:
    if len(notes) < 2:
        return None
    pitches, starts, ends = note_arrays(notes)
    phrase_start = min(starts, default=0.0)
    phrase_duration = max(1e-6, max(ends, default=phrase_start) - phrase_start)
    current_times = [(start + end) / 2.0 for start, end in zip(starts, ends)]
    current_chords = chord_contexts(
        current_times,
        phrase_start=phrase_start,
        phrase_duration=phrase_duration,
        chords=chords,
    )
    base_chord_ratio = chord_tone_ratio(pitches, current_chords)
    candidates: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    for insert_after_index in range(len(notes) - 1):
        left = notes[insert_after_index]
        right = notes[insert_after_index + 1]
        gap = float(right.start) - float(left.end)
        if gap < float(min_gap_seconds):
            continue
        duration = min(float(max_note_duration_seconds), max(0.06, gap * 0.45))
        if duration >= gap:
            duration = max(0.03, gap * 0.5)
        start = float(left.end) + max(0.0, (gap - duration) / 2.0)
        end = start + duration
        center_time = (start + end) / 2.0
        insert_chord = chord_contexts(
            [center_time],
            phrase_start=phrase_start,
            phrase_duration=phrase_duration,
            chords=chords,
        )[0]
        center_pitch = (int(left.pitch) + int(right.pitch)) / 2.0
        for pitch in chord_tone_pitch_options(
            center_pitch=center_pitch,
            chord=insert_chord,
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
        ):
            if abs(int(pitch) - int(left.pitch)) > int(max_interval):
                continue
            if abs(int(right.pitch) - int(pitch)) > int(max_interval):
                continue
            trial_pitches = pitches[: insert_after_index + 1] + [int(pitch)] + pitches[insert_after_index + 1 :]
            trial_starts = starts[: insert_after_index + 1] + [float(start)] + starts[insert_after_index + 1 :]
            trial_ends = ends[: insert_after_index + 1] + [float(end)] + ends[insert_after_index + 1 :]
            trial_times = [(start_item + end_item) / 2.0 for start_item, end_item in zip(trial_starts, trial_ends)]
            trial_chords = chord_contexts(
                trial_times,
                phrase_start=phrase_start,
                phrase_duration=phrase_duration,
                chords=chords,
            )
            trial_chord_ratio = chord_tone_ratio(trial_pitches, trial_chords)
            trial_direction = direction_change_ratio(trial_pitches)
            trial_interval = max_abs_interval(trial_pitches)
            if trial_chord_ratio + 1e-9 < max(float(min_chord_tone_ratio), base_chord_ratio):
                continue
            if trial_direction + 1e-9 < float(min_direction_change):
                continue
            if trial_interval > int(max_interval):
                continue
            score = (
                gap,
                trial_chord_ratio,
                trial_direction,
                -trial_interval,
                -abs(float(pitch) - center_pitch),
                -insert_after_index,
            )
            candidates.append(
                (
                    score,
                    {
                        "insert_after_index": int(insert_after_index),
                        "pitch": int(pitch),
                        "start": float(start),
                        "end": float(end),
                        "duration": float(duration),
                        "chord": insert_chord,
                        "gap_seconds": float(gap),
                        "chord_tone_ratio_after_insert": float(trial_chord_ratio),
                        "direction_change_ratio_after_insert": float(trial_direction),
                        "max_abs_interval_after_insert": int(trial_interval),
                    },
                )
            )
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def _target_midi_path(output_dir: Path, row: dict[str, Any]) -> Path:
    review_index = _int(row.get("review_index"))
    case_label = str(row.get("case_label") or "candidate")
    return output_dir / "midi" / f"candidate_{review_index:02d}_{case_label}_density_aftercare.mid"


def validate_sources(source_repair_sweep: dict[str, Any], objective_decision: dict[str, Any]) -> list[dict[str, Any]]:
    if str(source_repair_sweep.get("schema_version") or "") != SOURCE_REPAIR_SWEEP_SCHEMA_VERSION:
        raise SoloYieldDensityAftercareSweepError("source repair sweep schema required")
    if str(objective_decision.get("schema_version") or "") != OBJECTIVE_DECISION_SCHEMA_VERSION:
        raise SoloYieldDensityAftercareSweepError("objective decision schema required")
    _require_no_quality_claim(source_repair_sweep)
    _require_no_quality_claim(objective_decision)
    decision = _dict(objective_decision.get("decision"))
    if str(decision.get("selected_next_target") or "") != "density_aftercare":
        raise SoloYieldDensityAftercareSweepError("density aftercare target required")
    rows = [_dict(row) for row in _list(source_repair_sweep.get("candidate_repairs"))]
    if not rows:
        raise SoloYieldDensityAftercareSweepError("source repair rows required")
    return rows


def repair_candidate(
    row: dict[str, Any],
    *,
    output_dir: Path,
    min_note_count: int,
    min_direction_change: float,
    min_chord_tone_ratio: float,
    max_interval: int,
    pitch_min: int,
    pitch_max: int,
    min_gap_seconds: float,
    max_note_duration_seconds: float,
) -> dict[str, Any]:
    source_path = Path(str(row.get("repaired_midi_path") or ""))
    if not source_path.exists():
        raise SoloYieldDensityAftercareSweepError(f"source MIDI missing: {source_path}")
    before_profile = _dict(row.get("after_profile"))
    target_path = _target_midi_path(output_dir, row)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    low_density = _int(before_profile.get("midi_note_count")) < int(min_note_count)
    inserted_notes: list[dict[str, Any]] = []

    if low_density:
        midi = pretty_midi.PrettyMIDI(str(source_path))
        if not midi.instruments:
            raise SoloYieldDensityAftercareSweepError("source MIDI instrument required")
        chords = [item.strip() for item in str(row.get("chords") or "").split(",") if item.strip()]
        for _ in range(max(0, int(min_note_count) - _int(before_profile.get("midi_note_count")))):
            notes = _all_notes(midi)
            insert = choose_density_insert(
                notes,
                chords,
                min_direction_change=float(min_direction_change),
                min_chord_tone_ratio=float(min_chord_tone_ratio),
                max_interval=int(max_interval),
                pitch_min=int(pitch_min),
                pitch_max=int(pitch_max),
                min_gap_seconds=float(min_gap_seconds),
                max_note_duration_seconds=float(max_note_duration_seconds),
            )
            if insert is None:
                break
            non_drum = [instrument for instrument in midi.instruments if not instrument.is_drum]
            target_instrument = non_drum[0] if non_drum else midi.instruments[0]
            velocity_source = notes[insert["insert_after_index"]]
            target_instrument.notes.append(
                pretty_midi.Note(
                    velocity=int(velocity_source.velocity),
                    pitch=int(insert["pitch"]),
                    start=float(insert["start"]),
                    end=float(insert["end"]),
                )
            )
            inserted_notes.append(insert)
        midi.write(str(target_path))
    else:
        shutil.copy2(source_path, target_path)

    repaired_candidate = {
        "review_midi_path": str(target_path),
        "chords": str(row.get("chords") or ""),
    }
    after_profile = midi_profile(repaired_candidate)
    return {
        "review_index": _int(row.get("review_index")),
        "case_label": str(row.get("case_label") or ""),
        "chords": str(row.get("chords") or ""),
        "source_midi_path": str(source_path),
        "repaired_midi_path": str(target_path),
        "source_wav_path": str(row.get("source_wav_path") or ""),
        "before_profile": before_profile,
        "after_profile": after_profile,
        "repair": {
            "low_density_before": bool(low_density),
            "low_density_after": _int(after_profile.get("midi_note_count")) < int(min_note_count),
            "inserted_note_count": len(inserted_notes),
            "inserted_notes": inserted_notes,
            "note_count_delta": _int(after_profile.get("midi_note_count")) - _int(before_profile.get("midi_note_count")),
            "chord_tone_ratio_delta": (
                _float(after_profile.get("midi_chord_tone_ratio"))
                - _float(before_profile.get("midi_chord_tone_ratio"))
            ),
            "direction_change_after_valid": _float(after_profile.get("midi_direction_change_ratio"))
            >= float(min_direction_change),
            "max_interval_after_valid": _int(after_profile.get("midi_max_abs_interval")) <= int(max_interval),
            "final_landing_preserved": bool(after_profile.get("final_landing_is_chord_tone", False)),
        },
    }


def build_repair_sweep(
    *,
    source_repair_sweep: dict[str, Any],
    objective_decision: dict[str, Any],
    output_dir: Path,
    min_note_count: int = 30,
    min_direction_change: float = 0.50,
    min_chord_tone_ratio: float = 0.50,
    max_interval: int = 8,
    pitch_min: int = 55,
    pitch_max: int = 84,
    min_gap_seconds: float = 0.10,
    max_note_duration_seconds: float = 0.12,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = validate_sources(source_repair_sweep, objective_decision)
    repairs = [
        repair_candidate(
            row,
            output_dir=output_dir,
            min_note_count=int(min_note_count),
            min_direction_change=float(min_direction_change),
            min_chord_tone_ratio=float(min_chord_tone_ratio),
            max_interval=int(max_interval),
            pitch_min=int(pitch_min),
            pitch_max=int(pitch_max),
            min_gap_seconds=float(min_gap_seconds),
            max_note_duration_seconds=float(max_note_duration_seconds),
        )
        for row in rows
    ]
    low_before = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("low_density_before")))
    low_after = sum(1 for row in repairs if bool(_dict(row.get("repair")).get("low_density_after")))
    chord_ratio_decrease = sum(
        1 for row in repairs if _float(_dict(row.get("repair")).get("chord_tone_ratio_delta")) < -1e-9
    )
    weak_direction_after = sum(
        1 for row in repairs if _float(_dict(row.get("after_profile")).get("midi_direction_change_ratio")) < float(min_direction_change)
    )
    final_landing_not = sum(
        1 for row in repairs if not bool(_dict(row.get("after_profile")).get("final_landing_is_chord_tone", False))
    )
    wide_before = sum(
        1 for row in repairs if _int(_dict(row.get("before_profile")).get("midi_max_abs_interval")) >= int(max_interval)
    )
    wide_after = sum(
        1 for row in repairs if _int(_dict(row.get("after_profile")).get("midi_max_abs_interval")) >= int(max_interval)
    )
    inserted_total = sum(_int(_dict(row.get("repair")).get("inserted_note_count")) for row in repairs)
    target_supported = bool(
        low_before > low_after
        and low_after == 0
        and chord_ratio_decrease == 0
        and weak_direction_after == 0
        and final_landing_not == 0
        and wide_after <= wide_before
    )
    next_boundary = NEXT_BOUNDARY if target_supported else "music_transformer_solo_yield_objective_repair_decision"
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_repair_sweep": {
            "schema_version": source_repair_sweep.get("schema_version"),
            "output_dir": source_repair_sweep.get("output_dir"),
            "candidate_count": _int(_dict(source_repair_sweep.get("aggregate")).get("candidate_count")),
        },
        "source_objective_decision": {
            "schema_version": objective_decision.get("schema_version"),
            "output_dir": objective_decision.get("output_dir"),
            "selected_next_target": str(_dict(objective_decision.get("decision")).get("selected_next_target") or ""),
        },
        "candidate_repairs": repairs,
        "aggregate": {
            "candidate_count": len(repairs),
            "repaired_midi_count": len(repairs),
            "low_note_count_before": int(low_before),
            "low_note_count_after": int(low_after),
            "low_note_count_delta": int(low_before - low_after),
            "inserted_note_total": int(inserted_total),
            "chord_tone_ratio_decrease_count": int(chord_ratio_decrease),
            "weak_direction_change_count_after": int(weak_direction_after),
            "final_landing_not_chord_tone_count_after": int(final_landing_not),
            "wide_interval_review_count_before": int(wide_before),
            "wide_interval_review_count_after": int(wide_after),
            "target_supported": bool(target_supported),
        },
        "readiness": {
            "density_aftercare_sweep_completed": True,
            "target_supported": bool(target_supported),
            "validated_listening_input_present": False,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "artist_style_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "selected_next_target": SELECTED_TARGET if target_supported else "objective_repair_decision",
            "next_boundary": next_boundary,
            "critical_user_input_required": False,
            "reason": "low note-count residual reduced by gap-safe chord-tone inserts while preserving objective guards",
        },
        "not_proven": [
            "human_audio_preference",
            "stable_jazz_solo_quality",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
    }
    write_json(output_dir / "density_aftercare_sweep.json", report)
    write_json(output_dir / "density_aftercare_sweep_summary.json", validate_report(report))
    write_text(output_dir / "density_aftercare_sweep.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], *, min_candidates: int = 1) -> dict[str, Any]:
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise SoloYieldDensityAftercareSweepError("schema version mismatch")
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    decision = _dict(report.get("decision"))
    candidate_count = _int(aggregate.get("candidate_count"))
    if candidate_count < int(min_candidates):
        raise SoloYieldDensityAftercareSweepError("candidate count below requirement")
    if not bool(readiness.get("density_aftercare_sweep_completed", False)):
        raise SoloYieldDensityAftercareSweepError("repair sweep completion required")
    _require_no_quality_claim({"readiness": readiness})
    return {
        "schema_version": str(report.get("schema_version") or ""),
        "candidate_count": candidate_count,
        "repaired_midi_count": _int(aggregate.get("repaired_midi_count")),
        "low_note_count_before": _int(aggregate.get("low_note_count_before")),
        "low_note_count_after": _int(aggregate.get("low_note_count_after")),
        "low_note_count_delta": _int(aggregate.get("low_note_count_delta")),
        "inserted_note_total": _int(aggregate.get("inserted_note_total")),
        "chord_tone_ratio_decrease_count": _int(aggregate.get("chord_tone_ratio_decrease_count")),
        "weak_direction_change_count_after": _int(aggregate.get("weak_direction_change_count_after")),
        "final_landing_not_chord_tone_count_after": _int(
            aggregate.get("final_landing_not_chord_tone_count_after")
        ),
        "wide_interval_review_count_before": _int(aggregate.get("wide_interval_review_count_before")),
        "wide_interval_review_count_after": _int(aggregate.get("wide_interval_review_count_after")),
        "target_supported": bool(aggregate.get("target_supported", False)),
        "selected_next_target": str(decision.get("selected_next_target") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    decision = report["decision"]
    readiness = report["readiness"]
    lines = [
        "# Music Transformer Solo Yield Density Aftercare Sweep",
        "",
        "## Summary",
        "",
        f"- candidate count: `{aggregate['candidate_count']}`",
        f"- repaired MIDI count: `{aggregate['repaired_midi_count']}`",
        f"- low note count: `{aggregate['low_note_count_before']} -> {aggregate['low_note_count_after']}`",
        f"- inserted note total: `{aggregate['inserted_note_total']}`",
        f"- chord-tone ratio decrease count: `{aggregate['chord_tone_ratio_decrease_count']}`",
        f"- weak direction-change after: `{aggregate['weak_direction_change_count_after']}`",
        f"- final landing not chord-tone after: `{aggregate['final_landing_not_chord_tone_count_after']}`",
        f"- wide interval review: `{aggregate['wide_interval_review_count_before']} -> {aggregate['wide_interval_review_count_after']}`",
        f"- target supported: `{_bool_token(aggregate['target_supported'])}`",
        f"- selected next target: `{decision['selected_next_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Candidate Repairs",
        "",
    ]
    for row in report.get("candidate_repairs", []):
        repair = row["repair"]
        before = row["before_profile"]
        after = row["after_profile"]
        lines.extend(
            [
                f"- candidate `{row['review_index']}` / `{row['case_label']}`",
                f"  - note count: `{before['midi_note_count']}` -> `{after['midi_note_count']}`",
                f"  - chord-tone ratio: `{float(before['midi_chord_tone_ratio']):.4f}` -> `{float(after['midi_chord_tone_ratio']):.4f}`",
                f"  - direction-change ratio: `{float(before['midi_direction_change_ratio']):.4f}` -> `{float(after['midi_direction_change_ratio']):.4f}`",
                f"  - max interval: `{before['midi_max_abs_interval']}` -> `{after['midi_max_abs_interval']}`",
                f"  - inserted notes: `{repair['inserted_note_count']}`",
                f"  - final landing preserved: `{_bool_token(repair['final_landing_preserved'])}`",
                f"  - repaired MIDI: `{row['repaired_midi_path']}`",
            ]
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run density aftercare repair sweep")
    parser.add_argument(
        "--source_repair_sweep_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair/"
            "issue_1284_chord_role_balance_repair_sweep/chord_role_balance_repair_sweep.json"
        ),
    )
    parser.add_argument(
        "--objective_decision_report",
        type=str,
        default=(
            "outputs/music_transformer_finetune_mvp/solo_yield_chord_role_balance_repair_objective_next/"
            "issue_1292_chord_role_balance_objective_next/"
            "chord_role_balance_objective_next_decision.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/music_transformer_finetune_mvp/solo_yield_density_aftercare",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=8)
    parser.add_argument("--min_note_count", type=int, default=30)
    parser.add_argument("--min_direction_change", type=float, default=0.50)
    parser.add_argument("--min_chord_tone_ratio", type=float, default=0.50)
    parser.add_argument("--max_interval", type=int, default=8)
    parser.add_argument("--pitch_min", type=int, default=55)
    parser.add_argument("--pitch_max", type=int, default=84)
    parser.add_argument("--min_gap_seconds", type=float, default=0.10)
    parser.add_argument("--max_note_duration_seconds", type=float, default=0.12)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repair_sweep(
        source_repair_sweep=read_json(Path(args.source_repair_sweep_report)),
        objective_decision=read_json(Path(args.objective_decision_report)),
        output_dir=output_dir,
        min_note_count=int(args.min_note_count),
        min_direction_change=float(args.min_direction_change),
        min_chord_tone_ratio=float(args.min_chord_tone_ratio),
        max_interval=int(args.max_interval),
        pitch_min=int(args.pitch_min),
        pitch_max=int(args.pitch_max),
        min_gap_seconds=float(args.min_gap_seconds),
        max_note_duration_seconds=float(args.max_note_duration_seconds),
    )
    summary = validate_report(report, min_candidates=int(args.min_candidates))
    write_json(output_dir / "density_aftercare_sweep_summary.json", summary)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown_report(report))
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
