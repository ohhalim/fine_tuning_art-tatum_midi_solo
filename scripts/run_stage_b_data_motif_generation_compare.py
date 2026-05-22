"""Compare hand-written swing grammar with data-derived motif baseline generation."""

from __future__ import annotations

import argparse
import copy
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.metrics import max_simultaneous_notes  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.run_stage_b_generation_probe import (  # noqa: E402
    CHORD_TONE_INTERVALS,
    TENSION_INTERVALS,
    build_probe_summary,
    build_stage_b_primer,
    chord_aware_pitch_tokens,
    decode_stage_b_midi,
    jazz_rhythm_duration_tokens,
    jazz_rhythm_position_tokens,
    parse_chords,
    pitch_from_token,
    postprocess_stage_b_midi,
    sample_report,
)
from scripts.stage_b_tokens import (  # noqa: E402
    MAX_DURATION_STEPS,
    PIANO_PITCH_MAX,
    PIANO_PITCH_MIN,
    POSITIONS_PER_BAR,
    ROOT_TO_PC,
    TOKEN_BAR,
    TOKEN_END,
    chord_tokens,
    note_duration_token,
    note_pitch_token,
    note_velocity_token,
    parse_chord_symbol,
    pitch_from_token as stage_b_pitch_from_token,
    position_token,
)


VALID_BASELINE_MODES = {
    "straight_grid",
    "straight_guide_tones",
    "hand_written_swing",
    "data_motif",
    "data_motif_guide_tones",
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def run_command(cmd: list[str]) -> dict[str, Any]:
    completed = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True, check=False)
    return {
        "cmd": cmd,
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def run_template_extraction(args: argparse.Namespace, run_dir: Path) -> tuple[Path, dict[str, Any]]:
    template_run_id = f"{args.run_id}_templates"
    output_root = run_dir / "templates"
    cmd = [
        sys.executable,
        "scripts/run_stage_b_motif_template_extraction.py",
        "--run_id",
        template_run_id,
        "--output_root",
        str(output_root),
        "--input_dir",
        str(args.input_dir),
        "--max_files",
        str(args.max_files),
        "--window_bars",
        str(args.window_bars),
        "--window_stride_bars",
        str(args.window_stride_bars),
        "--min_window_target_notes",
        str(args.min_window_target_notes),
        "--motif_length",
        str(args.motif_length),
        "--max_bar_span",
        str(args.max_bar_span),
        "--max_records",
        str(args.max_records),
        "--top_n",
        str(args.template_top_n),
    ]
    result = run_command(cmd)
    report_path = output_root / template_run_id / "motif_template_report.json"
    return report_path, result


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_baseline_modes(raw: str) -> list[str]:
    modes = [mode.strip().lower() for mode in raw.split(",") if mode.strip()]
    invalid = [mode for mode in modes if mode not in VALID_BASELINE_MODES]
    if invalid:
        raise ValueError(f"Unknown baseline modes: {invalid}")
    return modes


def weighted_choice(rows: Sequence[dict[str, Any]], rng: random.Random, index: int) -> dict[str, Any]:
    if not rows:
        raise ValueError("template rows must not be empty")
    weights = [max(1, int(row.get("count", 1))) for row in rows]
    if index < len(rows):
        # Cycle through the highest-ranked rows first so tiny harness runs cover more than one template.
        return rows[index]
    return rng.choices(list(rows), weights=weights, k=1)[0]


def strictly_increasing_positions(raw_positions: Sequence[int], minimum: int, maximum: int) -> list[int]:
    result: list[int] = []
    previous = int(minimum) - 1
    for raw in raw_positions:
        value = max(int(minimum), min(int(maximum), int(raw)))
        if value <= previous:
            value = previous + 1
        if value > int(maximum):
            break
        result.append(value)
        previous = value
    return result


def normalize_position_deltas(
    position_deltas: Sequence[int],
    *,
    slot_start: int,
    slot_size: int,
) -> list[int]:
    deltas = [max(0, int(delta)) for delta in position_deltas]
    if not deltas:
        return []
    local_max = max(deltas)
    slot_size = max(1, int(slot_size))
    slot_end = min(int(POSITIONS_PER_BAR) - 1, int(slot_start) + slot_size - 1)
    if local_max <= 0:
        raw_positions = [int(slot_start) + index for index in range(len(deltas))]
    else:
        raw_positions = [
            int(slot_start) + int(round(delta * (slot_size - 1) / local_max))
            for delta in deltas
        ]
    return strictly_increasing_positions(raw_positions, int(slot_start), slot_end)


def duration_tokens_from_steps(duration_steps: Sequence[int], target_count: int) -> list[int]:
    steps = [max(1, min(int(MAX_DURATION_STEPS), int(step))) for step in duration_steps]
    if not steps:
        steps = [2]
    while len(steps) < int(target_count):
        steps.append(steps[-1])
    return [note_duration_token(step) for step in steps[: int(target_count)]]


def fit_duration_tokens_to_positions(
    positions: Sequence[int],
    duration_steps: Sequence[int],
    max_tail_duration: int = 4,
) -> list[int]:
    raw_steps = [max(1, min(int(MAX_DURATION_STEPS), int(step))) for step in duration_steps]
    if not raw_steps:
        raw_steps = [2]
    while len(raw_steps) < len(positions):
        raw_steps.append(raw_steps[-1])

    fitted: list[int] = []
    for index, position in enumerate(positions):
        if index + 1 < len(positions):
            max_duration = max(1, int(positions[index + 1]) - int(position))
        else:
            max_duration = max(1, min(int(max_tail_duration), int(POSITIONS_PER_BAR) - int(position)))
        fitted.append(max(1, min(raw_steps[index], max_duration)))
    return [note_duration_token(step) for step in fitted]


def nearest_allowed_pitch_token(
    target_pitch: int,
    allowed_tokens: Sequence[int],
    recent_pitches: Sequence[int] | None = None,
) -> int:
    if not allowed_tokens:
        return note_pitch_token(max(int(PIANO_PITCH_MIN), min(int(PIANO_PITCH_MAX), int(target_pitch))))
    blocked = set(int(pitch) for pitch in (recent_pitches or [])[-2:])
    candidates = [int(token) for token in allowed_tokens if stage_b_pitch_from_token(token) not in blocked]
    if not candidates:
        candidates = [int(token) for token in allowed_tokens]
    return min(
        candidates,
        key=lambda token: (
            abs(stage_b_pitch_from_token(token) - int(target_pitch)),
            abs(stage_b_pitch_from_token(token) - 67),
            stage_b_pitch_from_token(token),
        ),
    )


def pitch_class_set(chord: str | None, intervals_by_quality: dict[str, set[int]]) -> set[int]:
    root, quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root)
    if root_pc is None:
        return set()
    intervals = intervals_by_quality.get(quality, intervals_by_quality["unknown"])
    return {(int(root_pc) + int(interval)) % 12 for interval in intervals}


def chord_tone_pitch_classes(chord: str | None, *, include_root: bool = True) -> set[int]:
    tones = pitch_class_set(chord, CHORD_TONE_INTERVALS)
    if include_root:
        return tones
    root, _quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root)
    if root_pc is not None and len(tones) > 1:
        tones.discard(int(root_pc))
    return tones


def guide_tone_pitch_classes(chord: str | None) -> list[int]:
    root, quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root)
    if root_pc is None:
        return sorted(chord_tone_pitch_classes(chord, include_root=False))
    guide_intervals_by_quality = {
        "maj": [4, 7],
        "maj7": [4, 11],
        "min": [3, 7],
        "min7": [3, 10],
        "dom7": [4, 10],
        "dim": [3, 6],
        "halfdim": [3, 10],
        "sus": [5, 10],
        "unknown": [4, 7],
    }
    return [
        (int(root_pc) + int(interval)) % 12
        for interval in guide_intervals_by_quality.get(quality, guide_intervals_by_quality["unknown"])
    ]


def tension_pitch_classes(chord: str | None) -> set[int]:
    return pitch_class_set(chord, TENSION_INTERVALS)


def approach_pitch_class(target_pitch_class: int, recent_pitch_classes: Sequence[int]) -> int:
    for offset in (-1, 1, -2, 2):
        candidate = (int(target_pitch_class) + offset) % 12
        if candidate not in recent_pitch_classes[-1:]:
            return candidate
    return (int(target_pitch_class) - 1) % 12


def nearest_pitch_for_pitch_class(
    pitch_class: int,
    *,
    target_pitch: int,
    recent_pitches: Sequence[int] | None = None,
) -> int:
    recent = list(recent_pitches or [])
    candidates = [
        pitch
        for pitch in range(int(PIANO_PITCH_MIN), int(PIANO_PITCH_MAX) + 1)
        if pitch % 12 == int(pitch_class) % 12
    ]
    blocked = set(recent[-2:])
    filtered = [pitch for pitch in candidates if pitch not in blocked]
    if filtered:
        candidates = filtered
    if recent:
        previous = int(recent[-1])
        preferred = [pitch for pitch in candidates if 1 <= abs(pitch - previous) <= 7]
        if preferred:
            candidates = preferred
    return int(
        min(
            candidates,
            key=lambda pitch: (
                abs(int(pitch) - int(target_pitch)),
                abs(int(pitch) - 67),
                int(pitch),
            ),
        )
    )


def guide_tone_cadence_pitch_classes(
    chord: str | None,
    next_chord: str | None,
    *,
    bar_index: int,
) -> list[tuple[int, str]]:
    guides = guide_tone_pitch_classes(chord)
    non_root_tones = sorted(chord_tone_pitch_classes(chord, include_root=False))
    tensions = sorted(tension_pitch_classes(chord))
    next_guides = guide_tone_pitch_classes(next_chord)

    if not guides:
        guides = non_root_tones or sorted(chord_tone_pitch_classes(chord))
    if not non_root_tones:
        non_root_tones = guides or sorted(chord_tone_pitch_classes(chord))

    guide_a = guides[int(bar_index) % len(guides)]
    guide_b = guides[(int(bar_index) + 1) % len(guides)]
    color = tensions[int(bar_index) % len(tensions)] if tensions else non_root_tones[0]
    chord_color = non_root_tones[(int(bar_index) + 1) % len(non_root_tones)]
    next_target = next_guides[0] if next_guides else guide_a

    return [
        (guide_a, "guide"),
        (color, "color"),
        (guide_b, "guide"),
        (approach_pitch_class(guide_b, [guide_a, color]), "approach"),
        (guide_b, "guide"),
        (chord_color, "chord"),
        (guide_a, "guide"),
        (approach_pitch_class(next_target, [guide_b, chord_color, guide_a]), "approach_next"),
    ]


def guide_tone_cell_index_for_position(position: int) -> int:
    scaled = int(round(max(0, int(position)) * 8 / int(POSITIONS_PER_BAR)))
    return max(0, min(7, scaled))


def is_strong_beat_position(position: int) -> bool:
    return int(position) % 4 == 0


def guide_tone_pitch_for_position(
    chord: str | None,
    next_chord: str | None,
    *,
    bar_index: int,
    position: int,
    target_pitch: int,
    recent_pitches: Sequence[int],
) -> int:
    cells = guide_tone_cadence_pitch_classes(chord, next_chord, bar_index=bar_index)
    cell_index = guide_tone_cell_index_for_position(position)
    pitch_class, role = cells[cell_index]
    if is_strong_beat_position(position):
        guides = guide_tone_pitch_classes(chord)
        if guides:
            guide_index = (int(position) // 4 + int(bar_index)) % len(guides)
            pitch_class = guides[guide_index]
    if role.startswith("approach") and recent_pitches:
        target_pitch = int(recent_pitches[-1])
    return nearest_pitch_for_pitch_class(
        pitch_class,
        target_pitch=target_pitch,
        recent_pitches=recent_pitches,
    )


def base_pitch_token_for_chord(chord: str | None, rng: random.Random, recent_pitches: Sequence[int]) -> int:
    allowed = chord_aware_pitch_tokens(
        chord,
        pitch_mode="tones_tensions",
        recent_pitches=recent_pitches,
        repeat_window=2,
    )
    target = 64 + rng.choice([-5, -2, 0, 2, 5])
    return nearest_allowed_pitch_token(target, allowed, recent_pitches)


def pitch_tokens_from_contour(
    chord: str | None,
    pitch_intervals: Sequence[int],
    *,
    rng: random.Random,
    recent_pitches: list[int],
    group_offset: int,
) -> list[int]:
    start_token = base_pitch_token_for_chord(chord, rng, recent_pitches)
    start_pitch = pitch_from_token(start_token)
    tokens: list[int] = []
    intervals = [int(interval) for interval in pitch_intervals] or [0]
    for index, interval in enumerate(intervals):
        target_pitch = start_pitch + int(interval)
        allowed = chord_aware_pitch_tokens(
            chord,
            pitch_mode="approach_tensions",
            recent_pitches=recent_pitches,
            repeat_window=2,
            group_index=group_offset + index,
        )
        token = nearest_allowed_pitch_token(target_pitch, allowed, recent_pitches)
        tokens.append(token)
        recent_pitches.append(pitch_from_token(token))
    return tokens


def hand_written_swing_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    contour = [0, 2, 5, 3, 7, 5, 9, 7]
    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        base_token = base_pitch_token_for_chord(chord, rng, recent_pitches)
        base_pitch = pitch_from_token(base_token)
        for group_index in range(max(1, int(note_groups_per_bar))):
            position_candidates = jazz_rhythm_position_tokens(
                bar_index=bar_index,
                group_index=group_index,
                note_groups_per_bar=note_groups_per_bar,
                profile="swing_motif",
            )
            duration_candidates = jazz_rhythm_duration_tokens(
                bar_index=bar_index,
                group_index=group_index,
                note_groups_per_bar=note_groups_per_bar,
                profile="swing_motif",
            )
            target_pitch = base_pitch + contour[group_index % len(contour)]
            pitch_candidates = chord_aware_pitch_tokens(
                chord,
                pitch_mode="approach_tensions",
                recent_pitches=recent_pitches,
                repeat_window=2,
                group_index=group_index,
            )
            pitch_token_value = nearest_allowed_pitch_token(target_pitch, pitch_candidates, recent_pitches)
            recent_pitches.append(pitch_from_token(pitch_token_value))
            tokens.extend(
                [
                    position_candidates[len(position_candidates) // 2],
                    note_velocity_token(4),
                    pitch_token_value,
                    duration_candidates[len(duration_candidates) // 2],
                ]
            )
    tokens.append(TOKEN_END)
    return tokens


def straight_grid_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    contour = [0, 2, 4, 5, 7, 5, 4, 2]
    groups_per_bar = max(1, int(note_groups_per_bar))
    duration_steps = max(1, int(POSITIONS_PER_BAR) // groups_per_bar)
    positions = [min(int(POSITIONS_PER_BAR) - 1, index * duration_steps) for index in range(groups_per_bar)]

    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        base_token = base_pitch_token_for_chord(chord, rng, recent_pitches)
        base_pitch = pitch_from_token(base_token)
        for group_index, position in enumerate(positions):
            target_pitch = base_pitch + contour[group_index % len(contour)]
            pitch_candidates = chord_aware_pitch_tokens(
                chord,
                pitch_mode="approach_tensions",
                recent_pitches=recent_pitches,
                repeat_window=2,
                group_index=group_index,
            )
            pitch_token_value = nearest_allowed_pitch_token(target_pitch, pitch_candidates, recent_pitches)
            recent_pitches.append(pitch_from_token(pitch_token_value))
            tokens.extend(
                [
                    position_token(position),
                    note_velocity_token(4),
                    pitch_token_value,
                    note_duration_token(duration_steps),
                ]
            )
    tokens.append(TOKEN_END)
    return tokens


def straight_guide_tones_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    groups_per_bar = max(1, int(note_groups_per_bar))
    duration_steps = max(1, int(POSITIONS_PER_BAR) // groups_per_bar)
    positions = [min(int(POSITIONS_PER_BAR) - 1, index * duration_steps) for index in range(groups_per_bar)]

    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        next_chord = chords[(bar_index + 1) % len(chords)] if chords else chord
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        pitch_cells = guide_tone_cadence_pitch_classes(chord, next_chord, bar_index=bar_index)
        anchor = 64 + ((bar_index % 3) * 3) + rng.choice([-2, 0, 2])
        for group_index, position in enumerate(positions):
            pitch_class, role = pitch_cells[group_index % len(pitch_cells)]
            if role.startswith("approach") and group_index > 0:
                target_anchor = recent_pitches[-1] + rng.choice([-2, -1, 1, 2])
            else:
                target_anchor = anchor + (group_index % 4) * 2
            pitch = nearest_pitch_for_pitch_class(
                pitch_class,
                target_pitch=target_anchor,
                recent_pitches=recent_pitches,
            )
            recent_pitches.append(pitch)
            tokens.extend(
                [
                    position_token(position),
                    note_velocity_token(4),
                    note_pitch_token(pitch),
                    note_duration_token(duration_steps),
                ]
            )
    tokens.append(TOKEN_END)
    return tokens


def data_motif_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    template_report: dict[str, Any],
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    summary = template_report["summary"]
    rhythm_rows = summary["top_rhythm_templates"]
    contour_rows = summary["top_contour_templates"]
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    motif_length = 4
    motifs_per_bar = max(1, int(round(max(1, int(note_groups_per_bar)) / motif_length)))
    slot_size = max(1, int(POSITIONS_PER_BAR) // motifs_per_bar)

    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        emitted_in_bar = 0
        for motif_index in range(motifs_per_bar):
            row_index = bar_index * motifs_per_bar + motif_index
            rhythm = weighted_choice(rhythm_rows, rng, row_index)["key"]
            contour = weighted_choice(contour_rows, rng, row_index + int(seed))["key"]
            slot_start = min(int(POSITIONS_PER_BAR) - 1, motif_index * slot_size)
            positions = normalize_position_deltas(
                rhythm["position_deltas"],
                slot_start=slot_start,
                slot_size=slot_size,
            )
            durations = fit_duration_tokens_to_positions(positions, rhythm["duration_steps"])
            pitch_tokens = pitch_tokens_from_contour(
                chord,
                contour["pitch_intervals"],
                rng=rng,
                recent_pitches=recent_pitches,
                group_offset=emitted_in_bar,
            )
            for position, duration_token, pitch_token_value in zip(positions, durations, pitch_tokens):
                if emitted_in_bar >= int(note_groups_per_bar):
                    break
                tokens.extend(
                    [
                        position_token(position),
                        note_velocity_token(4),
                        pitch_token_value,
                        duration_token,
                    ]
                )
                emitted_in_bar += 1
    tokens.append(TOKEN_END)
    return tokens


def data_motif_guide_tones_tokens(
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    template_report: dict[str, Any],
    seed: int,
) -> list[int]:
    rng = random.Random(int(seed))
    summary = template_report["summary"]
    rhythm_rows = summary["top_rhythm_templates"]
    contour_rows = summary["top_contour_templates"]
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    motif_length = 4
    motifs_per_bar = max(1, int(round(max(1, int(note_groups_per_bar)) / motif_length)))
    slot_size = max(1, int(POSITIONS_PER_BAR) // motifs_per_bar)

    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        next_chord = chords[(bar_index + 1) % len(chords)] if chords else chord
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        emitted_in_bar = 0
        for motif_index in range(motifs_per_bar):
            row_index = bar_index * motifs_per_bar + motif_index
            rhythm = weighted_choice(rhythm_rows, rng, row_index)["key"]
            contour = weighted_choice(contour_rows, rng, row_index + int(seed))["key"]
            slot_start = min(int(POSITIONS_PER_BAR) - 1, motif_index * slot_size)
            positions = normalize_position_deltas(
                rhythm["position_deltas"],
                slot_start=slot_start,
                slot_size=slot_size,
            )
            durations = fit_duration_tokens_to_positions(positions, rhythm["duration_steps"])
            contour_steps = [int(step) for step in contour.get("pitch_intervals", [])] or [0]
            anchor = 64 + ((bar_index % 3) * 3) + rng.choice([-2, 0, 2])
            for local_index, (position, duration_token) in enumerate(zip(positions, durations)):
                if emitted_in_bar >= int(note_groups_per_bar):
                    break
                contour_offset = contour_steps[local_index % len(contour_steps)]
                target_pitch = anchor + contour_offset
                if recent_pitches:
                    target_pitch = int(round((target_pitch + int(recent_pitches[-1])) / 2))
                pitch = guide_tone_pitch_for_position(
                    chord,
                    next_chord,
                    bar_index=bar_index,
                    position=int(position),
                    target_pitch=target_pitch,
                    recent_pitches=recent_pitches,
                )
                recent_pitches.append(pitch)
                tokens.extend(
                    [
                        position_token(position),
                        note_velocity_token(4),
                        note_pitch_token(pitch),
                        duration_token,
                    ]
                )
                emitted_in_bar += 1
    tokens.append(TOKEN_END)
    return tokens


def generated_tokens_for_mode(
    mode: str,
    *,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bars: int,
    note_groups_per_bar: int,
    template_report: dict[str, Any],
    seed: int,
) -> list[int]:
    if mode == "straight_grid":
        return straight_grid_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            seed=seed,
        )
    if mode == "straight_guide_tones":
        return straight_guide_tones_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            seed=seed,
        )
    if mode == "hand_written_swing":
        return hand_written_swing_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            seed=seed,
        )
    if mode == "data_motif":
        return data_motif_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            template_report=template_report,
            seed=seed,
        )
    if mode == "data_motif_guide_tones":
        return data_motif_guide_tones_tokens(
            primer_tokens=primer_tokens,
            chords=chords,
            bars=bars,
            note_groups_per_bar=note_groups_per_bar,
            template_report=template_report,
            seed=seed,
        )
    raise ValueError(f"unknown baseline mode: {mode}")


def compact_summary(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_count": int(summary["sample_count"]),
        "valid_sample_count": int(summary["valid_sample_count"]),
        "strict_valid_sample_count": int(summary["strict_valid_sample_count"]),
        "avg_syncopated_onset_ratio": float(summary["avg_syncopated_onset_ratio"]),
        "avg_unique_bar_position_pattern_ratio": float(summary["avg_unique_bar_position_pattern_ratio"]),
        "avg_duration_diversity_ratio": float(summary["avg_duration_diversity_ratio"]),
        "avg_most_common_duration_ratio": float(summary["avg_most_common_duration_ratio"]),
        "avg_ioi_diversity_ratio": float(summary["avg_ioi_diversity_ratio"]),
        "avg_most_common_ioi_ratio": float(summary["avg_most_common_ioi_ratio"]),
        "avg_tension_ratio": float(summary["avg_tension_ratio"]),
        "avg_root_tone_ratio": float(summary["avg_root_tone_ratio"]),
        "passed_strict_review_gate": bool(summary["passed_strict_review_gate"]),
    }


def ordered_chord_pitches(chord: str | None, lower: int = 48, upper: int = 72) -> list[int]:
    root, quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root)
    if root_pc is None:
        root_pc = 0
    intervals = sorted(CHORD_TONE_INTERVALS.get(quality, CHORD_TONE_INTERVALS["unknown"]))
    base = lower + root_pc
    while base > lower + 11:
        base -= 12
    pitches: list[int] = []
    for interval in intervals:
        pitch = base + int(interval)
        while pitch > int(upper):
            pitch -= 12
        while pitch < int(lower):
            pitch += 12
        pitches.append(pitch)
    return sorted(set(pitches))


def bass_root_pitch(chord: str | None, lower: int = 36, upper: int = 47) -> int:
    root, _quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root, 0)
    pitch = lower + root_pc
    while pitch > upper:
        pitch -= 12
    while pitch < lower:
        pitch += 12
    return int(pitch)


def chord_guide_instruments(
    chords: Sequence[str],
    *,
    bpm: float | int,
    bars: int,
) -> list[pretty_midi.Instrument]:
    bar_duration_sec = 60.0 / max(1e-6, float(bpm)) * 4.0
    pad = pretty_midi.Instrument(program=4, is_drum=False, name="Chord Guide")
    bass = pretty_midi.Instrument(program=32, is_drum=False, name="Bass Root Guide")
    total_bars = max(1, int(bars))
    for bar_index in range(total_bars):
        chord = chords[bar_index % len(chords)] if chords else "Cmaj7"
        start = bar_index * bar_duration_sec
        end = start + bar_duration_sec
        for pitch in ordered_chord_pitches(chord):
            pad.notes.append(pretty_midi.Note(velocity=42, pitch=int(pitch), start=start, end=end))
        bass.notes.append(
            pretty_midi.Note(
                velocity=52,
                pitch=bass_root_pitch(chord),
                start=start,
                end=end,
            )
        )
    return [pad, bass]


def write_chord_guide_midi(
    output_path: Path,
    chords: Sequence[str],
    *,
    bpm: float | int,
    bars: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    midi.instruments.extend(chord_guide_instruments(chords, bpm=bpm, bars=bars))
    midi.write(str(output_path))
    return output_path


def write_context_midi(
    solo_midi_path: Path,
    output_path: Path,
    chords: Sequence[str],
    *,
    bpm: float | int,
    bars: int,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    context = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    context.instruments.extend(chord_guide_instruments(chords, bpm=bpm, bars=bars))
    solo = pretty_midi.PrettyMIDI(str(solo_midi_path))
    for instrument in solo.instruments:
        copied = copy.deepcopy(instrument)
        copied.name = f"Solo - {copied.name or 'Stage B'}"
        context.instruments.append(copied)
    context.write(str(output_path))
    return output_path


def _best_note_for_onset(notes: Sequence[pretty_midi.Note]) -> pretty_midi.Note:
    return max(
        notes,
        key=lambda note: (
            int(note.velocity),
            float(note.end) - float(note.start),
            -abs(int(note.pitch) - 67),
            -int(note.pitch),
        ),
    )


def overlap_free_solo_notes(
    notes: Sequence[pretty_midi.Note],
    *,
    time_precision: int = 6,
    min_duration_sec: float = 0.001,
) -> tuple[list[pretty_midi.Note], dict[str, Any]]:
    valid_notes = [note for note in notes if float(note.end) > float(note.start)]
    by_onset: dict[float, list[pretty_midi.Note]] = {}
    for note in valid_notes:
        onset = round(float(note.start), int(time_precision))
        by_onset.setdefault(onset, []).append(note)

    selected = [
        _best_note_for_onset(onset_notes)
        for _onset, onset_notes in sorted(by_onset.items(), key=lambda item: item[0])
    ]

    solo_notes: list[pretty_midi.Note] = []
    trimmed_note_count = 0
    for index, note in enumerate(selected):
        start = float(note.start)
        original_end = float(note.end)
        next_start = float(selected[index + 1].start) if index + 1 < len(selected) else None
        end = original_end
        if next_start is not None and end > next_start:
            end = next_start
            trimmed_note_count += 1
        if end <= start:
            continue
        if end - start < float(min_duration_sec):
            end = start + float(min_duration_sec)
            if next_start is not None and end > next_start:
                continue
        solo_notes.append(
            pretty_midi.Note(
                velocity=int(note.velocity),
                pitch=int(note.pitch),
                start=start,
                end=end,
            )
        )

    report = {
        "before_note_count": int(len(valid_notes)),
        "after_note_count": int(len(solo_notes)),
        "same_onset_dropped_note_count": int(max(0, len(valid_notes) - len(selected))),
        "trimmed_note_count": int(trimmed_note_count),
        "before_max_simultaneous_notes": int(max_simultaneous_notes(list(valid_notes))) if valid_notes else 0,
        "after_max_simultaneous_notes": int(max_simultaneous_notes(list(solo_notes))) if solo_notes else 0,
    }
    return solo_notes, report


def write_overlap_free_solo_midi(
    source_path: Path,
    output_path: Path,
    *,
    bpm: float | int | None = None,
) -> dict[str, Any]:
    source = pretty_midi.PrettyMIDI(str(source_path))
    tempo = float(bpm) if bpm is not None else float(source.estimate_tempo() or 120.0)
    result = pretty_midi.PrettyMIDI(initial_tempo=tempo)
    source_instruments = [instrument for instrument in source.instruments if not instrument.is_drum]
    source_notes = [note for instrument in source_instruments for note in instrument.notes]
    solo_notes, report = overlap_free_solo_notes(source_notes)
    program = int(source_instruments[0].program) if source_instruments else 0
    name = source_instruments[0].name if source_instruments else "Solo"
    copied = pretty_midi.Instrument(
        program=program,
        is_drum=False,
        name=f"Overlap-free {name or 'Solo'}",
    )
    copied.notes = solo_notes
    result.instruments.append(copied)
    total_report = {
        "enabled": True,
        "source_midi_path": str(source_path),
        "output_midi_path": str(output_path),
        "bpm": tempo,
        "before_note_count": int(report["before_note_count"]),
        "after_note_count": int(report["after_note_count"]),
        "same_onset_dropped_note_count": int(report["same_onset_dropped_note_count"]),
        "trimmed_note_count": int(report["trimmed_note_count"]),
        "before_max_simultaneous_notes": int(report["before_max_simultaneous_notes"]),
        "after_max_simultaneous_notes": int(report["after_max_simultaneous_notes"]),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.write(str(output_path))
    return total_report


def build_compare_summary(
    mode_summaries: dict[str, dict[str, Any]],
    min_strict_valid_samples: int,
) -> dict[str, Any]:
    hand = mode_summaries.get("hand_written_swing")
    data = mode_summaries.get("data_motif")
    ready = bool(hand and data)
    data_passed = bool(data and int(data["strict_valid_sample_count"]) >= int(min_strict_valid_samples))
    hand_passed = bool(hand and int(hand["strict_valid_sample_count"]) >= int(min_strict_valid_samples))
    def delta(metric: str) -> float:
        return float(data.get(metric, 0.0) - hand.get(metric, 0.0)) if ready and data and hand else 0.0

    return {
        "comparison_ready": ready,
        "passed_hand_written_swing_gate": hand_passed,
        "passed_data_motif_gate": data_passed,
        "passed_compare_gate": bool(ready and hand_passed and data_passed),
        "duration_diversity_delta_data_minus_hand": delta("avg_duration_diversity_ratio"),
        "ioi_diversity_delta_data_minus_hand": delta("avg_ioi_diversity_ratio"),
        "bar_pattern_delta_data_minus_hand": delta("avg_unique_bar_position_pattern_ratio"),
        "syncopation_delta_data_minus_hand": delta("avg_syncopated_onset_ratio"),
        "mode_summaries": {mode: compact_summary(summary) for mode, summary in sorted(mode_summaries.items())},
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Stage B Data Motif Generation Compare",
        "",
        f"- passed compare gate: `{str(summary['passed_compare_gate']).lower()}`",
        f"- duration diversity delta, data minus hand: `{summary['duration_diversity_delta_data_minus_hand']:.3f}`",
        f"- IOI diversity delta, data minus hand: `{summary['ioi_diversity_delta_data_minus_hand']:.3f}`",
        f"- bar-pattern delta, data minus hand: `{summary['bar_pattern_delta_data_minus_hand']:.3f}`",
        f"- syncopation delta, data minus hand: `{summary['syncopation_delta_data_minus_hand']:.3f}`",
        "",
        "| mode | samples | strict | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | tension | root |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, row in sorted(summary["mode_summaries"].items()):
        lines.append(
            "| {mode} | {sample_count} | {strict_valid_sample_count} | "
            "{avg_syncopated_onset_ratio:.3f} | {avg_unique_bar_position_pattern_ratio:.3f} | "
            "{avg_duration_diversity_ratio:.3f} | {avg_most_common_duration_ratio:.3f} | "
            "{avg_ioi_diversity_ratio:.3f} | {avg_most_common_ioi_ratio:.3f} | "
            "{avg_tension_ratio:.3f} | {avg_root_tone_ratio:.3f} |".format(mode=mode, **row)
        )
    return "\n".join(lines).rstrip() + "\n"


def candidate_sort_key(row: dict[str, Any]) -> tuple[int, int, float, float, float]:
    rhythm = row.get("rhythm_profile", {})
    return (
        int(bool(row.get("strict_valid"))),
        int(bool(row.get("valid"))),
        float(rhythm.get("unique_bar_position_pattern_ratio", 0.0) or 0.0),
        float(rhythm.get("duration_diversity_ratio", 0.0) or 0.0),
        -float(rhythm.get("most_common_duration_ratio", 0.0) or 0.0),
    )


def compact_review_candidate(
    mode: str,
    row: dict[str, Any],
    *,
    review_rank: int,
    review_midi_path: Path | None,
    context_midi_path: Path | None,
    review_variant: str = "original",
    review_postprocess_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rhythm = row.get("rhythm_profile", {})
    pitch_roles = row.get("pitch_roles", {})
    metrics = row.get("metrics", {})
    return {
        "mode": mode,
        "review_rank": int(review_rank),
        "sample_index": int(row["sample_index"]),
        "sample_seed": int(row["sample_seed"]),
        "valid": bool(row["valid"]),
        "strict_valid": bool(row["strict_valid"]),
        "midi_path": row["midi_path"],
        "review_midi_path": str(review_midi_path) if review_midi_path else None,
        "context_midi_path": str(context_midi_path) if context_midi_path else None,
        "review_variant": review_variant,
        "review_postprocess_report": review_postprocess_report or {},
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "syncopated_onset_ratio": float(rhythm.get("syncopated_onset_ratio", 0.0) or 0.0),
        "unique_bar_position_pattern_ratio": float(
            rhythm.get("unique_bar_position_pattern_ratio", 0.0) or 0.0
        ),
        "duration_diversity_ratio": float(rhythm.get("duration_diversity_ratio", 0.0) or 0.0),
        "most_common_duration_ratio": float(rhythm.get("most_common_duration_ratio", 0.0) or 0.0),
        "ioi_diversity_ratio": float(rhythm.get("ioi_diversity_ratio", 0.0) or 0.0),
        "most_common_ioi_ratio": float(rhythm.get("most_common_ioi_ratio", 0.0) or 0.0),
        "tension_ratio": float(pitch_roles.get("tension_ratio", 0.0) or 0.0),
        "root_tone_ratio": float(pitch_roles.get("root_tone_ratio", 0.0) or 0.0),
        "diagnostic_failure_reason": row.get("diagnostic_failure_reason"),
    }


def review_markdown_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# Stage B Data Motif Review Candidates",
        "",
        f"- candidate count: `{manifest['candidate_count']}`",
        f"- copy midi: `{str(manifest['copy_midi']).lower()}`",
        "",
        "| mode | rank | sample | variant | strict | notes | pitches | sync | bar-var | dur-var | dur-rep | ioi-var | ioi-rep | tension | solo/context MIDI |",
        "|---|---:|---:|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    guide_path = manifest.get("chord_guide_midi_path")
    if guide_path:
        lines.insert(4, f"- chord guide midi: `{guide_path}`")
        lines.insert(5, "")
    for row in manifest["candidates"]:
        midi_path = row.get("review_midi_path") or row.get("midi_path")
        context_path = row.get("context_midi_path") or ""
        format_row = dict(row)
        format_row["display_midi_path"] = midi_path
        format_row["display_context_midi_path"] = context_path
        lines.append(
            "| {mode} | {review_rank} | {sample_index} | {review_variant} | {strict_valid} | {note_count} | "
            "{unique_pitch_count} | {syncopated_onset_ratio:.3f} | "
            "{unique_bar_position_pattern_ratio:.3f} | {duration_diversity_ratio:.3f} | "
            "{most_common_duration_ratio:.3f} | {ioi_diversity_ratio:.3f} | "
            "{most_common_ioi_ratio:.3f} | {tension_ratio:.3f} | "
            "`{display_midi_path}`<br>`{display_context_midi_path}` |".format(**format_row)
        )
    return "\n".join(lines).rstrip() + "\n"


def build_review_export(
    samples_by_mode: dict[str, list[dict[str, Any]]],
    *,
    output_dir: Path,
    top_n: int,
    copy_midi: bool,
    chords: Sequence[str],
    bpm: float | int,
    bars: int,
    overlap_free_review_midi: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    named_dir = output_dir / "named_midi"
    context_dir = output_dir / "context_midi"
    if copy_midi:
        named_dir.mkdir(parents=True, exist_ok=True)
        context_dir.mkdir(parents=True, exist_ok=True)

    chord_guide_midi_path: Path | None = None
    if copy_midi:
        chord_guide_midi_path = write_chord_guide_midi(
            output_dir / "chord_guide.mid",
            chords,
            bpm=bpm,
            bars=bars,
        )

    candidates: list[dict[str, Any]] = []
    for mode_index, mode in enumerate(sorted(samples_by_mode), start=1):
        rows = sorted(samples_by_mode[mode], key=candidate_sort_key, reverse=True)
        for review_rank, row in enumerate(rows[: int(top_n)], start=1):
            review_midi_path: Path | None = None
            context_midi_path: Path | None = None
            review_variant = "original"
            review_postprocess_report: dict[str, Any] | None = None
            if copy_midi:
                source = Path(str(row["midi_path"]))
                if not source.is_absolute():
                    source = ROOT_DIR / source
                base_name = f"{mode_index:02d}_{mode}_rank_{review_rank:02d}_sample_{int(row['sample_index']):02d}"
                suffix = "_overlap_free.mid" if overlap_free_review_midi else ".mid"
                target = named_dir / f"{base_name}{suffix}"
                if source.exists():
                    if overlap_free_review_midi:
                        review_postprocess_report = write_overlap_free_solo_midi(source, target, bpm=bpm)
                        review_variant = "overlap_free_solo_line"
                    else:
                        shutil.copy2(source, target)
                    review_midi_path = target
                    context_suffix = "_overlap_free_with_context.mid" if overlap_free_review_midi else "_with_context.mid"
                    context_target = context_dir / f"{base_name}{context_suffix}"
                    context_midi_path = write_context_midi(
                        target,
                        context_target,
                        chords,
                        bpm=bpm,
                        bars=bars,
                    )
            candidates.append(
                compact_review_candidate(
                    mode,
                    row,
                    review_rank=review_rank,
                    review_midi_path=review_midi_path,
                    context_midi_path=context_midi_path,
                    review_variant=review_variant,
                    review_postprocess_report=review_postprocess_report,
                )
            )

    manifest = {
        "output_dir": str(output_dir),
        "copy_midi": bool(copy_midi),
        "overlap_free_review_midi": bool(overlap_free_review_midi),
        "top_n": int(top_n),
        "chord_progression": list(chords),
        "chord_guide_midi_path": str(chord_guide_midi_path) if chord_guide_midi_path else None,
        "candidate_count": int(len(candidates)),
        "candidates": candidates,
    }
    write_json(output_dir / "review_manifest.json", manifest)
    (output_dir / "review_candidates.md").write_text(review_markdown_report(manifest), encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B data-derived motif baseline generation compare")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_data_motif_compare"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio")
    parser.add_argument("--issue_number", type=int, default=65)
    parser.add_argument("--baseline_modes", type=str, default="straight_grid,hand_written_swing,data_motif")
    parser.add_argument("--max_files", type=int, default=4)
    parser.add_argument("--window_bars", type=int, default=8)
    parser.add_argument("--window_stride_bars", type=int, default=4)
    parser.add_argument("--min_window_target_notes", type=int, default=16)
    parser.add_argument("--motif_length", type=int, default=4)
    parser.add_argument("--max_bar_span", type=int, default=2)
    parser.add_argument("--max_records", type=int, default=64)
    parser.add_argument("--template_top_n", type=int, default=32)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--chords", type=str, default="Cm7,Fm7,Bb7,Ebmaj7")
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--max_sequence", type=int, default=384)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--review_top_n", type=int, default=3)
    parser.add_argument("--review_output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_data_motif_review"))
    parser.add_argument("--copy_review_midi", action="store_true")
    parser.add_argument("--overlap_free_review_midi", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / args.run_id
    samples_dir = run_dir / "samples"
    modes = parse_baseline_modes(args.baseline_modes)
    chords = parse_chords(args.chords)
    request = GenerationRequest(
        bpm=int(args.bpm),
        chord_progression=chords,
        bars=int(args.bars),
        density=args.density,
        energy=args.energy,
        temperature=float(args.temperature),
        top_k=int(args.top_k),
        seed=int(args.seed),
    )
    request.validate()

    template_report_path, template_command = run_template_extraction(args, run_dir)
    report: dict[str, Any] = {
        "run_id": args.run_id,
        "run_dir": str(run_dir),
        "issue": int(args.issue_number),
        "baseline_modes": modes,
        "template_report_path": str(template_report_path),
        "template_command": template_command,
        "chords": chords,
        "bars": int(args.bars),
        "note_groups_per_bar": int(args.note_groups_per_bar),
        "samples": {},
    }
    if template_command["returncode"] != 0:
        report["failure_reason"] = "motif template extraction failed"
        write_json(run_dir / "data_motif_compare_report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return int(template_command["returncode"])

    template_report = read_json(template_report_path)
    primer_tokens = build_stage_b_primer(chords, args.bpm)
    mode_summaries: dict[str, dict[str, Any]] = {}
    samples_by_mode: dict[str, list[dict[str, Any]]] = {}
    for mode in modes:
        rows: list[dict[str, Any]] = []
        for index in range(1, int(args.num_samples) + 1):
            sample_seed = int(args.seed) + index - 1
            tokens = generated_tokens_for_mode(
                mode,
                primer_tokens=primer_tokens,
                chords=chords,
                bars=int(args.bars),
                note_groups_per_bar=int(args.note_groups_per_bar),
                template_report=template_report,
                seed=sample_seed,
            )[: int(args.max_sequence)]
            midi_path = samples_dir / mode / f"{mode}_sample_{index}.mid"
            midi_path.parent.mkdir(parents=True, exist_ok=True)
            midi = decode_stage_b_midi(tokens, tempo_bpm=args.bpm)
            postprocess_report = postprocess_stage_b_midi(
                midi,
                simultaneous_limit=int(args.max_simultaneous_notes),
            )
            midi.write(str(midi_path))
            rows.append(
                sample_report(
                    sample_index=index,
                    sample_seed=sample_seed,
                    tokens=tokens,
                    primer_size=len(primer_tokens),
                    target_length=int(args.max_sequence),
                    midi_path=midi_path,
                    request=request,
                    postprocess_report=postprocess_report,
                )
            )
        summary = build_probe_summary(
            rows,
            min_valid_samples=1,
            min_strict_valid_samples=int(args.min_strict_valid_samples),
            require_all_grammar_samples=True,
        )
        report["samples"][mode] = rows
        samples_by_mode[mode] = rows
        mode_summaries[mode] = summary

    report["summary"] = build_compare_summary(
        mode_summaries,
        min_strict_valid_samples=int(args.min_strict_valid_samples),
    )
    report["review_export"] = build_review_export(
        samples_by_mode,
        output_dir=Path(args.review_output_root) / args.run_id,
        top_n=int(args.review_top_n),
        copy_midi=bool(args.copy_review_midi),
        overlap_free_review_midi=bool(args.overlap_free_review_midi),
        chords=chords,
        bpm=args.bpm,
        bars=args.bars,
    )
    write_json(run_dir / "data_motif_compare_report.json", report)
    (run_dir / "data_motif_compare_report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=True, indent=2))
    return 0 if report["summary"]["passed_compare_gate"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
