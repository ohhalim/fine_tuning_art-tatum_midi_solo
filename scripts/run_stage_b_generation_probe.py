"""
Run a Stage B decode/generation probe.

This script prepares short stage_b_v1 phrase windows, optionally trains a tiny
checkpoint, samples Stage B tokens with the full model vocabulary, decodes those
tokens back to MIDI, and records whether the output passes the existing MIDI
quality gates.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pretty_midi
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "music_transformer"))

from inference.app.metrics import compute_midi_metrics, max_simultaneous_notes, validate_metrics  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.control_tokens import control_prefix_tokens  # noqa: E402
from scripts.generate import load_model_with_lora  # noqa: E402
from scripts.run_stage_a_tiny_overfit import build_train_command, run_command, write_json  # noqa: E402
from scripts.run_stage_b_window_tiny_overfit import read_json, run_prepare_command, token_stats  # noqa: E402
from scripts.stage_b_tokens import (  # noqa: E402
    TOKEN_NOTE_DURATION_END,
    TOKEN_NOTE_DURATION_START,
    TOKEN_NOTE_PITCH_END,
    TOKEN_NOTE_PITCH_START,
    TOKEN_POSITION_END,
    TOKEN_POSITION_START,
    TOKEN_VELOCITY_END,
    TOKEN_VELOCITY_START,
    TOKEN_CHORD_QUALITY_END,
    TOKEN_CHORD_QUALITY_START,
    TOKEN_CHORD_ROOT_END,
    TOKEN_CHORD_ROOT_START,
    SEQUENCE_FORMAT_STAGE_B_V1,
    MAX_DURATION_STEPS,
    POSITIONS_PER_BAR,
    PIANO_PITCH_MAX,
    PIANO_PITCH_MIN,
    ROOT_TO_PC,
    chord_tokens,
    decode_stage_b_midi,
    duration_steps_from_token,
    is_note_duration_token,
    is_note_pitch_token,
    is_position_token,
    is_velocity_token,
    note_duration_token,
    note_pitch_token,
    parse_chord_symbol,
    pitch_from_token,
    position_from_token,
    stage_b_token_name,
)
from utilities.constants import TOKEN_BAR, TOKEN_END, VOCAB_SIZE  # noqa: E402
from utilities.device import get_device  # noqa: E402


DEFAULT_STRICT_MIN_UNIQUE_PITCHES = 3
DEFAULT_STRICT_MIN_UNIQUE_POSITIONS = 3
DEFAULT_STRICT_MIN_UNIQUE_POSITION_PITCH_PAIRS = 4
DEFAULT_STRICT_MAX_REPEATED_POSITION_PITCH_PAIR_RATIO = 0.49
DEFAULT_STRICT_MAX_POSTPROCESS_REMOVAL_RATIO = 0.49
DEFAULT_STRICT_MAX_COLLAPSE_WARNING_SAMPLE_RATE = 0.34

CHORD_TONE_INTERVALS = {
    "maj": {0, 4, 7},
    "maj7": {0, 4, 7, 11},
    "min": {0, 3, 7},
    "min7": {0, 3, 7, 10},
    "dom7": {0, 4, 7, 10},
    "dim": {0, 3, 6},
    "halfdim": {0, 3, 6, 10},
    "sus": {0, 5, 7},
    "unknown": {0, 4, 7},
}

TENSION_INTERVALS = {
    "maj": {2, 9},
    "maj7": {2, 6, 9},
    "min": {2, 5, 10},
    "min7": {2, 5},
    "dom7": {2, 5, 9},
    "dim": {9},
    "halfdim": {2, 5},
    "sus": {2, 10},
    "unknown": set(),
}


def parse_chords(raw_chords: str) -> list[str]:
    return [chord.strip() for chord in raw_chords.split(",") if chord.strip()]


def build_stage_b_primer(chords: Sequence[str], bpm: float | int, role: str = "lead") -> list[int]:
    first_chord = chords[0] if chords else None
    return control_prefix_tokens(role=role, tempo_bpm=bpm) + chord_tokens(first_chord)


def generate_stage_b_tokens(
    model: Any,
    primer_tokens: Sequence[int],
    target_length: int,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
) -> list[int]:
    primer = torch.tensor([int(token) for token in primer_tokens], dtype=torch.long, device=get_device())
    with torch.no_grad():
        generated = model.generate(
            primer=primer,
            target_seq_length=int(target_length),
            beam=0,
            beam_chance=1.0,
            temperature=float(temperature),
            top_k=top_k,
            top_p=top_p,
            sample_vocab_size=VOCAB_SIZE,
        )
    return [int(token) for token in generated[0].detach().cpu().tolist()]


def token_family(token: int) -> str:
    if int(token) == TOKEN_END:
        return "end"
    if int(token) == TOKEN_BAR:
        return "bar"
    if TOKEN_CHORD_ROOT_START <= int(token) <= TOKEN_CHORD_ROOT_END:
        return "chord"
    if TOKEN_CHORD_QUALITY_START <= int(token) <= TOKEN_CHORD_QUALITY_END:
        return "chord"
    if is_position_token(token):
        return "position"
    if is_velocity_token(token):
        return "velocity"
    if is_note_pitch_token(token):
        return "pitch"
    if is_note_duration_token(token):
        return "duration"
    return "other"


def analyze_stage_b_note_grammar(tokens: Sequence[int], primer_size: int = 0) -> dict[str, Any]:
    generated = [int(token) for token in tokens[int(primer_size) :]]
    expected = ["position", "velocity", "pitch", "duration"]
    expected_index = 0
    complete_groups = 0
    invalid_tokens: list[dict[str, Any]] = []
    family_counts: dict[str, int] = {}

    for offset, token in enumerate(generated):
        family = token_family(token)
        family_counts[family] = family_counts.get(family, 0) + 1
        if family == "end":
            break
        if family in {"bar", "chord"}:
            expected_index = 0
            continue
        expected_family = expected[expected_index]
        if family == expected_family:
            expected_index += 1
            if expected_index == len(expected):
                complete_groups += 1
                expected_index = 0
            continue
        invalid_tokens.append(
            {
                "offset": int(offset),
                "token": int(token),
                "token_name": stage_b_token_name(token),
                "family": family,
                "expected": expected_family,
            }
        )
        expected_index = 1 if family == "position" else 0

    return {
        "complete_note_groups": int(complete_groups),
        "incomplete_group_position": int(expected_index),
        "invalid_token_count": int(len(invalid_tokens)),
        "family_counts": family_counts,
        "invalid_tokens_head": invalid_tokens[:12],
        "grammar_valid": bool(complete_groups > 0 and not invalid_tokens and expected_index == 0),
    }


def extract_stage_b_note_groups(tokens: Sequence[int], primer_size: int = 0) -> list[dict[str, int]]:
    generated = [int(token) for token in tokens[int(primer_size) :]]
    note_groups: list[dict[str, int]] = []
    bar_index = 0
    current_position: int | None = None
    current_velocity: int | None = None
    current_pitch: int | None = None

    for token in generated:
        if int(token) == TOKEN_END:
            break
        if int(token) == TOKEN_BAR:
            bar_index += 1
            current_position = None
            current_velocity = None
            current_pitch = None
            continue
        if token_family(token) == "chord":
            continue
        if is_position_token(token):
            current_position = position_from_token(token)
            current_velocity = None
            current_pitch = None
            continue
        if is_velocity_token(token):
            current_velocity = int(token)
            continue
        if is_note_pitch_token(token):
            current_pitch = pitch_from_token(token)
            continue
        if is_note_duration_token(token) and current_position is not None and current_pitch is not None:
            note_groups.append(
                {
                    "bar": int(bar_index),
                    "position": int(current_position),
                    "pitch": int(current_pitch),
                    "velocity_token": int(current_velocity) if current_velocity is not None else -1,
                    "duration_token": int(token),
                    "duration_steps": int(duration_steps_from_token(token)),
                }
            )
            current_position = None
            current_velocity = None
            current_pitch = None

    return note_groups


def _counter(values: Sequence[Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return counts


def _longest_false_run(values: Sequence[bool]) -> int:
    longest = 0
    current = 0
    for value in values:
        if value:
            current = 0
            continue
        current += 1
        longest = max(longest, current)
    return longest


def analyze_stage_b_temporal_coverage(
    tokens: Sequence[int],
    primer_size: int = 0,
    bars: int = 2,
) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=primer_size)
    total_bars = max(1, int(bars))
    total_positions = total_bars * int(POSITIONS_PER_BAR)
    onset_occupied = [False] * total_positions
    sustained_occupied = [False] * total_positions
    per_bar_positions: dict[int, set[int]] = {bar: set() for bar in range(total_bars)}
    absolute_positions: list[int] = []

    for group in groups:
        bar = int(group["bar"])
        position = int(group["position"])
        if bar < 0 or bar >= total_bars:
            continue
        absolute_position = bar * int(POSITIONS_PER_BAR) + position
        absolute_positions.append(absolute_position)
        per_bar_positions.setdefault(bar, set()).add(position)
        if 0 <= absolute_position < total_positions:
            onset_occupied[absolute_position] = True
            duration_steps = max(1, int(group.get("duration_steps", 1)))
            for step in range(absolute_position, min(total_positions, absolute_position + duration_steps)):
                sustained_occupied[step] = True

    unique_absolute_positions = sorted(set(absolute_positions))
    earliest_position = unique_absolute_positions[0] if unique_absolute_positions else None
    latest_position = unique_absolute_positions[-1] if unique_absolute_positions else None
    span_steps = (
        int(latest_position - earliest_position + 1)
        if earliest_position is not None and latest_position is not None
        else 0
    )

    return {
        "bars": total_bars,
        "positions_per_bar": int(POSITIONS_PER_BAR),
        "total_positions": int(total_positions),
        "note_group_count": int(len(groups)),
        "unique_onset_position_count": int(len(unique_absolute_positions)),
        "onset_coverage_ratio": float(len(unique_absolute_positions) / total_positions),
        "sustained_coverage_ratio": float(sum(1 for value in sustained_occupied if value) / total_positions),
        "earliest_absolute_position": earliest_position,
        "latest_absolute_position": latest_position,
        "position_span_steps": span_steps,
        "position_span_ratio": float(span_steps / total_positions) if total_positions else 0.0,
        "head_empty_steps": int(earliest_position) if earliest_position is not None else total_positions,
        "tail_empty_steps": int(total_positions - latest_position - 1) if latest_position is not None else total_positions,
        "longest_onset_empty_run_steps": int(_longest_false_run(onset_occupied)),
        "longest_sustained_empty_run_steps": int(_longest_false_run(sustained_occupied)),
        "per_bar_unique_onset_positions": {
            str(bar): int(len(per_bar_positions.get(bar, set()))) for bar in range(total_bars)
        },
        "per_bar_onset_coverage_ratio": {
            str(bar): float(len(per_bar_positions.get(bar, set())) / int(POSITIONS_PER_BAR))
            for bar in range(total_bars)
        },
    }


def analyze_stage_b_collapse(
    tokens: Sequence[int],
    primer_size: int = 0,
    postprocess_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=primer_size)
    group_count = int(len(groups))
    pitches = [group["pitch"] for group in groups]
    positions = [group["position"] for group in groups]
    relative_pairs = [(group["position"], group["pitch"]) for group in groups]
    absolute_pairs = [(group["bar"], group["position"], group["pitch"]) for group in groups]
    bar_counts = _counter([group["bar"] for group in groups])
    pitch_counts = _counter(pitches)
    pair_counts = _counter(relative_pairs)

    unique_pitch_count = len(set(pitches))
    unique_position_count = len(set(positions))
    unique_relative_pair_count = len(set(relative_pairs))
    unique_absolute_pair_count = len(set(absolute_pairs))
    repeated_pitch_count = max(0, group_count - unique_pitch_count)
    repeated_position_pitch_pair_count = max(0, group_count - unique_relative_pair_count)
    repeated_absolute_position_pitch_pair_count = max(0, group_count - unique_absolute_pair_count)

    postprocess = postprocess_report or {}
    before_note_count = int(postprocess.get("before_note_count", 0) or 0)
    removed_note_count = int(postprocess.get("removed_note_count", 0) or 0)
    postprocess_removal_ratio = float(removed_note_count / before_note_count) if before_note_count else 0.0

    repeated_pair_ratio = float(repeated_position_pitch_pair_count / group_count) if group_count else 0.0
    repeated_pitch_ratio = float(repeated_pitch_count / group_count) if group_count else 0.0
    repeated_absolute_pair_ratio = (
        float(repeated_absolute_position_pitch_pair_count / group_count) if group_count else 0.0
    )
    max_same_pair_repeats = max(pair_counts.values()) if pair_counts else 0
    max_same_pitch_repeats = max(pitch_counts.values()) if pitch_counts else 0

    collapse_reasons: list[str] = []
    if group_count > 0 and unique_pitch_count <= 1:
        collapse_reasons.append("single_pitch")
    if group_count > 0 and unique_position_count <= 1:
        collapse_reasons.append("single_position")
    if repeated_pair_ratio >= 0.5:
        collapse_reasons.append("repeated_position_pitch")
    if postprocess_removal_ratio >= 0.5:
        collapse_reasons.append("postprocess_removed_majority")

    return {
        "note_group_count": group_count,
        "unique_pitch_count": int(unique_pitch_count),
        "unique_position_count": int(unique_position_count),
        "unique_position_pitch_pair_count": int(unique_relative_pair_count),
        "unique_absolute_position_pitch_pair_count": int(unique_absolute_pair_count),
        "repeated_pitch_count": int(repeated_pitch_count),
        "repeated_position_pitch_pair_count": int(repeated_position_pitch_pair_count),
        "repeated_absolute_position_pitch_pair_count": int(repeated_absolute_position_pitch_pair_count),
        "repeated_pitch_ratio": repeated_pitch_ratio,
        "repeated_position_pitch_pair_ratio": repeated_pair_ratio,
        "repeated_absolute_position_pitch_pair_ratio": repeated_absolute_pair_ratio,
        "max_same_pitch_repeats": int(max_same_pitch_repeats),
        "max_same_position_pitch_pair_repeats": int(max_same_pair_repeats),
        "postprocess_removal_ratio": postprocess_removal_ratio,
        "per_bar_note_counts": {str(key): int(value) for key, value in bar_counts.items()},
        "pitch_counts_head": dict(sorted(pitch_counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
        "position_pitch_pair_counts_head": dict(sorted(pair_counts.items(), key=lambda item: (-item[1], item[0]))[:8]),
        "collapse_warning": bool(collapse_reasons),
        "collapse_reasons": collapse_reasons,
    }


def _sign(value: int) -> int:
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def analyze_stage_b_phrase_contour(
    tokens: Sequence[int],
    primer_size: int = 0,
) -> dict[str, Any]:
    groups = sorted(
        extract_stage_b_note_groups(tokens, primer_size=primer_size),
        key=lambda group: (int(group["bar"]), int(group["position"])),
    )
    pitches = [int(group["pitch"]) for group in groups]
    intervals = [pitches[index + 1] - pitches[index] for index in range(max(0, len(pitches) - 1))]
    interval_count = len(intervals)
    nonzero_intervals = [interval for interval in intervals if interval != 0]
    signs = [_sign(interval) for interval in nonzero_intervals]
    direction_changes = sum(1 for index in range(1, len(signs)) if signs[index] != signs[index - 1])

    longest_same_pitch_run = 0
    current_same_pitch_run = 0
    previous_pitch: int | None = None
    for pitch in pitches:
        current_same_pitch_run = current_same_pitch_run + 1 if pitch == previous_pitch else 1
        longest_same_pitch_run = max(longest_same_pitch_run, current_same_pitch_run)
        previous_pitch = pitch

    adjacent_repeated_pitch_count = sum(1 for interval in intervals if interval == 0)
    stepwise_count = sum(1 for interval in nonzero_intervals if abs(interval) <= 2)
    leap_count = sum(1 for interval in nonzero_intervals if abs(interval) >= 5)

    warning_reasons: list[str] = []
    if interval_count > 0 and adjacent_repeated_pitch_count / interval_count >= 0.4:
        warning_reasons.append("adjacent_pitch_repetition")
    if longest_same_pitch_run >= 4:
        warning_reasons.append("long_same_pitch_run")
    if interval_count >= 8 and len(set(nonzero_intervals)) <= 2:
        warning_reasons.append("low_interval_variety")
    if len(signs) >= 8 and direction_changes / max(1, len(signs) - 1) <= 0.15:
        warning_reasons.append("low_direction_change")

    return {
        "note_group_count": int(len(groups)),
        "interval_count": int(interval_count),
        "unique_interval_count": int(len(set(nonzero_intervals))),
        "pitch_span": int(max(pitches) - min(pitches)) if pitches else 0,
        "adjacent_repeated_pitch_count": int(adjacent_repeated_pitch_count),
        "adjacent_repeated_pitch_ratio": (
            float(adjacent_repeated_pitch_count / interval_count) if interval_count else 0.0
        ),
        "nonzero_interval_ratio": float(len(nonzero_intervals) / interval_count) if interval_count else 0.0,
        "direction_change_count": int(direction_changes),
        "direction_change_ratio": float(direction_changes / max(1, len(signs) - 1)) if signs else 0.0,
        "stepwise_motion_ratio": float(stepwise_count / interval_count) if interval_count else 0.0,
        "leap_motion_ratio": float(leap_count / interval_count) if interval_count else 0.0,
        "longest_same_pitch_run": int(longest_same_pitch_run),
        "contour_warning": bool(warning_reasons),
        "contour_warning_reasons": warning_reasons,
    }


def evaluate_collapse_gate(
    collapse: dict[str, Any],
    min_unique_pitches: int = DEFAULT_STRICT_MIN_UNIQUE_PITCHES,
    min_unique_positions: int = DEFAULT_STRICT_MIN_UNIQUE_POSITIONS,
    min_unique_position_pitch_pairs: int = DEFAULT_STRICT_MIN_UNIQUE_POSITION_PITCH_PAIRS,
    max_repeated_position_pitch_pair_ratio: float = DEFAULT_STRICT_MAX_REPEATED_POSITION_PITCH_PAIR_RATIO,
    max_postprocess_removal_ratio: float = DEFAULT_STRICT_MAX_POSTPROCESS_REMOVAL_RATIO,
) -> dict[str, Any]:
    reasons: list[str] = []
    unique_pitch_count = int(collapse.get("unique_pitch_count", 0) or 0)
    unique_position_count = int(collapse.get("unique_position_count", 0) or 0)
    unique_pair_count = int(collapse.get("unique_position_pitch_pair_count", 0) or 0)
    repeated_pair_ratio = float(collapse.get("repeated_position_pitch_pair_ratio", 0.0) or 0.0)
    postprocess_removal_ratio = float(collapse.get("postprocess_removal_ratio", 0.0) or 0.0)

    if unique_pitch_count < int(min_unique_pitches):
        reasons.append(f"unique pitch count too low: {unique_pitch_count} < {int(min_unique_pitches)}")
    if unique_position_count < int(min_unique_positions):
        reasons.append(f"unique position count too low: {unique_position_count} < {int(min_unique_positions)}")
    if unique_pair_count < int(min_unique_position_pitch_pairs):
        reasons.append(
            f"unique position/pitch pair count too low: {unique_pair_count} < "
            f"{int(min_unique_position_pitch_pairs)}"
        )
    if repeated_pair_ratio > float(max_repeated_position_pitch_pair_ratio):
        reasons.append(
            "repeated position/pitch pair ratio too high: "
            f"{repeated_pair_ratio:.3f} > {float(max_repeated_position_pitch_pair_ratio):.3f}"
        )
    if postprocess_removal_ratio > float(max_postprocess_removal_ratio):
        reasons.append(
            "postprocess removal ratio too high: "
            f"{postprocess_removal_ratio:.3f} > {float(max_postprocess_removal_ratio):.3f}"
        )

    return {
        "passed": not reasons,
        "failure_reasons": reasons,
        "thresholds": {
            "min_unique_pitches": int(min_unique_pitches),
            "min_unique_positions": int(min_unique_positions),
            "min_unique_position_pitch_pairs": int(min_unique_position_pitch_pairs),
            "max_repeated_position_pitch_pair_ratio": float(max_repeated_position_pitch_pair_ratio),
            "max_postprocess_removal_ratio": float(max_postprocess_removal_ratio),
        },
    }


def choose_allowed_token(
    logits: torch.Tensor,
    allowed_tokens: Sequence[int],
    temperature: float,
    top_k: int | None,
) -> int:
    allowed = torch.tensor([int(token) for token in allowed_tokens], dtype=torch.long, device=logits.device)
    allowed_logits = logits.index_select(0, allowed)
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    allowed_logits = allowed_logits / float(temperature)
    if top_k is not None and int(top_k) > 0:
        k = min(int(top_k), int(allowed_logits.numel()))
        top_values, top_indices = torch.topk(allowed_logits, k=k)
        if k == 1:
            return int(allowed[int(top_indices[0])].item())
        probs = torch.softmax(top_values, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1)
        return int(allowed[int(top_indices[int(sampled.item())])].item())
    probs = torch.softmax(allowed_logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return int(allowed[int(sampled.item())].item())


def next_token_from_model(
    model: Any,
    tokens: Sequence[int],
    allowed_tokens: Sequence[int],
    temperature: float,
    top_k: int | None,
) -> int:
    device = get_device()
    input_tokens = torch.tensor([[int(token) for token in tokens]], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(input_tokens)[0, -1, :]
    return choose_allowed_token(logits, allowed_tokens, temperature=temperature, top_k=top_k)


def coverage_aware_position_tokens(
    group_index: int,
    note_groups_per_bar: int,
    position_window: int = 0,
) -> list[int]:
    groups_per_bar = max(1, int(note_groups_per_bar))
    cluster_count = max(1, (groups_per_bar + 1) // 2)
    cluster_index = int(group_index) // 2
    pair_offset = int(group_index) % 2
    anchor = int(round(cluster_index * int(POSITIONS_PER_BAR) / cluster_count))
    target = min(int(POSITIONS_PER_BAR) - 1, anchor + pair_offset)
    window = max(0, int(position_window))
    positions = range(max(0, target - window), min(int(POSITIONS_PER_BAR), target + window + 1))
    return [TOKEN_POSITION_START + int(position) for position in positions]


JAZZ_RHYTHM_POSITION_PATTERNS = {
    "swing_motif": [
        [0, 3, 5, 7, 10, 11, 13, 15],
        [1, 3, 4, 7, 9, 12, 14, 15],
        [0, 2, 5, 6, 8, 11, 13, 14],
        [2, 4, 5, 8, 10, 12, 13, 15],
    ],
    "compact_phrase": [
        [0, 1, 4, 7],
        [1, 2, 5, 8],
        [0, 1, 3, 6],
        [2, 3, 6, 9],
    ],
}

JAZZ_RHYTHM_DURATION_PATTERNS = {
    "swing_motif": [
        [2, 1, 3, 1, 2, 2, 1, 4],
        [1, 3, 1, 2, 2, 1, 3, 2],
        [3, 1, 2, 1, 4, 1, 2, 2],
        [1, 2, 2, 3, 1, 2, 1, 4],
    ],
    "compact_phrase": [
        [1, 2, 2, 3],
        [2, 1, 2, 3],
        [1, 2, 3, 2],
        [2, 1, 3, 2],
    ],
}


def _pattern_value(pattern: Sequence[int], group_index: int, note_groups_per_bar: int) -> int:
    if not pattern:
        return 0
    groups_per_bar = max(1, int(note_groups_per_bar))
    if groups_per_bar == len(pattern):
        index = int(group_index) % len(pattern)
    else:
        index = min(len(pattern) - 1, int(round(int(group_index) * (len(pattern) - 1) / max(1, groups_per_bar - 1))))
    return int(pattern[index])


def jazz_rhythm_position_tokens(
    bar_index: int,
    group_index: int,
    note_groups_per_bar: int,
    profile: str = "swing_motif",
    position_window: int = 0,
) -> list[int]:
    patterns = JAZZ_RHYTHM_POSITION_PATTERNS.get(profile, JAZZ_RHYTHM_POSITION_PATTERNS["swing_motif"])
    pattern = patterns[int(bar_index) % len(patterns)]
    target = max(0, min(int(POSITIONS_PER_BAR) - 1, _pattern_value(pattern, group_index, note_groups_per_bar)))
    window = max(0, int(position_window))
    positions = range(max(0, target - window), min(int(POSITIONS_PER_BAR), target + window + 1))
    return [TOKEN_POSITION_START + int(position) for position in positions]


def jazz_rhythm_duration_tokens(
    bar_index: int,
    group_index: int,
    note_groups_per_bar: int,
    profile: str = "swing_motif",
) -> list[int]:
    patterns = JAZZ_RHYTHM_DURATION_PATTERNS.get(profile, JAZZ_RHYTHM_DURATION_PATTERNS["swing_motif"])
    pattern = patterns[int(bar_index) % len(patterns)]
    target = max(1, min(int(MAX_DURATION_STEPS), _pattern_value(pattern, group_index, note_groups_per_bar)))
    duration_steps = sorted({max(1, min(int(MAX_DURATION_STEPS), target + offset)) for offset in (-1, 0, 1)})
    return [note_duration_token(step) for step in duration_steps]


def planned_position_tokens(
    *,
    bar_index: int,
    group_index: int,
    note_groups_per_bar: int,
    coverage_aware_positions: bool = False,
    coverage_position_window: int = 0,
    jazz_rhythm_positions: bool = False,
    jazz_rhythm_profile: str = "swing_motif",
) -> list[int] | None:
    if jazz_rhythm_positions:
        return jazz_rhythm_position_tokens(
            bar_index=bar_index,
            group_index=group_index,
            note_groups_per_bar=note_groups_per_bar,
            profile=jazz_rhythm_profile,
            position_window=coverage_position_window,
        )
    if coverage_aware_positions:
        return coverage_aware_position_tokens(
            group_index,
            note_groups_per_bar=note_groups_per_bar,
            position_window=coverage_position_window,
        )
    return None


def cap_duration_tokens_to_next_position(
    allowed_tokens: Sequence[int],
    *,
    current_position: int | None,
    bar_index: int,
    group_index: int,
    note_groups_per_bar: int,
    coverage_aware_positions: bool = False,
    coverage_position_window: int = 0,
    jazz_rhythm_positions: bool = False,
    jazz_rhythm_profile: str = "swing_motif",
) -> list[int]:
    allowed = [int(token) for token in allowed_tokens]
    if current_position is None:
        return allowed
    position = max(0, min(int(POSITIONS_PER_BAR) - 1, int(current_position)))
    groups_per_bar = max(1, int(note_groups_per_bar))
    if int(group_index) + 1 < groups_per_bar:
        next_tokens = planned_position_tokens(
            bar_index=bar_index,
            group_index=int(group_index) + 1,
            note_groups_per_bar=groups_per_bar,
            coverage_aware_positions=coverage_aware_positions,
            coverage_position_window=coverage_position_window,
            jazz_rhythm_positions=jazz_rhythm_positions,
            jazz_rhythm_profile=jazz_rhythm_profile,
        )
        next_positions = [
            position_from_token(token)
            for token in (next_tokens or [])
            if is_position_token(token) and position_from_token(token) > position
        ]
        next_position = min(next_positions) if next_positions else position + 1
    else:
        next_position = int(POSITIONS_PER_BAR)
    max_duration_steps = max(1, min(int(MAX_DURATION_STEPS), int(next_position) - position))
    filtered = [
        token
        for token in allowed
        if is_note_duration_token(token) and duration_steps_from_token(token) <= max_duration_steps
    ]
    return filtered or [note_duration_token(max_duration_steps)]


def fill_duration_token_to_next_position(
    *,
    current_position: int | None,
    bar_index: int,
    group_index: int,
    note_groups_per_bar: int,
    coverage_aware_positions: bool = False,
    coverage_position_window: int = 0,
    jazz_rhythm_positions: bool = False,
    jazz_rhythm_profile: str = "swing_motif",
) -> list[int]:
    return cap_duration_tokens_to_next_position(
        [note_duration_token(step) for step in range(1, int(MAX_DURATION_STEPS) + 1)],
        current_position=current_position,
        bar_index=bar_index,
        group_index=group_index,
        note_groups_per_bar=note_groups_per_bar,
        coverage_aware_positions=coverage_aware_positions,
        coverage_position_window=coverage_position_window,
        jazz_rhythm_positions=jazz_rhythm_positions,
        jazz_rhythm_profile=jazz_rhythm_profile,
    )[-1:]


def chord_pitch_classes(chord: str | None, pitch_mode: str = "tones_tensions") -> set[int]:
    root, quality = parse_chord_symbol(chord)
    root_pc = ROOT_TO_PC.get(root)
    if root_pc is None:
        return set(range(12))
    intervals = set(CHORD_TONE_INTERVALS.get(quality, CHORD_TONE_INTERVALS["unknown"]))
    if pitch_mode in {"tones_tensions", "approach_tensions"}:
        intervals.update(TENSION_INTERVALS.get(quality, set()))
    elif pitch_mode != "tones":
        raise ValueError(f"unknown chord pitch mode: {pitch_mode}")
    return {(root_pc + interval) % 12 for interval in intervals}


def chord_root_pitch_class(chord: str | None) -> int | None:
    root, _quality = parse_chord_symbol(chord)
    return ROOT_TO_PC.get(root)


def non_root_chord_pitch_classes(chord: str | None) -> set[int]:
    chord_tones = set(chord_pitch_classes(chord, pitch_mode="tones"))
    root_pc = chord_root_pitch_class(chord)
    if root_pc is not None and len(chord_tones) > 1:
        chord_tones.discard(root_pc)
    return chord_tones


def chord_approach_pitch_classes(chord: str | None) -> set[int]:
    target_pitch_classes = non_root_chord_pitch_classes(chord)
    approach_pitch_classes: set[int] = set()
    for pitch_class in target_pitch_classes:
        for step in (-2, -1, 1, 2):
            approach_pitch_classes.add((pitch_class + step) % 12)
    approach_pitch_classes.difference_update(chord_pitch_classes(chord, pitch_mode="tones"))
    return approach_pitch_classes or (chord_pitch_classes(chord, pitch_mode="tones_tensions") - chord_pitch_classes(chord, pitch_mode="tones"))


def analyze_stage_b_pitch_roles(
    tokens: Sequence[int],
    chords: Sequence[str],
    primer_size: int = 0,
) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=primer_size)
    group_count = int(len(groups))
    root_hits = 0
    chord_tone_hits = 0
    tension_hits = 0
    non_chord_hits = 0
    per_bar_counts: dict[int, int] = {}
    per_bar_root_hits: dict[int, int] = {}

    for group in groups:
        bar = int(group["bar"])
        pitch_class = int(group["pitch"]) % 12
        chord = chords[bar % len(chords)] if chords else None
        root_pc = chord_root_pitch_class(chord)
        chord_tones = chord_pitch_classes(chord, pitch_mode="tones")
        tones_with_tensions = chord_pitch_classes(chord, pitch_mode="tones_tensions")
        per_bar_counts[bar] = per_bar_counts.get(bar, 0) + 1
        if root_pc is not None and pitch_class == root_pc:
            root_hits += 1
            per_bar_root_hits[bar] = per_bar_root_hits.get(bar, 0) + 1
        if pitch_class in chord_tones:
            chord_tone_hits += 1
        elif pitch_class in tones_with_tensions:
            tension_hits += 1
        else:
            non_chord_hits += 1

    non_root_chord_tone_hits = max(0, chord_tone_hits - root_hits)
    return {
        "note_group_count": group_count,
        "root_tone_count": int(root_hits),
        "root_tone_ratio": float(root_hits / group_count) if group_count else 0.0,
        "chord_tone_count": int(chord_tone_hits),
        "chord_tone_ratio": float(chord_tone_hits / group_count) if group_count else 0.0,
        "non_root_chord_tone_count": int(non_root_chord_tone_hits),
        "non_root_chord_tone_ratio": float(non_root_chord_tone_hits / group_count) if group_count else 0.0,
        "tension_count": int(tension_hits),
        "tension_ratio": float(tension_hits / group_count) if group_count else 0.0,
        "non_chord_tone_count": int(non_chord_hits),
        "non_chord_tone_ratio": float(non_chord_hits / group_count) if group_count else 0.0,
        "per_bar_root_tone_ratio": {
            str(bar): float(per_bar_root_hits.get(bar, 0) / count if count else 0.0)
            for bar, count in sorted(per_bar_counts.items())
        },
    }


def analyze_stage_b_approach_resolution(
    tokens: Sequence[int],
    chords: Sequence[str],
    primer_size: int = 0,
    max_resolution_step: int = 2,
) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=primer_size)
    approach_candidates = 0
    resolved_approaches = 0
    unresolved_approaches = 0
    resolution_distances: list[int] = []

    for index, group in enumerate(groups[:-1]):
        pitch = int(group["pitch"])
        pitch_class = pitch % 12
        bar = int(group["bar"])
        chord = chords[bar % len(chords)] if chords else None
        chord_tones = chord_pitch_classes(chord, pitch_mode="tones")
        if pitch_class in chord_tones:
            continue

        next_group = groups[index + 1]
        next_pitch = int(next_group["pitch"])
        next_bar = int(next_group["bar"])
        next_chord = chords[next_bar % len(chords)] if chords else None
        next_chord_tones = chord_pitch_classes(next_chord, pitch_mode="tones")
        distance = abs(next_pitch - pitch)
        is_near_resolution = 1 <= distance <= int(max_resolution_step)
        is_resolved = is_near_resolution and (next_pitch % 12) in next_chord_tones

        near_current_tone = any(
            min((pitch_class - target) % 12, (target - pitch_class) % 12) <= int(max_resolution_step)
            for target in chord_tones
        )
        near_next_tone = any(
            min((pitch_class - target) % 12, (target - pitch_class) % 12) <= int(max_resolution_step)
            for target in next_chord_tones
        )
        if not (near_current_tone or near_next_tone):
            continue

        approach_candidates += 1
        if is_resolved:
            resolved_approaches += 1
            resolution_distances.append(distance)
        else:
            unresolved_approaches += 1

    group_count = int(len(groups))
    return {
        "note_group_count": group_count,
        "approach_candidate_count": int(approach_candidates),
        "approach_candidate_ratio": float(approach_candidates / group_count) if group_count else 0.0,
        "resolved_approach_count": int(resolved_approaches),
        "unresolved_approach_count": int(unresolved_approaches),
        "approach_resolution_ratio": (
            float(resolved_approaches / approach_candidates) if approach_candidates else 0.0
        ),
        "avg_resolution_distance": (
            float(sum(resolution_distances) / len(resolution_distances)) if resolution_distances else 0.0
        ),
    }


def analyze_stage_b_rhythm_profile(
    tokens: Sequence[int],
    primer_size: int = 0,
) -> dict[str, Any]:
    groups = extract_stage_b_note_groups(tokens, primer_size=primer_size)
    group_count = int(len(groups))
    if not groups:
        return {
            "note_group_count": 0,
            "unique_position_count": 0,
            "unique_position_ratio": 0.0,
            "syncopated_onset_ratio": 0.0,
            "unique_bar_position_pattern_count": 0,
            "unique_bar_position_pattern_ratio": 0.0,
            "duration_diversity_ratio": 0.0,
            "most_common_duration_ratio": 0.0,
            "ioi_diversity_ratio": 0.0,
            "most_common_ioi_ratio": 0.0,
            "bar_position_patterns": {},
        }

    positions = [int(group["position"]) for group in groups]
    durations = [int(group["duration_steps"]) for group in groups]
    absolute_positions = [
        int(group["bar"]) * int(POSITIONS_PER_BAR) + int(group["position"])
        for group in groups
    ]
    ioi_steps = [
        max(0, absolute_positions[index + 1] - absolute_positions[index])
        for index in range(len(absolute_positions) - 1)
    ]
    strong_positions = {0, 4, 8, 12}
    syncopated_count = sum(1 for position in positions if position not in strong_positions)

    per_bar_positions: dict[int, list[int]] = {}
    for group in groups:
        per_bar_positions.setdefault(int(group["bar"]), []).append(int(group["position"]))
    bar_patterns = {
        str(bar): tuple(sorted(values))
        for bar, values in sorted(per_bar_positions.items())
    }
    unique_bar_patterns = set(bar_patterns.values())
    duration_counts = Counter(durations)
    ioi_counts = Counter(ioi_steps)

    return {
        "note_group_count": group_count,
        "unique_position_count": int(len(set(positions))),
        "unique_position_ratio": float(len(set(positions)) / group_count),
        "syncopated_onset_ratio": float(syncopated_count / group_count),
        "unique_bar_position_pattern_count": int(len(unique_bar_patterns)),
        "unique_bar_position_pattern_ratio": (
            float(len(unique_bar_patterns) / len(bar_patterns)) if bar_patterns else 0.0
        ),
        "duration_diversity_ratio": float(len(duration_counts) / group_count),
        "most_common_duration_ratio": (
            float(max(duration_counts.values()) / group_count) if duration_counts else 0.0
        ),
        "ioi_diversity_ratio": float(len(ioi_counts) / max(1, len(ioi_steps))),
        "most_common_ioi_ratio": (
            float(max(ioi_counts.values()) / max(1, len(ioi_steps))) if ioi_counts else 0.0
        ),
        "bar_position_patterns": {
            bar: list(pattern)
            for bar, pattern in bar_patterns.items()
        },
    }


def chord_aware_pitch_tokens(
    chord: str | None,
    pitch_mode: str = "tones_tensions",
    recent_pitches: Sequence[int] | None = None,
    repeat_window: int = 2,
    group_index: int | None = None,
    pitch_min: int | None = None,
    pitch_max: int | None = None,
    max_adjacent_interval: int | None = None,
) -> list[int]:
    if pitch_mode == "approach_tensions":
        resolving_group = bool(group_index is not None and int(group_index) % 2 == 1)
        if resolving_group:
            pitch_classes = non_root_chord_pitch_classes(chord)
        else:
            pitch_classes = chord_approach_pitch_classes(chord)
    else:
        pitch_classes = chord_pitch_classes(chord, pitch_mode=pitch_mode)
    lower = max(int(PIANO_PITCH_MIN), int(pitch_min) if pitch_min is not None else int(PIANO_PITCH_MIN))
    upper = min(int(PIANO_PITCH_MAX), int(pitch_max) if pitch_max is not None else int(PIANO_PITCH_MAX))
    if lower > upper:
        raise ValueError(f"invalid pitch range: {lower}>{upper}")

    tokens = [
        note_pitch_token(pitch)
        for pitch in range(lower, upper + 1)
        if pitch % 12 in pitch_classes
    ]
    if not tokens:
        tokens = [note_pitch_token(pitch) for pitch in range(lower, upper + 1)]

    if pitch_mode == "approach_tensions" and recent_pitches and group_index is not None and int(group_index) % 2 == 1:
        last_pitch = int(list(recent_pitches)[-1])
        near_resolution_tokens = [
            token for token in tokens if 1 <= abs(pitch_from_token(token) - last_pitch) <= 2
        ]
        if near_resolution_tokens:
            tokens = near_resolution_tokens

    if recent_pitches and max_adjacent_interval is not None and int(max_adjacent_interval) >= 0:
        last_pitch = int(list(recent_pitches)[-1])
        interval_filtered = [
            token for token in tokens if abs(pitch_from_token(token) - last_pitch) <= int(max_adjacent_interval)
        ]
        if interval_filtered:
            tokens = interval_filtered

    window = max(0, int(repeat_window))
    if not recent_pitches or window == 0:
        return tokens

    blocked = set(int(pitch) for pitch in list(recent_pitches)[-window:])
    filtered = [token for token in tokens if pitch_from_token(token) not in blocked]
    return filtered or tokens


def generate_stage_b_constrained_tokens(
    model: Any,
    primer_tokens: Sequence[int],
    chords: Sequence[str],
    bpm: float | int,
    bars: int,
    note_groups_per_bar: int,
    max_sequence: int,
    temperature: float,
    top_k: int | None,
    coverage_aware_positions: bool = False,
    coverage_position_window: int = 0,
    chord_aware_pitches: bool = False,
    chord_pitch_mode: str = "tones_tensions",
    chord_pitch_repeat_window: int = 2,
    pitch_min: int | None = None,
    pitch_max: int | None = None,
    max_adjacent_interval: int | None = None,
    jazz_rhythm_positions: bool = False,
    jazz_duration_tokens: bool = False,
    jazz_rhythm_profile: str = "swing_motif",
    cap_duration_to_next_position: bool = False,
    fill_duration_to_next_position: bool = False,
) -> list[int]:
    tokens = [int(token) for token in primer_tokens]
    recent_pitches: list[int] = []
    families = [
        range(TOKEN_POSITION_START, TOKEN_POSITION_END + 1),
        range(TOKEN_VELOCITY_START, TOKEN_VELOCITY_END + 1),
        range(TOKEN_NOTE_PITCH_START, TOKEN_NOTE_PITCH_END + 1),
        range(TOKEN_NOTE_DURATION_START, TOKEN_NOTE_DURATION_END + 1),
    ]

    for bar_index in range(max(1, int(bars))):
        chord = chords[bar_index % len(chords)] if chords else None
        if bar_index > 0:
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        for group_index in range(max(1, int(note_groups_per_bar))):
            current_position: int | None = None
            for family_index, allowed_tokens in enumerate(families):
                if len(tokens) >= int(max_sequence) - 1:
                    tokens.append(TOKEN_END)
                    return tokens
                allowed = list(allowed_tokens)
                planned_positions = planned_position_tokens(
                    bar_index=bar_index,
                    group_index=group_index,
                    note_groups_per_bar=note_groups_per_bar,
                    coverage_aware_positions=coverage_aware_positions,
                    coverage_position_window=coverage_position_window,
                    jazz_rhythm_positions=jazz_rhythm_positions,
                    jazz_rhythm_profile=jazz_rhythm_profile,
                )
                if planned_positions is not None and family_index == 0:
                    allowed = planned_positions
                if chord_aware_pitches and family_index == 2:
                    allowed = chord_aware_pitch_tokens(
                        chord,
                        pitch_mode=chord_pitch_mode,
                        recent_pitches=recent_pitches,
                        repeat_window=chord_pitch_repeat_window,
                        group_index=group_index,
                        pitch_min=pitch_min,
                        pitch_max=pitch_max,
                        max_adjacent_interval=max_adjacent_interval,
                    )
                if jazz_duration_tokens and family_index == 3:
                    allowed = jazz_rhythm_duration_tokens(
                        bar_index=bar_index,
                        group_index=group_index,
                        note_groups_per_bar=note_groups_per_bar,
                        profile=jazz_rhythm_profile,
                    )
                if fill_duration_to_next_position and family_index == 3:
                    allowed = fill_duration_token_to_next_position(
                        current_position=current_position,
                        bar_index=bar_index,
                        group_index=group_index,
                        note_groups_per_bar=note_groups_per_bar,
                        coverage_aware_positions=coverage_aware_positions,
                        coverage_position_window=coverage_position_window,
                        jazz_rhythm_positions=jazz_rhythm_positions,
                        jazz_rhythm_profile=jazz_rhythm_profile,
                    )
                elif cap_duration_to_next_position and family_index == 3:
                    allowed = cap_duration_tokens_to_next_position(
                        allowed,
                        current_position=current_position,
                        bar_index=bar_index,
                        group_index=group_index,
                        note_groups_per_bar=note_groups_per_bar,
                        coverage_aware_positions=coverage_aware_positions,
                        coverage_position_window=coverage_position_window,
                        jazz_rhythm_positions=jazz_rhythm_positions,
                        jazz_rhythm_profile=jazz_rhythm_profile,
                    )
                token = next_token_from_model(
                    model,
                    tokens=tokens,
                    allowed_tokens=allowed,
                    temperature=temperature,
                    top_k=top_k,
                )
                tokens.append(token)
                if family_index == 0 and is_position_token(token):
                    current_position = position_from_token(token)
                if family_index == 2 and is_note_pitch_token(token):
                    recent_pitches.append(pitch_from_token(token))

    tokens.append(TOKEN_END)
    return tokens[: int(max_sequence)]


def dedupe_and_limit_notes(
    notes: Sequence[pretty_midi.Note],
    simultaneous_limit: int = 2,
    time_precision: int = 6,
) -> list[pretty_midi.Note]:
    best_by_onset_pitch: dict[tuple[float, int], pretty_midi.Note] = {}
    for note in notes:
        if float(note.end) <= float(note.start):
            continue
        key = (round(float(note.start), int(time_precision)), int(note.pitch))
        current = best_by_onset_pitch.get(key)
        if current is None:
            best_by_onset_pitch[key] = note
            continue
        current_score = (int(current.velocity), float(current.end) - float(current.start))
        note_score = (int(note.velocity), float(note.end) - float(note.start))
        if note_score > current_score:
            best_by_onset_pitch[key] = note

    selected: list[pretty_midi.Note] = []
    for note in sorted(best_by_onset_pitch.values(), key=lambda n: (float(n.start), -int(n.velocity), int(n.pitch))):
        active = [chosen for chosen in selected if float(chosen.end) > float(note.start)]
        if len(active) >= int(simultaneous_limit):
            continue
        selected.append(note)
    return sorted(selected, key=lambda n: (float(n.start), int(n.pitch)))


def postprocess_stage_b_midi(
    midi: pretty_midi.PrettyMIDI,
    simultaneous_limit: int = 2,
) -> dict[str, Any]:
    before_notes: list[pretty_midi.Note] = []
    after_notes: list[pretty_midi.Note] = []

    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        before_notes.extend(instrument.notes)
        instrument.notes = dedupe_and_limit_notes(instrument.notes, simultaneous_limit=simultaneous_limit)
        after_notes.extend(instrument.notes)

    return {
        "enabled": True,
        "simultaneous_limit": int(simultaneous_limit),
        "before_note_count": int(len(before_notes)),
        "after_note_count": int(len(after_notes)),
        "removed_note_count": int(max(0, len(before_notes) - len(after_notes))),
        "before_max_simultaneous_notes": int(max_simultaneous_notes(before_notes)) if before_notes else 0,
        "after_max_simultaneous_notes": int(max_simultaneous_notes(after_notes)) if after_notes else 0,
    }


def decode_tokens_to_midi(tokens: Sequence[int], output_path: Path, bpm: float | int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = decode_stage_b_midi(tokens, tempo_bpm=bpm)
    midi.write(str(output_path))
    return output_path


def sample_report(
    sample_index: int,
    sample_seed: int,
    tokens: Sequence[int],
    primer_size: int,
    target_length: int,
    midi_path: Path,
    request: GenerationRequest,
    postprocess_report: dict[str, Any] | None = None,
    strict_min_unique_pitches: int = DEFAULT_STRICT_MIN_UNIQUE_PITCHES,
    strict_min_unique_positions: int = DEFAULT_STRICT_MIN_UNIQUE_POSITIONS,
    strict_min_unique_position_pitch_pairs: int = DEFAULT_STRICT_MIN_UNIQUE_POSITION_PITCH_PAIRS,
    strict_max_repeated_position_pitch_pair_ratio: float = DEFAULT_STRICT_MAX_REPEATED_POSITION_PITCH_PAIR_RATIO,
    strict_max_postprocess_removal_ratio: float = DEFAULT_STRICT_MAX_POSTPROCESS_REMOVAL_RATIO,
) -> dict[str, Any]:
    raw_generated_tokens = [int(token) for token in tokens[primer_size:]]
    grammar = analyze_stage_b_note_grammar(tokens, primer_size=primer_size)
    collapse = analyze_stage_b_collapse(tokens, primer_size=primer_size, postprocess_report=postprocess_report)
    temporal_coverage = analyze_stage_b_temporal_coverage(tokens, primer_size=primer_size, bars=request.bars)
    phrase_contour = analyze_stage_b_phrase_contour(tokens, primer_size=primer_size)
    pitch_roles = analyze_stage_b_pitch_roles(tokens, chords=request.chord_progression, primer_size=primer_size)
    approach_resolution = analyze_stage_b_approach_resolution(
        tokens,
        chords=request.chord_progression,
        primer_size=primer_size,
    )
    rhythm_profile = analyze_stage_b_rhythm_profile(tokens, primer_size=primer_size)
    collapse_gate = evaluate_collapse_gate(
        collapse,
        min_unique_pitches=strict_min_unique_pitches,
        min_unique_positions=strict_min_unique_positions,
        min_unique_position_pitch_pairs=strict_min_unique_position_pitch_pairs,
        max_repeated_position_pitch_pair_ratio=strict_max_repeated_position_pitch_pair_ratio,
        max_postprocess_removal_ratio=strict_max_postprocess_removal_ratio,
    )
    metrics = compute_midi_metrics(midi_path, 0, False, request=request)
    valid, reason = validate_metrics(metrics, request.density, bars=request.bars)
    grammar_gate_passed = bool(grammar["complete_note_groups"] > 0 and metrics.note_count > 0)
    strict_valid = bool(valid and grammar_gate_passed and collapse_gate["passed"])
    diagnostic_failure_reason = reason
    if not valid and collapse["collapse_warning"]:
        diagnostic_failure_reason = (
            f"{reason}; collapse={','.join(collapse['collapse_reasons'])}" if reason else ",".join(collapse["collapse_reasons"])
        )
    if valid and not collapse_gate["passed"]:
        diagnostic_failure_reason = "; ".join(collapse_gate["failure_reasons"])
    return {
        "sample_index": int(sample_index),
        "sample_seed": int(sample_seed),
        "midi_path": str(midi_path),
        "token_count": int(len(tokens)),
        "generated_token_count": int(len(raw_generated_tokens)),
        "ended_early": bool(len(tokens) < int(target_length)),
        "hit_end_token": bool(TOKEN_END in raw_generated_tokens),
        "valid": bool(valid),
        "strict_valid": strict_valid,
        "grammar_gate_passed": grammar_gate_passed,
        "failure_reason": reason,
        "diagnostic_failure_reason": diagnostic_failure_reason,
        "grammar": grammar,
        "collapse": collapse,
        "temporal_coverage": temporal_coverage,
        "phrase_contour": phrase_contour,
        "pitch_roles": pitch_roles,
        "approach_resolution": approach_resolution,
        "rhythm_profile": rhythm_profile,
        "collapse_gate": collapse_gate,
        "postprocess": postprocess_report or {"enabled": False},
        "metrics": metrics.to_dict(),
        "generated_token_names_head": [stage_b_token_name(token) for token in raw_generated_tokens[:48]],
    }


def build_probe_summary(
    sample_rows: Sequence[dict[str, Any]],
    min_valid_samples: int = 1,
    min_strict_valid_samples: int = 1,
    require_all_grammar_samples: bool = False,
    max_collapse_warning_sample_rate: float = DEFAULT_STRICT_MAX_COLLAPSE_WARNING_SAMPLE_RATE,
) -> dict[str, Any]:
    sample_count = int(len(sample_rows))
    valid_indices = [int(row["sample_index"]) for row in sample_rows if row["valid"]]
    strict_valid_indices = [int(row["sample_index"]) for row in sample_rows if row.get("strict_valid")]
    grammar_indices = [int(row["sample_index"]) for row in sample_rows if row["grammar_gate_passed"]]
    valid_sample_count = int(len(valid_indices))
    strict_valid_sample_count = int(len(strict_valid_indices))
    grammar_gate_sample_count = int(len(grammar_indices))

    if require_all_grammar_samples:
        passed_grammar_gate = bool(sample_count > 0 and grammar_gate_sample_count == sample_count)
    else:
        passed_grammar_gate = bool(grammar_gate_sample_count > 0)

    failure_reasons: dict[str, int] = {}
    diagnostic_failure_reasons: dict[str, int] = {}
    strict_failure_reasons: dict[str, int] = {}
    collapse_warning_count = 0
    repeated_pair_ratios: list[float] = []
    postprocess_removal_ratios: list[float] = []
    onset_coverage_ratios: list[float] = []
    sustained_coverage_ratios: list[float] = []
    position_span_ratios: list[float] = []
    longest_sustained_empty_runs: list[int] = []
    adjacent_repeated_pitch_ratios: list[float] = []
    direction_change_ratios: list[float] = []
    longest_same_pitch_runs: list[int] = []
    root_tone_ratios: list[float] = []
    tension_ratios: list[float] = []
    approach_candidate_ratios: list[float] = []
    approach_resolution_ratios: list[float] = []
    syncopated_onset_ratios: list[float] = []
    unique_bar_pattern_ratios: list[float] = []
    duration_diversity_ratios: list[float] = []
    most_common_duration_ratios: list[float] = []
    ioi_diversity_ratios: list[float] = []
    most_common_ioi_ratios: list[float] = []
    for row in sample_rows:
        collapse = row.get("collapse", {})
        temporal_coverage = row.get("temporal_coverage", {})
        phrase_contour = row.get("phrase_contour", {})
        pitch_roles = row.get("pitch_roles", {})
        approach_resolution = row.get("approach_resolution", {})
        rhythm_profile = row.get("rhythm_profile", {})
        if collapse.get("collapse_warning"):
            collapse_warning_count += 1
        if not row.get("strict_valid", False):
            if not row.get("valid", False):
                strict_reason = f"midi_review_gate_failed: {row.get('failure_reason') or 'unknown'}"
                strict_failure_reasons[strict_reason] = strict_failure_reasons.get(strict_reason, 0) + 1
            if not row.get("grammar_gate_passed", False):
                strict_failure_reasons["grammar_gate_failed"] = (
                    strict_failure_reasons.get("grammar_gate_failed", 0) + 1
                )
            for strict_reason in row.get("collapse_gate", {}).get("failure_reasons", []):
                strict_failure_reasons[str(strict_reason)] = strict_failure_reasons.get(str(strict_reason), 0) + 1
        repeated_pair_ratios.append(float(collapse.get("repeated_position_pitch_pair_ratio", 0.0) or 0.0))
        postprocess_removal_ratios.append(float(collapse.get("postprocess_removal_ratio", 0.0) or 0.0))
        onset_coverage_ratios.append(float(temporal_coverage.get("onset_coverage_ratio", 0.0) or 0.0))
        sustained_coverage_ratios.append(float(temporal_coverage.get("sustained_coverage_ratio", 0.0) or 0.0))
        position_span_ratios.append(float(temporal_coverage.get("position_span_ratio", 0.0) or 0.0))
        longest_sustained_empty_runs.append(
            int(temporal_coverage.get("longest_sustained_empty_run_steps", 0) or 0)
        )
        adjacent_repeated_pitch_ratios.append(
            float(phrase_contour.get("adjacent_repeated_pitch_ratio", 0.0) or 0.0)
        )
        direction_change_ratios.append(float(phrase_contour.get("direction_change_ratio", 0.0) or 0.0))
        longest_same_pitch_runs.append(int(phrase_contour.get("longest_same_pitch_run", 0) or 0))
        root_tone_ratios.append(float(pitch_roles.get("root_tone_ratio", 0.0) or 0.0))
        tension_ratios.append(float(pitch_roles.get("tension_ratio", 0.0) or 0.0))
        approach_candidate_ratios.append(
            float(approach_resolution.get("approach_candidate_ratio", 0.0) or 0.0)
        )
        approach_resolution_ratios.append(
            float(approach_resolution.get("approach_resolution_ratio", 0.0) or 0.0)
        )
        syncopated_onset_ratios.append(float(rhythm_profile.get("syncopated_onset_ratio", 0.0) or 0.0))
        unique_bar_pattern_ratios.append(
            float(rhythm_profile.get("unique_bar_position_pattern_ratio", 0.0) or 0.0)
        )
        duration_diversity_ratios.append(float(rhythm_profile.get("duration_diversity_ratio", 0.0) or 0.0))
        most_common_duration_ratios.append(float(rhythm_profile.get("most_common_duration_ratio", 0.0) or 0.0))
        ioi_diversity_ratios.append(float(rhythm_profile.get("ioi_diversity_ratio", 0.0) or 0.0))
        most_common_ioi_ratios.append(float(rhythm_profile.get("most_common_ioi_ratio", 0.0) or 0.0))
        if not row["valid"]:
            reason = str(row.get("failure_reason") or "unknown")
            diagnostic_reason = str(row.get("diagnostic_failure_reason") or reason)
            failure_reasons[reason] = failure_reasons.get(reason, 0) + 1
            diagnostic_failure_reasons[diagnostic_reason] = diagnostic_failure_reasons.get(diagnostic_reason, 0) + 1

    collapse_warning_rate = float(collapse_warning_count / sample_count) if sample_count else 0.0
    passed_generation_gate = bool(valid_sample_count >= int(min_valid_samples))
    passed_strict_generation_gate = bool(strict_valid_sample_count >= int(min_strict_valid_samples))
    passed_collapse_rate_gate = bool(
        sample_count > 0 and collapse_warning_rate <= float(max_collapse_warning_sample_rate)
    )

    return {
        "sample_count": sample_count,
        "valid_sample_count": valid_sample_count,
        "strict_valid_sample_count": strict_valid_sample_count,
        "grammar_gate_sample_count": grammar_gate_sample_count,
        "valid_sample_rate": float(valid_sample_count / sample_count) if sample_count else 0.0,
        "strict_valid_sample_rate": float(strict_valid_sample_count / sample_count) if sample_count else 0.0,
        "grammar_gate_sample_rate": float(grammar_gate_sample_count / sample_count) if sample_count else 0.0,
        "valid_sample_indices": valid_indices,
        "strict_valid_sample_indices": strict_valid_indices,
        "grammar_gate_sample_indices": grammar_indices,
        "min_valid_samples": int(min_valid_samples),
        "min_strict_valid_samples": int(min_strict_valid_samples),
        "max_collapse_warning_sample_rate": float(max_collapse_warning_sample_rate),
        "require_all_grammar_samples": bool(require_all_grammar_samples),
        "passed_generation_gate": passed_generation_gate,
        "passed_strict_generation_gate": passed_strict_generation_gate,
        "passed_grammar_gate": passed_grammar_gate,
        "passed_strict_review_gate": bool(
            passed_generation_gate
            and passed_strict_generation_gate
            and passed_grammar_gate
            and passed_collapse_rate_gate
        ),
        "failure_reasons": failure_reasons,
        "diagnostic_failure_reasons": diagnostic_failure_reasons,
        "strict_failure_reasons": strict_failure_reasons,
        "collapse_warning_sample_count": int(collapse_warning_count),
        "collapse_warning_sample_rate": collapse_warning_rate,
        "passed_collapse_rate_gate": passed_collapse_rate_gate,
        "avg_repeated_position_pitch_pair_ratio": (
            float(sum(repeated_pair_ratios) / len(repeated_pair_ratios)) if repeated_pair_ratios else 0.0
        ),
        "max_repeated_position_pitch_pair_ratio": max(repeated_pair_ratios) if repeated_pair_ratios else 0.0,
        "avg_postprocess_removal_ratio": (
            float(sum(postprocess_removal_ratios) / len(postprocess_removal_ratios))
            if postprocess_removal_ratios
            else 0.0
        ),
        "max_postprocess_removal_ratio": max(postprocess_removal_ratios) if postprocess_removal_ratios else 0.0,
        "avg_onset_coverage_ratio": (
            float(sum(onset_coverage_ratios) / len(onset_coverage_ratios)) if onset_coverage_ratios else 0.0
        ),
        "avg_sustained_coverage_ratio": (
            float(sum(sustained_coverage_ratios) / len(sustained_coverage_ratios))
            if sustained_coverage_ratios
            else 0.0
        ),
        "avg_position_span_ratio": (
            float(sum(position_span_ratios) / len(position_span_ratios)) if position_span_ratios else 0.0
        ),
        "max_longest_sustained_empty_run_steps": (
            max(longest_sustained_empty_runs) if longest_sustained_empty_runs else 0
        ),
        "avg_adjacent_repeated_pitch_ratio": (
            float(sum(adjacent_repeated_pitch_ratios) / len(adjacent_repeated_pitch_ratios))
            if adjacent_repeated_pitch_ratios
            else 0.0
        ),
        "avg_direction_change_ratio": (
            float(sum(direction_change_ratios) / len(direction_change_ratios)) if direction_change_ratios else 0.0
        ),
        "max_longest_same_pitch_run": max(longest_same_pitch_runs) if longest_same_pitch_runs else 0,
        "avg_root_tone_ratio": (
            float(sum(root_tone_ratios) / len(root_tone_ratios)) if root_tone_ratios else 0.0
        ),
        "avg_tension_ratio": float(sum(tension_ratios) / len(tension_ratios)) if tension_ratios else 0.0,
        "avg_approach_candidate_ratio": (
            float(sum(approach_candidate_ratios) / len(approach_candidate_ratios))
            if approach_candidate_ratios
            else 0.0
        ),
        "avg_approach_resolution_ratio": (
            float(sum(approach_resolution_ratios) / len(approach_resolution_ratios))
            if approach_resolution_ratios
            else 0.0
        ),
        "avg_syncopated_onset_ratio": (
            float(sum(syncopated_onset_ratios) / len(syncopated_onset_ratios))
            if syncopated_onset_ratios
            else 0.0
        ),
        "avg_unique_bar_position_pattern_ratio": (
            float(sum(unique_bar_pattern_ratios) / len(unique_bar_pattern_ratios))
            if unique_bar_pattern_ratios
            else 0.0
        ),
        "avg_duration_diversity_ratio": (
            float(sum(duration_diversity_ratios) / len(duration_diversity_ratios))
            if duration_diversity_ratios
            else 0.0
        ),
        "avg_most_common_duration_ratio": (
            float(sum(most_common_duration_ratios) / len(most_common_duration_ratios))
            if most_common_duration_ratios
            else 0.0
        ),
        "avg_ioi_diversity_ratio": (
            float(sum(ioi_diversity_ratios) / len(ioi_diversity_ratios))
            if ioi_diversity_ratios
            else 0.0
        ),
        "avg_most_common_ioi_ratio": (
            float(sum(most_common_ioi_ratios) / len(most_common_ioi_ratios))
            if most_common_ioi_ratios
            else 0.0
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generation/decode probe")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio/Brad Mehldau")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_generation_probe"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=1)
    parser.add_argument("--window_bars", type=int, default=2)
    parser.add_argument("--window_stride_bars", type=int, default=2)
    parser.add_argument("--min_window_target_notes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_sequence", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--issue_number", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--chords", type=str, default="Cm7,Fm7,Bb7,Ebmaj7")
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--generation_mode", choices=("unconstrained", "constrained"), default="unconstrained")
    parser.add_argument("--constrained_note_groups_per_bar", type=int, default=4)
    parser.add_argument("--coverage_aware_positions", action="store_true")
    parser.add_argument("--coverage_position_window", type=int, default=0)
    parser.add_argument("--chord_aware_pitches", action="store_true")
    parser.add_argument(
        "--chord_pitch_mode",
        choices=("tones", "tones_tensions", "approach_tensions"),
        default="tones_tensions",
    )
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--constrained_pitch_min", type=int, default=None)
    parser.add_argument("--constrained_pitch_max", type=int, default=None)
    parser.add_argument("--constrained_max_adjacent_interval", type=int, default=None)
    parser.add_argument("--jazz_rhythm_positions", action="store_true")
    parser.add_argument("--jazz_duration_tokens", action="store_true")
    parser.add_argument(
        "--jazz_rhythm_profile",
        choices=tuple(sorted(JAZZ_RHYTHM_POSITION_PATTERNS)),
        default="swing_motif",
    )
    parser.add_argument("--cap_duration_to_next_position", action="store_true")
    parser.add_argument("--fill_duration_to_next_position", action="store_true")
    parser.add_argument("--postprocess_overlap", action="store_true")
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--strict_min_unique_pitches", type=int, default=DEFAULT_STRICT_MIN_UNIQUE_PITCHES)
    parser.add_argument("--strict_min_unique_positions", type=int, default=DEFAULT_STRICT_MIN_UNIQUE_POSITIONS)
    parser.add_argument(
        "--strict_min_unique_position_pitch_pairs",
        type=int,
        default=DEFAULT_STRICT_MIN_UNIQUE_POSITION_PITCH_PAIRS,
    )
    parser.add_argument(
        "--strict_max_repeated_position_pitch_pair_ratio",
        type=float,
        default=DEFAULT_STRICT_MAX_REPEATED_POSITION_PITCH_PAIR_RATIO,
    )
    parser.add_argument(
        "--strict_max_postprocess_removal_ratio",
        type=float,
        default=DEFAULT_STRICT_MAX_POSTPROCESS_REMOVAL_RATIO,
    )
    parser.add_argument(
        "--max_collapse_warning_sample_rate",
        type=float,
        default=DEFAULT_STRICT_MAX_COLLAPSE_WARNING_SAMPLE_RATE,
    )
    parser.add_argument("--require_all_grammar_samples", action="store_true")
    parser.add_argument("--require_note_groups", action="store_true")
    parser.add_argument("--require_valid_sample", action="store_true")
    parser.add_argument("--require_strict_valid_sample", action="store_true")
    parser.set_defaults(lora_only=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))

    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    roles_dir = run_dir / "roles"
    role_root = roles_dir / "lead"
    tokenized_dir = role_root / "tokenized"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    samples_dir = run_dir / "samples"

    chords = parse_chords(args.chords)
    request = GenerationRequest(
        bpm=int(args.bpm),
        chord_progression=chords,
        bars=int(args.bars),
        density=args.density,
        energy=args.energy,
        temperature=float(args.temperature),
        top_k=args.top_k,
        top_p=args.top_p,
        seed=int(args.seed),
    )
    request.validate()

    issue_number = args.issue_number
    if issue_number is None:
        issue_number = 22 if args.postprocess_overlap else 20 if args.generation_mode == "constrained" else 18

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "issue": int(issue_number),
        "sequence_format": SEQUENCE_FORMAT_STAGE_B_V1,
        "request": request.to_dict(),
        "checkpoint_dir": str(checkpoint_dir),
        "sample_vocab_size": int(VOCAB_SIZE),
        "generation_mode": args.generation_mode,
        "constrained_note_groups_per_bar": int(args.constrained_note_groups_per_bar),
        "coverage_aware_positions": bool(args.coverage_aware_positions),
        "coverage_position_window": int(args.coverage_position_window),
        "chord_aware_pitches": bool(args.chord_aware_pitches),
        "chord_pitch_mode": args.chord_pitch_mode,
        "chord_pitch_repeat_window": int(args.chord_pitch_repeat_window),
        "constrained_pitch_min": args.constrained_pitch_min,
        "constrained_pitch_max": args.constrained_pitch_max,
        "constrained_max_adjacent_interval": args.constrained_max_adjacent_interval,
        "jazz_rhythm_positions": bool(args.jazz_rhythm_positions),
        "jazz_duration_tokens": bool(args.jazz_duration_tokens),
        "jazz_rhythm_profile": args.jazz_rhythm_profile,
        "cap_duration_to_next_position": bool(args.cap_duration_to_next_position),
        "fill_duration_to_next_position": bool(args.fill_duration_to_next_position),
        "postprocess_overlap": bool(args.postprocess_overlap),
    }

    if not args.skip_prepare:
        prepare_result = run_prepare_command(args, roles_dir)
        report["prepare_result"] = prepare_result
        if prepare_result["returncode"] != 0:
            write_json(run_dir / "report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(prepare_result["returncode"])
        report["dataset_summary"] = read_json(role_root / "dataset_summary.json")
        report["token_stats"] = token_stats(tokenized_dir)
        if not report["token_stats"]["fits_vocab"]:
            report["failure_reason"] = "Stage B tokenized records do not fit model VOCAB_SIZE"
            write_json(run_dir / "report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return 2

    if not args.skip_train:
        train_result = run_command(build_train_command(args, tokenized_dir, checkpoint_dir), ROOT_DIR)
        report["train_result"] = train_result
        if train_result["returncode"] != 0:
            write_json(run_dir / "report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(train_result["returncode"])

    model = load_model_with_lora(
        lora_path=str(checkpoint_dir),
        prefer_full_checkpoint=True,
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        max_sequence=args.max_sequence,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )

    primer_tokens = build_stage_b_primer(chords, args.bpm)
    report["primer_tokens"] = [int(token) for token in primer_tokens]
    report["primer_token_names"] = [stage_b_token_name(token) for token in primer_tokens]

    sample_rows: list[dict[str, Any]] = []
    for index in range(1, int(args.num_samples) + 1):
        sample_seed = int(args.seed) + index - 1
        torch.manual_seed(sample_seed)
        if args.generation_mode == "constrained":
            generated_tokens = generate_stage_b_constrained_tokens(
                model=model,
                primer_tokens=primer_tokens,
                chords=chords,
                bpm=args.bpm,
                bars=args.bars,
                note_groups_per_bar=args.constrained_note_groups_per_bar,
                max_sequence=args.max_sequence,
                temperature=args.temperature,
                top_k=args.top_k,
                coverage_aware_positions=args.coverage_aware_positions,
                coverage_position_window=args.coverage_position_window,
                chord_aware_pitches=args.chord_aware_pitches,
                chord_pitch_mode=args.chord_pitch_mode,
                chord_pitch_repeat_window=args.chord_pitch_repeat_window,
                pitch_min=args.constrained_pitch_min,
                pitch_max=args.constrained_pitch_max,
                max_adjacent_interval=args.constrained_max_adjacent_interval,
                jazz_rhythm_positions=args.jazz_rhythm_positions,
                jazz_duration_tokens=args.jazz_duration_tokens,
                jazz_rhythm_profile=args.jazz_rhythm_profile,
                cap_duration_to_next_position=args.cap_duration_to_next_position,
                fill_duration_to_next_position=args.fill_duration_to_next_position,
            )
        else:
            generated_tokens = generate_stage_b_tokens(
                model=model,
                primer_tokens=primer_tokens,
                target_length=args.max_sequence,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
            )
        midi_path = samples_dir / f"stage_b_sample_{index}.mid"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi = decode_stage_b_midi(generated_tokens, tempo_bpm=args.bpm)
        postprocess_report = None
        if args.postprocess_overlap:
            postprocess_report = postprocess_stage_b_midi(
                midi,
                simultaneous_limit=args.max_simultaneous_notes,
            )
        midi.write(str(midi_path))
        sample_rows.append(
            sample_report(
                sample_index=index,
                sample_seed=sample_seed,
                tokens=generated_tokens,
                primer_size=len(primer_tokens),
                target_length=args.max_sequence,
                midi_path=midi_path,
                request=request,
                postprocess_report=postprocess_report,
                strict_min_unique_pitches=args.strict_min_unique_pitches,
                strict_min_unique_positions=args.strict_min_unique_positions,
                strict_min_unique_position_pitch_pairs=args.strict_min_unique_position_pitch_pairs,
                strict_max_repeated_position_pitch_pair_ratio=args.strict_max_repeated_position_pitch_pair_ratio,
                strict_max_postprocess_removal_ratio=args.strict_max_postprocess_removal_ratio,
            )
        )

    report["samples"] = sample_rows
    summary = build_probe_summary(
        sample_rows,
        min_valid_samples=args.min_valid_samples,
        min_strict_valid_samples=args.min_strict_valid_samples,
        require_all_grammar_samples=args.require_all_grammar_samples,
        max_collapse_warning_sample_rate=args.max_collapse_warning_sample_rate,
    )
    report["summary"] = summary
    report["valid_sample_count"] = summary["valid_sample_count"]
    report["grammar_gate_sample_count"] = summary["grammar_gate_sample_count"]
    report["sample_count"] = summary["sample_count"]
    report["passed_generation_gate"] = summary["passed_generation_gate"]
    report["passed_grammar_gate"] = summary["passed_grammar_gate"]
    report["passed_strict_review_gate"] = summary["passed_strict_review_gate"]
    if not report["passed_generation_gate"]:
        report["failure_reason"] = (
            f"Only {summary['valid_sample_count']} Stage B generated samples passed the MIDI review gate; "
            f"required {summary['min_valid_samples']}"
        )
    if not summary["passed_strict_generation_gate"]:
        report["strict_failure_reason"] = (
            f"Only {summary['strict_valid_sample_count']} Stage B generated samples passed the strict collapse gate; "
            f"required {summary['min_strict_valid_samples']}"
        )
    if not summary["passed_collapse_rate_gate"]:
        report["strict_failure_reason"] = (
            f"Collapse warning sample rate {summary['collapse_warning_sample_rate']:.3f} exceeded "
            f"{summary['max_collapse_warning_sample_rate']:.3f}"
        )
    if not report["passed_grammar_gate"]:
        report["failure_reason"] = "Stage B generated samples did not satisfy the configured grammar gate"

    write_json(run_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.require_note_groups and not report["passed_grammar_gate"]:
        return 4
    if args.require_valid_sample and not report["passed_generation_gate"]:
        return 3
    if args.require_strict_valid_sample and not report["passed_strict_review_gate"]:
        return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
