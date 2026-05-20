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
    POSITIONS_PER_BAR,
    chord_tokens,
    decode_stage_b_midi,
    duration_steps_from_token,
    is_note_duration_token,
    is_note_pitch_token,
    is_position_token,
    is_velocity_token,
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
) -> list[int]:
    tokens = [int(token) for token in primer_tokens]
    families = [
        range(TOKEN_POSITION_START, TOKEN_POSITION_END + 1),
        range(TOKEN_VELOCITY_START, TOKEN_VELOCITY_END + 1),
        range(TOKEN_NOTE_PITCH_START, TOKEN_NOTE_PITCH_END + 1),
        range(TOKEN_NOTE_DURATION_START, TOKEN_NOTE_DURATION_END + 1),
    ]

    for bar_index in range(max(1, int(bars))):
        if bar_index > 0:
            chord = chords[bar_index % len(chords)] if chords else None
            tokens.append(TOKEN_BAR)
            tokens.extend(chord_tokens(chord))
        for _group_index in range(max(1, int(note_groups_per_bar))):
            for allowed_tokens in families:
                if len(tokens) >= int(max_sequence) - 1:
                    tokens.append(TOKEN_END)
                    return tokens
                tokens.append(
                    next_token_from_model(
                        model,
                        tokens=tokens,
                        allowed_tokens=list(allowed_tokens),
                        temperature=temperature,
                        top_k=top_k,
                    )
                )

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
    for row in sample_rows:
        collapse = row.get("collapse", {})
        temporal_coverage = row.get("temporal_coverage", {})
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
