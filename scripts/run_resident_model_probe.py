#!/usr/bin/env python3
"""Measure a resident Music Transformer without scheduler or MIDI file output."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence

import pretty_midi


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import build_fallback_midi, phrase_duration_sec  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.generate import build_primer, generate_once, load_model_with_lora  # noqa: E402
from midi_processor.processor import (  # noqa: E402
    RANGE_NOTE_OFF,
    RANGE_NOTE_ON,
    RANGE_TIME_SHIFT,
    decode_midi,
)


REPORT_SCHEMA_VERSION = "resident_model_generation_probe_v6"
TIME_SHIFT_START = RANGE_NOTE_ON + RANGE_NOTE_OFF
TIME_SHIFT_END = TIME_SHIFT_START + RANGE_TIME_SHIFT - 1
TIME_STEP_MS = 1000.0 / RANGE_TIME_SHIFT


def percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def timing_summary(values_ms: Sequence[float]) -> dict[str, float | int | None]:
    return {
        "sample_count": len(values_ms),
        "minimum": min(values_ms) if values_ms else None,
        "p50": percentile(values_ms, 0.50),
        "p95": percentile(values_ms, 0.95),
        "p99": percentile(values_ms, 0.99),
        "maximum": max(values_ms) if values_ms else None,
    }


def stage_a_musical_duration_ms(tokens: Sequence[int]) -> float:
    return sum(
        (int(token) - TIME_SHIFT_START + 1) * TIME_STEP_MS
        for token in tokens
        if TIME_SHIFT_START <= int(token) <= TIME_SHIFT_END
    )


def validate_generated_token_block(
    tokens: Sequence[int],
    *,
    lookahead_ms: float,
    allow_rest_bar: bool = False,
) -> dict[str, object]:
    active_pitches: set[int] = set()
    orphan_note_off_count = 0
    duplicate_note_on_count = 0
    for raw_token in tokens:
        token = int(raw_token)
        if 0 <= token < RANGE_NOTE_ON:
            if token in active_pitches:
                duplicate_note_on_count += 1
            active_pitches.add(token)
        elif RANGE_NOTE_ON <= token < RANGE_NOTE_ON + RANGE_NOTE_OFF:
            pitch = token - RANGE_NOTE_ON
            if pitch not in active_pitches:
                orphan_note_off_count += 1
            active_pitches.discard(pitch)

    musical_duration_ms = stage_a_musical_duration_ms(tokens)
    quantized_target_ms = math.ceil(lookahead_ms / TIME_STEP_MS) * TIME_STEP_MS
    decoded_note_count = 0
    decoded_note_end_max_ms = None
    silent_note_count = 0
    decode_error = None
    try:
        decoded = decode_midi([int(token) for token in tokens])
        notes = [
            note
            for instrument in decoded.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ]
        decoded_note_count = len(notes)
        # Stage A decodes velocity as ``value * 4`` and carries it as state, so a
        # block can decode to MIDI velocity 0. That is a note-off by the MIDI
        # spec and ``build_scheduled_midi_block`` rejects it; without this check
        # the validator would pass a block the scheduler cannot play.
        silent_note_count = sum(1 for note in notes if int(note.velocity) <= 0)
        if notes:
            decoded_note_end_max_ms = max(float(note.end) * 1000.0 for note in notes)
    except Exception as exc:
        decode_error = f"{type(exc).__name__}: {exc}"

    duration_matches_target = math.isclose(
        musical_duration_ms,
        quantized_target_ms,
        abs_tol=1e-6,
    )
    decoded_notes_within_target = (
        decoded_note_end_max_ms is not None
        and decoded_note_end_max_ms <= quantized_target_ms + 1e-6
    )
    # A whole-bar rest fills the bar exactly and decodes to no notes. That is
    # music, not truncation, and it is distinguishable from token-budget
    # underfill, which leaves duration_matches_target False. Callers that can
    # play silence opt in; the default still requires a note.
    is_rest_bar = decoded_note_count == 0 and duration_matches_target and decode_error is None
    rest_bar_accepted = is_rest_bar and allow_rest_bar
    valid = all(
        (
            decode_error is None,
            decoded_note_count > 0 or rest_bar_accepted,
            duration_matches_target,
            decoded_notes_within_target or rest_bar_accepted,
            orphan_note_off_count == 0,
            duplicate_note_on_count == 0,
            silent_note_count == 0,
            not active_pitches,
        )
    )
    return {
        "valid": valid,
        "musical_duration_ms": musical_duration_ms,
        "quantized_target_ms": quantized_target_ms,
        "duration_matches_target": duration_matches_target,
        "decoded_note_count": decoded_note_count,
        "decoded_note_end_max_ms": decoded_note_end_max_ms,
        "decoded_notes_within_target": decoded_notes_within_target,
        "orphan_note_off_count": orphan_note_off_count,
        "duplicate_note_on_count": duplicate_note_on_count,
        "is_rest_bar": is_rest_bar,
        "rest_bar_accepted": rest_bar_accepted,
        "silent_note_count": silent_note_count,
        "stuck_note_count": len(active_pitches),
        "decode_error": decode_error,
    }


def validate_in_memory_midi_block(
    midi: pretty_midi.PrettyMIDI,
    *,
    lookahead_ms: float,
    block_duration_ms: float,
) -> dict[str, object]:
    notes = sorted(
        (
            note
            for instrument in midi.instruments
            if not instrument.is_drum
            for note in instrument.notes
        ),
        key=lambda note: (note.start, note.pitch, note.end),
    )
    target_seconds = lookahead_ms / 1000.0
    invalid_pitch_count = sum(not 0 <= int(note.pitch) <= 127 for note in notes)
    invalid_velocity_count = sum(not 1 <= int(note.velocity) <= 127 for note in notes)
    invalid_time_count = sum(
        float(note.start) < 0.0 or float(note.end) <= float(note.start)
        for note in notes
    )
    target_overrun_count = sum(float(note.end) > target_seconds + 1e-9 for note in notes)
    duration_matches_target = math.isclose(
        block_duration_ms,
        lookahead_ms,
        abs_tol=1e-6,
    )

    last_end_by_pitch: dict[int, float] = {}
    same_pitch_overlap_count = 0
    for note in notes:
        pitch = int(note.pitch)
        if float(note.start) < last_end_by_pitch.get(pitch, 0.0) - 1e-9:
            same_pitch_overlap_count += 1
        last_end_by_pitch[pitch] = max(last_end_by_pitch.get(pitch, 0.0), float(note.end))

    valid = bool(notes) and duration_matches_target and all(
        count == 0
        for count in (
            invalid_pitch_count,
            invalid_velocity_count,
            invalid_time_count,
            target_overrun_count,
            same_pitch_overlap_count,
        )
    )
    return {
        "valid": valid,
        "decoded_note_count": len(notes),
        "decoded_note_end_max_ms": (
            max(float(note.end) for note in notes) * 1000.0 if notes else None
        ),
        "target_window_ms": lookahead_ms,
        "block_duration_ms": block_duration_ms,
        "duration_matches_target": duration_matches_target,
        "invalid_pitch_count": invalid_pitch_count,
        "invalid_velocity_count": invalid_velocity_count,
        "invalid_time_count": invalid_time_count,
        "target_overrun_count": target_overrun_count,
        "same_pitch_overlap_count": same_pitch_overlap_count,
    }


def evaluate_generation_arm(
    *,
    target_total_tokens: int,
    primer_tokens: int,
    generated_token_counts: Sequence[int],
    final_block_durations_ms: Sequence[float],
    generation_times_ms: Sequence[float],
    model_validation_times_ms: Sequence[float],
    fallback_generation_times_ms: Sequence[float],
    fallback_validation_times_ms: Sequence[float],
    lookahead_ms: float,
    scheduling_margin_ms: float,
    minimum_gate_samples: int,
    one_bar_contract_validated: bool,
) -> dict[str, object]:
    summary = timing_summary(generation_times_ms)
    model_validation_summary = timing_summary(model_validation_times_ms)
    fallback_generation_summary = timing_summary(fallback_generation_times_ms)
    fallback_validation_summary = timing_summary(fallback_validation_times_ms)
    block_resolution_times_ms = [
        model_validation_ms + fallback_generation_ms + fallback_validation_ms
        for model_validation_ms, fallback_generation_ms, fallback_validation_ms in zip(
            model_validation_times_ms,
            fallback_generation_times_ms,
            fallback_validation_times_ms,
        )
    ]
    block_resolution_summary = timing_summary(block_resolution_times_ms)
    block_ready_times_ms = [
        generation_ms + resolution_ms
        for generation_ms, resolution_ms in zip(
            generation_times_ms,
            block_resolution_times_ms,
        )
    ]
    block_ready_summary = timing_summary(block_ready_times_ms)
    timing_samples_aligned = len(
        {
            len(generation_times_ms),
            len(model_validation_times_ms),
            len(fallback_generation_times_ms),
            len(fallback_validation_times_ms),
        }
    ) == 1
    enough_timing_samples = (
        timing_samples_aligned and len(generation_times_ms) >= minimum_gate_samples
    )
    all_samples_reach_target_duration = bool(final_block_durations_ms) and all(
        duration_ms >= lookahead_ms for duration_ms in final_block_durations_ms
    )
    samples_reaching_target_duration = sum(
        duration_ms >= lookahead_ms for duration_ms in final_block_durations_ms
    )
    generation_p99 = summary["p99"]
    block_resolution_p99 = block_resolution_summary["p99"]
    deadline_measurement_ms = None
    if generation_p99 is not None and block_resolution_p99 is not None:
        deadline_measurement_ms = float(generation_p99) + float(block_resolution_p99)
    deadline_budget_ms = lookahead_ms - scheduling_margin_ms
    passed = None
    measurement_sufficient = enough_timing_samples and one_bar_contract_validated
    if measurement_sufficient and deadline_measurement_ms is not None:
        passed = all_samples_reach_target_duration and deadline_measurement_ms <= deadline_budget_ms
    operating_headroom_passed = None
    if measurement_sufficient and block_ready_summary["p99"] is not None:
        operating_headroom_passed = float(block_ready_summary["p99"]) <= lookahead_ms * 0.5
    return {
        "target_total_tokens": target_total_tokens,
        "primer_tokens": primer_tokens,
        "requested_generated_tokens": target_total_tokens - primer_tokens,
        "generated_token_counts": list(generated_token_counts),
        "final_block_duration_ms": timing_summary(final_block_durations_ms),
        "samples_reaching_target_duration": samples_reaching_target_duration,
        "samples_under_target_duration": (
            len(final_block_durations_ms) - samples_reaching_target_duration
        ),
        "all_samples_reach_target_duration": all_samples_reach_target_duration,
        "generation_time_ms": summary,
        "model_validation_time_ms": model_validation_summary,
        "fallback_generation_time_ms": fallback_generation_summary,
        "fallback_validation_time_ms": fallback_validation_summary,
        "block_resolution_time_ms": block_resolution_summary,
        "block_ready_time_ms": block_ready_summary,
        "lookahead_ms": lookahead_ms,
        "scheduling_margin_ms": scheduling_margin_ms,
        "generation_deadline_budget_ms": deadline_budget_ms,
        "generation_p99_plus_block_resolution_p99_ms": deadline_measurement_ms,
        "minimum_gate_samples": minimum_gate_samples,
        "block_ready_sample_counts_aligned": timing_samples_aligned,
        "timing_sample_count_sufficient": enough_timing_samples,
        "one_bar_musical_duration_contract_validated": one_bar_contract_validated,
        "measurement_sufficient_for_r2_gate": measurement_sufficient,
        "passed_r2_generation_deadline_gate": passed,
        "passed_r2_operating_headroom_gate": operating_headroom_passed,
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_target_lengths(value: str) -> list[int]:
    lengths = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not lengths or any(length <= 0 for length in lengths):
        raise argparse.ArgumentTypeError("target lengths must be positive comma-separated integers")
    return lengths


def parse_chords(value: str) -> list[str]:
    chords = [item.strip() for item in value.split(",") if item.strip()]
    if not chords:
        raise argparse.ArgumentTypeError("fallback chords must not be empty")
    return chords


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--conditioning-midi", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--bpm", type=float, default=128.0)
    parser.add_argument("--beats-per-bar", type=int, default=4)
    parser.add_argument("--primer-max-tokens", type=int, default=32)
    parser.add_argument("--target-total-tokens", type=parse_target_lengths, default=[48, 64, 96, 128])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--minimum-gate-samples", type=int, default=20)
    parser.add_argument("--scheduling-margin-ms", type=float, default=20.0)
    parser.add_argument("--warmup-target-total-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--fallback-chords",
        type=parse_chords,
        default=["Dm7", "G7", "Cmaj7", "A7"],
    )
    parser.add_argument("--fallback-density", choices=("sparse", "medium", "dense"), default="medium")
    parser.add_argument("--fallback-energy", choices=("low", "mid", "high"), default="mid")
    parser.add_argument("--no-grammar-mask", action="store_false", dest="grammar_mask")
    parser.set_defaults(grammar_mask=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint not found: {args.checkpoint}")
    if not args.conditioning_midi.is_file():
        raise FileNotFoundError(f"conditioning MIDI not found: {args.conditioning_midi}")
    if args.bpm <= 0 or args.beats_per_bar <= 0:
        raise ValueError("bpm and beats-per-bar must be positive")
    if args.repetitions <= 0 or args.minimum_gate_samples <= 0:
        raise ValueError("repetitions and minimum-gate-samples must be positive")

    import torch

    load_started_ns = time.perf_counter_ns()
    model, model_metadata = load_model_with_lora(
        lora_path=str(args.checkpoint.parent),
        checkpoint_path=str(args.checkpoint),
        prefer_full_checkpoint=True,
        max_sequence=max(args.target_total_tokens),
        return_metadata=True,
    )
    load_time_ms = (time.perf_counter_ns() - load_started_ns) / 1_000_000
    device = str(next(model.parameters()).device)
    primer = build_primer(
        conditioning_midi=str(args.conditioning_midi),
        primer_max_tokens=args.primer_max_tokens,
        append_sep_token=True,
        control_format="control_v1",
        role="lead",
        tempo_bpm=args.bpm,
    )
    primer_tokens = len(primer)
    if any(target <= primer_tokens for target in args.target_total_tokens):
        raise ValueError(
            f"every target total length must exceed primer length {primer_tokens}: "
            f"{args.target_total_tokens}"
        )

    warmup_target = max(primer_tokens + 1, args.warmup_target_total_tokens)
    lookahead_ms = args.beats_per_bar * 60_000.0 / args.bpm
    torch.manual_seed(args.seed)
    generate_once(
        model=model,
        primer=primer,
        target_length=warmup_target,
        strip_primer=True,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        grammar_mask=args.grammar_mask,
        target_duration_seconds=lookahead_ms / 1000.0,
    )

    arms = []
    for target_total_tokens in args.target_total_tokens:
        generation_times_ms: list[float] = []
        model_validation_times_ms: list[float] = []
        fallback_generation_times_ms: list[float] = []
        fallback_validation_times_ms: list[float] = []
        generated_token_counts: list[int] = []
        model_musical_durations_ms: list[float] = []
        final_block_durations_ms: list[float] = []
        model_block_validations: list[dict[str, object]] = []
        final_block_validations: list[dict[str, object]] = []
        fallback_used: list[bool] = []
        generation_metadata: list[dict[str, object]] = []
        for repetition in range(args.repetitions):
            sample_seed = args.seed + repetition
            torch.manual_seed(sample_seed)
            started_ns = time.perf_counter_ns()
            tokens, sample_metadata = generate_once(
                model=model,
                primer=primer,
                target_length=target_total_tokens,
                strip_primer=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                grammar_mask=args.grammar_mask,
                target_duration_seconds=lookahead_ms / 1000.0,
                return_metadata=True,
            )
            generation_times_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)
            generation_metadata.append(sample_metadata)
            generated_token_counts.append(len(tokens))
            model_duration_ms = stage_a_musical_duration_ms(tokens)
            model_musical_durations_ms.append(model_duration_ms)
            validation_started_ns = time.perf_counter_ns()
            model_validation = validate_generated_token_block(tokens, lookahead_ms=lookahead_ms)
            model_validation_times_ms.append(
                (time.perf_counter_ns() - validation_started_ns) / 1_000_000
            )
            model_block_validations.append(model_validation)

            used_fallback = not bool(model_validation["valid"])
            fallback_used.append(used_fallback)
            fallback_generation_ms = 0.0
            fallback_validation_ms = 0.0
            if used_fallback:
                fallback_request = GenerationRequest(
                    bpm=args.bpm,
                    chord_progression=list(args.fallback_chords),
                    bars=1,
                    time_signature=f"{args.beats_per_bar}/4",
                    energy=args.fallback_energy,
                    density=args.fallback_density,
                    style="deterministic_fallback",
                    seed=sample_seed,
                    job_id=f"resident-fallback-{sample_seed}",
                )
                fallback_request.validate()
                fallback_started_ns = time.perf_counter_ns()
                fallback_midi = build_fallback_midi(fallback_request)
                fallback_generation_ms = (
                    time.perf_counter_ns() - fallback_started_ns
                ) / 1_000_000
                fallback_validation_started_ns = time.perf_counter_ns()
                fallback_duration_ms = phrase_duration_sec(fallback_request) * 1000.0
                fallback_validation = validate_in_memory_midi_block(
                    fallback_midi,
                    lookahead_ms=lookahead_ms,
                    block_duration_ms=fallback_duration_ms,
                )
                fallback_validation_ms = (
                    time.perf_counter_ns() - fallback_validation_started_ns
                ) / 1_000_000
                final_block_validations.append(
                    {
                        "source": "deterministic_fallback",
                        "valid": bool(fallback_validation["valid"]),
                        "validation": fallback_validation,
                    }
                )
                final_block_durations_ms.append(fallback_duration_ms)
            else:
                final_block_validations.append(
                    {
                        "source": "resident_model",
                        "valid": True,
                        "validation": model_validation,
                    }
                )
                final_block_durations_ms.append(model_duration_ms)
            fallback_generation_times_ms.append(fallback_generation_ms)
            fallback_validation_times_ms.append(fallback_validation_ms)

        one_bar_contract_validated = bool(final_block_validations) and all(
            bool(validation["valid"]) for validation in final_block_validations
        )
        arm = evaluate_generation_arm(
            target_total_tokens=target_total_tokens,
            primer_tokens=primer_tokens,
            generated_token_counts=generated_token_counts,
            final_block_durations_ms=final_block_durations_ms,
            generation_times_ms=generation_times_ms,
            model_validation_times_ms=model_validation_times_ms,
            fallback_generation_times_ms=fallback_generation_times_ms,
            fallback_validation_times_ms=fallback_validation_times_ms,
            lookahead_ms=lookahead_ms,
            scheduling_margin_ms=args.scheduling_margin_ms,
            minimum_gate_samples=args.minimum_gate_samples,
            one_bar_contract_validated=one_bar_contract_validated,
        )
        arm["model_generated_musical_duration_ms"] = timing_summary(
            model_musical_durations_ms
        )
        arm["model_valid_block_count"] = sum(
            bool(validation["valid"]) for validation in model_block_validations
        )
        arm["model_invalid_block_count"] = len(model_block_validations) - int(
            arm["model_valid_block_count"]
        )
        arm["fallback_count"] = sum(fallback_used)
        arm["fallback_generation_time_ms_used"] = timing_summary(
            [
                duration_ms
                for duration_ms, used in zip(fallback_generation_times_ms, fallback_used)
                if used
            ]
        )
        arm["fallback_validation_time_ms_used"] = timing_summary(
            [
                duration_ms
                for duration_ms, used in zip(fallback_validation_times_ms, fallback_used)
                if used
            ]
        )
        arm["fallback_failure_count"] = sum(
            validation["source"] == "deterministic_fallback"
            and not bool(validation["valid"])
            for validation in final_block_validations
        )
        arm["final_valid_block_count"] = sum(
            bool(validation["valid"]) for validation in final_block_validations
        )
        arm["final_invalid_block_count"] = len(final_block_validations) - int(
            arm["final_valid_block_count"]
        )
        arm["model_block_validation_failure_counts"] = {
            "duration_mismatch": sum(
                not bool(validation["duration_matches_target"])
                for validation in model_block_validations
            ),
            "empty_decoded_block": sum(
                int(validation["decoded_note_count"]) == 0
                for validation in model_block_validations
            ),
            "decoded_target_overrun": sum(
                not bool(validation["decoded_notes_within_target"])
                and int(validation["decoded_note_count"]) > 0
                for validation in model_block_validations
            ),
            "orphan_note_off": sum(
                int(validation["orphan_note_off_count"])
                for validation in model_block_validations
            ),
            "duplicate_note_on": sum(
                int(validation["duplicate_note_on_count"])
                for validation in model_block_validations
            ),
            "stuck_note": sum(
                int(validation["stuck_note_count"])
                for validation in model_block_validations
            ),
            "decode_error": sum(
                validation["decode_error"] is not None
                for validation in model_block_validations
            ),
        }
        model_forward_step_counts = [
            int(metadata["model_forward_step_count"]) for metadata in generation_metadata
        ]
        arm["model_forward_step_counts"] = model_forward_step_counts
        arm["sampled_output_token_counts"] = [
            int(metadata["sampled_output_token_count"]) for metadata in generation_metadata
        ]
        arm["boundary_note_off_counts"] = [
            int(metadata["boundary_note_off_count"]) for metadata in generation_metadata
        ]
        arm["returned_generated_token_counts"] = [
            int(metadata["returned_generated_token_count"]) for metadata in generation_metadata
        ]
        arm["generation_time_per_model_forward_step_ms"] = timing_summary(
            [
                generation_ms / forward_steps
                for generation_ms, forward_steps in zip(
                    generation_times_ms,
                    model_forward_step_counts,
                )
                if forward_steps > 0
            ]
        )
        arm["stop_reason_counts"] = {
            reason: sum(metadata["stop_reason"] == reason for metadata in generation_metadata)
            for reason in ("duration_target", "token_budget", "end_token")
        }
        arm["underfill_stop_reason_counts"] = {
            reason: sum(
                metadata["stop_reason"] == reason
                and not bool(validation["duration_matches_target"])
                for metadata, validation in zip(generation_metadata, model_block_validations)
            )
            for reason in ("duration_target", "token_budget", "end_token")
        }
        arm["generation_metadata"] = generation_metadata
        arm["model_block_validations"] = model_block_validations
        arm["final_block_validations"] = final_block_validations
        arms.append(arm)

    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "scope": "resident_model_generation_and_decode_validation",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "conditioning_midi": str(args.conditioning_midi.resolve()),
        "device": device,
        "model_max_sequence": model_metadata["model_max_sequence"],
        "model_rpr": model_metadata["rpr"],
        "model_shape": {
            "n_layers": model_metadata["n_layers"],
            "num_heads": model_metadata["num_heads"],
            "d_model": model_metadata["d_model"],
            "dim_feedforward": model_metadata["dim_feedforward"],
        },
        "checkpoint_model_config_present": model_metadata[
            "checkpoint_model_config_present"
        ],
        "resized_token_layer_keys": model_metadata["resized_token_layer_keys"],
        "rpr_embedding_shapes": model_metadata["rpr_embedding_shapes"],
        "model_load_time_ms": load_time_ms,
        "bpm": args.bpm,
        "beats_per_bar": args.beats_per_bar,
        "grammar_mask": args.grammar_mask,
        "generation_duration_target_enabled": True,
        "boundary_time_shift_crop_enabled": True,
        "generated_note_off_closure_enabled": True,
        "invalid_block_fallback_enabled": True,
        "fallback": {
            "mode": "in_memory_deterministic",
            "chord_progression": list(args.fallback_chords),
            "density": args.fallback_density,
            "energy": args.fallback_energy,
            "bars": 1,
            "time_signature": f"{args.beats_per_bar}/4",
            "filesystem_io_included": False,
        },
        "primer_max_tokens": args.primer_max_tokens,
        "primer_tokens": primer_tokens,
        "warmup_target_total_tokens": warmup_target,
        "repetitions": args.repetitions,
        "seed_start": args.seed,
        "sample_seeds": [args.seed + repetition for repetition in range(args.repetitions)],
        "sampling": {
            "temperature": args.temperature,
            "top_k": args.top_k,
            "top_p": args.top_p,
        },
        "one_bar_musical_duration_contract_validated": bool(arms) and all(
            bool(arm["one_bar_musical_duration_contract_validated"]) for arm in arms
        ),
        "model_decode_validation_included": True,
        "fallback_validation_included": True,
        "scheduler_included": False,
        "arms": arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(args.output_json), **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
