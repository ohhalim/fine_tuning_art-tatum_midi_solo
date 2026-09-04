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


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.generate import build_primer, generate_once, load_model_with_lora  # noqa: E402
from midi_processor.processor import (  # noqa: E402
    RANGE_NOTE_OFF,
    RANGE_NOTE_ON,
    RANGE_TIME_SHIFT,
    decode_midi,
)


REPORT_SCHEMA_VERSION = "resident_model_generation_probe_v4"
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
    valid = all(
        (
            decode_error is None,
            decoded_note_count > 0,
            duration_matches_target,
            decoded_notes_within_target,
            orphan_note_off_count == 0,
            duplicate_note_on_count == 0,
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
        "stuck_note_count": len(active_pitches),
        "decode_error": decode_error,
    }


def evaluate_generation_arm(
    *,
    target_total_tokens: int,
    primer_tokens: int,
    generated_token_counts: Sequence[int],
    musical_durations_ms: Sequence[float],
    generation_times_ms: Sequence[float],
    decode_validation_times_ms: Sequence[float],
    lookahead_ms: float,
    scheduling_margin_ms: float,
    minimum_gate_samples: int,
    one_bar_contract_validated: bool,
) -> dict[str, object]:
    summary = timing_summary(generation_times_ms)
    decode_validation_summary = timing_summary(decode_validation_times_ms)
    block_ready_times_ms = [
        generation_ms + decode_ms
        for generation_ms, decode_ms in zip(
            generation_times_ms,
            decode_validation_times_ms,
        )
    ]
    block_ready_summary = timing_summary(block_ready_times_ms)
    timing_samples_aligned = len(generation_times_ms) == len(decode_validation_times_ms)
    enough_timing_samples = (
        timing_samples_aligned and len(generation_times_ms) >= minimum_gate_samples
    )
    covers_target_bar = bool(musical_durations_ms) and all(
        duration_ms >= lookahead_ms for duration_ms in musical_durations_ms
    )
    samples_covering_target_bar = sum(
        duration_ms >= lookahead_ms for duration_ms in musical_durations_ms
    )
    generation_p99 = summary["p99"]
    decode_validation_p99 = decode_validation_summary["p99"]
    deadline_measurement_ms = None
    if generation_p99 is not None and decode_validation_p99 is not None:
        deadline_measurement_ms = float(generation_p99) + float(decode_validation_p99)
    deadline_budget_ms = lookahead_ms - scheduling_margin_ms
    passed = None
    measurement_sufficient = enough_timing_samples and one_bar_contract_validated
    if measurement_sufficient and deadline_measurement_ms is not None:
        passed = covers_target_bar and deadline_measurement_ms <= deadline_budget_ms
    operating_headroom_passed = None
    if measurement_sufficient and block_ready_summary["p99"] is not None:
        operating_headroom_passed = float(block_ready_summary["p99"]) <= lookahead_ms * 0.5
    return {
        "target_total_tokens": target_total_tokens,
        "primer_tokens": primer_tokens,
        "requested_generated_tokens": target_total_tokens - primer_tokens,
        "generated_token_counts": list(generated_token_counts),
        "generated_musical_duration_ms": timing_summary(musical_durations_ms),
        "samples_covering_target_bar": samples_covering_target_bar,
        "samples_under_target_bar": len(musical_durations_ms) - samples_covering_target_bar,
        "all_samples_cover_target_bar": covers_target_bar,
        "generation_time_ms": summary,
        "decode_validation_time_ms": decode_validation_summary,
        "block_ready_time_ms": block_ready_summary,
        "lookahead_ms": lookahead_ms,
        "scheduling_margin_ms": scheduling_margin_ms,
        "generation_deadline_budget_ms": deadline_budget_ms,
        "generation_plus_decode_p99_ms": deadline_measurement_ms,
        "minimum_gate_samples": minimum_gate_samples,
        "generation_decode_sample_counts_aligned": timing_samples_aligned,
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
        decode_validation_times_ms: list[float] = []
        generated_token_counts: list[int] = []
        musical_durations_ms: list[float] = []
        block_validations: list[dict[str, object]] = []
        for repetition in range(args.repetitions):
            torch.manual_seed(args.seed + repetition)
            started_ns = time.perf_counter_ns()
            tokens = generate_once(
                model=model,
                primer=primer,
                target_length=target_total_tokens,
                strip_primer=True,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                grammar_mask=args.grammar_mask,
                target_duration_seconds=lookahead_ms / 1000.0,
            )
            generation_times_ms.append((time.perf_counter_ns() - started_ns) / 1_000_000)
            generated_token_counts.append(len(tokens))
            musical_durations_ms.append(stage_a_musical_duration_ms(tokens))
            validation_started_ns = time.perf_counter_ns()
            block_validations.append(
                validate_generated_token_block(tokens, lookahead_ms=lookahead_ms)
            )
            decode_validation_times_ms.append(
                (time.perf_counter_ns() - validation_started_ns) / 1_000_000
            )
        one_bar_contract_validated = bool(block_validations) and all(
            bool(validation["valid"]) for validation in block_validations
        )
        arm = evaluate_generation_arm(
            target_total_tokens=target_total_tokens,
            primer_tokens=primer_tokens,
            generated_token_counts=generated_token_counts,
            musical_durations_ms=musical_durations_ms,
            generation_times_ms=generation_times_ms,
            decode_validation_times_ms=decode_validation_times_ms,
            lookahead_ms=lookahead_ms,
            scheduling_margin_ms=args.scheduling_margin_ms,
            minimum_gate_samples=args.minimum_gate_samples,
            one_bar_contract_validated=one_bar_contract_validated,
        )
        arm["valid_block_count"] = sum(
            bool(validation["valid"]) for validation in block_validations
        )
        arm["invalid_block_count"] = len(block_validations) - int(
            arm["valid_block_count"]
        )
        arm["block_validation_failure_counts"] = {
            "duration_mismatch": sum(
                not bool(validation["duration_matches_target"])
                for validation in block_validations
            ),
            "empty_decoded_block": sum(
                int(validation["decoded_note_count"]) == 0
                for validation in block_validations
            ),
            "decoded_target_overrun": sum(
                not bool(validation["decoded_notes_within_target"])
                and int(validation["decoded_note_count"]) > 0
                for validation in block_validations
            ),
            "orphan_note_off": sum(
                int(validation["orphan_note_off_count"])
                for validation in block_validations
            ),
            "duplicate_note_on": sum(
                int(validation["duplicate_note_on_count"])
                for validation in block_validations
            ),
            "stuck_note": sum(
                int(validation["stuck_note_count"])
                for validation in block_validations
            ),
            "decode_error": sum(
                validation["decode_error"] is not None
                for validation in block_validations
            ),
        }
        arm["block_validations"] = block_validations
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
        "underfill_fallback_enabled": False,
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
        "decode_validation_included": True,
        "scheduler_included": False,
        "arms": arms,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"report_path": str(args.output_json), **report}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
