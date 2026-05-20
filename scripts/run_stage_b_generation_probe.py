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
    chord_tokens,
    decode_stage_b_midi,
    is_note_duration_token,
    is_note_pitch_token,
    is_position_token,
    is_velocity_token,
    stage_b_token_name,
)
from utilities.constants import TOKEN_BAR, TOKEN_END, VOCAB_SIZE  # noqa: E402
from utilities.device import get_device  # noqa: E402


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
) -> dict[str, Any]:
    raw_generated_tokens = [int(token) for token in tokens[primer_size:]]
    grammar = analyze_stage_b_note_grammar(tokens, primer_size=primer_size)
    metrics = compute_midi_metrics(midi_path, 0, False, request=request)
    valid, reason = validate_metrics(metrics, request.density, bars=request.bars)
    grammar_gate_passed = bool(grammar["complete_note_groups"] > 0 and metrics.note_count > 0)
    return {
        "sample_index": int(sample_index),
        "sample_seed": int(sample_seed),
        "midi_path": str(midi_path),
        "token_count": int(len(tokens)),
        "generated_token_count": int(len(raw_generated_tokens)),
        "ended_early": bool(len(tokens) < int(target_length)),
        "hit_end_token": bool(TOKEN_END in raw_generated_tokens),
        "valid": bool(valid),
        "grammar_gate_passed": grammar_gate_passed,
        "failure_reason": reason,
        "grammar": grammar,
        "postprocess": postprocess_report or {"enabled": False},
        "metrics": metrics.to_dict(),
        "generated_token_names_head": [stage_b_token_name(token) for token in raw_generated_tokens[:48]],
    }


def build_probe_summary(
    sample_rows: Sequence[dict[str, Any]],
    min_valid_samples: int = 1,
    require_all_grammar_samples: bool = False,
) -> dict[str, Any]:
    sample_count = int(len(sample_rows))
    valid_indices = [int(row["sample_index"]) for row in sample_rows if row["valid"]]
    grammar_indices = [int(row["sample_index"]) for row in sample_rows if row["grammar_gate_passed"]]
    valid_sample_count = int(len(valid_indices))
    grammar_gate_sample_count = int(len(grammar_indices))

    if require_all_grammar_samples:
        passed_grammar_gate = bool(sample_count > 0 and grammar_gate_sample_count == sample_count)
    else:
        passed_grammar_gate = bool(grammar_gate_sample_count > 0)

    failure_reasons: dict[str, int] = {}
    for row in sample_rows:
        if row["valid"]:
            continue
        reason = str(row.get("failure_reason") or "unknown")
        failure_reasons[reason] = failure_reasons.get(reason, 0) + 1

    return {
        "sample_count": sample_count,
        "valid_sample_count": valid_sample_count,
        "grammar_gate_sample_count": grammar_gate_sample_count,
        "valid_sample_rate": float(valid_sample_count / sample_count) if sample_count else 0.0,
        "grammar_gate_sample_rate": float(grammar_gate_sample_count / sample_count) if sample_count else 0.0,
        "valid_sample_indices": valid_indices,
        "grammar_gate_sample_indices": grammar_indices,
        "min_valid_samples": int(min_valid_samples),
        "require_all_grammar_samples": bool(require_all_grammar_samples),
        "passed_generation_gate": bool(valid_sample_count >= int(min_valid_samples)),
        "passed_grammar_gate": passed_grammar_gate,
        "failure_reasons": failure_reasons,
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
    parser.add_argument("--require_all_grammar_samples", action="store_true")
    parser.add_argument("--require_note_groups", action="store_true")
    parser.add_argument("--require_valid_sample", action="store_true")
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
            )
        )

    report["samples"] = sample_rows
    summary = build_probe_summary(
        sample_rows,
        min_valid_samples=args.min_valid_samples,
        require_all_grammar_samples=args.require_all_grammar_samples,
    )
    report["summary"] = summary
    report["valid_sample_count"] = summary["valid_sample_count"]
    report["grammar_gate_sample_count"] = summary["grammar_gate_sample_count"]
    report["sample_count"] = summary["sample_count"]
    report["passed_generation_gate"] = summary["passed_generation_gate"]
    report["passed_grammar_gate"] = summary["passed_grammar_gate"]
    if not report["passed_generation_gate"]:
        report["failure_reason"] = (
            f"Only {summary['valid_sample_count']} Stage B generated samples passed the MIDI review gate; "
            f"required {summary['min_valid_samples']}"
        )
    if not report["passed_grammar_gate"]:
        report["failure_reason"] = "Stage B generated samples did not satisfy the configured grammar gate"

    write_json(run_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.require_note_groups and not report["passed_grammar_gate"]:
        return 4
    if args.require_valid_sample and not report["passed_generation_gate"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
