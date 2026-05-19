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

import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "music_transformer"))

from inference.app.metrics import compute_midi_metrics, validate_metrics  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.control_tokens import control_prefix_tokens  # noqa: E402
from scripts.generate import load_model_with_lora  # noqa: E402
from scripts.run_stage_a_tiny_overfit import build_train_command, run_command, write_json  # noqa: E402
from scripts.run_stage_b_window_tiny_overfit import read_json, run_prepare_command, token_stats  # noqa: E402
from scripts.stage_b_tokens import (  # noqa: E402
    SEQUENCE_FORMAT_STAGE_B_V1,
    chord_tokens,
    decode_stage_b_midi,
    stage_b_token_name,
)
from utilities.constants import TOKEN_END, VOCAB_SIZE  # noqa: E402
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


def decode_tokens_to_midi(tokens: Sequence[int], output_path: Path, bpm: float | int) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = decode_stage_b_midi(tokens, tempo_bpm=bpm)
    midi.write(str(output_path))
    return output_path


def sample_report(
    sample_index: int,
    tokens: Sequence[int],
    primer_size: int,
    target_length: int,
    midi_path: Path,
    request: GenerationRequest,
) -> dict[str, Any]:
    raw_generated_tokens = [int(token) for token in tokens[primer_size:]]
    metrics = compute_midi_metrics(midi_path, 0, False, request=request)
    valid, reason = validate_metrics(metrics, request.density, bars=request.bars)
    return {
        "sample_index": int(sample_index),
        "midi_path": str(midi_path),
        "token_count": int(len(tokens)),
        "generated_token_count": int(len(raw_generated_tokens)),
        "ended_early": bool(len(tokens) < int(target_length)),
        "hit_end_token": bool(TOKEN_END in raw_generated_tokens),
        "valid": bool(valid),
        "failure_reason": reason,
        "metrics": metrics.to_dict(),
        "generated_token_names_head": [stage_b_token_name(token) for token in raw_generated_tokens[:48]],
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
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--bars", type=int, default=2)
    parser.add_argument("--chords", type=str, default="Cm7,Fm7,Bb7,Ebmaj7")
    parser.add_argument("--density", type=str, default="medium")
    parser.add_argument("--energy", type=str, default="mid")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--top_p", type=float, default=None)
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

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "issue": 18,
        "sequence_format": SEQUENCE_FORMAT_STAGE_B_V1,
        "request": request.to_dict(),
        "checkpoint_dir": str(checkpoint_dir),
        "sample_vocab_size": int(VOCAB_SIZE),
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
        generated_tokens = generate_stage_b_tokens(
            model=model,
            primer_tokens=primer_tokens,
            target_length=args.max_sequence,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        midi_path = samples_dir / f"stage_b_sample_{index}.mid"
        decode_tokens_to_midi(generated_tokens, midi_path, bpm=args.bpm)
        sample_rows.append(
            sample_report(
                sample_index=index,
                tokens=generated_tokens,
                primer_size=len(primer_tokens),
                target_length=args.max_sequence,
                midi_path=midi_path,
                request=request,
            )
        )

    report["samples"] = sample_rows
    report["valid_sample_count"] = sum(1 for row in sample_rows if row["valid"])
    report["sample_count"] = len(sample_rows)
    report["passed_generation_gate"] = bool(report["valid_sample_count"] > 0)
    if not report["passed_generation_gate"]:
        report["failure_reason"] = "No Stage B generated sample passed the MIDI review gate"

    write_json(run_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    if args.require_valid_sample and not report["passed_generation_gate"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
