"""
Run a focused Stage A tiny-overfit smoke test.

This script creates 1-3 deterministic, known-good MIDI solo phrases, tokenizes
them into the existing Music Transformer event format, optionally trains a tiny
checkpoint, generates fixed-seed samples, and writes a report with the same
quality gates used by the MVP inference path.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from generate import encode_midi_simple, resolve_full_checkpoint_path  # noqa: E402
from control_tokens import SEQUENCE_FORMAT_LEGACY_SEP  # noqa: E402
from inference.app.generator import generate_midi_phrase  # noqa: E402
from inference.app.metrics import compute_midi_metrics, validate_metrics  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402


DEFAULT_CHORDS = ["Cm7", "Fm7", "Bb7", "Ebmaj7"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n")


def write_known_good_midi(path: Path, variant: int, bpm: int = 124) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    beat_sec = 60.0 / float(bpm)
    chord_duration = beat_sec * 2.0
    step = chord_duration / 6.0
    duration = step * 0.72
    velocity_base = 78 + (variant * 4)

    chord_lines = [
        [60, 63, 67, 70, 67, 63],
        [65, 68, 72, 75, 72, 68],
        [70, 74, 77, 80, 77, 74],
        [75, 79, 82, 86, 82, 79],
    ]

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    piano = pretty_midi.Instrument(program=0, is_drum=False, name=f"tiny_overfit_solo_{variant}")

    for chord_index, chord_pitches in enumerate(chord_lines):
        if variant == 1:
            chord_pitches = chord_pitches[2:] + chord_pitches[:2]
        elif variant == 2:
            chord_pitches = list(reversed(chord_pitches))
        for note_index, pitch in enumerate(chord_pitches):
            start = (chord_index * chord_duration) + (note_index * step)
            if variant == 1 and note_index == 2:
                pitch += 12
            elif variant == 2 and note_index == 1:
                pitch -= 12
            piano.notes.append(
                pretty_midi.Note(
                    velocity=min(110, velocity_base + note_index * 3),
                    pitch=int(pitch),
                    start=float(start),
                    end=float(start + duration),
                )
            )

    pm.instruments.append(piano)
    pm.write(str(path))
    return path


def prepare_tiny_dataset(
    run_dir: Path,
    sample_count: int = 3,
    bpm: int = 124,
) -> dict[str, Any]:
    sample_count = max(1, min(3, int(sample_count)))
    midi_dir = run_dir / "input_midi"
    data_dir = run_dir / "tokenized"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        midi_path = write_known_good_midi(midi_dir / f"tiny_solo_{index + 1}.mid", index, bpm=bpm)
        tokens = encode_midi_simple(str(midi_path))
        if not tokens:
            raise RuntimeError(f"failed to tokenize tiny MIDI: {midi_path}")
        token_array = np.array(tokens, dtype=np.int32)

        train_path = train_dir / f"{index:05d}.npy"
        val_path = val_dir / f"{index:05d}.npy"
        np.save(train_path, token_array)
        np.save(val_path, token_array)
        samples.append(
            {
                "midi_path": str(midi_path),
                "train_tokens_path": str(train_path),
                "val_tokens_path": str(val_path),
                "token_count": int(len(tokens)),
            }
        )

    manifest = {
        "bpm": int(bpm),
        "chord_progression": DEFAULT_CHORDS,
        "sample_count": sample_count,
        "data_dir": str(data_dir),
        "samples": samples,
    }
    write_json(run_dir / "tiny_dataset_manifest.json", manifest)
    return manifest


def run_command(cmd: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(
        cmd,
        cwd=str(cwd),
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "cmd": cmd,
        "returncode": completed.returncode,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def parse_best_validation_loss(text: str) -> float | None:
    matches = re.findall(r"Best validation loss:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(matches[-1]) if matches else None


def summarize_report(report: dict[str, Any]) -> dict[str, Any]:
    inference_result = report.get("inference_result") or {}
    raw_rows = report.get("raw_sample_metrics") or []
    train_result = report.get("train_result") or {}
    stdout_tail = str(train_result.get("stdout_tail") or "")
    stderr_tail = str(train_result.get("stderr_tail") or "")
    best_validation_loss = parse_best_validation_loss(stdout_tail + "\n" + stderr_tail)
    valid_raw_count = sum(1 for row in raw_rows if row.get("valid") is True)
    passed = inference_result.get("status") == "COMPLETED" and inference_result.get("fallback_used") is False
    return {
        "training_mode": report.get("training_mode"),
        "passed_mvp_gate": bool(passed),
        "best_validation_loss": best_validation_loss,
        "valid_raw_sample_count": int(valid_raw_count),
        "raw_sample_count": int(len(raw_rows)),
        "inference_status": inference_result.get("status"),
        "fallback_used": inference_result.get("fallback_used"),
        "model_failure_reason": inference_result.get("model_failure_reason"),
    }


def build_train_command(args: argparse.Namespace, data_dir: Path, checkpoint_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/train_qlora.py",
        "--data_dir",
        str(data_dir),
        "--output_dir",
        str(checkpoint_dir),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--gradient_accumulation",
        "1",
        "--num_workers",
        "0",
        "--label_smoothing",
        "0.0",
        "--lr",
        str(args.lr),
        "--seed",
        str(args.seed),
        "--max_sequence",
        str(args.max_sequence),
        "--n_layers",
        str(args.n_layers),
        "--num_heads",
        str(args.num_heads),
        "--d_model",
        str(args.d_model),
        "--dim_feedforward",
        str(args.dim_feedforward),
        "--lora_r",
        str(args.lora_r),
        "--lora_alpha",
        str(args.lora_alpha),
        "--lora_dropout",
        "0.0",
    ]
    if not args.lora_only:
        cmd.append("--train_full_model")
    return cmd


def build_generate_command(args: argparse.Namespace, checkpoint_dir: Path, conditioning_midi: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/generate.py",
        "--lora_path",
        str(checkpoint_dir),
        "--conditioning_midi",
        str(conditioning_midi),
        "--control_format",
        SEQUENCE_FORMAT_LEGACY_SEP,
        "--primer_max_tokens",
        str(args.primer_max_tokens),
        "--length",
        str(args.max_sequence),
        "--max_sequence",
        str(args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--output",
        str(output_dir),
    ]
    if args.top_k is not None:
        cmd.extend(["--top_k", str(args.top_k)])
    if args.top_p is not None:
        cmd.extend(["--top_p", str(args.top_p)])
    return cmd


def evaluate_raw_samples(sample_dir: Path, request: GenerationRequest) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for midi_path in sorted(sample_dir.glob("jazz_sample_*.mid")):
        metrics = compute_midi_metrics(
            midi_path,
            generation_time_ms=0,
            fallback_used=False,
            request=request,
        )
        valid, reason = validate_metrics(metrics, request.density, bars=request.bars)
        rows.append(
            {
                "midi_path": str(midi_path),
                "valid": bool(valid),
                "failure_reason": reason,
                "metrics": metrics.to_dict(),
            }
        )
    return rows


def write_markdown_report(report_path: Path, report: dict[str, Any]) -> None:
    inference_result = report.get("inference_result") or {}
    metrics = inference_result.get("metrics") or {}
    summary = report.get("summary") or {}
    fallback_used = inference_result.get("fallback_used")
    passed = inference_result.get("status") == "COMPLETED" and fallback_used is False
    decision = (
        "Current tokenization/training path can continue to the next controlled experiment."
        if passed
        else "Do not expand conditioning yet; inspect tiny-overfit failure before changing product scope."
    )

    lines = [
        "# Stage A Tiny Overfit Report",
        "",
        f"- Result: {'PASS' if passed else 'FAIL'}",
        f"- Training mode: `{summary.get('training_mode')}`",
        f"- Best validation loss: {summary.get('best_validation_loss')}",
        f"- Valid raw samples: {summary.get('valid_raw_sample_count')}/{summary.get('raw_sample_count')}",
        f"- Fallback used: {fallback_used}",
        f"- Checkpoint: `{report.get('checkpoint_path')}`",
        f"- Report JSON: `{report.get('report_json')}`",
        "",
        "## Inference Gate Metrics",
        "",
        f"- note_count: {metrics.get('note_count')}",
        f"- note_density: {metrics.get('note_density')}",
        f"- dead_air_ratio: {metrics.get('dead_air_ratio')}",
        f"- max_note_duration_ratio: {metrics.get('max_note_duration_ratio')}",
        f"- chord_tone_ratio: {metrics.get('chord_tone_ratio')}",
        f"- model_failure_reason: {inference_result.get('model_failure_reason')}",
        "",
        "## Decision",
        "",
        decision,
        "",
        "## Raw Model Samples",
        "",
    ]
    for row in report.get("raw_sample_metrics", []):
        row_metrics = row.get("metrics", {})
        lines.append(
            "- "
            f"`{row.get('midi_path')}` "
            f"valid={row.get('valid')} "
            f"notes={row_metrics.get('note_count')} "
            f"max_note_duration_ratio={row_metrics.get('max_note_duration_ratio')} "
            f"reason={row.get('failure_reason')}"
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage A tiny-overfit smoke test")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_a_tiny_overfit"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--sample_count", type=int, default=3)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_sequence", type=int, default=128)
    parser.add_argument("--primer_max_tokens", type=int, default=24)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=1)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_only", action="store_true", help="Keep legacy random-base LoRA-only behavior")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    raw_samples_dir = run_dir / "raw_samples"
    inference_output_dir = run_dir / "generated"

    manifest = prepare_tiny_dataset(run_dir, sample_count=args.sample_count, bpm=args.bpm)
    if args.prepare_only:
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, ensure_ascii=True, indent=2))
        return 0

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "training_mode": "lora_only" if args.lora_only else "full_model_tiny",
        "train_full_model": not args.lora_only,
    }

    if not args.skip_train:
        train_result = run_command(
            build_train_command(args, Path(manifest["data_dir"]), checkpoint_dir),
            ROOT_DIR,
        )
        report["train_result"] = train_result
        if train_result["returncode"] != 0:
            write_json(run_dir / "report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(train_result["returncode"])

    checkpoint_path = resolve_full_checkpoint_path(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint_epoch*.pt found under {checkpoint_dir}")
    report["checkpoint_path"] = str(checkpoint_path)

    conditioning_midi = Path(manifest["samples"][0]["midi_path"])
    request = GenerationRequest(
        bpm=int(args.bpm),
        chord_progression=DEFAULT_CHORDS,
        bars=2,
        key="C minor",
        section="drop",
        energy="mid",
        density="medium",
        style="tiny_overfit",
        temperature=float(args.temperature),
        top_k=args.top_k,
        top_p=args.top_p,
        job_id=f"tiny_overfit_gate_s{args.seed}",
        seed=int(args.seed),
    )

    if not args.skip_generation:
        raw_generate_result = run_command(
            build_generate_command(args, checkpoint_dir, conditioning_midi, raw_samples_dir),
            ROOT_DIR,
        )
        report["raw_generate_result"] = raw_generate_result
        if raw_generate_result["returncode"] != 0:
            write_json(run_dir / "report.json", report)
            print(json.dumps(report, ensure_ascii=True, indent=2))
            return int(raw_generate_result["returncode"])

        report["raw_sample_metrics"] = evaluate_raw_samples(raw_samples_dir, request)
        inference_result = generate_midi_phrase(
            request=request,
            output_dir=inference_output_dir,
            use_model=True,
            lora_path=checkpoint_dir,
            conditioning_midi=conditioning_midi,
            primer_max_tokens=args.primer_max_tokens,
            max_sequence=args.max_sequence,
            model_candidates=args.num_samples,
            control_format=SEQUENCE_FORMAT_LEGACY_SEP,
        )
        report["inference_result"] = inference_result.to_dict()

    report_json = run_dir / "report.json"
    report_md = run_dir / "report.md"
    report["report_json"] = str(report_json)
    report["report_md"] = str(report_md)
    report["summary"] = summarize_report(report)
    write_json(report_json, report)
    write_markdown_report(report_md, report)
    print(json.dumps(report, ensure_ascii=True, indent=2))

    inference_result = report.get("inference_result") or {}
    passed = inference_result.get("status") == "COMPLETED" and inference_result.get("fallback_used") is False
    return 0 if args.skip_generation or passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
