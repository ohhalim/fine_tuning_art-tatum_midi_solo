"""
Run a focused control_v1 Stage A tiny-overfit smoke test.

This keeps the existing legacy tiny-compare path intact while proving that the
new explicit control-token prompt can be trained and used for generation.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pretty_midi


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from scripts.control_tokens import SEQUENCE_FORMAT_CONTROL_V1, build_control_sequence, control_prefix_tokens, token_names  # noqa: E402
from scripts.generate import encode_midi_simple, resolve_full_checkpoint_path  # noqa: E402
from inference.app.generator import generate_midi_phrase  # noqa: E402
from inference.app.schemas import GenerationRequest  # noqa: E402
from scripts.run_stage_a_tiny_overfit import (  # noqa: E402
    DEFAULT_CHORDS,
    build_train_command,
    evaluate_raw_samples,
    run_command,
    summarize_report,
    write_json,
    write_known_good_midi,
    write_markdown_report,
)


def write_tiny_conditioning_midi(path: Path, bpm: int = 124) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    beat_sec = 60.0 / float(bpm)
    chord_duration = beat_sec * 2.0
    chord_voicings = [
        [48, 51, 55, 58],
        [53, 56, 60, 63],
        [46, 50, 53, 57],
        [51, 55, 58, 62],
    ]

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="control_v1_conditioning")
    for chord_index, pitches in enumerate(chord_voicings):
        start = chord_index * chord_duration
        end = start + (chord_duration * 0.85)
        for pitch in pitches:
            piano.notes.append(
                pretty_midi.Note(
                    velocity=58,
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                )
            )
    pm.instruments.append(piano)
    pm.write(str(path))
    return path


def prepare_control_v1_tiny_dataset(
    run_dir: Path,
    sample_count: int = 1,
    bpm: int = 124,
) -> dict[str, Any]:
    sample_count = max(1, min(3, int(sample_count)))
    midi_dir = run_dir / "input_midi"
    data_dir = run_dir / "tokenized"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    conditioning_path = write_tiny_conditioning_midi(midi_dir / "control_conditioning.mid", bpm=bpm)
    conditioning_tokens = encode_midi_simple(str(conditioning_path))
    if not conditioning_tokens:
        raise RuntimeError(f"failed to tokenize control conditioning MIDI: {conditioning_path}")

    samples: list[dict[str, Any]] = []
    for index in range(sample_count):
        target_path = write_known_good_midi(midi_dir / f"tiny_solo_{index + 1}.mid", index, bpm=bpm)
        target_tokens = encode_midi_simple(str(target_path))
        if not target_tokens:
            raise RuntimeError(f"failed to tokenize tiny target MIDI: {target_path}")

        tokens = build_control_sequence(
            conditioning_tokens,
            target_tokens,
            role="lead",
            tempo_bpm=bpm,
        )
        token_array = np.array(tokens, dtype=np.int32)

        train_path = train_dir / f"{index:05d}.npy"
        val_path = val_dir / f"{index:05d}.npy"
        np.save(train_path, token_array)
        np.save(val_path, token_array)
        samples.append(
            {
                "conditioning_path": str(conditioning_path),
                "target_midi_path": str(target_path),
                "train_tokens_path": str(train_path),
                "val_tokens_path": str(val_path),
                "conditioning_token_count": int(len(conditioning_tokens)),
                "target_token_count": int(len(target_tokens)),
                "token_count": int(len(tokens)),
            }
        )

    prefix = control_prefix_tokens(role="lead", tempo_bpm=bpm)
    manifest = {
        "bpm": int(bpm),
        "chord_progression": DEFAULT_CHORDS,
        "sequence_format": SEQUENCE_FORMAT_CONTROL_V1,
        "control_tokens": prefix,
        "control_token_names": token_names(prefix),
        "sample_count": sample_count,
        "data_dir": str(data_dir),
        "conditioning_midi": str(conditioning_path),
        "samples": samples,
    }
    write_json(run_dir / "control_v1_tiny_dataset_manifest.json", manifest)
    return manifest


def build_generate_command(args: argparse.Namespace, checkpoint_dir: Path, conditioning_midi: Path, output_dir: Path) -> list[str]:
    cmd = [
        sys.executable,
        "scripts/generate.py",
        "--lora_path",
        str(checkpoint_dir),
        "--conditioning_midi",
        str(conditioning_midi),
        "--control_format",
        SEQUENCE_FORMAT_CONTROL_V1,
        "--role",
        "lead",
        "--tempo_bpm",
        str(args.bpm),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage A control_v1 tiny-overfit smoke test")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_a_control_v1_tiny_overfit"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default=None)
    parser.add_argument("--sample_count", type=int, default=1)
    parser.add_argument("--bpm", type=int, default=124)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_sequence", type=int, default=192)
    parser.add_argument("--primer_max_tokens", type=int, default=96)
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
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.set_defaults(lora_only=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    run_dir = output_root / run_id
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else run_dir / "checkpoints"
    raw_samples_dir = run_dir / "raw_samples"
    inference_output_dir = run_dir / "generated"

    manifest = prepare_control_v1_tiny_dataset(run_dir, sample_count=args.sample_count, bpm=args.bpm)
    if args.prepare_only:
        print(json.dumps({"run_dir": str(run_dir), "manifest": manifest}, ensure_ascii=True, indent=2))
        return 0

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "manifest": manifest,
        "training_mode": "control_v1_full_model_tiny",
        "sequence_format": SEQUENCE_FORMAT_CONTROL_V1,
        "train_full_model": True,
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

    conditioning_midi = Path(manifest["conditioning_midi"])
    request = GenerationRequest(
        bpm=int(args.bpm),
        chord_progression=DEFAULT_CHORDS,
        bars=2,
        key="C minor",
        section="drop",
        energy="mid",
        density="medium",
        style="control_v1_tiny_overfit",
        temperature=float(args.temperature),
        top_k=args.top_k,
        top_p=args.top_p,
        job_id=f"control_v1_tiny_gate_s{args.seed}",
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
            control_format=SEQUENCE_FORMAT_CONTROL_V1,
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
