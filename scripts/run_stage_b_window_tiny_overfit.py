"""
Stage B windowed tiny-overfit smoke.

This script prepares short stage_b_v1 phrase windows, verifies that their token
ids fit the model vocabulary, and optionally runs the existing full-model tiny
training path against those records.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "scripts"))
sys.path.insert(0, str(ROOT_DIR / "music_transformer"))

from scripts.run_stage_a_tiny_overfit import build_train_command, run_command, write_json  # noqa: E402
from utilities.constants import VOCAB_SIZE  # noqa: E402


def run_prepare_command(args: argparse.Namespace, roles_dir: Path) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "scripts/prepare_role_dataset.py",
        "--input_dir",
        str(args.input_dir),
        "--output_dir",
        str(roles_dir),
        "--role",
        "lead",
        "--sequence_format",
        "stage_b_v1",
        "--stage_b_window_bars",
        str(args.window_bars),
        "--stage_b_window_stride_bars",
        str(args.window_stride_bars),
        "--stage_b_min_window_target_notes",
        str(args.min_window_target_notes),
        "--overwrite",
    ]
    if args.max_files is not None:
        cmd.extend(["--max_files", str(args.max_files)])
    return run_command(cmd, ROOT_DIR)


def token_stats(tokenized_dir: Path) -> dict[str, Any]:
    lengths: list[int] = []
    max_token_id = -1
    files = sorted(tokenized_dir.glob("*/*.npy"))
    non_empty_files = 0
    for path in files:
        tokens = np.load(path)
        if len(tokens) == 0:
            continue
        non_empty_files += 1
        lengths.append(int(len(tokens)))
        max_token_id = max(max_token_id, int(tokens.max()))

    sorted_lengths = sorted(lengths)
    p50 = sorted_lengths[len(sorted_lengths) // 2] if sorted_lengths else 0
    return {
        "files": len(files),
        "min_length": min(lengths) if lengths else 0,
        "p50_length": int(p50),
        "max_length": max(lengths) if lengths else 0,
        "mean_length": float(sum(lengths) / len(lengths)) if lengths else 0.0,
        "max_token_id": int(max_token_id),
        "vocab_size": int(VOCAB_SIZE),
        "non_empty_files": int(non_empty_files),
        "has_tokenized_records": bool(non_empty_files > 0),
        "fits_vocab": bool(non_empty_files > 0 and max_token_id < VOCAB_SIZE),
    }


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B windowed tiny-overfit smoke")
    parser.add_argument("--input_dir", type=str, default="./midi_dataset/midi/studio/Brad Mehldau")
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_window_tiny_overfit"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--max_files", type=int, default=2)
    parser.add_argument("--window_bars", type=int, default=2)
    parser.add_argument("--window_stride_bars", type=int, default=2)
    parser.add_argument("--min_window_target_notes", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--max_sequence", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--dim_feedforward", type=int, default=256)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--skip_train", action="store_true")
    parser.set_defaults(lora_only=False)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    roles_dir = run_dir / "roles"
    role_root = roles_dir / "lead"
    tokenized_dir = role_root / "tokenized"
    checkpoint_dir = run_dir / "checkpoints"

    report: dict[str, Any] = {
        "run_id": run_id,
        "run_dir": str(run_dir),
        "sequence_format": "stage_b_v1",
        "training_mode": "stage_b_window_full_model_tiny",
        "train_full_model": True,
    }

    prepare_result = run_prepare_command(args, roles_dir)
    report["prepare_result"] = prepare_result
    if prepare_result["returncode"] != 0:
        write_json(run_dir / "report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return int(prepare_result["returncode"])

    summary_path = role_root / "dataset_summary.json"
    report["dataset_summary"] = read_json(summary_path)
    report["token_stats"] = token_stats(tokenized_dir)
    if not report["token_stats"]["has_tokenized_records"]:
        report["failure_reason"] = "No Stage B tokenized records produced"
        write_json(run_dir / "report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2
    if not report["token_stats"]["fits_vocab"]:
        report["failure_reason"] = "Stage B token id exceeds model VOCAB_SIZE"
        write_json(run_dir / "report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 2

    if args.prepare_only or args.skip_train:
        write_json(run_dir / "report.json", report)
        print(json.dumps(report, ensure_ascii=True, indent=2))
        return 0

    train_args = argparse.Namespace(**vars(args))
    train_result = run_command(build_train_command(train_args, tokenized_dir, checkpoint_dir), ROOT_DIR)
    report["train_result"] = train_result
    report["checkpoint_dir"] = str(checkpoint_dir)
    write_json(run_dir / "report.json", report)
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return int(train_result["returncode"])


if __name__ == "__main__":
    raise SystemExit(main())
