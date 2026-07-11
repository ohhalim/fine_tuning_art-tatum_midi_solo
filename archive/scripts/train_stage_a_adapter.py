"""
Explicit Stage A adapter-training entrypoint.

This wrapper requires a checkpoint so adapter training cannot silently start
from a random base model.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def build_train_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/train_qlora.py",
        "--data_dir",
        str(args.data_dir),
        "--output_dir",
        str(args.output_dir),
        "--checkpoint",
        str(args.checkpoint),
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--lr",
        str(args.lr),
        "--gradient_accumulation",
        str(args.gradient_accumulation),
        "--num_workers",
        str(args.num_workers),
        "--label_smoothing",
        str(args.label_smoothing),
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
        str(args.lora_dropout),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Stage A adapter from a pretrained/full base checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data/roles/lead/tokenized")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/stage_a_adapter")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cmd = build_train_command(args)
    print("Running:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT_DIR), check=False)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
