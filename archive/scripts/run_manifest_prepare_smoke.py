"""
Run a small audit -> manifest -> role-dataset preparation smoke test.

This is a local contract check before broad training. It verifies that the
manifest split builder and prepare_role_dataset.py agree on train/val inputs
and produce tokenized control_v1 records.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def read_manifest_paths(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def write_limited_manifest(source: Path, target: Path, limit: int) -> int:
    paths = read_manifest_paths(source)
    selected = paths[: max(0, int(limit))]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(f"{path}\n" for path in selected), encoding="utf-8")
    return len(selected)


def count_npy_files(path: Path) -> int:
    return len(sorted(path.glob("*.npy")))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run manifest-based role dataset preparation smoke")
    parser.add_argument("--output_root", type=str, default="./outputs/manifest_prepare_smoke")
    parser.add_argument("--run_id", type=str, default="default")
    parser.add_argument("--audit_max_files", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_files", type=int, default=4)
    parser.add_argument("--val_files", type=int, default=2)
    parser.add_argument("--sequence_format", type=str, default="control_v1")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_dir = Path(args.output_root) / args.run_id
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Run directory already exists: {run_dir}. Use --overwrite.")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = run_dir / "audit"
    manifests_dir = run_dir / "manifests"
    smoke_manifests_dir = run_dir / "smoke_manifests"
    roles_dir = run_dir / "roles"

    audit_json = audit_dir / "jazz_piano_dataset_audit.json"
    audit_md = audit_dir / "jazz_piano_dataset_audit.md"
    run_command(
        [
            sys.executable,
            "scripts/audit_jazz_piano_dataset.py",
            "--max_files",
            str(args.audit_max_files),
            "--output_json",
            str(audit_json),
            "--output_md",
            str(audit_md),
        ]
    )

    run_command(
        [
            sys.executable,
            "scripts/build_jazz_training_manifests.py",
            "--audit_json",
            str(audit_json),
            "--output_dir",
            str(manifests_dir),
            "--seed",
            str(args.seed),
        ]
    )

    train_manifest = smoke_manifests_dir / "generic_jazz_train.txt"
    val_manifest = smoke_manifests_dir / "generic_jazz_val.txt"
    train_selected = write_limited_manifest(
        manifests_dir / "generic_jazz_train.txt",
        train_manifest,
        args.train_files,
    )
    val_selected = write_limited_manifest(
        manifests_dir / "generic_jazz_val.txt",
        val_manifest,
        args.val_files,
    )
    if train_selected <= 0:
        raise ValueError("Smoke train manifest is empty")
    if val_selected <= 0:
        raise ValueError("Smoke val manifest is empty")

    run_command(
        [
            sys.executable,
            "scripts/prepare_role_dataset.py",
            "--train_manifest",
            str(train_manifest),
            "--val_manifest",
            str(val_manifest),
            "--output_dir",
            str(roles_dir),
            "--role",
            "lead",
            "--sequence_format",
            str(args.sequence_format),
            "--overwrite",
        ]
    )

    role_root = roles_dir / "lead"
    dataset_summary_path = role_root / "dataset_summary.json"
    dataset_summary = json.loads(dataset_summary_path.read_text(encoding="utf-8"))
    tokenized_train = count_npy_files(role_root / "tokenized" / "train")
    tokenized_val = count_npy_files(role_root / "tokenized" / "val")
    if tokenized_train <= 0:
        raise ValueError("No tokenized train records produced")
    if tokenized_val <= 0:
        raise ValueError("No tokenized val records produced")

    summary = {
        "run_dir": str(run_dir),
        "audit_json": str(audit_json),
        "manifests_dir": str(manifests_dir),
        "smoke_train_manifest": str(train_manifest),
        "smoke_val_manifest": str(val_manifest),
        "roles_dir": str(roles_dir),
        "audit_max_files": int(args.audit_max_files),
        "selected_train_files": int(train_selected),
        "selected_val_files": int(val_selected),
        "tokenized_train": int(tokenized_train),
        "tokenized_val": int(tokenized_val),
        "dataset_summary": dataset_summary,
    }
    summary_path = run_dir / "manifest_prepare_smoke_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
