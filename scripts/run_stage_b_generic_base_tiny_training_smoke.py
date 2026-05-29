"""Run a tiny Stage B generic-base training smoke from prepared window records."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "music_transformer"))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text
from utilities.constants import VOCAB_SIZE


class StageBGenericBaseTinyTrainingSmokeError(ValueError):
    pass


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT_DIR),
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "cmd": list(command),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def parse_best_validation_loss(text: str) -> float | None:
    matches = re.findall(r"Best validation loss:\s*([0-9]+(?:\.[0-9]+)?)", text)
    return float(matches[-1]) if matches else None


def copy_limited_records(source_split_dir: Path, target_split_dir: Path, limit: int) -> int:
    files = sorted(source_split_dir.glob("*.npy"))[: max(0, int(limit))]
    target_split_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files):
        shutil.copy2(source, target_split_dir / f"{index:05d}.npy")
    return len(files)


def token_subset_stats(tokenized_dir: Path) -> dict[str, Any]:
    files = sorted(tokenized_dir.glob("*/*.npy"))
    lengths: list[int] = []
    max_token_id = -1
    for path in files:
        tokens = np.load(path)
        if len(tokens) <= 0:
            continue
        lengths.append(int(len(tokens)))
        max_token_id = max(max_token_id, int(tokens.max()))
    return {
        "files": len(files),
        "non_empty_files": len(lengths),
        "max_token_id": int(max_token_id),
        "vocab_size": int(VOCAB_SIZE),
        "fits_vocab": bool(lengths and max_token_id < VOCAB_SIZE),
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
    }


def build_train_command(args: argparse.Namespace, tokenized_dir: Path, checkpoint_dir: Path) -> list[str]:
    return [
        sys.executable,
        "scripts/train_qlora.py",
        "--data_dir",
        str(tokenized_dir),
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
        "--train_full_model",
    ]


def build_training_smoke_report(
    *,
    run_dir: Path,
    source_tokenized_dir: Path,
    tokenized_dir: Path,
    checkpoint_dir: Path,
    selected_train_records: int,
    selected_val_records: int,
    train_result: dict[str, Any],
    token_stats: dict[str, Any],
) -> dict[str, Any]:
    best_validation_loss = parse_best_validation_loss(
        str(train_result.get("stdout_tail") or "") + "\n" + str(train_result.get("stderr_tail") or "")
    )
    training_smoke_passed = (
        selected_train_records > 0
        and selected_val_records > 0
        and bool(token_stats.get("fits_vocab", False))
        and int(train_result.get("returncode", 1)) == 0
        and best_validation_loss is not None
    )
    boundary = "stage_b_generic_base_tiny_training_smoke"
    next_boundary = "stage_b_generic_tiny_checkpoint_generation_probe"
    return {
        "schema_version": "stage_b_generic_base_tiny_training_smoke_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "source_tokenized_dir": str(source_tokenized_dir),
        "tokenized_dir": str(tokenized_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "input": {
            "selected_train_records": int(selected_train_records),
            "selected_val_records": int(selected_val_records),
        },
        "token_stats": token_stats,
        "training": {
            "returncode": int(train_result.get("returncode", 1)),
            "best_validation_loss": best_validation_loss,
            "stdout_tail": str(train_result.get("stdout_tail") or ""),
            "stderr_tail": str(train_result.get("stderr_tail") or ""),
        },
        "readiness": {
            "boundary": boundary,
            "tiny_training_smoke_passed": training_smoke_passed,
            "generic_base_training_path_smoked": training_smoke_passed,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": boundary,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "generic Stage B window records can enter the tiny training path; "
                "this is a training contract smoke, not broad model quality evidence"
            ),
        },
        "not_proven": [
            "generic_base_generation_quality",
            "generic_base_multi_seed_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic tiny checkpoint generation probe",
    }


def validate_training_smoke_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_training_smoke_passed: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    training = _dict(report.get("training"))
    token = _dict(report.get("token_stats"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseTinyTrainingSmokeError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseTinyTrainingSmokeError(f"expected next boundary {expected_next_boundary}, got {next_boundary}")
    if require_training_smoke_passed and not bool(readiness.get("tiny_training_smoke_passed", False)):
        raise StageBGenericBaseTinyTrainingSmokeError("tiny training smoke should pass")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseTinyTrainingSmokeError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseTinyTrainingSmokeError("Brad style adaptation must not be claimed")
    if not bool(token.get("fits_vocab", False)):
        raise StageBGenericBaseTinyTrainingSmokeError("token ids must fit vocab")
    if _int(training.get("returncode")) != 0:
        raise StageBGenericBaseTinyTrainingSmokeError("training command must succeed")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "tiny_training_smoke_passed": bool(readiness.get("tiny_training_smoke_passed", False)),
        "generic_base_training_path_smoked": bool(readiness.get("generic_base_training_path_smoked", False)),
        "best_validation_loss": training.get("best_validation_loss"),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "broad_training_execution_ready": bool(readiness.get("broad_training_execution_ready", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    inputs = report["input"]
    token = report["token_stats"]
    training = report["training"]
    lines = [
        "# Stage B Generic Base Tiny Training Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- tiny training smoke passed: `{_bool_token(readiness['tiny_training_smoke_passed'])}`",
        f"- broad training execution ready: `{_bool_token(readiness['broad_training_execution_ready'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Input",
        "",
        f"- selected train records: `{inputs['selected_train_records']}`",
        f"- selected val records: `{inputs['selected_val_records']}`",
        f"- token files: `{token['files']}`",
        f"- max token id: `{token['max_token_id']}`",
        f"- vocab size: `{token['vocab_size']}`",
        f"- fits vocab: `{_bool_token(token['fits_vocab'])}`",
        "",
        "## Training",
        "",
        f"- returncode: `{training['returncode']}`",
        f"- best validation loss: `{training['best_validation_loss']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic base tiny training smoke")
    parser.add_argument(
        "--source_tokenized_dir",
        type=str,
        default="outputs/stage_b_generic_manifest_window_smoke/"
        "harness_stage_b_generic_manifest_window_smoke/roles/lead/tokenized",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_tiny_training_smoke")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--train_records", type=int, default=32)
    parser.add_argument("--val_records", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_training_smoke_passed", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    source_tokenized_dir = Path(args.source_tokenized_dir)
    tokenized_dir = run_dir / "tokenized"
    selected_train = copy_limited_records(source_tokenized_dir / "train", tokenized_dir / "train", args.train_records)
    selected_val = copy_limited_records(source_tokenized_dir / "val", tokenized_dir / "val", args.val_records)
    if selected_train <= 0:
        raise StageBGenericBaseTinyTrainingSmokeError("selected train token records required")
    if selected_val <= 0:
        raise StageBGenericBaseTinyTrainingSmokeError("selected val token records required")
    stats = token_subset_stats(tokenized_dir)
    checkpoint_dir = run_dir / "checkpoints"
    train_result = run_command(build_train_command(args, tokenized_dir, checkpoint_dir))
    report = build_training_smoke_report(
        run_dir=run_dir,
        source_tokenized_dir=source_tokenized_dir,
        tokenized_dir=tokenized_dir,
        checkpoint_dir=checkpoint_dir,
        selected_train_records=selected_train,
        selected_val_records=selected_val,
        train_result=train_result,
        token_stats=stats,
    )
    summary = validate_training_smoke_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_training_smoke_passed=bool(args.require_training_smoke_passed),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_base_tiny_training_smoke.json", report)
    write_json(run_dir / "stage_b_generic_base_tiny_training_smoke_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_base_tiny_training_smoke.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
