"""Run a controlled Stage B generic-base training scale smoke."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / "music_transformer"))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_base_tiny_training_smoke import (  # noqa: E402
    build_train_command,
    copy_limited_records,
    parse_best_validation_loss,
    run_command,
    token_subset_stats,
)


class StageBGenericBaseTrainingScaleSmokeError(ValueError):
    pass


FULL_WINDOW_BOUNDARY = "stage_b_generic_full_manifest_window_preparation"
BOUNDARY = "stage_b_generic_base_training_scale_smoke"
NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_generation_probe"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_full_window_preparation(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    token = _dict(report.get("token_stats"))
    inputs = _dict(report.get("input"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if boundary != FULL_WINDOW_BOUNDARY:
        raise StageBGenericBaseTrainingScaleSmokeError("full window preparation boundary required")
    if next_boundary != BOUNDARY:
        raise StageBGenericBaseTrainingScaleSmokeError("full window preparation must route to scale smoke")
    if not bool(readiness.get("generic_base_training_scale_smoke_ready", False)):
        raise StageBGenericBaseTrainingScaleSmokeError("scale smoke readiness required")
    if not bool(token.get("fits_vocab", False)):
        raise StageBGenericBaseTrainingScaleSmokeError("full window token ids must fit vocab")
    if _int(token.get("tokenized_train_files")) <= 0:
        raise StageBGenericBaseTrainingScaleSmokeError("full window train token records required")
    if _int(token.get("tokenized_val_files")) <= 0:
        raise StageBGenericBaseTrainingScaleSmokeError("full window val token records required")
    if bool(readiness.get("full_training_executed", True)):
        raise StageBGenericBaseTrainingScaleSmokeError("full training must not be claimed before scale smoke")
    if bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseTrainingScaleSmokeError("broad trained-model quality must not be claimed")
    roles_dir = Path(str(report.get("roles_dir") or ""))
    source_tokenized_dir = roles_dir / "lead" / "tokenized"
    if not source_tokenized_dir.exists():
        raise StageBGenericBaseTrainingScaleSmokeError(f"tokenized source missing: {source_tokenized_dir}")
    return {
        "train_file_count": _int(inputs.get("train_file_count")),
        "val_file_count": _int(inputs.get("val_file_count")),
        "source_tokenized_dir": str(source_tokenized_dir),
        "source_tokenized_train_files": _int(token.get("tokenized_train_files")),
        "source_tokenized_val_files": _int(token.get("tokenized_val_files")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
    }


def build_training_scale_smoke_report(
    *,
    args: argparse.Namespace,
    run_dir: Path,
    source_report: dict[str, Any],
    source_window_summary: dict[str, Any],
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
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_epoch*.pt"))
    lora_weights = checkpoint_dir / "lora_weights.pt"
    training_scale_smoke_passed = (
        selected_train_records >= int(args.min_train_records)
        and selected_val_records >= int(args.min_val_records)
        and selected_train_records > 32
        and selected_val_records > 8
        and bool(token_stats.get("fits_vocab", False))
        and int(train_result.get("returncode", 1)) == 0
        and best_validation_loss is not None
        and bool(checkpoint_files)
        and lora_weights.exists()
    )
    return {
        "schema_version": "stage_b_generic_base_training_scale_smoke_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "source_full_window_schema": str(source_report.get("schema_version") or ""),
        "source_tokenized_dir": str(source_tokenized_dir),
        "tokenized_dir": str(tokenized_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "source_window_summary": source_window_summary,
        "input": {
            "selected_train_records": int(selected_train_records),
            "selected_val_records": int(selected_val_records),
            "requested_train_records": int(args.train_records),
            "requested_val_records": int(args.val_records),
            "min_train_records": int(args.min_train_records),
            "min_val_records": int(args.min_val_records),
        },
        "training_config": {
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "seed": int(args.seed),
            "max_sequence": int(args.max_sequence),
            "n_layers": int(args.n_layers),
            "num_heads": int(args.num_heads),
            "d_model": int(args.d_model),
            "dim_feedforward": int(args.dim_feedforward),
            "train_full_model": True,
        },
        "token_stats": token_stats,
        "training": {
            "returncode": int(train_result.get("returncode", 1)),
            "best_validation_loss": best_validation_loss,
            "stdout_tail": str(train_result.get("stdout_tail") or ""),
            "stderr_tail": str(train_result.get("stderr_tail") or ""),
        },
        "artifacts": {
            "checkpoint_count": len(checkpoint_files),
            "checkpoint_files": [str(path) for path in checkpoint_files],
            "lora_weights": str(lora_weights),
            "lora_weights_exists": lora_weights.exists(),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "training_scale_smoke_passed": training_scale_smoke_passed,
            "generic_base_training_scale_smoked": training_scale_smoke_passed,
            "generic_base_scale_checkpoint_generation_probe_ready": training_scale_smoke_passed,
            "scale_training_smoke_executed": True,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "full generic Stage B window output can enter a larger local training subset; "
                "next step is checkpoint generation evidence, not quality claim"
            ),
        },
        "proven": [
            "full_window_output_reused_as_training_source",
            "larger_than_tiny_training_subset_executed",
            "checkpoint_artifacts_created",
            "validation_loss_recorded",
        ],
        "not_proven": [
            "full_generic_training_run",
            "generic_base_generation_quality",
            "generic_base_multi_seed_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base scale checkpoint generation probe",
    }


def validate_training_scale_smoke_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_training_scale_smoke_passed: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
    min_train_records: int,
    min_val_records: int,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    training = _dict(report.get("training"))
    token = _dict(report.get("token_stats"))
    inputs = _dict(report.get("input"))
    artifacts = _dict(report.get("artifacts"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseTrainingScaleSmokeError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseTrainingScaleSmokeError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_training_scale_smoke_passed and not bool(readiness.get("training_scale_smoke_passed", False)):
        raise StageBGenericBaseTrainingScaleSmokeError("training scale smoke should pass")
    if _int(inputs.get("selected_train_records")) < int(min_train_records):
        raise StageBGenericBaseTrainingScaleSmokeError("selected train records below threshold")
    if _int(inputs.get("selected_val_records")) < int(min_val_records):
        raise StageBGenericBaseTrainingScaleSmokeError("selected val records below threshold")
    if not bool(token.get("fits_vocab", False)):
        raise StageBGenericBaseTrainingScaleSmokeError("token ids must fit vocab")
    if _int(training.get("returncode")) != 0:
        raise StageBGenericBaseTrainingScaleSmokeError("training command must succeed")
    if training.get("best_validation_loss") is None:
        raise StageBGenericBaseTrainingScaleSmokeError("best validation loss must be recorded")
    if _int(artifacts.get("checkpoint_count")) <= 0:
        raise StageBGenericBaseTrainingScaleSmokeError("checkpoint artifact required")
    if not bool(artifacts.get("lora_weights_exists", False)):
        raise StageBGenericBaseTrainingScaleSmokeError("lora weights artifact required")
    if bool(readiness.get("full_generic_training_executed", True)):
        raise StageBGenericBaseTrainingScaleSmokeError("full generic training must not be claimed")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseTrainingScaleSmokeError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseTrainingScaleSmokeError("Brad style adaptation must not be claimed")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "training_scale_smoke_passed": bool(readiness.get("training_scale_smoke_passed", False)),
        "generic_base_training_scale_smoked": bool(
            readiness.get("generic_base_training_scale_smoked", False)
        ),
        "generic_base_scale_checkpoint_generation_probe_ready": bool(
            readiness.get("generic_base_scale_checkpoint_generation_probe_ready", False)
        ),
        "selected_train_records": _int(inputs.get("selected_train_records")),
        "selected_val_records": _int(inputs.get("selected_val_records")),
        "source_tokenized_train_files": _int(
            _dict(report.get("source_window_summary")).get("source_tokenized_train_files")
        ),
        "source_tokenized_val_files": _int(
            _dict(report.get("source_window_summary")).get("source_tokenized_val_files")
        ),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "training_returncode": _int(training.get("returncode")),
        "best_validation_loss": training.get("best_validation_loss"),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
        "full_generic_training_executed": bool(readiness.get("full_generic_training_executed", True)),
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
    source = report["source_window_summary"]
    inputs = report["input"]
    token = report["token_stats"]
    training = report["training"]
    artifacts = report["artifacts"]
    lines = [
        "# Stage B Generic Base Training Scale Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- training scale smoke passed: `{_bool_token(readiness['training_scale_smoke_passed'])}`",
        f"- scale training smoke executed: `{_bool_token(readiness['scale_training_smoke_executed'])}`",
        f"- full generic training executed: `{_bool_token(readiness['full_generic_training_executed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Source Window",
        "",
        f"- train / val manifest files: `{source['train_file_count']}` / `{source['val_file_count']}`",
        (
            "- source tokenized train / val records: "
            f"`{source['source_tokenized_train_files']}` / `{source['source_tokenized_val_files']}`"
        ),
        f"- source max token id / vocab size: `{source['max_token_id']}` / `{source['vocab_size']}`",
        "",
        "## Input",
        "",
        f"- selected train / val records: `{inputs['selected_train_records']}` / `{inputs['selected_val_records']}`",
        f"- min train / val records: `{inputs['min_train_records']}` / `{inputs['min_val_records']}`",
        f"- token files: `{token['files']}`",
        f"- max token id / vocab size: `{token['max_token_id']}` / `{token['vocab_size']}`",
        f"- fits vocab: `{_bool_token(token['fits_vocab'])}`",
        "",
        "## Training",
        "",
        f"- returncode: `{training['returncode']}`",
        f"- best validation loss: `{training['best_validation_loss']}`",
        f"- checkpoint count: `{artifacts['checkpoint_count']}`",
        f"- lora weights exists: `{_bool_token(artifacts['lora_weights_exists'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic base training scale smoke")
    parser.add_argument(
        "--full_window_preparation",
        type=str,
        default="outputs/stage_b_generic_full_manifest_window_preparation/"
        "harness_stage_b_generic_full_manifest_window_preparation/"
        "stage_b_generic_full_manifest_window_preparation.json",
    )
    parser.add_argument("--source_tokenized_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_training_scale_smoke")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--train_records", type=int, default=128)
    parser.add_argument("--val_records", type=int, default=32)
    parser.add_argument("--min_train_records", type=int, default=64)
    parser.add_argument("--min_val_records", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--dim_feedforward", type=int, default=128)
    parser.add_argument("--lora_r", type=int, default=4)
    parser.add_argument("--lora_alpha", type=int, default=8)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_training_scale_smoke_passed", action="store_true")
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

    full_window_report = read_json(Path(args.full_window_preparation))
    source_window_summary = validate_full_window_preparation(full_window_report)
    source_tokenized_dir = Path(args.source_tokenized_dir or source_window_summary["source_tokenized_dir"])
    tokenized_dir = run_dir / "tokenized"
    selected_train = copy_limited_records(source_tokenized_dir / "train", tokenized_dir / "train", args.train_records)
    selected_val = copy_limited_records(source_tokenized_dir / "val", tokenized_dir / "val", args.val_records)
    if selected_train <= 32:
        raise StageBGenericBaseTrainingScaleSmokeError("selected train records must exceed tiny smoke size")
    if selected_val <= 8:
        raise StageBGenericBaseTrainingScaleSmokeError("selected val records must exceed tiny smoke size")
    stats = token_subset_stats(tokenized_dir)
    checkpoint_dir = run_dir / "checkpoints"
    train_result = run_command(build_train_command(args, tokenized_dir, checkpoint_dir))
    report = build_training_scale_smoke_report(
        args=args,
        run_dir=run_dir,
        source_report=full_window_report,
        source_window_summary=source_window_summary,
        source_tokenized_dir=source_tokenized_dir,
        tokenized_dir=tokenized_dir,
        checkpoint_dir=checkpoint_dir,
        selected_train_records=selected_train,
        selected_val_records=selected_val,
        train_result=train_result,
        token_stats=stats,
    )
    summary = validate_training_scale_smoke_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_training_scale_smoke_passed=bool(args.require_training_scale_smoke_passed),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
        min_train_records=int(args.min_train_records),
        min_val_records=int(args.min_val_records),
    )
    write_json(run_dir / "stage_b_generic_base_training_scale_smoke.json", report)
    write_json(run_dir / "stage_b_generic_base_training_scale_smoke_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_base_training_scale_smoke.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
