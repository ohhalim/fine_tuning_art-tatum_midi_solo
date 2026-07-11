"""Prepare full generic manifests as Stage B duration-explicit windows."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.run_manifest_prepare_smoke import read_manifest_paths  # noqa: E402
from scripts.run_stage_b_window_tiny_overfit import token_stats  # noqa: E402


class StageBGenericFullManifestWindowPreparationError(ValueError):
    pass


PLAN_BOUNDARY = "stage_b_generic_model_core_training_data_plan"
BOUNDARY = "stage_b_generic_full_manifest_window_preparation"
NEXT_BOUNDARY = "stage_b_generic_base_training_scale_smoke"


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


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def count_npy_files(path: Path) -> int:
    return len(sorted(path.glob("*.npy")))


def validate_training_data_plan(report: dict[str, Any]) -> dict[str, Any]:
    claim = _dict(report.get("claim_boundary"))
    decision = _dict(report.get("decision"))
    plan = _dict(report.get("training_data_plan"))
    params = _dict(plan.get("window_parameters"))
    if str(claim.get("boundary") or "") != PLAN_BOUNDARY:
        raise StageBGenericFullManifestWindowPreparationError("training data plan boundary required")
    if not bool(claim.get("training_data_plan_ready", False)):
        raise StageBGenericFullManifestWindowPreparationError("training data plan must be ready")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericFullManifestWindowPreparationError("training data plan must route to full window preparation")
    if bool(claim.get("full_training_executed", True)):
        raise StageBGenericFullManifestWindowPreparationError("full training must not be executed before window prep")
    if bool(claim.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericFullManifestWindowPreparationError("broad quality claim must remain false")
    return {
        "generic_train_file_count": _int(plan.get("generic_train_file_count")),
        "generic_val_file_count": _int(plan.get("generic_val_file_count")),
        "window_bars": _int(params.get("window_bars")),
        "window_stride_bars": _int(params.get("window_stride_bars")),
        "min_window_target_notes": _int(params.get("min_window_target_notes")),
    }


def copy_manifest(src: Path, dst: Path) -> int:
    if not src.exists():
        raise StageBGenericFullManifestWindowPreparationError(f"manifest missing: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return len(read_manifest_paths(dst))


def build_full_window_report(
    *,
    run_dir: Path,
    training_data_plan: dict[str, Any],
    source_manifests_dir: Path,
    train_manifest: Path,
    val_manifest: Path,
    roles_dir: Path,
    train_file_count: int,
    val_file_count: int,
    window_bars: int,
    window_stride_bars: int,
    min_window_target_notes: int,
) -> dict[str, Any]:
    plan = validate_training_data_plan(training_data_plan)
    role_root = roles_dir / "lead"
    summary_path = role_root / "dataset_summary.json"
    dataset_summary = read_json(summary_path)
    tokenized_dir = role_root / "tokenized"
    stats = token_stats(tokenized_dir)
    tokenized_train = count_npy_files(tokenized_dir / "train")
    tokenized_val = count_npy_files(tokenized_dir / "val")
    split_counts = _dict(dataset_summary.get("input_split_file_counts"))
    ready = (
        train_file_count == plan["generic_train_file_count"]
        and val_file_count == plan["generic_val_file_count"]
        and _int(split_counts.get("train")) == train_file_count
        and _int(split_counts.get("val")) == val_file_count
        and tokenized_train > 0
        and tokenized_val > 0
        and bool(stats.get("fits_vocab", False))
        and str(dataset_summary.get("sequence_format") or "") == "stage_b_v1"
        and _int(dataset_summary.get("stage_b_window_bars")) == int(window_bars)
        and _int(dataset_summary.get("stage_b_window_stride_bars")) == int(window_stride_bars)
    )
    return {
        "schema_version": "stage_b_generic_full_manifest_window_preparation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "source_training_data_plan_schema": str(training_data_plan.get("schema_version") or ""),
        "source_manifests_dir": str(source_manifests_dir),
        "full_train_manifest": str(train_manifest),
        "full_val_manifest": str(val_manifest),
        "roles_dir": str(roles_dir),
        "input": {
            "train_file_count": int(train_file_count),
            "val_file_count": int(val_file_count),
            "window_bars": int(window_bars),
            "window_stride_bars": int(window_stride_bars),
            "min_window_target_notes": int(min_window_target_notes),
        },
        "dataset_summary": dataset_summary,
        "token_stats": {
            **stats,
            "tokenized_train_files": int(tokenized_train),
            "tokenized_val_files": int(tokenized_val),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "full_manifest_window_preparation_ready": ready,
            "generic_base_training_scale_smoke_ready": ready,
            "full_training_executed": False,
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
                "full non-Brad generic train/val manifests prepared as Stage B windows; "
                "next step is controlled training scale smoke, not broad quality claim"
            ),
        },
        "proven": [
            "full_generic_manifest_train_val_boundary_preserved",
            "stage_b_full_window_token_records_created",
            "full_window_token_vocab_guard_passed",
        ],
        "not_proven": [
            "generic_base_training_scale_smoke",
            "full_generic_training_run",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base training scale smoke",
    }


def validate_full_window_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_ready: bool,
    require_no_training_claim: bool,
    require_no_quality_claim: bool,
    min_tokenized_train_files: int,
    min_tokenized_val_files: int,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    token = _dict(report.get("token_stats"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericFullManifestWindowPreparationError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericFullManifestWindowPreparationError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_ready and not bool(readiness.get("full_manifest_window_preparation_ready", False)):
        raise StageBGenericFullManifestWindowPreparationError("full manifest window preparation must be ready")
    if _int(token.get("tokenized_train_files")) < min_tokenized_train_files:
        raise StageBGenericFullManifestWindowPreparationError("tokenized train records below threshold")
    if _int(token.get("tokenized_val_files")) < min_tokenized_val_files:
        raise StageBGenericFullManifestWindowPreparationError("tokenized val records below threshold")
    if not bool(token.get("fits_vocab", False)):
        raise StageBGenericFullManifestWindowPreparationError("token ids must fit vocab")
    if require_no_training_claim and bool(readiness.get("full_training_executed", True)):
        raise StageBGenericFullManifestWindowPreparationError("full training must not be claimed")
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
            bool(readiness.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericFullManifestWindowPreparationError("quality claims must remain false")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "full_manifest_window_preparation_ready": bool(
            readiness.get("full_manifest_window_preparation_ready", False)
        ),
        "generic_base_training_scale_smoke_ready": bool(
            readiness.get("generic_base_training_scale_smoke_ready", False)
        ),
        "train_file_count": _int(_dict(report.get("input")).get("train_file_count")),
        "val_file_count": _int(_dict(report.get("input")).get("val_file_count")),
        "tokenized_train_files": _int(token.get("tokenized_train_files")),
        "tokenized_val_files": _int(token.get("tokenized_val_files")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "full_training_executed": bool(readiness.get("full_training_executed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    inputs = report["input"]
    token = report["token_stats"]
    summary = report["dataset_summary"]
    lines = [
        "# Stage B Generic Full Manifest Window Preparation",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- full manifest window preparation ready: `{_bool_token(readiness['full_manifest_window_preparation_ready'])}`",
        f"- training scale smoke ready: `{_bool_token(readiness['generic_base_training_scale_smoke_ready'])}`",
        f"- full training executed: `{_bool_token(readiness['full_training_executed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Input",
        "",
        f"- train files: `{inputs['train_file_count']}`",
        f"- val files: `{inputs['val_file_count']}`",
        f"- window bars: `{inputs['window_bars']}`",
        f"- window stride bars: `{inputs['window_stride_bars']}`",
        f"- min window target notes: `{inputs['min_window_target_notes']}`",
        "",
        "## Tokenized Records",
        "",
        f"- tokenized train files: `{token['tokenized_train_files']}`",
        f"- tokenized val files: `{token['tokenized_val_files']}`",
        f"- max token id: `{token['max_token_id']}`",
        f"- vocab size: `{token['vocab_size']}`",
        f"- fits vocab: `{_bool_token(token['fits_vocab'])}`",
        f"- dataset train samples: `{summary.get('train_samples')}`",
        f"- dataset val samples: `{summary.get('val_samples')}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare full generic manifests as Stage B windows")
    parser.add_argument(
        "--training_data_plan",
        type=str,
        default="outputs/stage_b_generic_model_core_training_data_plan/"
        "harness_stage_b_generic_model_core_training_data_plan/"
        "stage_b_generic_model_core_training_data_plan.json",
    )
    parser.add_argument(
        "--manifests_dir",
        type=str,
        default="outputs/stage_b_generic_base_manifest_contract/"
        "harness_stage_b_generic_base_manifest_contract/manifests",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_full_manifest_window_preparation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default=BOUNDARY)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_ready", action="store_true")
    parser.add_argument("--require_no_training_claim", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    parser.add_argument("--min_tokenized_train_files", type=int, default=1)
    parser.add_argument("--min_tokenized_val_files", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    training_data_plan = read_json(Path(args.training_data_plan))
    plan = validate_training_data_plan(training_data_plan)

    manifests_dir = Path(args.manifests_dir)
    train_manifest = run_dir / "manifests" / "generic_jazz_train.txt"
    val_manifest = run_dir / "manifests" / "generic_jazz_val.txt"
    train_count = copy_manifest(manifests_dir / "generic_jazz_train.txt", train_manifest)
    val_count = copy_manifest(manifests_dir / "generic_jazz_val.txt", val_manifest)

    roles_dir = run_dir / "roles"
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
            "stage_b_v1",
            "--stage_b_window_bars",
            str(plan["window_bars"]),
            "--stage_b_window_stride_bars",
            str(plan["window_stride_bars"]),
            "--stage_b_min_window_target_notes",
            str(plan["min_window_target_notes"]),
            "--overwrite",
        ]
    )

    report = build_full_window_report(
        run_dir=run_dir,
        training_data_plan=training_data_plan,
        source_manifests_dir=manifests_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        roles_dir=roles_dir,
        train_file_count=train_count,
        val_file_count=val_count,
        window_bars=plan["window_bars"],
        window_stride_bars=plan["window_stride_bars"],
        min_window_target_notes=plan["min_window_target_notes"],
    )
    summary = validate_full_window_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_ready=bool(args.require_ready),
        require_no_training_claim=bool(args.require_no_training_claim),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_tokenized_train_files=int(args.min_tokenized_train_files),
        min_tokenized_val_files=int(args.min_tokenized_val_files),
    )
    write_json(run_dir / "stage_b_generic_full_manifest_window_preparation.json", report)
    write_json(run_dir / "stage_b_generic_full_manifest_window_preparation_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_full_manifest_window_preparation.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
