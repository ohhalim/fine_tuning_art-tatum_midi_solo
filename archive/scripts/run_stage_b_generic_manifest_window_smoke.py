"""Run Stage B duration-explicit window preparation smoke from generic manifests."""

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

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text
from scripts.run_manifest_prepare_smoke import read_manifest_paths, write_limited_manifest
from scripts.run_stage_b_window_tiny_overfit import token_stats


class StageBGenericManifestWindowSmokeError(ValueError):
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


def run_command(command: Sequence[str]) -> None:
    print("+ " + " ".join(str(part) for part in command))
    subprocess.run(command, cwd=ROOT_DIR, check=True)


def count_npy_files(path: Path) -> int:
    return len(sorted(path.glob("*.npy")))


def build_smoke_report(
    *,
    run_dir: Path,
    source_manifests_dir: Path,
    train_manifest: Path,
    val_manifest: Path,
    roles_dir: Path,
    selected_train_files: int,
    selected_val_files: int,
    window_bars: int,
    window_stride_bars: int,
    min_window_target_notes: int,
) -> dict[str, Any]:
    role_root = roles_dir / "lead"
    summary_path = role_root / "dataset_summary.json"
    dataset_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    tokenized_dir = role_root / "tokenized"
    stats = token_stats(tokenized_dir)
    tokenized_train = count_npy_files(tokenized_dir / "train")
    tokenized_val = count_npy_files(tokenized_dir / "val")
    smoke_ready = (
        selected_train_files > 0
        and selected_val_files > 0
        and tokenized_train > 0
        and tokenized_val > 0
        and bool(stats.get("fits_vocab", False))
        and str(dataset_summary.get("sequence_format") or "") == "stage_b_v1"
        and _int(dataset_summary.get("stage_b_window_bars")) == int(window_bars)
        and _int(dataset_summary.get("stage_b_window_stride_bars")) == int(window_stride_bars)
    )
    boundary = "stage_b_generic_stage_b_window_prepare_smoke"
    next_boundary = "stage_b_generic_base_tiny_training_smoke"
    return {
        "schema_version": "stage_b_generic_manifest_window_smoke_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "source_manifests_dir": str(source_manifests_dir),
        "smoke_train_manifest": str(train_manifest),
        "smoke_val_manifest": str(val_manifest),
        "roles_dir": str(roles_dir),
        "input": {
            "selected_train_files": int(selected_train_files),
            "selected_val_files": int(selected_val_files),
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
            "boundary": boundary,
            "stage_b_window_prepare_smoke_ready": smoke_ready,
            "generic_base_training_execution_ready": False,
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
                "generic manifest prefix produced stage_b_v1 duration-explicit train/val windows; "
                "next step is a tiny training smoke, not broad quality claim"
            ),
        },
        "not_proven": [
            "generic_base_training_run",
            "generic_base_multi_seed_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base tiny training smoke",
    }


def validate_smoke_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_smoke_ready: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    token = _dict(report.get("token_stats"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericManifestWindowSmokeError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericManifestWindowSmokeError(f"expected next boundary {expected_next_boundary}, got {next_boundary}")
    if require_smoke_ready and not bool(readiness.get("stage_b_window_prepare_smoke_ready", False)):
        raise StageBGenericManifestWindowSmokeError("Stage B generic window preparation smoke should be ready")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericManifestWindowSmokeError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericManifestWindowSmokeError("Brad style adaptation must not be claimed")
    if not bool(token.get("fits_vocab", False)):
        raise StageBGenericManifestWindowSmokeError("token ids must fit the model vocab")
    if _int(token.get("tokenized_train_files")) <= 0:
        raise StageBGenericManifestWindowSmokeError("tokenized train records required")
    if _int(token.get("tokenized_val_files")) <= 0:
        raise StageBGenericManifestWindowSmokeError("tokenized val records required")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "stage_b_window_prepare_smoke_ready": bool(
            readiness.get("stage_b_window_prepare_smoke_ready", False)
        ),
        "tokenized_train_files": _int(token.get("tokenized_train_files")),
        "tokenized_val_files": _int(token.get("tokenized_val_files")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "generic_base_training_execution_ready": bool(
            readiness.get("generic_base_training_execution_ready", True)
        ),
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
    summary = report["dataset_summary"]
    lines = [
        "# Stage B Generic Split Window Preparation Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- stage_b window prepare smoke ready: `{_bool_token(readiness['stage_b_window_prepare_smoke_ready'])}`",
        f"- generic base training execution ready: `{_bool_token(readiness['generic_base_training_execution_ready'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Input",
        "",
        f"- selected train files: `{inputs['selected_train_files']}`",
        f"- selected val files: `{inputs['selected_val_files']}`",
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
    parser = argparse.ArgumentParser(description="Run Stage B generic manifest window preparation smoke")
    parser.add_argument(
        "--manifests_dir",
        type=str,
        default="outputs/stage_b_generic_base_manifest_contract/"
        "harness_stage_b_generic_base_manifest_contract/manifests",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_manifest_window_smoke")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--train_files", type=int, default=6)
    parser.add_argument("--val_files", type=int, default=3)
    parser.add_argument("--window_bars", type=int, default=2)
    parser.add_argument("--window_stride_bars", type=int, default=2)
    parser.add_argument("--min_window_target_notes", type=int, default=4)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_smoke_ready", action="store_true")
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

    manifests_dir = Path(args.manifests_dir)
    train_manifest = run_dir / "smoke_manifests" / "generic_jazz_train.txt"
    val_manifest = run_dir / "smoke_manifests" / "generic_jazz_val.txt"
    selected_train = write_limited_manifest(manifests_dir / "generic_jazz_train.txt", train_manifest, args.train_files)
    selected_val = write_limited_manifest(manifests_dir / "generic_jazz_val.txt", val_manifest, args.val_files)
    if selected_train <= 0:
        raise StageBGenericManifestWindowSmokeError("generic smoke train manifest is empty")
    if selected_val <= 0:
        raise StageBGenericManifestWindowSmokeError("generic smoke val manifest is empty")

    # Explicit read validates the generated manifest files are parseable after the prefix copy.
    read_manifest_paths(train_manifest)
    read_manifest_paths(val_manifest)

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
            str(args.window_bars),
            "--stage_b_window_stride_bars",
            str(args.window_stride_bars),
            "--stage_b_min_window_target_notes",
            str(args.min_window_target_notes),
            "--overwrite",
        ]
    )
    report = build_smoke_report(
        run_dir=run_dir,
        source_manifests_dir=manifests_dir,
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        roles_dir=roles_dir,
        selected_train_files=selected_train,
        selected_val_files=selected_val,
        window_bars=args.window_bars,
        window_stride_bars=args.window_stride_bars,
        min_window_target_notes=args.min_window_target_notes,
    )
    summary = validate_smoke_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_smoke_ready=bool(args.require_smoke_ready),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_manifest_window_smoke.json", report)
    write_json(run_dir / "stage_b_generic_manifest_window_smoke_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_manifest_window_smoke.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
