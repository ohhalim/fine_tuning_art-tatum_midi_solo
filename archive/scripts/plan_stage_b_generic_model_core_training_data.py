"""Build the generic model-core training data plan after repair-loop stop."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBGenericModelCoreTrainingDataPlanError(ValueError):
    pass


MODEL_CORE_DECISION_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_model_core_review_decision"
)
MANIFEST_CONTRACT_BOUNDARY = "stage_b_generic_base_manifest_contract"
WINDOW_SMOKE_BOUNDARY = "stage_b_generic_stage_b_window_prepare_smoke"
TINY_TRAINING_BOUNDARY = "stage_b_generic_base_tiny_training_smoke"
PLAN_BOUNDARY = "stage_b_generic_model_core_training_data_plan"
NEXT_BOUNDARY = "stage_b_generic_full_manifest_window_preparation"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def validate_model_core_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("methodology_evidence"))
    if str(decision.get("current_boundary") or "") != MODEL_CORE_DECISION_BOUNDARY:
        raise StageBGenericModelCoreTrainingDataPlanError("model-core review decision boundary required")
    if str(decision.get("next_boundary") or "") != PLAN_BOUNDARY:
        raise StageBGenericModelCoreTrainingDataPlanError("model-core review decision must route to training data plan")
    if bool(decision.get("continue_constraint_postprocess_repair_loop", True)):
        raise StageBGenericModelCoreTrainingDataPlanError("constraint/postprocess repair loop must be stopped")
    if str(decision.get("tiny_checkpoint_role") or "") != "diagnostic_only":
        raise StageBGenericModelCoreTrainingDataPlanError("tiny checkpoint must be diagnostic-only")
    if not bool(decision.get("model_core_transition_required", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("model-core transition must be required")
    if not bool(evidence.get("objective_proxy_gap_recorded", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("objective proxy gap evidence required")
    blocked = [
        "musical_quality_claimed",
        "quality_root_cause_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked if bool(claim.get(name, True))]
    if claimed:
        raise StageBGenericModelCoreTrainingDataPlanError(f"unsupported model-core claims: {claimed}")
    return {
        "decision": str(decision.get("decision") or ""),
        "tiny_checkpoint_role": str(decision.get("tiny_checkpoint_role") or ""),
        "candidate_without_objective_flag_count": _int(
            evidence.get("candidate_without_objective_flag_count")
        ),
    }


def validate_manifest_contract(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    guards = _dict(report.get("guards"))
    counts = _dict(report.get("split_counts"))
    if str(readiness.get("boundary") or "") != MANIFEST_CONTRACT_BOUNDARY:
        raise StageBGenericModelCoreTrainingDataPlanError("manifest contract boundary required")
    if not bool(readiness.get("manifest_contract_ready", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("manifest contract must be ready")
    if bool(readiness.get("broad_training_execution_ready", True)):
        raise StageBGenericModelCoreTrainingDataPlanError("broad training must remain blocked at contract stage")
    leak_fields = [
        "generic_brad_leak_count",
        "brad_non_brad_leak_count",
        "overlap_path_count",
        "duplicate_exact_hash_group_count",
        "duplicate_exact_file_count",
    ]
    leaks = {name: _int(guards.get(name)) for name in leak_fields}
    if any(value != 0 for value in leaks.values()):
        raise StageBGenericModelCoreTrainingDataPlanError(f"manifest leakage guard failed: {leaks}")
    return {
        "generic_jazz_train": _int(counts.get("generic_jazz_train")),
        "generic_jazz_val": _int(counts.get("generic_jazz_val")),
        "brad_adaptation_train": _int(counts.get("brad_adaptation_train")),
        "brad_adaptation_val": _int(counts.get("brad_adaptation_val")),
        "brad_test_holdout": _int(counts.get("brad_test_holdout")),
        "generic_leak_count": leaks["generic_brad_leak_count"],
        "overlap_path_count": leaks["overlap_path_count"],
    }


def validate_window_smoke(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    input_config = _dict(report.get("input"))
    token_stats = _dict(report.get("token_stats"))
    dataset_summary = _dict(report.get("dataset_summary"))
    if str(readiness.get("boundary") or "") != WINDOW_SMOKE_BOUNDARY:
        raise StageBGenericModelCoreTrainingDataPlanError("window smoke boundary required")
    if not bool(readiness.get("stage_b_window_prepare_smoke_ready", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("window smoke must be ready")
    if bool(readiness.get("generic_base_training_execution_ready", True)):
        raise StageBGenericModelCoreTrainingDataPlanError("generic base training must remain blocked after smoke")
    if _int(token_stats.get("max_token_id")) >= _int(token_stats.get("vocab_size")):
        raise StageBGenericModelCoreTrainingDataPlanError("window smoke token ids must fit vocab")
    return {
        "selected_train_files": _int(input_config.get("selected_train_files")),
        "selected_val_files": _int(input_config.get("selected_val_files")),
        "window_bars": _int(input_config.get("window_bars")),
        "window_stride_bars": _int(input_config.get("window_stride_bars")),
        "min_window_target_notes": _int(input_config.get("min_window_target_notes")),
        "token_files": _int(token_stats.get("files")),
        "non_empty_token_files": _int(token_stats.get("non_empty_files")),
        "max_token_id": _int(token_stats.get("max_token_id")),
        "vocab_size": _int(token_stats.get("vocab_size")),
        "train_manifest": str(dataset_summary.get("train_manifest") or ""),
        "val_manifest": str(dataset_summary.get("val_manifest") or ""),
    }


def validate_tiny_training_smoke(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    token_stats = _dict(report.get("token_stats"))
    training = _dict(report.get("training"))
    input_config = _dict(report.get("input"))
    if str(readiness.get("boundary") or "") != TINY_TRAINING_BOUNDARY:
        raise StageBGenericModelCoreTrainingDataPlanError("tiny training smoke boundary required")
    if not bool(readiness.get("tiny_training_smoke_passed", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("tiny training smoke must pass")
    if not bool(readiness.get("generic_base_training_path_smoked", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("generic training path smoke required")
    if bool(readiness.get("broad_training_execution_ready", True)):
        raise StageBGenericModelCoreTrainingDataPlanError("broad training execution must remain blocked")
    if _int(training.get("returncode")) != 0:
        raise StageBGenericModelCoreTrainingDataPlanError("tiny training command must succeed")
    if not bool(token_stats.get("fits_vocab", False)):
        raise StageBGenericModelCoreTrainingDataPlanError("tiny training tokens must fit vocab")
    return {
        "selected_train_records": _int(input_config.get("selected_train_records")),
        "selected_val_records": _int(input_config.get("selected_val_records")),
        "token_files": _int(token_stats.get("files")),
        "non_empty_token_files": _int(token_stats.get("non_empty_files")),
        "max_token_id": _int(token_stats.get("max_token_id")),
        "vocab_size": _int(token_stats.get("vocab_size")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "returncode": _int(training.get("returncode")),
    }


def build_training_data_plan(
    model_core_decision: dict[str, Any],
    manifest_contract: dict[str, Any],
    window_smoke: dict[str, Any],
    tiny_training_smoke: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    model = validate_model_core_decision(model_core_decision)
    manifest = validate_manifest_contract(manifest_contract)
    window = validate_window_smoke(window_smoke)
    tiny = validate_tiny_training_smoke(tiny_training_smoke)
    return {
        "schema_version": "stage_b_generic_model_core_training_data_plan_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "input_boundaries": {
            "model_core_decision": MODEL_CORE_DECISION_BOUNDARY,
            "manifest_contract": MANIFEST_CONTRACT_BOUNDARY,
            "window_smoke": WINDOW_SMOKE_BOUNDARY,
            "tiny_training_smoke": TINY_TRAINING_BOUNDARY,
        },
        "methodology_reset": {
            "repair_loop_status": "stopped",
            "tiny_checkpoint_role": model["tiny_checkpoint_role"],
            "objective_proxy_gap_candidate_count": model["candidate_without_objective_flag_count"],
            "next_method": "generic_manifest_full_window_preparation_then_training",
        },
        "evidence_summary": {
            "manifest": manifest,
            "window_smoke": window,
            "tiny_training_smoke": tiny,
        },
        "training_data_plan": {
            "plan_boundary": PLAN_BOUNDARY,
            "data_source": "generic_jazz_train_val_manifest",
            "generic_train_file_count": manifest["generic_jazz_train"],
            "generic_val_file_count": manifest["generic_jazz_val"],
            "brad_files_excluded_from_generic_base": True,
            "brad_holdout_preserved": True,
            "window_parameters": {
                "sequence_format": "stage_b_v1",
                "role": "lead",
                "window_bars": window["window_bars"],
                "window_stride_bars": window["window_stride_bars"],
                "min_window_target_notes": window["min_window_target_notes"],
                "token_vocab_guard": "max_token_id_lt_vocab_size",
            },
            "execution_order": [
                {
                    "step": 1,
                    "name": "full_generic_manifest_window_preparation",
                    "goal": "convert full non-Brad generic train/val manifests to Stage B windows",
                    "stop_condition": "token ids exceed vocab or train/val boundary changes",
                },
                {
                    "step": 2,
                    "name": "full_window_token_guard",
                    "goal": "record train/val window counts, non-empty token counts, max token id, vocab fit",
                    "stop_condition": "empty validation split or vocab overflow",
                },
                {
                    "step": 3,
                    "name": "generic_base_training_scale_smoke",
                    "goal": "run controlled larger-than-tiny training with validation loss and checkpoint metadata",
                    "stop_condition": "training returncode nonzero or validation artifact missing",
                },
                {
                    "step": 4,
                    "name": "generic_base_generation_probe",
                    "goal": "evaluate raw model output before constrained rescue",
                    "stop_condition": "raw generation fails structural gate; record as model-core failure",
                },
                {
                    "step": 5,
                    "name": "review_package_and_audio_boundary",
                    "goal": "render only structurally reviewable candidates for listening review",
                    "stop_condition": "no candidate passes objective review gate",
                },
            ],
        },
        "claim_boundary": {
            "boundary": PLAN_BOUNDARY,
            "training_data_plan_ready": True,
            "full_training_executed": False,
            "full_window_preparation_executed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": PLAN_BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "next_recommended_issue": "Stage B generic full manifest window preparation",
        },
        "proven": [
            "repair_loop_stopped_before_new_training_plan",
            "generic_brad_manifest_contract_ready",
            "stage_b_window_smoke_ready",
            "generic_tiny_training_path_smoked",
        ],
        "not_proven": [
            "full_generic_window_preparation",
            "full_generic_training_run",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
    }


def validate_training_data_plan(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_stop_repair_loop: bool,
    require_no_quality_claim: bool,
    min_generic_train_files: int,
    min_generic_val_files: int,
) -> dict[str, Any]:
    claim = _dict(report.get("claim_boundary"))
    decision = _dict(report.get("decision"))
    reset = _dict(report.get("methodology_reset"))
    plan = _dict(report.get("training_data_plan"))
    boundary = str(claim.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericModelCoreTrainingDataPlanError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericModelCoreTrainingDataPlanError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_stop_repair_loop and str(reset.get("repair_loop_status") or "") != "stopped":
        raise StageBGenericModelCoreTrainingDataPlanError("repair loop must be stopped")
    if _int(plan.get("generic_train_file_count")) < min_generic_train_files:
        raise StageBGenericModelCoreTrainingDataPlanError("generic train file count below threshold")
    if _int(plan.get("generic_val_file_count")) < min_generic_val_files:
        raise StageBGenericModelCoreTrainingDataPlanError("generic val file count below threshold")
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("full_training_executed", True)),
            bool(claim.get("full_window_preparation_executed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
            bool(claim.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericModelCoreTrainingDataPlanError("training or quality claims must remain false")
    return {
        "boundary": boundary,
        "repair_loop_status": str(reset.get("repair_loop_status") or ""),
        "tiny_checkpoint_role": str(reset.get("tiny_checkpoint_role") or ""),
        "generic_train_file_count": _int(plan.get("generic_train_file_count")),
        "generic_val_file_count": _int(plan.get("generic_val_file_count")),
        "brad_files_excluded_from_generic_base": bool(plan.get("brad_files_excluded_from_generic_base", False)),
        "training_data_plan_ready": bool(claim.get("training_data_plan_ready", False)),
        "full_training_executed": bool(claim.get("full_training_executed", True)),
        "broad_trained_model_quality_claimed": bool(claim.get("broad_trained_model_quality_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": next_boundary,
        "next_recommended_issue": str(decision.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    reset = report["methodology_reset"]
    evidence = report["evidence_summary"]
    plan = report["training_data_plan"]
    claim = report["claim_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Model-Core Training Data Plan",
        "",
        "## Summary",
        "",
        f"- boundary: `{claim['boundary']}`",
        f"- repair loop status: `{reset['repair_loop_status']}`",
        f"- tiny checkpoint role: `{reset['tiny_checkpoint_role']}`",
        f"- next method: `{reset['next_method']}`",
        f"- generic train / val files: `{plan['generic_train_file_count']}` / `{plan['generic_val_file_count']}`",
        f"- Brad files excluded from generic base: `{_bool_token(plan['brad_files_excluded_from_generic_base'])}`",
        f"- full window preparation executed: `{_bool_token(claim['full_window_preparation_executed'])}`",
        f"- full training executed: `{_bool_token(claim['full_training_executed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(claim['broad_trained_model_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Evidence",
        "",
        f"- manifest generic train / val: `{evidence['manifest']['generic_jazz_train']}` / `{evidence['manifest']['generic_jazz_val']}`",
        f"- manifest Brad split: `{evidence['manifest']['brad_adaptation_train']}` / `{evidence['manifest']['brad_adaptation_val']}` / `{evidence['manifest']['brad_test_holdout']}`",
        f"- window smoke selected train / val files: `{evidence['window_smoke']['selected_train_files']}` / `{evidence['window_smoke']['selected_val_files']}`",
        f"- window smoke token max / vocab: `{evidence['window_smoke']['max_token_id']}` / `{evidence['window_smoke']['vocab_size']}`",
        f"- tiny training selected train / val records: `{evidence['tiny_training_smoke']['selected_train_records']}` / `{evidence['tiny_training_smoke']['selected_val_records']}`",
        f"- tiny training best validation loss: `{evidence['tiny_training_smoke']['best_validation_loss']:.4f}`",
        "",
        "## Execution Order",
        "",
        "| step | name | goal | stop condition |",
        "|---:|---|---|---|",
    ]
    for item in plan["execution_order"]:
        lines.append(
            f"| {item['step']} | `{item['name']}` | {item['goal']} | {item['stop_condition']} |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage B generic model-core training data plan")
    parser.add_argument(
        "--model_core_decision",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_model_core_review_decision/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_model_core_review_decision/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_model_core_review_decision.json",
    )
    parser.add_argument(
        "--manifest_contract",
        type=str,
        default="outputs/stage_b_generic_base_manifest_contract/"
        "harness_stage_b_generic_base_manifest_contract/stage_b_generic_base_manifest_contract.json",
    )
    parser.add_argument(
        "--window_smoke",
        type=str,
        default="outputs/stage_b_generic_manifest_window_smoke/"
        "harness_stage_b_generic_manifest_window_smoke/stage_b_generic_manifest_window_smoke.json",
    )
    parser.add_argument(
        "--tiny_training_smoke",
        type=str,
        default="outputs/stage_b_generic_base_tiny_training_smoke/"
        "harness_stage_b_generic_base_tiny_training_smoke/stage_b_generic_base_tiny_training_smoke.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_model_core_training_data_plan",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default=PLAN_BOUNDARY)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--min_generic_train_files", type=int, default=2000)
    parser.add_argument("--min_generic_val_files", type=int, default=200)
    parser.add_argument("--require_stop_repair_loop", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_training_data_plan(
        read_json(Path(args.model_core_decision)),
        read_json(Path(args.manifest_contract)),
        read_json(Path(args.window_smoke)),
        read_json(Path(args.tiny_training_smoke)),
        output_dir=output_dir,
    )
    summary = validate_training_data_plan(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_stop_repair_loop=bool(args.require_stop_repair_loop),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_generic_train_files=int(args.min_generic_train_files),
        min_generic_val_files=int(args.min_generic_val_files),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_generic_model_core_training_data_plan.json", report)
    write_json(output_dir / "stage_b_generic_model_core_training_data_plan_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_model_core_training_data_plan.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
