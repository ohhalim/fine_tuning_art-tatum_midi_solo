"""Assess Stage B readiness before moving toward a generic jazz base."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBGenericBaseReadinessError(ValueError):
    pass


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


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_inputs(dataset_audit: dict[str, Any], final_decision: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    dataset_summary = _dict(dataset_audit.get("summary"))
    decision = _dict(final_decision.get("decision"))
    claim = _dict(final_decision.get("claim_boundary"))

    if not dataset_summary:
        raise StageBGenericBaseReadinessError("dataset audit summary required")
    if str(decision.get("final_boundary") or "") != "outside_soloing_repair_objective_path_complete":
        raise StageBGenericBaseReadinessError("outside-soloing repair final objective boundary required")

    blocked_claims = [
        "human_audio_preference_claimed",
        "multi_reviewer_preference_claimed",
        "broad_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(claim.get(name, True))]
    if claimed:
        raise StageBGenericBaseReadinessError(f"unexpected claim for readiness audit: {claimed}")
    return dataset_summary, claim


def build_readiness_report(
    dataset_audit: dict[str, Any],
    final_decision: dict[str, Any],
    *,
    output_dir: Path,
    min_non_brad_candidates: int,
    min_brad_holdout_candidates: int,
) -> dict[str, Any]:
    dataset_summary, final_claim = validate_inputs(dataset_audit, final_decision)
    final_decision_payload = _dict(final_decision.get("decision"))
    objective = _dict(final_decision.get("objective_repeatability"))

    readable = _int(dataset_summary.get("readable_file_count"))
    unreadable = _int(dataset_summary.get("unreadable_file_count"))
    candidate = _int(dataset_summary.get("candidate_file_count"))
    non_brad = _int(dataset_summary.get("candidate_non_brad_file_count"))
    brad = _int(dataset_summary.get("candidate_brad_file_count"))
    duplicate_groups = _int(dataset_summary.get("duplicate_exact_hash_group_count"))
    duplicate_files = _int(dataset_summary.get("duplicate_exact_file_count"))

    dataset_pool_ready = (
        readable > 0
        and unreadable == 0
        and candidate > 0
        and non_brad >= min_non_brad_candidates
        and brad >= min_brad_holdout_candidates
        and duplicate_groups == 0
        and duplicate_files == 0
    )
    objective_path_ready = bool(final_claim.get("outside_soloing_repair_objective_path_claimed", False))
    phase4_prep_ready = dataset_pool_ready and objective_path_ready

    missing_for_training_run = [
        "Stage B generic train/val manifest contract refresh",
        "generic split duration-explicit window preparation smoke",
        "generic-base training run and multi-sample review gate",
    ]
    missing_for_style_adaptation = [
        "generic base review-gate pass-rate",
        "Brad adaptation or retrieval path experiment",
        "Brad holdout evaluation boundary",
    ]

    boundary = "stage_b_generic_base_readiness_audit"
    next_boundary = "stage_b_generic_base_manifest_contract"
    return {
        "schema_version": "stage_b_generic_base_readiness_audit_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_dataset_audit": str(dataset_audit.get("input_dir") or ""),
        "source_final_decision_schema": str(final_decision.get("schema_version") or ""),
        "dataset_pool": {
            "readable_file_count": readable,
            "unreadable_file_count": unreadable,
            "candidate_file_count": candidate,
            "candidate_non_brad_file_count": non_brad,
            "candidate_brad_file_count": brad,
            "duplicate_exact_hash_group_count": duplicate_groups,
            "duplicate_exact_file_count": duplicate_files,
            "min_non_brad_candidates": int(min_non_brad_candidates),
            "min_brad_holdout_candidates": int(min_brad_holdout_candidates),
            "generic_candidate_pool_ready": dataset_pool_ready,
            "brad_holdout_available": brad >= min_brad_holdout_candidates,
        },
        "stage_b_objective_path": {
            "final_boundary": str(final_decision_payload.get("final_boundary") or ""),
            "objective_path_ready": objective_path_ready,
            "source_candidate_count": _int(objective.get("source_candidate_count")),
            "qualified_source_candidate_count": _int(objective.get("qualified_source_candidate_count")),
            "supported_repair_policy_count": _int(objective.get("supported_repair_policy_count")),
            "total_qualified_variant_count": _int(objective.get("total_qualified_variant_count")),
        },
        "readiness": {
            "boundary": boundary,
            "phase4_prep_ready": phase4_prep_ready,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_ready": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": boundary,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "dataset pool and Stage B objective path support Phase 4 preparation; "
                "actual generic-base training and Brad adaptation remain unrun"
            ),
        },
        "missing_for_broad_training_execution": missing_for_training_run,
        "missing_for_brad_style_adaptation": missing_for_style_adaptation,
        "proven": [
            "dataset_audit_readable_candidate_pool",
            "non_brad_generic_candidate_pool_available",
            "brad_holdout_pool_available",
            "stage_b_objective_repair_path_complete",
        ],
        "not_proven": [
            "generic_base_training_run",
            "generic_base_multi_seed_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B generic base manifest contract",
    }


def validate_readiness_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_phase4_prep_ready: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    dataset = _dict(report.get("dataset_pool"))
    stage_b = _dict(report.get("stage_b_objective_path"))

    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseReadinessError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseReadinessError(f"expected next boundary {expected_next_boundary}, got {next_boundary}")
    if require_phase4_prep_ready and not bool(readiness.get("phase4_prep_ready", False)):
        raise StageBGenericBaseReadinessError("Phase 4 preparation should be ready")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseReadinessError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseReadinessError("Brad style adaptation must not be claimed")

    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "phase4_prep_ready": bool(readiness.get("phase4_prep_ready", False)),
        "generic_candidate_pool_ready": bool(dataset.get("generic_candidate_pool_ready", False)),
        "brad_holdout_available": bool(dataset.get("brad_holdout_available", False)),
        "stage_b_objective_path_ready": bool(stage_b.get("objective_path_ready", False)),
        "broad_training_execution_ready": bool(readiness.get("broad_training_execution_ready", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    dataset = report["dataset_pool"]
    stage_b = report["stage_b_objective_path"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Base Readiness Audit",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- phase4 prep ready: `{_bool_token(readiness['phase4_prep_ready'])}`",
        f"- broad training execution ready: `{_bool_token(readiness['broad_training_execution_ready'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Dataset Pool",
        "",
        f"- readable files: `{dataset['readable_file_count']}`",
        f"- candidate files: `{dataset['candidate_file_count']}`",
        f"- non-Brad candidate files: `{dataset['candidate_non_brad_file_count']}`",
        f"- Brad holdout candidates: `{dataset['candidate_brad_file_count']}`",
        f"- duplicate exact hash groups: `{dataset['duplicate_exact_hash_group_count']}`",
        f"- generic candidate pool ready: `{_bool_token(dataset['generic_candidate_pool_ready'])}`",
        "",
        "## Stage B Objective Path",
        "",
        f"- final boundary: `{stage_b['final_boundary']}`",
        f"- objective path ready: `{_bool_token(stage_b['objective_path_ready'])}`",
        f"- source candidates: `{stage_b['source_candidate_count']}`",
        f"- qualified source candidates: `{stage_b['qualified_source_candidate_count']}`",
        f"- supported repair policies: `{stage_b['supported_repair_policy_count']}`",
        f"- qualified variants: `{stage_b['total_qualified_variant_count']}`",
        "",
        "## Missing For Broad Training Execution",
        "",
    ]
    for item in report["missing_for_broad_training_execution"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Missing For Brad Style Adaptation",
            "",
        ]
    )
    for item in report["missing_for_brad_style_adaptation"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## Not Proven",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assess Stage B generic base readiness")
    parser.add_argument(
        "--dataset_audit",
        type=str,
        default="outputs/dataset_audit/jazz_piano_dataset_audit.json",
    )
    parser.add_argument(
        "--final_decision",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_final_decision/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_final_decision/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_final_decision.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_readiness_audit")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--min_non_brad_candidates", type=int, default=1000)
    parser.add_argument("--min_brad_holdout_candidates", type=int, default=20)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_phase4_prep_ready", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_readiness_report(
        read_json(Path(args.dataset_audit)),
        read_json(Path(args.final_decision)),
        output_dir=output_dir,
        min_non_brad_candidates=args.min_non_brad_candidates,
        min_brad_holdout_candidates=args.min_brad_holdout_candidates,
    )
    summary = validate_readiness_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_phase4_prep_ready=bool(args.require_phase4_prep_ready),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(output_dir / "stage_b_generic_base_readiness_audit.json", report)
    write_json(output_dir / "stage_b_generic_base_readiness_audit_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_base_readiness_audit.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
