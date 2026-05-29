"""Build and validate the Stage B generic base manifest contract."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text
from scripts.build_jazz_training_manifests import build_manifest_payload, write_outputs


class StageBGenericBaseManifestContractError(ValueError):
    pass


GENERIC_SPLITS = ("generic_jazz_train", "generic_jazz_val")
BRAD_SPLITS = ("brad_adaptation_train", "brad_adaptation_val", "brad_test_holdout")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def split_paths(payload: dict[str, Any], split_names: tuple[str, ...]) -> set[str]:
    splits = _dict(payload.get("splits"))
    paths: set[str] = set()
    for split_name in split_names:
        for row in splits.get(split_name, []):
            paths.add(str(row.get("path") or ""))
    paths.discard("")
    return paths


def count_leaks(payload: dict[str, Any]) -> dict[str, int]:
    splits = _dict(payload.get("splits"))
    generic_brad_leaks = 0
    brad_non_brad_leaks = 0
    for split_name in GENERIC_SPLITS:
        generic_brad_leaks += sum(1 for row in splits.get(split_name, []) if bool(row.get("is_brad_mehldau")))
    for split_name in BRAD_SPLITS:
        brad_non_brad_leaks += sum(1 for row in splits.get(split_name, []) if not bool(row.get("is_brad_mehldau")))
    return {
        "generic_brad_leak_count": int(generic_brad_leaks),
        "brad_non_brad_leak_count": int(brad_non_brad_leaks),
    }


def overlapping_paths(payload: dict[str, Any]) -> list[str]:
    splits = _dict(payload.get("splits"))
    owners: dict[str, str] = {}
    overlaps: set[str] = set()
    for split_name, rows in splits.items():
        for row in rows:
            path = str(row.get("path") or "")
            if not path:
                continue
            owner = owners.get(path)
            if owner is None:
                owners[path] = split_name
            elif owner != split_name:
                overlaps.add(path)
    return sorted(overlaps)


def validate_readiness(readiness: dict[str, Any]) -> None:
    readiness_payload = _dict(readiness.get("readiness"))
    if str(readiness_payload.get("boundary") or "") != "stage_b_generic_base_readiness_audit":
        raise StageBGenericBaseManifestContractError("generic base readiness audit boundary required")
    if not bool(readiness_payload.get("phase4_prep_ready", False)):
        raise StageBGenericBaseManifestContractError("phase4 prep readiness required")
    blocked_claims = [
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness_payload.get(name, True))]
    if claimed:
        raise StageBGenericBaseManifestContractError(f"unexpected readiness claim: {claimed}")


def build_contract_report(
    manifest_payload: dict[str, Any],
    readiness: dict[str, Any],
    *,
    output_dir: Path,
    min_generic_train: int,
    min_generic_val: int,
    min_brad_holdout: int,
) -> dict[str, Any]:
    validate_readiness(readiness)
    counts = _dict(_dict(manifest_payload.get("counts")).get("splits"))
    audit_summary = _dict(manifest_payload.get("audit_summary"))
    leaks = count_leaks(manifest_payload)
    overlaps = overlapping_paths(manifest_payload)

    generic_train = _int(counts.get("generic_jazz_train"))
    generic_val = _int(counts.get("generic_jazz_val"))
    brad_train = _int(counts.get("brad_adaptation_train"))
    brad_val = _int(counts.get("brad_adaptation_val"))
    brad_holdout = _int(counts.get("brad_test_holdout"))
    duplicate_groups = _int(audit_summary.get("duplicate_exact_hash_group_count"))
    duplicate_files = _int(audit_summary.get("duplicate_exact_file_count"))
    expected_non_brad = _int(audit_summary.get("candidate_non_brad_file_count"))
    expected_brad = _int(audit_summary.get("candidate_brad_file_count"))
    actual_non_brad = generic_train + generic_val
    actual_brad = brad_train + brad_val + brad_holdout

    contract_ready = (
        generic_train >= min_generic_train
        and generic_val >= min_generic_val
        and brad_holdout >= min_brad_holdout
        and actual_non_brad == expected_non_brad
        and actual_brad == expected_brad
        and duplicate_groups == 0
        and duplicate_files == 0
        and leaks["generic_brad_leak_count"] == 0
        and leaks["brad_non_brad_leak_count"] == 0
        and not overlaps
    )

    boundary = "stage_b_generic_base_manifest_contract"
    next_boundary = "stage_b_generic_stage_b_window_prepare_smoke"
    return {
        "schema_version": "stage_b_generic_base_manifest_contract_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_manifest_schema": str(manifest_payload.get("schema_version") or ""),
        "source_readiness_schema": str(readiness.get("schema_version") or ""),
        "split_counts": {
            "generic_jazz_train": generic_train,
            "generic_jazz_val": generic_val,
            "brad_adaptation_train": brad_train,
            "brad_adaptation_val": brad_val,
            "brad_test_holdout": brad_holdout,
            "expected_non_brad_candidates": expected_non_brad,
            "actual_non_brad_split_count": actual_non_brad,
            "expected_brad_candidates": expected_brad,
            "actual_brad_split_count": actual_brad,
        },
        "guards": {
            **leaks,
            "overlap_path_count": len(overlaps),
            "duplicate_exact_hash_group_count": duplicate_groups,
            "duplicate_exact_file_count": duplicate_files,
            "min_generic_train": int(min_generic_train),
            "min_generic_val": int(min_generic_val),
            "min_brad_holdout": int(min_brad_holdout),
        },
        "readiness": {
            "boundary": boundary,
            "manifest_contract_ready": contract_ready,
            "stage_b_window_prepare_smoke_ready": contract_ready,
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
                "generic and Brad split manifests are separated and count-complete; "
                "next step is Stage B duration-explicit window preparation smoke"
            ),
        },
        "not_proven": [
            "stage_b_generic_window_prepare_smoke",
            "generic_base_training_run",
            "generic_base_multi_seed_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic split duration-explicit window preparation smoke",
    }


def validate_contract_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_contract_ready: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guards = _dict(report.get("guards"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseManifestContractError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseManifestContractError(f"expected next boundary {expected_next_boundary}, got {next_boundary}")
    if require_contract_ready and not bool(readiness.get("manifest_contract_ready", False)):
        raise StageBGenericBaseManifestContractError("manifest contract should be ready")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseManifestContractError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseManifestContractError("Brad style adaptation must not be claimed")
    if _int(guards.get("generic_brad_leak_count")) != 0:
        raise StageBGenericBaseManifestContractError("Brad rows leaked into generic splits")
    if _int(guards.get("brad_non_brad_leak_count")) != 0:
        raise StageBGenericBaseManifestContractError("non-Brad rows leaked into Brad splits")
    if _int(guards.get("overlap_path_count")) != 0:
        raise StageBGenericBaseManifestContractError("manifest paths overlap across splits")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "manifest_contract_ready": bool(readiness.get("manifest_contract_ready", False)),
        "stage_b_window_prepare_smoke_ready": bool(readiness.get("stage_b_window_prepare_smoke_ready", False)),
        "broad_training_execution_ready": bool(readiness.get("broad_training_execution_ready", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "generic_brad_leak_count": _int(guards.get("generic_brad_leak_count")),
        "brad_non_brad_leak_count": _int(guards.get("brad_non_brad_leak_count")),
        "overlap_path_count": _int(guards.get("overlap_path_count")),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    counts = report["split_counts"]
    guards = report["guards"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Base Manifest Contract",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- manifest contract ready: `{_bool_token(readiness['manifest_contract_ready'])}`",
        f"- stage_b window prepare smoke ready: `{_bool_token(readiness['stage_b_window_prepare_smoke_ready'])}`",
        f"- broad training execution ready: `{_bool_token(readiness['broad_training_execution_ready'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Split Counts",
        "",
        f"- generic_jazz_train: `{counts['generic_jazz_train']}`",
        f"- generic_jazz_val: `{counts['generic_jazz_val']}`",
        f"- expected non-Brad candidates: `{counts['expected_non_brad_candidates']}`",
        f"- actual non-Brad split count: `{counts['actual_non_brad_split_count']}`",
        f"- brad_adaptation_train: `{counts['brad_adaptation_train']}`",
        f"- brad_adaptation_val: `{counts['brad_adaptation_val']}`",
        f"- brad_test_holdout: `{counts['brad_test_holdout']}`",
        f"- expected Brad candidates: `{counts['expected_brad_candidates']}`",
        f"- actual Brad split count: `{counts['actual_brad_split_count']}`",
        "",
        "## Guards",
        "",
        f"- generic Brad leak count: `{guards['generic_brad_leak_count']}`",
        f"- Brad non-Brad leak count: `{guards['brad_non_brad_leak_count']}`",
        f"- overlap path count: `{guards['overlap_path_count']}`",
        f"- duplicate exact hash group count: `{guards['duplicate_exact_hash_group_count']}`",
        f"- duplicate exact file count: `{guards['duplicate_exact_file_count']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check Stage B generic base manifest contract")
    parser.add_argument("--audit_json", type=str, default="outputs/dataset_audit/jazz_piano_dataset_audit.json")
    parser.add_argument(
        "--readiness_audit",
        type=str,
        default="outputs/stage_b_generic_base_readiness_audit/"
        "harness_stage_b_generic_base_readiness_audit/"
        "stage_b_generic_base_readiness_audit.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_manifest_contract")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_generic_train", type=int, default=1000)
    parser.add_argument("--min_generic_val", type=int, default=100)
    parser.add_argument("--min_brad_holdout", type=int, default=10)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_contract_ready", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    manifests_dir = output_dir / "manifests"
    audit_json = Path(args.audit_json)
    readiness = read_json(Path(args.readiness_audit))
    manifest_payload = build_manifest_payload(
        read_json(audit_json),
        audit_json=audit_json,
        seed=args.seed,
        generic_train_ratio=0.9,
        generic_val_ratio=0.1,
        brad_train_ratio=0.65,
        brad_val_ratio=0.15,
        brad_holdout_ratio=0.2,
        group_fields=["artist", "album"],
    )
    write_outputs(manifests_dir, manifest_payload)
    report = build_contract_report(
        manifest_payload,
        readiness,
        output_dir=output_dir,
        min_generic_train=args.min_generic_train,
        min_generic_val=args.min_generic_val,
        min_brad_holdout=args.min_brad_holdout,
    )
    summary = validate_contract_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_contract_ready=bool(args.require_contract_ready),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(output_dir / "stage_b_generic_base_manifest_contract.json", report)
    write_json(output_dir / "stage_b_generic_base_manifest_contract_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_base_manifest_contract.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
