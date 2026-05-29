"""Decide final objective-only boundary after outside-soloing repair repeatability consolidation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_inputs(repeatability_consolidation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    boundary = _dict(repeatability_consolidation.get("consolidated_claim_boundary"))
    objective = _dict(repeatability_consolidation.get("selected_source_objective_support"))
    repeatability = _dict(repeatability_consolidation.get("policy_repeatability_support"))
    review = _dict(repeatability_consolidation.get("pending_user_review"))

    if str(boundary.get("boundary") or "") != "outside_soloing_repair_objective_repeatability_support":
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(
            "objective repeatability support boundary required"
        )
    if not bool(boundary.get("objective_repair_repeatability_claimed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(
            "objective repair repeatability claim required"
        )
    if int(objective.get("source_candidate_count", 0) or 0) < 2:
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError("two objective source candidates required")
    if int(repeatability.get("supported_repair_policy_count", 0) or 0) < 3:
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError("three supported repair policies required")
    if bool(review.get("review_input_present", True)):
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(
            "review input must remain absent for objective-only final decision"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "multi_reviewer_preference_claimed",
        "broad_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(boundary.get(name, True))]
    if claimed:
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(f"unexpected claim: {claimed}")
    return objective, repeatability, review


def build_final_decision(
    repeatability_consolidation: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    objective, repeatability, review = validate_inputs(repeatability_consolidation)
    final_boundary = "outside_soloing_repair_objective_path_complete"
    next_boundary = "stage_b_model_core_evidence_readme_refresh"
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_final_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_repeatability_consolidation_schema": str(repeatability_consolidation.get("schema_version") or ""),
        "input_boundary": "outside_soloing_repair_objective_repeatability_support",
        "objective_repeatability": {
            "source_candidate_count": int(objective.get("source_candidate_count", 0) or 0),
            "qualified_source_candidate_count": int(objective.get("qualified_source_candidate_count", 0) or 0),
            "dead_air_preserved_source_candidate_count": int(
                objective.get("dead_air_preserved_source_candidate_count", 0) or 0
            ),
            "supported_repair_policy_count": int(repeatability.get("supported_repair_policy_count", 0) or 0),
            "total_variant_count": int(repeatability.get("total_variant_count", 0) or 0),
            "total_qualified_variant_count": int(repeatability.get("total_qualified_variant_count", 0) or 0),
            "selected_min_chord_tone_ratio": float(
                repeatability.get("selected_min_chord_tone_ratio", 0.0) or 0.0
            ),
            "selected_max_non_chord_tone_run": int(
                repeatability.get("selected_max_non_chord_tone_run", 0) or 0
            ),
            "selected_max_interval": int(repeatability.get("selected_max_interval", 0) or 0),
        },
        "review_boundary": {
            "boundary": str(review.get("boundary") or ""),
            "review_input_present": bool(review.get("review_input_present", False)),
            "human_audio_preference_claimed": False,
            "preference_claim_requires_user_review": True,
        },
        "decision": {
            "final_boundary": final_boundary,
            "next_boundary": next_boundary,
            "reason": (
                "outside-soloing repair has objective selected-source and policy repeatability support; "
                "human/audio preference remains pending and must not be inferred from MIDI metrics"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "blocked_reason": "",
        },
        "claim_boundary": {
            "boundary": final_boundary,
            "outside_soloing_repair_objective_path_claimed": True,
            "human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "outside_soloing_repair_selected_source_objective_support",
            "outside_soloing_repair_policy_repeatability_support",
            "human_audio_preference_claim_blocked_without_review_input",
        ],
        "not_proven": [
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B model-core evidence README refresh",
    }


def validate_final_decision(
    report: dict[str, Any],
    *,
    expected_final_boundary: str | None,
    expected_next_boundary: str | None,
    require_auto_progress_allowed: bool,
    require_no_critical_user_input: bool,
    require_no_preference_claim: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    review = _dict(report.get("review_boundary"))
    final_boundary = str(decision.get("final_boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")

    if expected_final_boundary and final_boundary != expected_final_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(
            f"expected final boundary {expected_final_boundary}, got {final_boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError("auto progress must be allowed")
    if require_no_critical_user_input and bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError("critical user input must not be required")
    if require_no_preference_claim:
        if bool(claim.get("human_audio_preference_claimed", True)):
            raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError(
                "human/audio preference must not be claimed"
            )
        if bool(review.get("review_input_present", True)):
            raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError("review input must remain absent")
    if require_no_broad_quality_claim and bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairFinalDecisionError("broad model quality must not be claimed")
    return {
        "input_boundary": str(report.get("input_boundary") or ""),
        "final_boundary": final_boundary,
        "next_boundary": next_boundary,
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "review_input_present": bool(review.get("review_input_present", False)),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "broad_model_quality_claimed": bool(claim.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    objective = report["objective_repeatability"]
    review = report["review_boundary"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Final Decision",
        "",
        f"- input boundary: `{report['input_boundary']}`",
        f"- final boundary: `{decision['final_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- auto progress allowed: `{decision['auto_progress_allowed']}`",
        f"- critical user input required: `{decision['critical_user_input_required']}`",
        f"- human/audio preference claimed: `{claim['human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{claim['broad_model_quality_claimed']}`",
        "",
        "## Objective Repeatability",
        "",
        f"- source candidates: `{objective['source_candidate_count']}`",
        f"- qualified source candidates: `{objective['qualified_source_candidate_count']}`",
        f"- dead-air preserved source candidates: `{objective['dead_air_preserved_source_candidate_count']}`",
        f"- supported repair policies: `{objective['supported_repair_policy_count']}`",
        f"- total variants: `{objective['total_variant_count']}`",
        f"- qualified variants: `{objective['total_qualified_variant_count']}`",
        "",
        "## Review Boundary",
        "",
        f"- review input present: `{review['review_input_present']}`",
        f"- preference claim requires user review: `{review['preference_claim_requires_user_review']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide final boundary after outside-soloing repair repeatability")
    parser.add_argument(
        "--repeatability_consolidation",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_final_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_final_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_auto_progress_allowed", action="store_true")
    parser.add_argument("--require_no_critical_user_input", action="store_true")
    parser.add_argument("--require_no_preference_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_final_decision(
        read_json(Path(args.repeatability_consolidation)),
        output_dir=output_dir,
    )
    summary = validate_final_decision(
        report,
        expected_final_boundary=str(args.expected_final_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_auto_progress_allowed=bool(args.require_auto_progress_allowed),
        require_no_critical_user_input=bool(args.require_no_critical_user_input),
        require_no_preference_claim=bool(args.require_no_preference_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_final_decision.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_final_decision.md"
    write_json(report_path, report)
    write_json(
        output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_final_decision_validation_summary.json",
        summary,
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
