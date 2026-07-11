"""Consolidate outside-soloing repair objective and repeatability boundaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_inputs(
    *,
    objective_evidence: dict[str, Any],
    broader_repeatability: dict[str, Any],
    user_review_fill: dict[str, Any],
) -> None:
    objective_summary = _dict(objective_evidence.get("objective_evidence_summary"))
    objective_claim = _dict(objective_evidence.get("claim_boundary"))
    repeatability_summary = _dict(broader_repeatability.get("repeatability_summary"))
    repeatability_claim = _dict(broader_repeatability.get("claim_boundary"))
    review_claim = _dict(user_review_fill.get("claim_boundary"))
    review_decision = _dict(user_review_fill.get("decision"))

    if str(objective_summary.get("boundary") or "") != "outside_soloing_repair_objective_evidence_support":
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "objective evidence support boundary required"
        )
    if not bool(objective_claim.get("objective_midi_evidence_claimed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "objective MIDI evidence claim required"
        )
    if bool(objective_claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "objective evidence must not claim human/audio preference"
        )
    if bool(objective_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "objective evidence must not claim broad model quality"
        )

    if str(repeatability_summary.get("boundary") or "") != "outside_soloing_repair_policy_repeatability_support":
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "policy repeatability support boundary required"
        )
    if not bool(repeatability_claim.get("policy_repeatability_claimed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "policy repeatability claim required"
        )
    if bool(repeatability_claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "repeatability sweep must not claim human/audio preference"
        )
    if bool(repeatability_claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "repeatability sweep must not claim broad model quality"
        )

    if str(review_claim.get("boundary") or "") != "outside_soloing_repair_audio_review_pending":
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "pending audio review boundary required"
        )
    if bool(user_review_fill.get("review_input_present", True)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "review input must remain absent for objective-only consolidation"
        )
    if bool(review_claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "pending review must not claim human/audio preference"
        )
    if not bool(review_decision.get("objective_auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "objective auto progress must be allowed"
        )


def build_repeatability_consolidation_report(
    *,
    objective_evidence: dict[str, Any],
    broader_repeatability: dict[str, Any],
    user_review_fill: dict[str, Any],
    output_dir: Path,
    min_source_candidates: int,
    min_policy_repeatability_count: int,
) -> dict[str, Any]:
    validate_inputs(
        objective_evidence=objective_evidence,
        broader_repeatability=broader_repeatability,
        user_review_fill=user_review_fill,
    )

    objective_summary = _dict(objective_evidence.get("objective_evidence_summary"))
    repeatability_summary = _dict(broader_repeatability.get("repeatability_summary"))
    review_claim = _dict(user_review_fill.get("claim_boundary"))

    objective_source_count = int(objective_summary.get("source_candidate_count", 0) or 0)
    objective_qualified_count = int(objective_summary.get("qualified_source_candidate_count", 0) or 0)
    objective_dead_air_count = int(objective_summary.get("dead_air_preserved_source_candidate_count", 0) or 0)
    objective_chord_count = int(objective_summary.get("chord_tone_pass_source_candidate_count", 0) or 0)
    objective_non_chord_count = int(objective_summary.get("non_chord_run_pass_source_candidate_count", 0) or 0)
    objective_interval_count = int(objective_summary.get("interval_pass_source_candidate_count", 0) or 0)
    supported_policy_count = int(repeatability_summary.get("supported_repair_policy_count", 0) or 0)
    total_variant_count = int(repeatability_summary.get("total_variant_count", 0) or 0)
    total_qualified_variant_count = int(repeatability_summary.get("total_qualified_variant_count", 0) or 0)

    selected_source_support = (
        objective_source_count >= int(min_source_candidates)
        and objective_qualified_count >= int(min_source_candidates)
        and objective_dead_air_count >= int(min_source_candidates)
        and objective_chord_count >= int(min_source_candidates)
        and objective_non_chord_count >= int(min_source_candidates)
        and objective_interval_count >= int(min_source_candidates)
    )
    policy_repeatability_support = supported_policy_count >= int(min_policy_repeatability_count)
    objective_repeatability_support = selected_source_support and policy_repeatability_support
    boundary = (
        "outside_soloing_repair_objective_repeatability_support"
        if objective_repeatability_support
        else "outside_soloing_repair_objective_repeatability_incomplete"
    )

    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schemas": {
            "objective_evidence": str(objective_evidence.get("schema_version") or ""),
            "broader_repeatability": str(broader_repeatability.get("schema_version") or ""),
            "user_review_fill": str(user_review_fill.get("schema_version") or ""),
        },
        "thresholds": {
            "min_source_candidates": int(min_source_candidates),
            "min_policy_repeatability_count": int(min_policy_repeatability_count),
        },
        "input_boundaries": {
            "objective_evidence": str(objective_summary.get("boundary") or ""),
            "broader_repeatability": str(repeatability_summary.get("boundary") or ""),
            "user_review": str(review_claim.get("boundary") or ""),
        },
        "selected_source_objective_support": {
            "source_candidate_count": objective_source_count,
            "qualified_source_candidate_count": objective_qualified_count,
            "dead_air_preserved_source_candidate_count": objective_dead_air_count,
            "chord_tone_pass_source_candidate_count": objective_chord_count,
            "non_chord_run_pass_source_candidate_count": objective_non_chord_count,
            "interval_pass_source_candidate_count": objective_interval_count,
            "selected_min_chord_tone_ratio": float(
                objective_summary.get("selected_min_chord_tone_ratio", 0.0) or 0.0
            ),
            "selected_max_non_chord_tone_run": int(
                objective_summary.get("selected_max_non_chord_tone_run", 0) or 0
            ),
            "selected_max_interval": int(objective_summary.get("selected_max_interval", 0) or 0),
            "support_claimed": bool(selected_source_support),
        },
        "policy_repeatability_support": {
            "source_candidate_count": int(repeatability_summary.get("source_candidate_count", 0) or 0),
            "repair_policy_count": int(repeatability_summary.get("repair_policy_count", 0) or 0),
            "supported_repair_policy_count": supported_policy_count,
            "total_variant_count": total_variant_count,
            "total_qualified_variant_count": total_qualified_variant_count,
            "selected_min_chord_tone_ratio": float(
                repeatability_summary.get("selected_min_chord_tone_ratio", 0.0) or 0.0
            ),
            "selected_max_non_chord_tone_run": int(
                repeatability_summary.get("selected_max_non_chord_tone_run", 0) or 0
            ),
            "selected_max_interval": int(repeatability_summary.get("selected_max_interval", 0) or 0),
            "support_claimed": bool(policy_repeatability_support),
        },
        "pending_user_review": {
            "review_input_present": bool(user_review_fill.get("review_input_present", False)),
            "fill_status": str(user_review_fill.get("fill_status") or ""),
            "boundary": str(review_claim.get("boundary") or ""),
            "human_audio_preference_claimed": False,
        },
        "consolidated_claim_boundary": {
            "boundary": boundary,
            "objective_repair_repeatability_claimed": bool(objective_repeatability_support),
            "selected_source_objective_support_claimed": bool(selected_source_support),
            "policy_repeatability_claimed": bool(policy_repeatability_support),
            "human_audio_preference_claimed": False,
            "multi_reviewer_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "selected_source_objective_repair_support",
            "policy_level_objective_repeatability_support",
            "pending_review_boundary_preserved",
        ]
        if objective_repeatability_support
        else [],
        "not_proven": [
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair final decision"
        ),
    }


def validate_repeatability_consolidation(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_source_candidates: int,
    min_policy_repeatability_count: int,
    require_pending_review_guard: bool,
    require_no_preference_claim: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("consolidated_claim_boundary"))
    objective = _dict(report.get("selected_source_objective_support"))
    repeatability = _dict(report.get("policy_repeatability_support"))
    review = _dict(report.get("pending_user_review"))
    boundary_name = str(boundary.get("boundary") or "")

    if expected_boundary and boundary_name != expected_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary_name}"
        )
    if int(objective.get("source_candidate_count", 0) or 0) < int(min_source_candidates):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "not enough objective source candidates"
        )
    if int(repeatability.get("supported_repair_policy_count", 0) or 0) < int(min_policy_repeatability_count):
        raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
            "not enough supported repair policies"
        )
    if require_pending_review_guard:
        if bool(review.get("review_input_present", True)):
            raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
                "review input must remain absent"
            )
        if str(review.get("boundary") or "") != "outside_soloing_repair_audio_review_pending":
            raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
                "pending review boundary required"
            )
    if require_no_preference_claim:
        blocked = ["human_audio_preference_claimed", "multi_reviewer_preference_claimed"]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
                f"unexpected preference claim: {claimed}"
            )
    if require_no_broad_quality_claim:
        blocked = [
            "broad_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBDurationCoverageOutsideSoloingRepairRepeatabilityConsolidationError(
                f"unexpected broad claim: {claimed}"
            )

    return {
        "boundary": boundary_name,
        "objective_repair_repeatability_claimed": bool(
            boundary.get("objective_repair_repeatability_claimed", False)
        ),
        "objective_source_candidate_count": int(objective.get("source_candidate_count", 0) or 0),
        "qualified_source_candidate_count": int(objective.get("qualified_source_candidate_count", 0) or 0),
        "dead_air_preserved_source_candidate_count": int(
            objective.get("dead_air_preserved_source_candidate_count", 0) or 0
        ),
        "supported_repair_policy_count": int(repeatability.get("supported_repair_policy_count", 0) or 0),
        "total_variant_count": int(repeatability.get("total_variant_count", 0) or 0),
        "total_qualified_variant_count": int(repeatability.get("total_qualified_variant_count", 0) or 0),
        "review_input_present": bool(review.get("review_input_present", False)),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "broad_model_quality_claimed": bool(boundary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["consolidated_claim_boundary"]
    objective = report["selected_source_objective_support"]
    repeatability = report["policy_repeatability_support"]
    review = report["pending_user_review"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Repeatability Consolidation",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- objective repair repeatability claimed: `{boundary['objective_repair_repeatability_claimed']}`",
        f"- human/audio preference claimed: `{boundary['human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{boundary['broad_model_quality_claimed']}`",
        "",
        "## Selected Source Objective Support",
        "",
        f"- source candidates: `{objective['source_candidate_count']}`",
        f"- qualified source candidates: `{objective['qualified_source_candidate_count']}`",
        f"- dead-air preserved source candidates: `{objective['dead_air_preserved_source_candidate_count']}`",
        f"- chord-tone pass source candidates: `{objective['chord_tone_pass_source_candidate_count']}`",
        f"- non-chord run pass source candidates: `{objective['non_chord_run_pass_source_candidate_count']}`",
        f"- interval pass source candidates: `{objective['interval_pass_source_candidate_count']}`",
        "",
        "## Policy Repeatability Support",
        "",
        f"- repair policies supported: `{repeatability['supported_repair_policy_count']}`",
        f"- total variants: `{repeatability['total_variant_count']}`",
        f"- qualified variants: `{repeatability['total_qualified_variant_count']}`",
        f"- selected min chord-tone ratio: `{repeatability['selected_min_chord_tone_ratio']:.3f}`",
        f"- selected max non-chord run: `{repeatability['selected_max_non_chord_tone_run']}`",
        f"- selected max interval: `{repeatability['selected_max_interval']}`",
        "",
        "## Pending Review Boundary",
        "",
        f"- review input present: `{review['review_input_present']}`",
        f"- boundary: `{review['boundary']}`",
        f"- human/audio preference claimed: `{review['human_audio_preference_claimed']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate outside-soloing repair repeatability evidence")
    parser.add_argument(
        "--objective_evidence",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.json",
    )
    parser.add_argument(
        "--broader_repeatability",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.json",
    )
    parser.add_argument(
        "--user_review_fill",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_source_candidates", type=int, default=2)
    parser.add_argument("--min_policy_repeatability_count", type=int, default=3)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_pending_review_guard", action="store_true")
    parser.add_argument("--require_no_preference_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repeatability_consolidation_report(
        objective_evidence=read_json(Path(args.objective_evidence)),
        broader_repeatability=read_json(Path(args.broader_repeatability)),
        user_review_fill=read_json(Path(args.user_review_fill)),
        output_dir=output_dir,
        min_source_candidates=int(args.min_source_candidates),
        min_policy_repeatability_count=int(args.min_policy_repeatability_count),
    )
    summary = validate_repeatability_consolidation(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_source_candidates=int(args.min_source_candidates),
        min_policy_repeatability_count=int(args.min_policy_repeatability_count),
        require_pending_review_guard=bool(args.require_pending_review_guard),
        require_no_preference_claim=bool(args.require_no_preference_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation.md"
    write_json(report_path, report)
    write_json(
        output_dir
        / "stage_b_duration_coverage_fill_outside_soloing_repair_repeatability_consolidation_validation_summary.json",
        summary,
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
