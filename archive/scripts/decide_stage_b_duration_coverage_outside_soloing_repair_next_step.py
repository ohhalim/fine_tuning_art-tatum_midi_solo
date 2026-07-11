"""Decide next step after outside-soloing repair objective evidence consolidation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairNextDecisionError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def validate_inputs(objective_evidence: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(objective_evidence.get("objective_evidence_summary"))
    claim = _dict(objective_evidence.get("claim_boundary"))
    boundary = str(summary.get("boundary") or "")
    if boundary != "outside_soloing_repair_objective_evidence_support":
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError(
            f"objective support boundary required, got {boundary}"
        )
    if int(summary.get("source_candidate_count", 0) or 0) < 2:
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError("two source candidates required")
    if int(summary.get("qualified_source_candidate_count", 0) or 0) < 2:
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError("two qualified source candidates required")
    if bool(claim.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError(
            "human/audio preference must not be claimed"
        )
    if bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError(
            "broad model quality must not be claimed"
        )
    return summary


def build_outside_soloing_repair_next_decision(
    objective_evidence: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    summary = validate_inputs(objective_evidence)
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_next_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_objective_evidence_schema": str(objective_evidence.get("schema_version") or ""),
        "input_boundary": str(summary.get("boundary") or ""),
        "objective_evidence": {
            "source_candidate_count": int(summary.get("source_candidate_count", 0) or 0),
            "qualified_source_candidate_count": int(summary.get("qualified_source_candidate_count", 0) or 0),
            "dead_air_preserved_source_candidate_count": int(
                summary.get("dead_air_preserved_source_candidate_count", 0) or 0
            ),
            "chord_tone_pass_source_candidate_count": int(
                summary.get("chord_tone_pass_source_candidate_count", 0) or 0
            ),
            "non_chord_run_pass_source_candidate_count": int(
                summary.get("non_chord_run_pass_source_candidate_count", 0) or 0
            ),
            "interval_pass_source_candidate_count": int(summary.get("interval_pass_source_candidate_count", 0) or 0),
            "selected_min_chord_tone_ratio": float(summary.get("selected_min_chord_tone_ratio", 0.0) or 0.0),
            "selected_max_non_chord_tone_run": int(summary.get("selected_max_non_chord_tone_run", 0) or 0),
            "selected_max_interval": int(summary.get("selected_max_interval", 0) or 0),
        },
        "decision": {
            "next_boundary": "outside_soloing_repair_broader_repeatability_sweep",
            "reason": (
                "selected repaired sources have objective pitch-role support, but listening preference remains "
                "pending; broaden objective repeatability before any preference or broad quality claim"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "blocked_reason": "",
        },
        "selection_constraints": {
            "require_dead_air_preservation": True,
            "require_chord_tone_ratio_gate": True,
            "require_non_chord_run_gate": True,
            "require_interval_gate": True,
            "do_not_claim_human_audio_preference": True,
            "do_not_claim_broad_model_quality": True,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_next_decision",
            "human_audio_preference_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "not_proven": [
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair "
            "broader repeatability sweep"
        ),
    }


def validate_outside_soloing_repair_next_decision(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None,
    require_auto_progress_allowed: bool,
    require_no_critical_user_input: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError("auto progress must be allowed")
    if require_no_critical_user_input and bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError("critical user input must not be required")
    if require_no_broad_quality_claim and bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairNextDecisionError("broad model quality must not be claimed")
    return {
        "input_boundary": str(report.get("input_boundary") or ""),
        "next_boundary": next_boundary,
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "broad_model_quality_claimed": bool(claim.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    claim = report["claim_boundary"]
    objective = report["objective_evidence"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Next Decision",
        "",
        f"- input boundary: `{report['input_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- auto progress allowed: `{decision['auto_progress_allowed']}`",
        f"- critical user input required: `{decision['critical_user_input_required']}`",
        f"- human/audio preference claimed: `{claim['human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{claim['broad_model_quality_claimed']}`",
        "",
        "## Objective Evidence",
        "",
        f"- source candidates: `{objective['source_candidate_count']}`",
        f"- qualified source candidates: `{objective['qualified_source_candidate_count']}`",
        f"- dead-air preserved source candidates: `{objective['dead_air_preserved_source_candidate_count']}`",
        f"- chord-tone pass source candidates: `{objective['chord_tone_pass_source_candidate_count']}`",
        f"- non-chord run pass source candidates: `{objective['non_chord_run_pass_source_candidate_count']}`",
        f"- interval pass source candidates: `{objective['interval_pass_source_candidate_count']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide next step after outside-soloing repair objective evidence")
    parser.add_argument(
        "--objective_evidence",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_objective_evidence_consolidation.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_auto_progress_allowed", action="store_true")
    parser.add_argument("--require_no_critical_user_input", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_outside_soloing_repair_next_decision(
        read_json(Path(args.objective_evidence)),
        output_dir=output_dir,
    )
    summary = validate_outside_soloing_repair_next_decision(
        report,
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_auto_progress_allowed=bool(args.require_auto_progress_allowed),
        require_no_critical_user_input=bool(args.require_no_critical_user_input),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_next_decision.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_next_decision.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_next_decision_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
