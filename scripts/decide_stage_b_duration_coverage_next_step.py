"""Decide next Stage B duration/coverage fill boundary after user review consolidation."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageNextDecisionError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_next_decision(
    consolidation: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    boundary = (
        consolidation.get("consolidated_claim_boundary")
        if isinstance(consolidation.get("consolidated_claim_boundary"), dict)
        else {}
    )
    alignment = (
        consolidation.get("evidence_alignment")
        if isinstance(consolidation.get("evidence_alignment"), dict)
        else {}
    )
    preferred = str(boundary.get("preferred_candidate") or "")
    if preferred != "duration_coverage_fill_keep":
        raise StageBDurationCoverageNextDecisionError(f"unexpected preferred candidate: {preferred}")
    if not bool(alignment.get("same_preferred_candidate", False)):
        raise StageBDurationCoverageNextDecisionError("MIDI/user evidence must prefer the same candidate")
    if bool(boundary.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageNextDecisionError("broad model quality must not be claimed")
    if not bool(boundary.get("single_user_human_audio_preference_claimed", False)):
        raise StageBDurationCoverageNextDecisionError("single-user human/audio preference is required")
    decision = "broader_repeatability_sweep"
    return {
        "schema_version": "stage_b_duration_coverage_fill_next_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_consolidation_schema": str(consolidation.get("schema_version") or ""),
        "candidate_id": str(consolidation.get("candidate_id") or ""),
        "current_evidence": {
            "preferred_candidate": preferred,
            "midi_and_user_preference_aligned": True,
            "rendered_audio_file_count": int(alignment.get("rendered_audio_file_count", 0) or 0),
            "single_user_review": bool(alignment.get("single_user_review", False)),
            "broad_model_quality_claimed": False,
        },
        "decision": {
            "next_boundary": decision,
            "reason": "fill candidate has aligned MIDI evidence and single-user listening support, but repeatability is not proven",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "blocked_reason": "",
        },
        "repeatability_requirements": {
            "compare_against_current_keep": True,
            "require_midi_evidence_gate": True,
            "require_no_broad_quality_claim": True,
            "requires_new_user_review": False,
            "suggested_scope": "duration_coverage_fill_broader_repeatability_sweep",
        },
        "not_proven": [
            "multi_seed_repeatability",
            "multi_reviewer_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill broader repeatability sweep",
    }


def validate_next_decision(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None,
    require_auto_progress_allowed: bool,
    require_no_critical_user_input: bool,
) -> dict[str, Any]:
    decision = report.get("decision") if isinstance(report.get("decision"), dict) else {}
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBDurationCoverageNextDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageNextDecisionError("auto progress must be allowed")
    if require_no_critical_user_input and bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageNextDecisionError("critical user input must not be required")
    current = report.get("current_evidence") if isinstance(report.get("current_evidence"), dict) else {}
    if bool(current.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageNextDecisionError("broad model quality must not be claimed")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "next_boundary": next_boundary,
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "preferred_candidate": str(current.get("preferred_candidate") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    current = report["current_evidence"]
    decision = report["decision"]
    lines = [
        "# Stage B Duration Coverage Fill Next Decision",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- preferred candidate: `{current['preferred_candidate']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- auto progress allowed: `{decision['auto_progress_allowed']}`",
        f"- critical user input required: `{decision['critical_user_input_required']}`",
        f"- broad model quality claimed: `{current['broad_model_quality_claimed']}`",
        "",
        "## Reason",
        "",
        f"- {decision['reason']}",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide next Stage B duration coverage boundary")
    parser.add_argument("--consolidation", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_auto_progress_allowed", action="store_true")
    parser.add_argument("--require_no_critical_user_input", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_next_decision(read_json(Path(args.consolidation)), output_dir=output_dir)
    summary = validate_next_decision(
        report,
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_auto_progress_allowed=bool(args.require_auto_progress_allowed),
        require_no_critical_user_input=bool(args.require_no_critical_user_input),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_next_decision.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_next_decision.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_next_decision_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
