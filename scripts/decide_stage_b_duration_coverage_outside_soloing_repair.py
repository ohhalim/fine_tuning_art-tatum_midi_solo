"""Decide next repair boundary after outside-soloing user review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairDecisionError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def build_outside_soloing_repair_decision(
    user_listening_review: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    review = _dict(user_listening_review.get("user_listening_review"))
    claim = _dict(user_listening_review.get("claim_boundary"))
    if str(claim.get("boundary") or "") != "repeatability_audio_review_needs_followup":
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("needs-followup audio boundary is required")
    if str(review.get("overall_decision") or "") != "reject_all":
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("overall reject_all decision is required")
    if bool(claim.get("repeatability_human_audio_keep_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("human/audio keep must not be claimed")
    if bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("broad model quality must not be claimed")
    candidate_reviews = _list(review.get("candidate_reviews"))
    if len(candidate_reviews) != 2:
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("expected two candidate reviews")
    if any(str(item.get("decision") or "") != "needs_followup" for item in candidate_reviews if isinstance(item, dict)):
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("all candidate reviews must need follow-up")

    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_user_listening_review_schema": str(user_listening_review.get("schema_version") or ""),
        "candidate_id": str(user_listening_review.get("candidate_id") or ""),
        "input_boundary": str(claim.get("boundary") or ""),
        "user_review_summary": {
            "overall_decision": str(review.get("overall_decision") or ""),
            "candidate_decision": str(review.get("candidate_decision") or ""),
            "timing": str(review.get("timing") or ""),
            "phrase": str(review.get("phrase") or ""),
            "vocabulary": str(review.get("vocabulary") or ""),
            "assessment": str(review.get("assessment") or ""),
            "reviewed_audio_file_count": len(_list(user_listening_review.get("reviewed_audio_files"))),
        },
        "decision": {
            "next_boundary": "outside_soloing_pitch_role_phrase_clarity_repair",
            "reason": "MIDI/dead-air repeatability passed, but user listening rejected both repeatability WAVs as difficult and outside-soloing-like",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "blocked_reason": "",
        },
        "repair_targets": [
            "reduce_outside_sounding_pitch_choices",
            "increase_chord_tone_or_guide_tone_landing",
            "limit_non_chord_tone_run_length",
            "penalize_large_interval_after_fill",
            "prefer_phrase_contour_resolution_over_density",
        ],
        "selection_constraints": {
            "keep_dead_air_gain_gate": True,
            "keep_monophonic_gate": True,
            "require_audio_review_after_repair": True,
            "do_not_claim_broad_model_quality": True,
        },
        "claim_boundary": {
            "boundary": "outside_soloing_repair_decision",
            "human_audio_keep_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "not_proven": [
            "human_audio_keep",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair sweep"
        ),
    }


def validate_outside_soloing_repair_decision(
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
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("auto progress must be allowed")
    if require_no_critical_user_input and bool(decision.get("critical_user_input_required", True)):
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("critical user input must not be required")
    if require_no_broad_quality_claim and bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairDecisionError("broad model quality must not be claimed")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "input_boundary": str(report.get("input_boundary") or ""),
        "next_boundary": next_boundary,
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "repair_target_count": len(_list(report.get("repair_targets"))),
        "human_audio_keep_claimed": bool(claim.get("human_audio_keep_claimed", True)),
        "broad_model_quality_claimed": bool(claim.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    review = report["user_review_summary"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair Decision",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- input boundary: `{report['input_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- auto progress allowed: `{decision['auto_progress_allowed']}`",
        f"- critical user input required: `{decision['critical_user_input_required']}`",
        f"- human/audio keep claimed: `{claim['human_audio_keep_claimed']}`",
        f"- broad model quality claimed: `{claim['broad_model_quality_claimed']}`",
        "",
        "## User Review",
        "",
        f"- overall decision: `{review['overall_decision']}`",
        f"- candidate decision: `{review['candidate_decision']}`",
        f"- timing: `{review['timing']}`",
        f"- phrase: `{review['phrase']}`",
        f"- vocabulary: `{review['vocabulary']}`",
        f"- assessment: {review['assessment']}",
        "",
        "## Repair Targets",
        "",
    ]
    for item in report.get("repair_targets", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide outside-soloing repair boundary")
    parser.add_argument(
        "--user_listening_review",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_repeatability_user_listening_review_fill/"
        "harness_stage_b_duration_coverage_fill_repeatability_user_listening_review_fill/"
        "stage_b_duration_coverage_fill_repeatability_user_listening_review_fill.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_decision",
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
    report = build_outside_soloing_repair_decision(
        read_json(Path(args.user_listening_review)),
        output_dir=output_dir,
    )
    summary = validate_outside_soloing_repair_decision(
        report,
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_auto_progress_allowed=bool(args.require_auto_progress_allowed),
        require_no_critical_user_input=bool(args.require_no_critical_user_input),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_decision.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_decision.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_decision_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
