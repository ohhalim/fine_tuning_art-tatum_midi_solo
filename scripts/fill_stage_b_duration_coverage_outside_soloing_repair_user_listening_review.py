"""Fill or guard outside-soloing repair user listening review state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(ValueError):
    pass


OVERALL_DECISIONS = {
    "keep_both",
    "prefer_sample_seed_155",
    "prefer_sample_seed_131",
    "reject_all",
    "unclear",
}
ATTRIBUTE_VALUES = {
    "improved",
    "acceptable",
    "outside_or_unclear",
    "needs_followup",
    "unclear",
}
CANDIDATE_DECISIONS = {"keep", "needs_followup", "reject", "unclear"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def review_items(audio_review_package: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(audio_review_package.get("audio_review_boundary"))
    if str(boundary.get("status") or "") != "ready_for_user_listening_review":
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            "audio review package must be ready for user listening review"
        )
    if bool(boundary.get("audio_rendered_quality_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            "audio rendered quality must not be claimed before review"
        )
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            "human/audio preference must not be claimed before review"
        )
    items = [dict(item) for item in _list(audio_review_package.get("review_items")) if isinstance(item, dict)]
    if len(items) != 2:
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            f"expected 2 review items, got {len(items)}"
        )
    for item in items:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
                f"missing wav for {item.get('role')}"
            )
    return items


def pending_candidate_reviews(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "role": str(item.get("role") or ""),
            "candidate_id": str(item.get("candidate_id") or ""),
            "sample_seed": int(item.get("sample_seed", 0) or 0),
            "decision": "pending",
            "notes": "",
        }
        for item in items
    ]


def validate_review_input(review_input: dict[str, Any], *, expected_candidate_id: str, items: list[dict[str, Any]]) -> dict[str, Any]:
    candidate_id = str(review_input.get("candidate_id") or "")
    if candidate_id != expected_candidate_id:
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            f"expected review candidate {expected_candidate_id}, got {candidate_id}"
        )
    reviewer = str(review_input.get("reviewer") or "").strip()
    if not reviewer:
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError("reviewer is required")
    if not bool(review_input.get("audio_render_used", False)):
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError("audio_render_used must be true")
    overall_decision = str(review_input.get("overall_decision") or "")
    if overall_decision not in OVERALL_DECISIONS:
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            f"invalid overall decision: {overall_decision}"
        )
    timing = str(review_input.get("timing") or "")
    phrase = str(review_input.get("phrase") or "")
    vocabulary = str(review_input.get("vocabulary") or "")
    for field, value in (("timing", timing), ("phrase", phrase), ("vocabulary", vocabulary)):
        if value not in ATTRIBUTE_VALUES:
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
                f"invalid {field}: {value}"
            )
    candidate_reviews = _list(review_input.get("candidate_reviews"))
    expected_roles = {str(item.get("role") or "") for item in items}
    if len(candidate_reviews) != len(items):
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError("candidate review count mismatch")
    compact_reviews: list[dict[str, Any]] = []
    for review in candidate_reviews:
        if not isinstance(review, dict):
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError("candidate review must be an object")
        role = str(review.get("role") or "")
        decision = str(review.get("decision") or "")
        if role not in expected_roles:
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(f"unexpected review role: {role}")
        if decision not in CANDIDATE_DECISIONS:
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
                f"invalid candidate decision: {decision}"
            )
        compact_reviews.append(
            {
                "role": role,
                "decision": decision,
                "notes": str(review.get("notes") or ""),
            }
        )
    return {
        "status": "reviewed",
        "reviewer": reviewer,
        "audio_render_used": True,
        "overall_decision": overall_decision,
        "timing": timing,
        "phrase": phrase,
        "vocabulary": vocabulary,
        "assessment": str(review_input.get("assessment") or ""),
        "candidate_reviews": compact_reviews,
    }


def pending_review(items: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "status": "pending_review_input",
        "reviewer": "",
        "audio_render_used": False,
        "overall_decision": "pending",
        "timing": "pending",
        "phrase": "pending",
        "vocabulary": "pending",
        "assessment": "",
        "candidate_reviews": pending_candidate_reviews(items),
    }


def build_user_listening_review_fill(
    audio_review_package: dict[str, Any],
    review_input: dict[str, Any] | None,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    items = review_items(audio_review_package)
    candidate_id = str(audio_review_package.get("candidate_id") or "")
    if candidate_id != "outside_soloing_repair_candidates":
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            f"unexpected candidate id: {candidate_id}"
        )
    if review_input is None:
        fill_status = "pending_review_input"
        review = pending_review(items)
        preference_claimed = False
    else:
        fill_status = "review_input_applied"
        review = validate_review_input(review_input, expected_candidate_id=candidate_id, items=items)
        preference_claimed = True
    return {
        "schema_version": "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_audio_review_package_schema": str(audio_review_package.get("schema_version") or ""),
        "candidate_id": candidate_id,
        "review_input_present": review_input is not None,
        "fill_status": fill_status,
        "reviewed_audio_files": [
            {
                "role": str(item.get("role") or ""),
                "candidate_id": str(item.get("candidate_id") or ""),
                "sample_seed": int(item.get("sample_seed", 0) or 0),
                "wav_path": str(_dict(item.get("wav_file")).get("path") or ""),
                "metrics": _dict(item.get("metrics")),
            }
            for item in items
        ],
        "user_listening_review": review,
        "decision": {
            "objective_auto_progress_allowed": True,
            "critical_user_input_required": False,
            "human_preference_input_required_for_preference_claim": review_input is None,
            "next_boundary": (
                "outside_soloing_repair_audio_review_pending"
                if review_input is None
                else "outside_soloing_repair_audio_review_applied"
            ),
        },
        "claim_boundary": {
            "boundary": (
                "outside_soloing_repair_audio_review_pending"
                if review_input is None
                else "outside_soloing_repair_audio_review_applied"
            ),
            "human_audio_preference_claimed": preference_claimed,
            "pending_without_review_input": review_input is None,
            "objective_only_followup_allowed": True,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "not_proven": []
        if review_input is not None
        else [
            "human_audio_preference",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair "
            "objective evidence consolidation"
            if review_input is None
            else "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair "
            "listening review consolidation"
        ),
    }


def validate_user_listening_review_fill(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_pending_without_input: bool,
    require_no_preference_without_input: bool,
    require_objective_auto_progress_allowed: bool,
) -> dict[str, Any]:
    claim = _dict(report.get("claim_boundary"))
    decision = _dict(report.get("decision"))
    review = _dict(report.get("user_listening_review"))
    boundary = str(claim.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_objective_auto_progress_allowed and not bool(decision.get("objective_auto_progress_allowed", False)):
        raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
            "objective auto progress must be allowed"
        )
    review_input_present = bool(report.get("review_input_present", False))
    if require_pending_without_input and not review_input_present:
        if str(report.get("fill_status") or "") != "pending_review_input":
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError("expected pending fill status")
        if str(review.get("status") or "") != "pending_review_input":
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError("expected pending review status")
    if require_no_preference_without_input and not review_input_present:
        if bool(claim.get("human_audio_preference_claimed", True)):
            raise StageBDurationCoverageOutsideSoloingRepairUserListeningReviewError(
                "preference must not be claimed without input"
            )
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "boundary": boundary,
        "review_input_present": review_input_present,
        "fill_status": str(report.get("fill_status") or ""),
        "user_listening_status": str(review.get("status") or ""),
        "overall_decision": str(review.get("overall_decision") or ""),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "objective_auto_progress_allowed": bool(decision.get("objective_auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["user_listening_review"]
    claim = report["claim_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B Duration Coverage Fill Outside-Soloing Repair User Listening Review Fill",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- review input present: `{report['review_input_present']}`",
        f"- fill status: `{report['fill_status']}`",
        f"- boundary: `{claim['boundary']}`",
        f"- user listening status: `{review['status']}`",
        f"- overall decision: `{review['overall_decision']}`",
        f"- human/audio preference claimed: `{claim['human_audio_preference_claimed']}`",
        f"- objective auto progress allowed: `{decision['objective_auto_progress_allowed']}`",
        f"- critical user input required: `{decision['critical_user_input_required']}`",
        "",
        "Human/audio preference is not claimed unless validated listening input is provided.",
        "",
        "## Reviewed Audio Files",
        "",
        "| role | sample seed | wav path |",
        "|---|---:|---|",
    ]
    for item in report.get("reviewed_audio_files", []):
        lines.append(
            "| {role} | {sample_seed} | `{wav}` |".format(
                role=item["role"],
                sample_seed=int(item["sample_seed"]),
                wav=item["wav_path"],
            )
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill or guard outside-soloing repair user listening review")
    parser.add_argument(
        "--audio_review_package",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/"
        "harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/"
        "stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.json",
    )
    parser.add_argument("--review_input", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_pending_without_input", action="store_true")
    parser.add_argument("--require_no_preference_without_input", action="store_true")
    parser.add_argument("--require_objective_auto_progress_allowed", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    review_input = read_json(Path(args.review_input)) if args.review_input else None
    report = build_user_listening_review_fill(
        read_json(Path(args.audio_review_package)),
        review_input,
        output_dir=output_dir,
    )
    summary = validate_user_listening_review_fill(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_pending_without_input=bool(args.require_pending_without_input),
        require_no_preference_without_input=bool(args.require_no_preference_without_input),
        require_objective_auto_progress_allowed=bool(args.require_objective_auto_progress_allowed),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.md"
    write_json(report_path, report)
    write_json(
        output_dir / "stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill_validation_summary.json",
        summary,
    )
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
