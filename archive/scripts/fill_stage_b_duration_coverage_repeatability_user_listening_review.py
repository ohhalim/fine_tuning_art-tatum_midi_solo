"""Fill user listening review for duration/coverage repeatability audio package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageRepeatabilityUserListeningReviewError(ValueError):
    pass


OVERALL_DECISIONS = {"keep_any", "reject_all", "unclear"}
CANDIDATE_DECISIONS = {"keep", "needs_followup", "reject", "unclear"}
ATTRIBUTE_VALUES = {"acceptable", "outside_or_unclear", "weak", "unclear"}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def compact_review_item(item: dict[str, Any]) -> dict[str, Any]:
    wav_file = _dict(item.get("wav_file"))
    midi_file = _dict(item.get("midi_file"))
    metrics = _dict(item.get("metrics"))
    if not bool(wav_file.get("exists", False)):
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError(f"missing WAV for {item.get('role')}")
    return {
        "role": str(item.get("role") or ""),
        "candidate_id": str(item.get("candidate_id") or ""),
        "source_candidate_id": str(item.get("source_candidate_id") or ""),
        "sample_seed": int(item.get("sample_seed", 0) or 0),
        "midi_path": str(midi_file.get("path") or ""),
        "wav_path": str(wav_file.get("path") or ""),
        "duration_seconds": float(wav_file.get("duration_seconds", 0.0) or 0.0),
        "sample_rate": int(wav_file.get("sample_rate", 0) or 0),
        "sha256": str(wav_file.get("sha256") or ""),
        "selected_dead_air_ratio": float(metrics.get("selected_dead_air_ratio", 0.0) or 0.0),
        "selected_focused_unique_pitch_count": int(metrics.get("selected_focused_unique_pitch_count", 0) or 0),
    }


def build_candidate_reviews(items: list[dict[str, Any]], *, decision: str, assessment: str) -> list[dict[str, Any]]:
    if decision not in CANDIDATE_DECISIONS:
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError(f"invalid candidate decision: {decision}")
    return [
        {
            "role": item["role"],
            "candidate_id": item["candidate_id"],
            "sample_seed": item["sample_seed"],
            "decision": decision,
            "assessment": assessment.strip(),
        }
        for item in items
    ]


def build_repeatability_user_listening_review(
    audio_review_package: dict[str, Any],
    *,
    output_dir: Path,
    reviewer: str,
    overall_decision: str,
    candidate_decision: str,
    timing: str,
    phrase: str,
    vocabulary: str,
    assessment: str,
    notes: str,
) -> dict[str, Any]:
    boundary = _dict(audio_review_package.get("audio_review_boundary"))
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError("technical WAV validation is required")
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError("source package must not claim preference")
    if bool(boundary.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError("source package must not claim broad quality")
    if overall_decision not in OVERALL_DECISIONS:
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError(f"invalid overall decision: {overall_decision}")
    for field, value in (("timing", timing), ("phrase", phrase), ("vocabulary", vocabulary)):
        if value not in ATTRIBUTE_VALUES:
            raise StageBDurationCoverageRepeatabilityUserListeningReviewError(f"invalid {field}: {value}")
    items = [compact_review_item(item) for item in _list(audio_review_package.get("review_items")) if isinstance(item, dict)]
    if len(items) != 2:
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError(f"expected 2 review items, got {len(items)}")
    keep_claimed = overall_decision == "keep_any"
    return {
        "schema_version": "stage_b_duration_coverage_fill_repeatability_user_listening_review_fill_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_audio_review_package_schema": str(audio_review_package.get("schema_version") or ""),
        "candidate_id": str(audio_review_package.get("candidate_id") or ""),
        "reviewed_audio_files": items,
        "user_listening_review": {
            "status": "reviewed",
            "reviewer": reviewer.strip(),
            "review_basis": "user_listening_review_of_repeatability_wav_files",
            "overall_decision": overall_decision,
            "candidate_decision": candidate_decision,
            "timing": timing,
            "phrase": phrase,
            "vocabulary": vocabulary,
            "assessment": assessment.strip(),
            "notes": notes.strip(),
            "candidate_reviews": build_candidate_reviews(
                items,
                decision=candidate_decision,
                assessment=assessment,
            ),
        },
        "claim_boundary": {
            "single_user_review_completed": True,
            "repeatability_human_audio_keep_claimed": keep_claimed,
            "human_audio_preference_claimed": keep_claimed,
            "audio_render_used": True,
            "audio_rendered_quality_claimed": False,
            "broad_model_quality_claimed": False,
            "boundary": (
                "repeatability_audio_review_needs_followup"
                if not keep_claimed
                else "repeatability_audio_review_single_user_keep_support"
            ),
        },
        "proven": [
            "repeatability_wav_user_review_completed",
        ],
        "not_proven": [
            "repeatability_human_audio_keep",
            "multi_reviewer_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair decision"
        ),
    }


def validate_repeatability_user_listening_review(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_overall_decision: str | None,
    require_no_keep_claim: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    review = _dict(report.get("user_listening_review"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(claim.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    overall = str(review.get("overall_decision") or "")
    if expected_overall_decision and overall != expected_overall_decision:
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError(
            f"expected decision {expected_overall_decision}, got {overall}"
        )
    if require_no_keep_claim and bool(claim.get("repeatability_human_audio_keep_claimed", True)):
        raise StageBDurationCoverageRepeatabilityUserListeningReviewError("human/audio keep must not be claimed")
    if require_no_broad_quality_claim:
        blocked = [
            "audio_rendered_quality_claimed",
            "broad_model_quality_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBDurationCoverageRepeatabilityUserListeningReviewError(f"unexpected quality claim: {claimed}")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "boundary": boundary,
        "review_status": str(review.get("status") or ""),
        "overall_decision": overall,
        "candidate_decision": str(review.get("candidate_decision") or ""),
        "timing": str(review.get("timing") or ""),
        "phrase": str(review.get("phrase") or ""),
        "vocabulary": str(review.get("vocabulary") or ""),
        "repeatability_human_audio_keep_claimed": bool(
            claim.get("repeatability_human_audio_keep_claimed", True)
        ),
        "broad_model_quality_claimed": bool(claim.get("broad_model_quality_claimed", True)),
        "reviewed_audio_file_count": len(_list(report.get("reviewed_audio_files"))),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["user_listening_review"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Duration Coverage Fill Repeatability User Listening Review Fill",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- boundary: `{claim['boundary']}`",
        f"- review status: `{review['status']}`",
        f"- overall decision: `{review['overall_decision']}`",
        f"- candidate decision: `{review['candidate_decision']}`",
        f"- timing: `{review['timing']}`",
        f"- phrase: `{review['phrase']}`",
        f"- vocabulary: `{review['vocabulary']}`",
        f"- repeatability human/audio keep claimed: `{claim['repeatability_human_audio_keep_claimed']}`",
        f"- broad model quality claimed: `{claim['broad_model_quality_claimed']}`",
        "",
        "## Assessment",
        "",
        f"- {review['assessment']}",
        f"- notes: {review['notes']}",
        "",
        "## Reviewed Files",
        "",
    ]
    for item in report.get("reviewed_audio_files", []):
        lines.append(
            f"- sample seed `{item['sample_seed']}`: `{item['wav_path']}`"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill repeatability source user listening review")
    parser.add_argument("--audio_review_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_repeatability_user_listening_review_fill",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--reviewer", type=str, required=True)
    parser.add_argument("--overall_decision", type=str, required=True)
    parser.add_argument("--candidate_decision", type=str, required=True)
    parser.add_argument("--timing", type=str, required=True)
    parser.add_argument("--phrase", type=str, required=True)
    parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--assessment", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_overall_decision", type=str, default="")
    parser.add_argument("--require_no_keep_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repeatability_user_listening_review(
        read_json(Path(args.audio_review_package)),
        output_dir=output_dir,
        reviewer=str(args.reviewer),
        overall_decision=str(args.overall_decision),
        candidate_decision=str(args.candidate_decision),
        timing=str(args.timing),
        phrase=str(args.phrase),
        vocabulary=str(args.vocabulary),
        assessment=str(args.assessment),
        notes=str(args.notes),
    )
    summary = validate_repeatability_user_listening_review(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_overall_decision=str(args.expected_overall_decision or ""),
        require_no_keep_claim=bool(args.require_no_keep_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_repeatability_user_listening_review_fill.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_repeatability_user_listening_review_fill.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_repeatability_user_listening_review_fill_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
