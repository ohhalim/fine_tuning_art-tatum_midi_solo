"""Fill user listening review from rendered duration/coverage fill WAV evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageUserListeningReviewFillError(ValueError):
    pass


PREFERENCE_VALUES = {
    "source_constrained_partial",
    "duration_coverage_fill_keep",
    "tie",
    "reject_both",
}

ATTRIBUTE_VALUES = {
    "source_constrained_partial",
    "duration_coverage_fill_keep",
    "tie",
    "unclear",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def validate_audio_render_report(audio_render_report: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = (
        audio_render_report.get("audio_render_boundary")
        if isinstance(audio_render_report.get("audio_render_boundary"), dict)
        else {}
    )
    if not bool(boundary.get("render_attempted", False)):
        raise StageBDurationCoverageUserListeningReviewFillError("audio render must be attempted")
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBDurationCoverageUserListeningReviewFillError("technical WAV validation is required")
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBDurationCoverageUserListeningReviewFillError("source audio render report must not claim preference")
    rendered_files = audio_render_report.get("rendered_audio_files")
    if not isinstance(rendered_files, list) or len(rendered_files) != 2:
        raise StageBDurationCoverageUserListeningReviewFillError("audio render report must contain two WAV files")
    compacted = []
    for item in rendered_files:
        wav_file = item.get("wav_file") if isinstance(item.get("wav_file"), dict) else {}
        if not bool(wav_file.get("exists", False)):
            raise StageBDurationCoverageUserListeningReviewFillError(f"missing WAV for {item.get('role')}")
        compacted.append(
            {
                "role": str(item.get("role") or ""),
                "candidate_id": str(item.get("candidate_id") or ""),
                "source_midi_path": str(item.get("source_midi_path") or ""),
                "wav_path": str(wav_file.get("path") or ""),
                "duration_seconds": float(wav_file.get("duration_seconds", 0.0) or 0.0),
                "sample_rate": int(wav_file.get("sample_rate", 0) or 0),
                "sha256": str(wav_file.get("sha256") or ""),
            }
        )
    return compacted


def build_user_listening_review_fill(
    audio_render_report: dict[str, Any],
    *,
    output_dir: Path,
    reviewer: str,
    preference: str,
    timing: str,
    phrase: str,
    vocabulary: str,
    source_assessment: str,
    fill_assessment: str,
    notes: str,
) -> dict[str, Any]:
    rendered_files = validate_audio_render_report(audio_render_report)
    if not reviewer.strip():
        raise StageBDurationCoverageUserListeningReviewFillError("reviewer is required")
    if preference not in PREFERENCE_VALUES:
        raise StageBDurationCoverageUserListeningReviewFillError(f"invalid preference: {preference}")
    for field, value in (("timing", timing), ("phrase", phrase), ("vocabulary", vocabulary)):
        if value not in ATTRIBUTE_VALUES:
            raise StageBDurationCoverageUserListeningReviewFillError(f"invalid {field}: {value}")
    candidate_id = str(audio_render_report.get("candidate_id") or "")
    return {
        "schema_version": "stage_b_duration_coverage_fill_user_listening_review_fill_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_audio_render_schema": str(audio_render_report.get("schema_version") or ""),
        "candidate_id": candidate_id,
        "rendered_audio_files": rendered_files,
        "user_listening_review": {
            "status": "reviewed",
            "reviewer": reviewer.strip(),
            "audio_render_used": True,
            "review_basis": "user_listening_review_of_rendered_wav",
            "preference": preference,
            "timing": timing,
            "phrase": phrase,
            "vocabulary": vocabulary,
            "source_assessment": source_assessment.strip(),
            "fill_assessment": fill_assessment.strip(),
            "notes": notes.strip(),
        },
        "claim_boundary": {
            "human_audio_preference_claimed": True,
            "single_user_review": True,
            "audio_render_used": True,
            "audio_rendered_quality_claimed": False,
            "broad_model_quality_claimed": False,
        },
        "proven": [
            "single_user_preference_for_duration_coverage_fill_keep",
            "source_fill_audio_review_completed",
        ],
        "not_proven": [
            "multi_reviewer_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill user listening review consolidation",
    }


def validate_user_listening_review_fill(
    report: dict[str, Any],
    *,
    expected_preference: str | None,
    require_human_audio_preference: bool,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    review = report.get("user_listening_review") if isinstance(report.get("user_listening_review"), dict) else {}
    preference = str(review.get("preference") or "")
    if expected_preference and preference != expected_preference:
        raise StageBDurationCoverageUserListeningReviewFillError(
            f"expected preference {expected_preference}, got {preference}"
        )
    claim = report.get("claim_boundary") if isinstance(report.get("claim_boundary"), dict) else {}
    if require_human_audio_preference and not bool(claim.get("human_audio_preference_claimed", False)):
        raise StageBDurationCoverageUserListeningReviewFillError("human/audio preference must be claimed")
    if require_no_broad_quality_claim and bool(claim.get("broad_model_quality_claimed", True)):
        raise StageBDurationCoverageUserListeningReviewFillError("broad model quality must not be claimed")
    if require_no_broad_quality_claim and bool(claim.get("audio_rendered_quality_claimed", True)):
        raise StageBDurationCoverageUserListeningReviewFillError("audio rendered quality must not be claimed")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "review_status": str(review.get("status") or ""),
        "preference": preference,
        "timing": str(review.get("timing") or ""),
        "phrase": str(review.get("phrase") or ""),
        "vocabulary": str(review.get("vocabulary") or ""),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", False)),
        "broad_model_quality_claimed": bool(claim.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["user_listening_review"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Duration Coverage Fill User Listening Review Fill",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- review status: `{review['status']}`",
        f"- reviewer: `{review['reviewer']}`",
        f"- preference: `{review['preference']}`",
        f"- timing: `{review['timing']}`",
        f"- phrase: `{review['phrase']}`",
        f"- vocabulary: `{review['vocabulary']}`",
        f"- human/audio preference claimed: `{claim['human_audio_preference_claimed']}`",
        f"- broad model quality claimed: `{claim['broad_model_quality_claimed']}`",
        "",
        "## Assessment",
        "",
        f"- source: {review['source_assessment']}",
        f"- fill: {review['fill_assessment']}",
        f"- notes: {review['notes']}",
        "",
        "## Rendered Files",
        "",
    ]
    for item in report.get("rendered_audio_files", []):
        lines.append(f"- `{item['role']}`: `{item['wav_path']}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill Stage B duration coverage user listening review")
    parser.add_argument("--audio_render_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_user_listening_review_fill",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--reviewer", type=str, required=True)
    parser.add_argument("--preference", type=str, required=True)
    parser.add_argument("--timing", type=str, required=True)
    parser.add_argument("--phrase", type=str, required=True)
    parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--source_assessment", type=str, required=True)
    parser.add_argument("--fill_assessment", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--expected_preference", type=str, default="")
    parser.add_argument("--require_human_audio_preference", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_user_listening_review_fill(
        read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        reviewer=str(args.reviewer),
        preference=str(args.preference),
        timing=str(args.timing),
        phrase=str(args.phrase),
        vocabulary=str(args.vocabulary),
        source_assessment=str(args.source_assessment),
        fill_assessment=str(args.fill_assessment),
        notes=str(args.notes or ""),
    )
    summary = validate_user_listening_review_fill(
        report,
        expected_preference=str(args.expected_preference or ""),
        require_human_audio_preference=bool(args.require_human_audio_preference),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_user_listening_review_fill.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_user_listening_review_fill.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_user_listening_review_fill_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
