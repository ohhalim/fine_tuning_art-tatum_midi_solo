"""Fill or guard the duration/coverage fill human/audio review state."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillHumanAudioReviewFillError(ValueError):
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


def selected_item(boundary: dict[str, Any]) -> dict[str, Any]:
    items = boundary.get("review_items")
    if not isinstance(items, list):
        raise DurationCoverageFillHumanAudioReviewFillError("boundary must contain review_items")
    selected = [item for item in items if item.get("role") == "duration_coverage_fill_keep"]
    if len(selected) != 1:
        raise DurationCoverageFillHumanAudioReviewFillError("boundary must contain one duration fill keep item")
    return selected[0]


def validate_review_input(review_input: dict[str, Any], *, expected_candidate_id: str) -> dict[str, Any]:
    candidate_id = str(review_input.get("candidate_id") or "")
    if candidate_id != expected_candidate_id:
        raise DurationCoverageFillHumanAudioReviewFillError(
            f"expected review candidate {expected_candidate_id}, got {candidate_id}"
        )
    reviewer = str(review_input.get("reviewer") or "").strip()
    if not reviewer:
        raise DurationCoverageFillHumanAudioReviewFillError("reviewer is required for human/audio review fill")
    if not bool(review_input.get("audio_render_used", False)):
        raise DurationCoverageFillHumanAudioReviewFillError("audio_render_used must be true for review fill")
    preference = str(review_input.get("preference") or "")
    if preference not in PREFERENCE_VALUES:
        raise DurationCoverageFillHumanAudioReviewFillError(f"invalid preference: {preference}")
    timing = str(review_input.get("timing") or "")
    phrase = str(review_input.get("phrase") or "")
    vocabulary = str(review_input.get("vocabulary") or "")
    for field, value in (("timing", timing), ("phrase", phrase), ("vocabulary", vocabulary)):
        if value not in ATTRIBUTE_VALUES:
            raise DurationCoverageFillHumanAudioReviewFillError(f"invalid {field}: {value}")
    return {
        "candidate_id": candidate_id,
        "status": "reviewed",
        "reviewer": reviewer,
        "audio_render_used": True,
        "preference": preference,
        "timing": timing,
        "phrase": phrase,
        "vocabulary": vocabulary,
        "notes": str(review_input.get("notes") or ""),
    }


def pending_review(*, candidate_id: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "status": "pending",
        "reviewer": "",
        "audio_render_used": False,
        "preference": "pending",
        "timing": "pending",
        "phrase": "pending",
        "vocabulary": "pending",
        "notes": "",
    }


def build_review_fill(
    boundary: dict[str, Any],
    review_input: dict[str, Any] | None,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    item = selected_item(boundary)
    candidate_id = str(item.get("candidate_id") or "")
    boundary_status = str(boundary.get("human_audio_boundary", {}).get("status") or "")
    if boundary_status != "pending":
        raise DurationCoverageFillHumanAudioReviewFillError(f"expected pending boundary, got {boundary_status}")
    if review_input is None:
        fill_status = "pending_review_input"
        review = pending_review(candidate_id=candidate_id)
        preference_claimed = False
    else:
        fill_status = "review_input_applied"
        review = validate_review_input(review_input, expected_candidate_id=candidate_id)
        preference_claimed = True
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_fill_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_boundary_schema": str(boundary.get("schema_version") or ""),
        "candidate_id": candidate_id,
        "review_input_present": review_input is not None,
        "fill_status": fill_status,
        "human_audio_review": review,
        "claim_boundary": {
            "preference_claimed": preference_claimed,
            "pending_without_review_input": review_input is None,
            "requires_human_or_audio_review_input": review_input is None,
        },
        "not_proven": []
        if review_input is not None
        else [
            "human_audio_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill audio review package"
            if review_input is None
            else "Stage B margin-recovered phrase/vocabulary duration coverage fill review consolidation"
        ),
    }


def validate_review_fill(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    require_pending_without_input: bool,
    require_no_preference_without_input: bool,
) -> dict[str, Any]:
    candidate_id = str(report.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillHumanAudioReviewFillError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    review_input_present = bool(report.get("review_input_present", False))
    claim = report.get("claim_boundary") if isinstance(report.get("claim_boundary"), dict) else {}
    review = report.get("human_audio_review") if isinstance(report.get("human_audio_review"), dict) else {}
    if require_pending_without_input and not review_input_present:
        if str(report.get("fill_status") or "") != "pending_review_input":
            raise DurationCoverageFillHumanAudioReviewFillError("expected pending_review_input fill status")
        if str(review.get("status") or "") != "pending":
            raise DurationCoverageFillHumanAudioReviewFillError("expected pending review status")
    if require_no_preference_without_input and not review_input_present:
        if bool(claim.get("preference_claimed", True)):
            raise DurationCoverageFillHumanAudioReviewFillError("preference must not be claimed without input")
        if str(review.get("preference") or "") != "pending":
            raise DurationCoverageFillHumanAudioReviewFillError("preference must remain pending without input")
    return {
        "candidate_id": candidate_id,
        "review_input_present": review_input_present,
        "fill_status": str(report.get("fill_status") or ""),
        "human_audio_status": str(review.get("status") or ""),
        "preference": str(review.get("preference") or ""),
        "preference_claimed": bool(claim.get("preference_claimed", False)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["human_audio_review"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Human/Audio Review Fill",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- review input present: `{report['review_input_present']}`",
        f"- fill status: `{report['fill_status']}`",
        f"- human/audio status: `{review['status']}`",
        f"- preference: `{review['preference']}`",
        f"- preference claimed: `{claim['preference_claimed']}`",
        "",
        "No preference is claimed unless a validated human/audio review input is provided.",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill or guard duration/coverage fill human/audio review state")
    parser.add_argument("--human_audio_boundary", type=str, required=True)
    parser.add_argument("--review_input", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_fill",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--require_pending_without_input", action="store_true")
    parser.add_argument("--require_no_preference_without_input", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    review_input = read_json(Path(args.review_input)) if args.review_input else None
    report = build_review_fill(
        read_json(Path(args.human_audio_boundary)),
        review_input,
        output_dir=output_dir,
    )
    summary = validate_review_fill(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        require_pending_without_input=bool(args.require_pending_without_input),
        require_no_preference_without_input=bool(args.require_no_preference_without_input),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_human_audio_review_fill.json"
    markdown_path = output_dir / "duration_coverage_fill_human_audio_review_fill.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_human_audio_review_fill_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
