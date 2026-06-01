"""Record user listening review for sparse phrase repair WAV candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _int,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
    ValueError
):
    pass


AUDIO_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_local_audio_render_attempt"
)
REVIEW_BOUNDARY_REJECT_ALL = (
    "generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_audio_review_reject_all"
)
NEXT_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_rejection_analysis"
)
OVERALL_DECISIONS = {"keep_any", "reject_all", "unclear"}
CANDIDATE_DECISIONS = {"keep", "needs_followup", "reject", "unclear"}
PRIMARY_FAILURES = {"subjective_not_musical", "outside_or_unclear", "unclear"}
ATTRIBUTE_VALUES = {"acceptable", "outside_or_unclear", "weak", "not_musical", "unclear"}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_audio_render_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(report.get("audio_render_boundary"))
    if str(boundary.get("boundary") or "") != AUDIO_BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "unexpected sparse phrase audio render boundary"
        )
    if not bool(boundary.get("render_attempted", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "render attempt required"
        )
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "technical WAV validation required"
        )
    if bool(boundary.get("audio_rendered_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "source report must not claim audio quality"
        )
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "source report must not claim human preference"
        )
    if bool(boundary.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "source report must not claim musical quality"
        )
    rendered = [dict(item) for item in _list(report.get("rendered_audio_files")) if isinstance(item, dict)]
    if not rendered:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "rendered audio files required"
        )
    return rendered


def compact_rendered_item(item: dict[str, Any]) -> dict[str, Any]:
    wav_file = _dict(item.get("wav_file"))
    if not bool(wav_file.get("exists", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"missing WAV for rank {item.get('review_rank')}"
        )
    return {
        "review_rank": _int(item.get("review_rank")),
        "interval_cap": _int(item.get("interval_cap")),
        "sample_seed": _int(item.get("sample_seed")),
        "sample_index": _int(item.get("sample_index")),
        "source_midi_path": str(item.get("source_midi_path") or ""),
        "wav_path": str(wav_file.get("path") or ""),
        "duration_seconds": float(wav_file.get("duration_seconds", 0.0) or 0.0),
        "sample_rate": int(wav_file.get("sample_rate", 0) or 0),
        "sha256": str(wav_file.get("sha256") or ""),
    }


def build_candidate_reviews(
    items: list[dict[str, Any]],
    *,
    decision: str,
    primary_failure: str,
    assessment: str,
) -> list[dict[str, Any]]:
    if decision not in CANDIDATE_DECISIONS:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"invalid candidate decision: {decision}"
        )
    if primary_failure not in PRIMARY_FAILURES:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"invalid primary failure: {primary_failure}"
        )
    return [
        {
            "review_rank": item["review_rank"],
            "interval_cap": item["interval_cap"],
            "sample_seed": item["sample_seed"],
            "sample_index": item["sample_index"],
            "decision": decision,
            "primary_failure": primary_failure,
            "assessment": assessment.strip(),
        }
        for item in items
    ]


def build_user_listening_review(
    audio_render_report: dict[str, Any],
    *,
    output_dir: Path,
    reviewer: str,
    overall_decision: str,
    candidate_decision: str,
    primary_failure: str,
    timing: str,
    phrase: str,
    vocabulary: str,
    assessment: str,
    notes: str,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "reviewer is required"
        )
    if overall_decision not in OVERALL_DECISIONS:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"invalid overall decision: {overall_decision}"
        )
    if candidate_decision not in CANDIDATE_DECISIONS:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"invalid candidate decision: {candidate_decision}"
        )
    if primary_failure not in PRIMARY_FAILURES:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"invalid primary failure: {primary_failure}"
        )
    for field, value in (("timing", timing), ("phrase", phrase), ("vocabulary", vocabulary)):
        if value not in ATTRIBUTE_VALUES:
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
                f"invalid {field}: {value}"
            )
    rendered = validate_audio_render_report(audio_render_report)
    items = [compact_rendered_item(item) for item in rendered]
    keep_claimed = overall_decision == "keep_any"
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_user_listening_review_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_audio_render_schema": str(audio_render_report.get("schema_version") or ""),
        "reviewed_audio_files": items,
        "user_listening_review": {
            "status": "reviewed",
            "reviewer": reviewer.strip(),
            "review_basis": "single_user_listening_review_of_sparse_phrase_wav_files",
            "overall_decision": overall_decision,
            "candidate_decision": candidate_decision,
            "primary_failure": primary_failure,
            "timing": timing,
            "phrase": phrase,
            "vocabulary": vocabulary,
            "assessment": assessment.strip(),
            "notes": notes.strip(),
            "candidate_reviews": build_candidate_reviews(
                items,
                decision=candidate_decision,
                primary_failure=primary_failure,
                assessment=assessment,
            ),
        },
        "claim_boundary": {
            "single_user_review_completed": True,
            "human_audio_keep_claimed": keep_claimed,
            "human_audio_preference_claimed": keep_claimed,
            "human_audio_reject_all_recorded": overall_decision == "reject_all",
            "audio_render_used": True,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "boundary": (
                "generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
                "sparse_phrase_audio_review_single_user_keep_support"
                if keep_claimed
                else REVIEW_BOUNDARY_REJECT_ALL
            ),
        },
        "decision": {
            "current_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
                "sparse_phrase_user_listening_review_input"
            ),
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "repair_target": primary_failure,
        },
        "proven": [
            "single_user_audio_review_completed",
            "sparse_phrase_candidates_rejected",
        ],
        "not_proven": [
            "human_audio_keep",
            "multi_reviewer_preference",
            "audio_rendered_quality",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair phrase continuation range interval guard "
            "sparse phrase rejection analysis"
        ),
    }


def validate_user_listening_review(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_overall_decision: str | None,
    expected_primary_failure: str | None,
    expected_file_count: int,
    require_no_keep_claim: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    review = _dict(report.get("user_listening_review"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(claim.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    overall = str(review.get("overall_decision") or "")
    if expected_overall_decision and overall != expected_overall_decision:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"expected overall decision {expected_overall_decision}, got {overall}"
        )
    primary_failure = str(review.get("primary_failure") or "")
    if expected_primary_failure and primary_failure != expected_primary_failure:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"expected primary failure {expected_primary_failure}, got {primary_failure}"
        )
    reviewed_file_count = len(_list(report.get("reviewed_audio_files")))
    if reviewed_file_count != expected_file_count:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            f"expected reviewed file count {expected_file_count}, got {reviewed_file_count}"
        )
    if require_no_keep_claim and bool(claim.get("human_audio_keep_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
            "human/audio keep must not be claimed"
        )
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("audio_rendered_quality_claimed", True)),
            bool(claim.get("musical_quality_claimed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseUserListeningReviewError(
                "quality claims must not be set"
            )
    decision = _dict(report.get("decision"))
    return {
        "boundary": boundary,
        "review_status": str(review.get("status") or ""),
        "overall_decision": overall,
        "candidate_decision": str(review.get("candidate_decision") or ""),
        "primary_failure": primary_failure,
        "timing": str(review.get("timing") or ""),
        "phrase": str(review.get("phrase") or ""),
        "vocabulary": str(review.get("vocabulary") or ""),
        "reviewed_audio_file_count": reviewed_file_count,
        "human_audio_keep_claimed": bool(claim.get("human_audio_keep_claimed", True)),
        "human_audio_reject_all_recorded": bool(claim.get("human_audio_reject_all_recorded", False)),
        "audio_rendered_quality_claimed": bool(claim.get("audio_rendered_quality_claimed", True)),
        "musical_quality_claimed": bool(claim.get("musical_quality_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["user_listening_review"]
    claim = report["claim_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase User Listening Review",
        "",
        "## Summary",
        "",
        f"- boundary: `{claim['boundary']}`",
        f"- review status: `{review['status']}`",
        f"- overall decision: `{review['overall_decision']}`",
        f"- candidate decision: `{review['candidate_decision']}`",
        f"- primary failure: `{review['primary_failure']}`",
        f"- timing: `{review['timing']}`",
        f"- phrase: `{review['phrase']}`",
        f"- vocabulary: `{review['vocabulary']}`",
        f"- human/audio keep claimed: `{_bool_token(claim['human_audio_keep_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(claim['musical_quality_claimed'])}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        "",
        "## Assessment",
        "",
        f"- {review['assessment']}",
        f"- notes: {review['notes']}",
        "",
        "## Reviewed Files",
        "",
        "| rank | cap | seed | sample | duration | wav |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("reviewed_audio_files", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("review_rank", "")),
                    str(item.get("interval_cap", "")),
                    str(item.get("sample_seed", "")),
                    str(item.get("sample_index", "")),
                    f"{float(item.get('duration_seconds', 0.0) or 0.0):.3f}",
                    str(item.get("wav_path", "")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fill sparse phrase user listening review"
    )
    parser.add_argument(
        "--audio_render_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_local_audio_render_attempt/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_local_audio_render_attempt/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_local_audio_render_attempt.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_user_listening_review"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--reviewer", type=str, required=True)
    parser.add_argument("--overall_decision", type=str, required=True)
    parser.add_argument("--candidate_decision", type=str, required=True)
    parser.add_argument("--primary_failure", type=str, required=True)
    parser.add_argument("--timing", type=str, required=True)
    parser.add_argument("--phrase", type=str, required=True)
    parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--assessment", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_overall_decision", type=str, default="")
    parser.add_argument("--expected_primary_failure", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=3)
    parser.add_argument("--require_no_keep_claim", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_user_listening_review(
        read_json(Path(args.audio_render_report)),
        output_dir=output_dir,
        reviewer=str(args.reviewer),
        overall_decision=str(args.overall_decision),
        candidate_decision=str(args.candidate_decision),
        primary_failure=str(args.primary_failure),
        timing=str(args.timing),
        phrase=str(args.phrase),
        vocabulary=str(args.vocabulary),
        assessment=str(args.assessment),
        notes=str(args.notes),
    )
    summary = validate_user_listening_review(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_overall_decision=str(args.expected_overall_decision or ""),
        expected_primary_failure=str(args.expected_primary_failure or ""),
        expected_file_count=int(args.expected_file_count),
        require_no_keep_claim=bool(args.require_no_keep_claim),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_user_listening_review.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
