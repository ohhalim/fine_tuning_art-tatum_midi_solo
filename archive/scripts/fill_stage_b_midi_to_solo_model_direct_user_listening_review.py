"""Fill model-direct user listening review from rendered WAV feedback."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.build_stage_b_midi_to_solo_model_direct_listening_review_package import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.guard_stage_b_midi_to_solo_model_direct_user_listening_review_input import (  # noqa: E402
    BOUNDARY as GUARD_BOUNDARY,
)


class StageBMidiToSoloModelDirectUserListeningReviewFillError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_direct_user_listening_review_fill"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_songlike_melody_rejection_analysis"
CLAIM_BOUNDARY = "model_direct_listening_review_songlike_rejection"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_user_listening_review_fill_v1"

OVERALL_DECISIONS = {"keep_any", "reject_all", "unclear"}
CANDIDATE_DECISIONS = {"keep", "relative_best_needs_followup", "needs_followup", "reject", "unclear"}
ATTRIBUTE_VALUES = {"acceptable", "songlike_not_soloing", "weak", "unclear"}
PRIMARY_FAILURES = {
    "songlike_melody_not_soloing",
    "simple_scale_melody",
    "not_jazz_phrase_vocabulary",
    "too_regular_contour",
    "rhythm_too_plain",
    "unclear",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def compact_rendered_candidate(item: dict[str, Any]) -> dict[str, Any]:
    wav_file = _dict(item.get("wav_file"))
    if not bool(wav_file.get("exists", False)):
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
            f"missing WAV for rank {item.get('rank')}"
        )
    return {
        "rank": _int(item.get("rank")),
        "sample_index": _int(item.get("sample_index")),
        "midi_path": str(item.get("package_midi_path") or item.get("source_midi_path") or ""),
        "wav_path": str(wav_file.get("path") or ""),
        "duration_seconds": _float(wav_file.get("duration_seconds")),
        "sample_rate": _int(wav_file.get("sample_rate")),
        "sha256": str(wav_file.get("sha256") or ""),
        "note_count": _int(item.get("source_note_count")),
        "unique_pitch_count": _int(item.get("source_unique_pitch_count")),
        "max_interval": _int(item.get("source_max_interval")),
        "dead_air_ratio": _float(item.get("source_dead_air_ratio")),
        "source_diagnostic_flags": _list(item.get("source_diagnostic_flags")),
    }


def validate_source_package(source_package: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(source_package.get("listening_review_package_boundary"))
    decision = _dict(source_package.get("decision"))
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("listening review package boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("source package must route to review fill")
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("technical WAV validation required")
    blocked_claims = [
        "listening_review_completed",
        "human_audio_preference_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(boundary.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(f"unexpected upstream claim: {claimed}")
    candidates = [
        compact_rendered_candidate(item)
        for item in _list(source_package.get("rendered_audio_files"))
        if isinstance(item, dict)
    ]
    if _int(boundary.get("candidate_count")) != len(candidates):
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("candidate count mismatch")
    if len(candidates) < 1:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("rendered candidates required")
    return candidates


def validate_guard_report(guard_report: dict[str, Any] | None) -> dict[str, Any]:
    if guard_report is None:
        return {
            "guard_report_used": False,
            "validated_review_input_present": None,
            "preference_fill_allowed": None,
        }
    if str(guard_report.get("boundary") or "") != GUARD_BOUNDARY:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("input guard boundary required")
    guard = _dict(guard_report.get("guard_result"))
    readiness = _dict(guard_report.get("readiness"))
    if bool(readiness.get("human_audio_preference_claimed", False)):
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("guard report must not claim preference")
    return {
        "guard_report_used": True,
        "validated_review_input_present": bool(guard.get("validated_review_input_present", False)),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
        "pending_status_field_count": _int(guard.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(guard.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(guard.get("pending_candidate_field_count")),
    }


def build_candidate_reviews(
    candidates: list[dict[str, Any]],
    *,
    preferred_rank: int,
    overall_decision: str,
    candidate_decision: str,
    primary_failure: str,
    assessment: str,
) -> list[dict[str, Any]]:
    if overall_decision not in OVERALL_DECISIONS:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
            f"invalid overall decision: {overall_decision}"
        )
    if candidate_decision not in CANDIDATE_DECISIONS:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
            f"invalid candidate decision: {candidate_decision}"
        )
    ranks = {item["rank"] for item in candidates}
    if preferred_rank not in ranks:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
            f"preferred rank not in candidates: {preferred_rank}"
        )
    reviews: list[dict[str, Any]] = []
    for item in candidates:
        if overall_decision == "reject_all":
            decision = candidate_decision if item["rank"] == preferred_rank else "reject"
            musical_acceptance = "reject"
        elif item["rank"] == preferred_rank:
            decision = "keep"
            musical_acceptance = "keep"
        else:
            decision = "needs_followup"
            musical_acceptance = "needs_followup"
        reviews.append(
            {
                "rank": item["rank"],
                "decision": decision,
                "musical_acceptance": musical_acceptance,
                "relative_best": item["rank"] == preferred_rank,
                "primary_failure": primary_failure,
                "assessment": assessment.strip(),
            }
        )
    return reviews


def build_user_listening_review_fill(
    source_package: dict[str, Any],
    *,
    output_dir: Path,
    reviewer: str,
    preferred_rank: int,
    overall_decision: str,
    candidate_decision: str,
    timing: str,
    phrase: str,
    vocabulary: str,
    primary_failure: str,
    assessment: str,
    notes: str,
    guard_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("reviewer is required")
    for field, value in (("timing", timing), ("phrase", phrase), ("vocabulary", vocabulary)):
        if value not in ATTRIBUTE_VALUES:
            raise StageBMidiToSoloModelDirectUserListeningReviewFillError(f"invalid {field}: {value}")
    if primary_failure not in PRIMARY_FAILURES:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
            f"invalid primary failure: {primary_failure}"
        )
    candidates = validate_source_package(source_package)
    guard_summary = validate_guard_report(guard_report)
    keep_claimed = overall_decision == "keep_any"
    candidate_reviews = build_candidate_reviews(
        candidates,
        preferred_rank=int(preferred_rank),
        overall_decision=overall_decision,
        candidate_decision=candidate_decision,
        primary_failure=primary_failure,
        assessment=assessment,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "reviewed_candidates": candidates,
        "input_guard_summary": guard_summary,
        "user_listening_review": {
            "status": "reviewed",
            "reviewer": reviewer.strip(),
            "review_basis": "single_user_listening_review_of_rendered_wav",
            "preferred_rank": int(preferred_rank),
            "relative_preference_recorded": True,
            "overall_decision": overall_decision,
            "candidate_decision_for_preferred_rank": candidate_decision,
            "timing": timing,
            "phrase": phrase,
            "vocabulary": vocabulary,
            "primary_failure": primary_failure,
            "assessment": assessment.strip(),
            "notes": notes.strip(),
            "candidate_reviews": candidate_reviews,
        },
        "claim_boundary": {
            "boundary": CLAIM_BOUNDARY if not keep_claimed else "model_direct_listening_review_single_user_keep",
            "single_user_review_completed": True,
            "relative_preference_recorded": True,
            "overall_reject_all": overall_decision == "reject_all",
            "human_audio_preference_claimed": keep_claimed,
            "model_direct_candidate_keep_claimed": keep_claimed,
            "audio_render_used": True,
            "audio_rendered_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_model_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "user_listening_review_completed": True,
            "relative_preference_recorded": True,
            "human_audio_preference_claimed": keep_claimed,
            "model_direct_candidate_keep_claimed": keep_claimed,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "single-user review rejected current model-direct candidates as songlike rather than soloing",
        },
        "proven": [
            "single_user_listening_review_recorded",
            "relative_best_rank_recorded",
            "current_candidate_set_rejected_for_songlike_melody",
        ],
        "not_proven": [
            "human_audio_keep_preference",
            "jazz_solo_musical_quality",
            "model_direct_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct songlike melody rejection analysis",
    }


def validate_user_listening_review_fill(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_overall_decision: str | None,
    expected_preferred_rank: int | None,
    require_review_completed: bool,
    require_no_keep_claim: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    review = _dict(report.get("user_listening_review"))
    claim = _dict(report.get("claim_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("unexpected next boundary")
    if expected_overall_decision and str(review.get("overall_decision") or "") != expected_overall_decision:
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("unexpected overall decision")
    if expected_preferred_rank is not None and _int(review.get("preferred_rank")) != int(expected_preferred_rank):
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("unexpected preferred rank")
    if require_review_completed and not bool(readiness.get("user_listening_review_completed", False)):
        raise StageBMidiToSoloModelDirectUserListeningReviewFillError("review completion required")
    if require_no_keep_claim:
        blocked = [
            "human_audio_preference_claimed",
            "model_direct_candidate_keep_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBMidiToSoloModelDirectUserListeningReviewFillError(f"unexpected keep claim: {claimed}")
    if require_no_quality_claim:
        blocked_quality = [
            "audio_rendered_quality_claimed",
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_model_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed_quality = [name for name in blocked_quality if bool(claim.get(name, True))]
        if claimed_quality:
            raise StageBMidiToSoloModelDirectUserListeningReviewFillError(
                f"unexpected quality claim: {claimed_quality}"
            )
    return {
        "boundary": boundary,
        "claim_boundary": str(claim.get("boundary") or ""),
        "review_status": str(review.get("status") or ""),
        "preferred_rank": _int(review.get("preferred_rank")),
        "relative_preference_recorded": bool(review.get("relative_preference_recorded", False)),
        "overall_decision": str(review.get("overall_decision") or ""),
        "candidate_decision_for_preferred_rank": str(review.get("candidate_decision_for_preferred_rank") or ""),
        "timing": str(review.get("timing") or ""),
        "phrase": str(review.get("phrase") or ""),
        "vocabulary": str(review.get("vocabulary") or ""),
        "primary_failure": str(review.get("primary_failure") or ""),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "model_direct_candidate_keep_claimed": bool(claim.get("model_direct_candidate_keep_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            claim.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "reviewed_candidate_count": len(_list(report.get("reviewed_candidates"))),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["user_listening_review"]
    claim = report["claim_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct User Listening Review Fill",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- claim boundary: `{claim['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- review status: `{review['status']}`",
        f"- preferred rank: `{review['preferred_rank']}`",
        f"- overall decision: `{review['overall_decision']}`",
        f"- primary failure: `{review['primary_failure']}`",
        f"- timing: `{review['timing']}`",
        f"- phrase: `{review['phrase']}`",
        f"- vocabulary: `{review['vocabulary']}`",
        f"- human/audio preference claimed: `{_bool_token(claim['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(claim['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Assessment",
        "",
        f"- {review['assessment']}",
        f"- notes: {review['notes']}",
        "",
        "## Candidate Reviews",
        "",
        "| rank | decision | relative best | musical acceptance | primary failure |",
        "|---:|---|---|---|---|",
    ]
    for item in review.get("candidate_reviews", []):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    f"`{item['decision']}`",
                    f"`{_bool_token(item['relative_best'])}`",
                    f"`{item['musical_acceptance']}`",
                    f"`{item['primary_failure']}`",
                ]
            )
            + " |"
        )
    lines.extend(["", "## Reviewed Files", ""])
    for item in report.get("reviewed_candidates", []):
        lines.append(f"- rank `{item['rank']}`: `{item['wav_path']}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill model-direct user listening review")
    parser.add_argument("--listening_review_package", type=str, required=True)
    parser.add_argument("--input_guard_report", type=str, default="")
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_user_listening_review_fill",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--reviewer", type=str, required=True)
    parser.add_argument("--preferred_rank", type=int, required=True)
    parser.add_argument("--overall_decision", type=str, required=True)
    parser.add_argument("--candidate_decision", type=str, required=True)
    parser.add_argument("--timing", type=str, required=True)
    parser.add_argument("--phrase", type=str, required=True)
    parser.add_argument("--vocabulary", type=str, required=True)
    parser.add_argument("--primary_failure", type=str, required=True)
    parser.add_argument("--assessment", type=str, required=True)
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_overall_decision", type=str, default="")
    parser.add_argument("--expected_preferred_rank", type=int, default=0)
    parser.add_argument("--require_review_completed", action="store_true")
    parser.add_argument("--require_no_keep_claim", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    guard_report = read_json(Path(args.input_guard_report)) if args.input_guard_report else None
    report = build_user_listening_review_fill(
        read_json(Path(args.listening_review_package)),
        output_dir=output_dir,
        reviewer=str(args.reviewer),
        preferred_rank=int(args.preferred_rank),
        overall_decision=str(args.overall_decision),
        candidate_decision=str(args.candidate_decision),
        timing=str(args.timing),
        phrase=str(args.phrase),
        vocabulary=str(args.vocabulary),
        primary_failure=str(args.primary_failure),
        assessment=str(args.assessment),
        notes=str(args.notes or ""),
        guard_report=guard_report,
    )
    summary = validate_user_listening_review_fill(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_overall_decision=str(args.expected_overall_decision or ""),
        expected_preferred_rank=int(args.expected_preferred_rank) if int(args.expected_preferred_rank) else None,
        require_review_completed=bool(args.require_review_completed),
        require_no_keep_claim=bool(args.require_no_keep_claim),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_user_listening_review_fill.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_user_listening_review_fill_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_user_listening_review_fill.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
