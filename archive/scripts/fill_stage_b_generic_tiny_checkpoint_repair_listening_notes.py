"""Fill or guard generic tiny checkpoint repair listening notes."""

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


class StageBGenericTinyCheckpointRepairListeningFillError(ValueError):
    pass


PHRASE_VALUES = {"natural", "acceptable", "stiff", "unclear"}
TIME_FEEL_VALUES = {"swinging", "acceptable", "stiff", "unclear"}
INSIDE_OUTSIDE_VALUES = {"inside", "outside_but_usable", "too_outside", "unclear"}
REPETITION_VALUES = {"lively", "acceptable", "repetitive", "unclear"}
KEEP_VALUES = {"keep", "needs_followup", "reject", "unclear"}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def note_items(listening_notes_report: dict[str, Any]) -> list[dict[str, Any]]:
    readiness = _dict(listening_notes_report.get("readiness"))
    notes = _dict(listening_notes_report.get("listening_notes"))
    if str(readiness.get("boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_listening_notes":
        raise StageBGenericTinyCheckpointRepairListeningFillError("unexpected listening notes boundary")
    if bool(readiness.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairListeningFillError("musical quality must not be claimed before fill")
    if bool(readiness.get("human_review_filled", True)):
        raise StageBGenericTinyCheckpointRepairListeningFillError("source notes must not already be filled")
    if str(notes.get("status") or "") != "pending_human_review":
        raise StageBGenericTinyCheckpointRepairListeningFillError("source notes must be pending human review")
    items = [dict(item) for item in _list(notes.get("notes")) if isinstance(item, dict)]
    if not items:
        raise StageBGenericTinyCheckpointRepairListeningFillError("listening notes must contain candidates")
    return items


def pending_review_for_note(note: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_rank": _int(note.get("review_rank")),
        "sample_seed": _int(note.get("sample_seed")),
        "sample_index": _int(note.get("sample_index")),
        "midi_path": str(note.get("midi_path") or ""),
        "status": "pending_review_input",
        "phrase_naturalness": "pending",
        "time_feel": "pending",
        "inside_outside_balance": "pending",
        "repetition_or_liveliness": "pending",
        "keep_decision": "pending",
        "notes": "",
    }


def validate_review_input(review_input: dict[str, Any], *, notes: list[dict[str, Any]]) -> dict[str, Any]:
    reviewer = str(review_input.get("reviewer") or "").strip()
    if not reviewer:
        raise StageBGenericTinyCheckpointRepairListeningFillError("reviewer is required")
    reviews = _list(review_input.get("candidate_reviews"))
    expected_ranks = {_int(note.get("review_rank")) for note in notes}
    if len(reviews) != len(notes):
        raise StageBGenericTinyCheckpointRepairListeningFillError("candidate review count mismatch")
    compact_reviews: list[dict[str, Any]] = []
    seen_ranks: set[int] = set()
    for review in reviews:
        if not isinstance(review, dict):
            raise StageBGenericTinyCheckpointRepairListeningFillError("candidate review must be an object")
        rank = _int(review.get("review_rank"))
        if rank not in expected_ranks:
            raise StageBGenericTinyCheckpointRepairListeningFillError(f"unexpected review rank: {rank}")
        if rank in seen_ranks:
            raise StageBGenericTinyCheckpointRepairListeningFillError(f"duplicate review rank: {rank}")
        seen_ranks.add(rank)
        phrase = str(review.get("phrase_naturalness") or "")
        time_feel = str(review.get("time_feel") or "")
        inside_outside = str(review.get("inside_outside_balance") or "")
        repetition = str(review.get("repetition_or_liveliness") or "")
        keep = str(review.get("keep_decision") or "")
        if phrase not in PHRASE_VALUES:
            raise StageBGenericTinyCheckpointRepairListeningFillError(f"invalid phrase_naturalness: {phrase}")
        if time_feel not in TIME_FEEL_VALUES:
            raise StageBGenericTinyCheckpointRepairListeningFillError(f"invalid time_feel: {time_feel}")
        if inside_outside not in INSIDE_OUTSIDE_VALUES:
            raise StageBGenericTinyCheckpointRepairListeningFillError(
                f"invalid inside_outside_balance: {inside_outside}"
            )
        if repetition not in REPETITION_VALUES:
            raise StageBGenericTinyCheckpointRepairListeningFillError(
                f"invalid repetition_or_liveliness: {repetition}"
            )
        if keep not in KEEP_VALUES:
            raise StageBGenericTinyCheckpointRepairListeningFillError(f"invalid keep_decision: {keep}")
        compact_reviews.append(
            {
                "review_rank": rank,
                "status": "reviewed",
                "phrase_naturalness": phrase,
                "time_feel": time_feel,
                "inside_outside_balance": inside_outside,
                "repetition_or_liveliness": repetition,
                "keep_decision": keep,
                "notes": str(review.get("notes") or ""),
            }
        )
    return {
        "reviewer": reviewer,
        "overall_notes": str(review_input.get("overall_notes") or ""),
        "candidate_reviews": sorted(compact_reviews, key=lambda item: item["review_rank"]),
    }


def build_review_rows(notes: list[dict[str, Any]], review_input: dict[str, Any] | None) -> list[dict[str, Any]]:
    if review_input is None:
        return [pending_review_for_note(note) for note in notes]
    validated = validate_review_input(review_input, notes=notes)
    review_by_rank = {review["review_rank"]: review for review in validated["candidate_reviews"]}
    rows: list[dict[str, Any]] = []
    for note in notes:
        rank = _int(note.get("review_rank"))
        rows.append(
            {
                "review_rank": rank,
                "sample_seed": _int(note.get("sample_seed")),
                "sample_index": _int(note.get("sample_index")),
                "midi_path": str(note.get("midi_path") or ""),
                **review_by_rank[rank],
            }
        )
    return rows


def build_listening_fill_report(
    *,
    run_dir: Path,
    listening_notes_report_path: Path,
    listening_notes_report: dict[str, Any],
    review_input: dict[str, Any] | None,
    review_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    review_input_present = review_input is not None
    keep_count = sum(1 for row in review_rows if row.get("keep_decision") == "keep")
    fill_status = "review_input_applied" if review_input_present else "pending_review_input"
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_listening_fill_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "listening_notes_report_path": str(listening_notes_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "review_input_present": review_input_present,
        },
        "source_summary": {
            "candidate_count": _int(_dict(listening_notes_report.get("listening_notes")).get("candidate_count")),
            "notes_status": str(_dict(listening_notes_report.get("listening_notes")).get("status") or ""),
        },
        "review_input_present": review_input_present,
        "fill_status": fill_status,
        "listening_fill": {
            "status": "reviewed" if review_input_present else "pending_review_input",
            "reviewer": str(review_input.get("reviewer") or "") if review_input_present else "",
            "candidate_count": len(review_rows),
            "keep_count": keep_count,
            "candidate_reviews": review_rows,
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_listening_fill",
            "human_review_filled": review_input_present,
            "pending_without_review_input": not review_input_present,
            "musical_quality_claimed": review_input_present and keep_count > 0,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_listening_fill",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_listening_consolidation"
                if review_input_present
                else "stage_b_generic_tiny_checkpoint_repair_audio_render_package"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "human_review_input_required_for_quality_claim": not review_input_present,
            "reason": (
                "human review input is required before musical quality or keep decisions are claimed"
                if not review_input_present
                else "validated human review input was applied to listening notes"
            ),
        },
        "not_proven": []
        if review_input_present
        else [
            "human_listening_result",
            "musical_quality",
            "unconstrained_raw_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair listening consolidation"
            if review_input_present
            else "Stage B generic tiny checkpoint repair audio render package"
        ),
    }


def validate_listening_fill_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_pending_without_input: bool,
    require_no_quality_without_input: bool,
    require_objective_auto_progress_allowed: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    fill = _dict(report.get("listening_fill"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairListeningFillError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    review_input_present = bool(report.get("review_input_present", False))
    if require_pending_without_input and not review_input_present:
        if str(report.get("fill_status") or "") != "pending_review_input":
            raise StageBGenericTinyCheckpointRepairListeningFillError("expected pending fill status")
        if str(fill.get("status") or "") != "pending_review_input":
            raise StageBGenericTinyCheckpointRepairListeningFillError("expected pending review status")
    if require_no_quality_without_input and not review_input_present:
        if bool(readiness.get("musical_quality_claimed", True)):
            raise StageBGenericTinyCheckpointRepairListeningFillError(
                "musical quality must not be claimed without input"
            )
    if require_objective_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBGenericTinyCheckpointRepairListeningFillError("objective auto progress must be allowed")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairListeningFillError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointRepairListeningFillError("Brad style adaptation must not be claimed")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "review_input_present": review_input_present,
        "fill_status": str(report.get("fill_status") or ""),
        "listening_fill_status": str(fill.get("status") or ""),
        "candidate_count": _int(fill.get("candidate_count")),
        "keep_count": _int(fill.get("keep_count")),
        "human_review_filled": bool(readiness.get("human_review_filled", True)),
        "pending_without_review_input": bool(readiness.get("pending_without_review_input", False)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    fill = report["listening_fill"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Listening Fill",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- review input present: `{_bool_token(report['review_input_present'])}`",
        f"- fill status: `{report['fill_status']}`",
        f"- listening fill status: `{fill['status']}`",
        f"- human review filled: `{_bool_token(readiness['human_review_filled'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        "",
        "## Candidate Reviews",
        "",
        f"- candidate count: `{fill['candidate_count']}`",
        f"- keep count: `{fill['keep_count']}`",
        "",
    ]
    for row in fill["candidate_reviews"]:
        lines.append(
            "- "
            f"rank `{row['review_rank']}` "
            f"seed `{row['sample_seed']}` "
            f"sample `{row['sample_index']}` "
            f"status `{row['status']}` "
            f"keep `{row['keep_decision']}`"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fill or guard generic tiny checkpoint repair listening notes")
    parser.add_argument(
        "--listening_notes_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_listening_notes/"
        "harness_stage_b_generic_tiny_checkpoint_repair_listening_notes/"
        "stage_b_generic_tiny_checkpoint_repair_listening_notes.json",
    )
    parser.add_argument("--review_input", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_repair_listening_fill")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=403)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_pending_without_input", action="store_true")
    parser.add_argument("--require_no_quality_without_input", action="store_true")
    parser.add_argument("--require_objective_auto_progress_allowed", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    listening_notes_report_path = Path(args.listening_notes_report)
    if not listening_notes_report_path.exists():
        raise StageBGenericTinyCheckpointRepairListeningFillError("listening notes report required")
    listening_notes_report = read_json(listening_notes_report_path)
    review_input = read_json(Path(args.review_input)) if args.review_input else None
    notes = note_items(listening_notes_report)
    review_rows = build_review_rows(notes, review_input)
    report = build_listening_fill_report(
        run_dir=run_dir,
        listening_notes_report_path=listening_notes_report_path,
        listening_notes_report=listening_notes_report,
        review_input=review_input,
        review_rows=review_rows,
        args=args,
    )
    summary = validate_listening_fill_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_pending_without_input=bool(args.require_pending_without_input),
        require_no_quality_without_input=bool(args.require_no_quality_without_input),
        require_objective_auto_progress_allowed=bool(args.require_objective_auto_progress_allowed),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_listening_fill.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_listening_fill_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_repair_listening_fill.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
