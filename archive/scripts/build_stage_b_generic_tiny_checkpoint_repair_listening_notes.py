"""Build pending listening notes for generic tiny checkpoint repair candidates."""

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
    _float,
    _int,
)


class StageBGenericTinyCheckpointRepairListeningNotesError(ValueError):
    pass


def build_listening_note(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "review_rank": _int(candidate.get("review_rank")),
        "sample_seed": _int(candidate.get("sample_seed")),
        "sample_index": _int(candidate.get("sample_index")),
        "midi_path": str(candidate.get("package_midi_path") or candidate.get("source_midi_path") or ""),
        "objective_context": {
            "dead_air_ratio": _float(candidate.get("dead_air_ratio")),
            "phrase_coverage_ratio": _float(candidate.get("phrase_coverage_ratio")),
            "chord_tone_ratio": _float(candidate.get("chord_tone_ratio")),
            "unique_pitch_count": _int(candidate.get("unique_pitch_count")),
            "max_simultaneous_notes": _int(candidate.get("max_simultaneous_notes")),
            "adjacent_repeated_pitch_ratio": _float(candidate.get("adjacent_repeated_pitch_ratio")),
            "direction_change_ratio": _float(candidate.get("direction_change_ratio")),
            "root_tone_ratio": _float(candidate.get("root_tone_ratio")),
            "tension_ratio": _float(candidate.get("tension_ratio")),
        },
        "human_review": {
            "status": "pending",
            "phrase_naturalness": "",
            "time_feel": "",
            "inside_outside_balance": "",
            "repetition_or_liveliness": "",
            "keep_decision": "",
            "notes": "",
        },
    }


def build_notes_report(
    *,
    run_dir: Path,
    review_package_report_path: Path,
    review_package_report: dict[str, Any],
    notes: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    package = _dict(review_package_report.get("review_package"))
    candidate_count = len(notes)
    notes_ready = candidate_count >= int(args.min_candidate_count)
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_listening_notes_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "review_package_report_path": str(review_package_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "min_candidate_count": int(args.min_candidate_count),
        },
        "source_summary": {
            "candidate_count": _int(package.get("candidate_count")),
            "failed_candidate_count": _int(package.get("failed_candidate_count")),
            "midi_dir": str(package.get("midi_dir") or ""),
        },
        "listening_notes": {
            "candidate_count": candidate_count,
            "status": "pending_human_review",
            "notes": notes,
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_listening_notes",
            "listening_notes_ready": notes_ready,
            "human_review_filled": False,
            "musical_quality_claimed": False,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_listening_notes",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_listening_fill"
                if notes_ready
                else "stage_b_generic_tiny_checkpoint_repair_listening_notes_rebuild"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "listening notes are prepared as pending human-review inputs without quality claims",
        },
        "not_proven": [
            "human_listening_result",
            "musical_quality",
            "unconstrained_raw_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair listening fill"
            if notes_ready
            else "Stage B generic tiny checkpoint repair listening notes rebuild"
        ),
    }


def validate_notes_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_listening_notes_ready: bool,
    require_pending_human_review: bool,
    require_no_musical_quality_claim: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    notes = _dict(report.get("listening_notes"))
    source = _dict(report.get("source_summary"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairListeningNotesError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_listening_notes_ready and not bool(readiness.get("listening_notes_ready", False)):
        raise StageBGenericTinyCheckpointRepairListeningNotesError("listening notes should be ready")
    if require_pending_human_review and notes.get("status") != "pending_human_review":
        raise StageBGenericTinyCheckpointRepairListeningNotesError("listening notes must stay pending")
    if require_pending_human_review and bool(readiness.get("human_review_filled", True)):
        raise StageBGenericTinyCheckpointRepairListeningNotesError("human review must not be marked filled")
    if require_no_musical_quality_claim and bool(readiness.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairListeningNotesError("musical quality must not be claimed")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairListeningNotesError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointRepairListeningNotesError("Brad style adaptation must not be claimed")
    if _int(notes.get("candidate_count")) <= 0:
        raise StageBGenericTinyCheckpointRepairListeningNotesError("candidate notes count must be positive")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "source_candidate_count": _int(source.get("candidate_count")),
        "source_failed_candidate_count": _int(source.get("failed_candidate_count")),
        "notes_candidate_count": _int(notes.get("candidate_count")),
        "notes_status": str(notes.get("status") or ""),
        "listening_notes_ready": bool(readiness.get("listening_notes_ready", False)),
        "human_review_filled": bool(readiness.get("human_review_filled", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "raw_generation_quality_claimed": bool(readiness.get("raw_generation_quality_claimed", True)),
        "constrained_generation_quality_claimed": bool(
            readiness.get("constrained_generation_quality_claimed", True)
        ),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    notes = report["listening_notes"]
    source = report["source_summary"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Listening Notes",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- listening notes ready: `{_bool_token(readiness['listening_notes_ready'])}`",
        f"- human review filled: `{_bool_token(readiness['human_review_filled'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Source",
        "",
        f"- candidate count: `{source['candidate_count']}`",
        f"- failed candidate count: `{source['failed_candidate_count']}`",
        f"- midi dir: `{source['midi_dir']}`",
        "",
        "## Notes",
        "",
        f"- notes status: `{notes['status']}`",
        f"- notes candidate count: `{notes['candidate_count']}`",
        "",
    ]
    for note in notes["notes"]:
        context = note["objective_context"]
        review = note["human_review"]
        lines.extend(
            [
                f"### Candidate {note['review_rank']}",
                "",
                f"- midi: `{note['midi_path']}`",
                f"- seed / sample: `{note['sample_seed']}/{note['sample_index']}`",
                f"- dead-air / coverage / chord-tone: `{context['dead_air_ratio']}/"
                f"{context['phrase_coverage_ratio']}/{context['chord_tone_ratio']}`",
                f"- unique pitch / max simultaneous: `{context['unique_pitch_count']}/"
                f"{context['max_simultaneous_notes']}`",
                f"- status: `{review['status']}`",
                f"- phrase naturalness: `{review['phrase_naturalness']}`",
                f"- time feel: `{review['time_feel']}`",
                f"- inside/outside balance: `{review['inside_outside_balance']}`",
                f"- repetition or liveliness: `{review['repetition_or_liveliness']}`",
                f"- keep decision: `{review['keep_decision']}`",
                f"- notes: `{review['notes']}`",
                "",
            ]
        )
    lines.extend(["## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pending listening notes for repair review candidates")
    parser.add_argument(
        "--review_package_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_review_package/"
        "harness_stage_b_generic_tiny_checkpoint_repair_review_package/"
        "stage_b_generic_tiny_checkpoint_repair_review_package.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_repair_listening_notes")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=401)
    parser.add_argument("--min_candidate_count", type=int, default=5)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_listening_notes_ready", action="store_true")
    parser.add_argument("--require_pending_human_review", action="store_true")
    parser.add_argument("--require_no_musical_quality_claim", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    review_package_report_path = Path(args.review_package_report)
    if not review_package_report_path.exists():
        raise StageBGenericTinyCheckpointRepairListeningNotesError("review package report required")
    review_package_report = read_json(review_package_report_path)
    candidates = _dict(review_package_report.get("review_package")).get("candidates") or []
    notes = [build_listening_note(candidate) for candidate in candidates]
    report = build_notes_report(
        run_dir=run_dir,
        review_package_report_path=review_package_report_path,
        review_package_report=review_package_report,
        notes=notes,
        args=args,
    )
    summary = validate_notes_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_listening_notes_ready=bool(args.require_listening_notes_ready),
        require_pending_human_review=bool(args.require_pending_human_review),
        require_no_musical_quality_claim=bool(args.require_no_musical_quality_claim),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_listening_notes.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_listening_notes_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_repair_listening_notes.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
