"""Record phrase-continuation candidate rejection with MIDI note evidence."""

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
from scripts.review_midi_note_objectives import pitch_name, read_midi_notes  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(ValueError):
    pass


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_audio_render_report(report: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(report.get("audio_render_boundary"))
    if str(boundary.get("boundary") or "") != (
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt"
    ):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "unexpected phrase-continuation audio render boundary"
        )
    if not bool(boundary.get("render_attempted", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError("render attempt required")
    if not bool(boundary.get("technical_wav_validation", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "technical WAV validation required"
        )
    if bool(boundary.get("audio_rendered_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "source report must not claim audio quality"
        )
    if bool(boundary.get("human_audio_preference_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "source report must not claim human preference"
        )
    if bool(boundary.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "source report must not claim musical quality"
        )
    rendered = [dict(item) for item in _list(report.get("rendered_audio_files")) if isinstance(item, dict)]
    if not rendered:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError("rendered files required")
    return rendered


def interval_metrics(pitches: list[int], *, large_interval_threshold: int, severe_interval_threshold: int) -> dict[str, Any]:
    intervals = [pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1)]
    abs_intervals = [abs(interval) for interval in intervals]
    large_intervals = [interval for interval in abs_intervals if interval >= large_interval_threshold]
    severe_intervals = [interval for interval in abs_intervals if interval >= severe_interval_threshold]
    return {
        "intervals": intervals,
        "abs_intervals": abs_intervals,
        "max_abs_interval": max(abs_intervals) if abs_intervals else 0,
        "avg_abs_interval": sum(abs_intervals) / len(abs_intervals) if abs_intervals else 0.0,
        "large_interval_count": len(large_intervals),
        "large_interval_ratio": len(large_intervals) / len(abs_intervals) if abs_intervals else 0.0,
        "severe_interval_count": len(severe_intervals),
        "severe_interval_ratio": len(severe_intervals) / len(abs_intervals) if abs_intervals else 0.0,
    }


def note_audit(
    midi_path: Path,
    *,
    max_pitch_span: int,
    max_abs_interval: int,
    max_large_interval_ratio: float,
    large_interval_threshold: int,
    severe_interval_threshold: int,
) -> dict[str, Any]:
    if not midi_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            f"MIDI not found: {midi_path}"
        )
    ticks_per_beat, notes = read_midi_notes(midi_path)
    pitches = [int(note["pitch"]) for note in notes]
    pitch_span = (max(pitches) - min(pitches)) if pitches else 0
    intervals = interval_metrics(
        pitches,
        large_interval_threshold=large_interval_threshold,
        severe_interval_threshold=severe_interval_threshold,
    )
    failure_reasons: list[str] = []
    if pitch_span > max_pitch_span:
        failure_reasons.append("pitch_span_above_target")
    if _int(intervals.get("max_abs_interval")) > max_abs_interval:
        failure_reasons.append("max_interval_above_target")
    if _float(intervals.get("large_interval_ratio")) > max_large_interval_ratio:
        failure_reasons.append("large_interval_ratio_above_target")
    if _int(intervals.get("severe_interval_count")) > 0:
        failure_reasons.append("severe_interval_present")
    note_sequence = [
        {
            "index": index,
            "start_beats": note["start"] / ticks_per_beat,
            "duration_beats": (note["end"] - note["start"]) / ticks_per_beat,
            "pitch": int(note["pitch"]),
            "pitch_name": pitch_name(int(note["pitch"])),
        }
        for index, note in enumerate(notes, start=1)
    ]
    return {
        "midi_path": str(midi_path),
        "ticks_per_beat": int(ticks_per_beat),
        "note_count": len(notes),
        "pitch_min": min(pitches) if pitches else None,
        "pitch_max": max(pitches) if pitches else None,
        "pitch_span": pitch_span,
        "pitch_sequence": pitches,
        "pitch_name_sequence": [pitch_name(pitch) for pitch in pitches],
        "note_sequence": note_sequence,
        **intervals,
        "targets": {
            "max_pitch_span": int(max_pitch_span),
            "max_abs_interval": int(max_abs_interval),
            "max_large_interval_ratio": float(max_large_interval_ratio),
            "large_interval_threshold": int(large_interval_threshold),
            "severe_interval_threshold": int(severe_interval_threshold),
        },
        "failed": bool(failure_reasons),
        "failure_reasons": failure_reasons,
    }


def build_failure_review(
    audio_render_report: dict[str, Any],
    *,
    output_dir: Path,
    reviewer: str,
    assessment: str,
    notes: str,
    max_pitch_span: int,
    max_abs_interval: int,
    max_large_interval_ratio: float,
    large_interval_threshold: int,
    severe_interval_threshold: int,
) -> dict[str, Any]:
    if not reviewer.strip():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError("reviewer is required")
    rendered = validate_audio_render_report(audio_render_report)
    reviewed_items: list[dict[str, Any]] = []
    for item in rendered:
        wav_file = _dict(item.get("wav_file"))
        if not bool(wav_file.get("exists", False)):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
                f"missing WAV for rank {item.get('review_rank')}"
            )
        source_midi_path = Path(str(item.get("source_midi_path") or ""))
        reviewed_items.append(
            {
                "review_rank": _int(item.get("review_rank")),
                "sample_seed": _int(item.get("sample_seed")),
                "sample_index": _int(item.get("sample_index")),
                "wav_path": str(wav_file.get("path") or ""),
                "duration_seconds": float(wav_file.get("duration_seconds", 0.0) or 0.0),
                "sample_rate": int(wav_file.get("sample_rate", 0) or 0),
                "source_midi_path": str(source_midi_path),
                "midi_note_audit": note_audit(
                    source_midi_path,
                    max_pitch_span=max_pitch_span,
                    max_abs_interval=max_abs_interval,
                    max_large_interval_ratio=max_large_interval_ratio,
                    large_interval_threshold=large_interval_threshold,
                    severe_interval_threshold=severe_interval_threshold,
                ),
            }
        )
    all_failed = all(bool(_dict(item.get("midi_note_audit")).get("failed", False)) for item in reviewed_items)
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_audio_render_schema": str(audio_render_report.get("schema_version") or ""),
        "reviewed_audio_files": reviewed_items,
        "user_listening_review": {
            "status": "reviewed",
            "reviewer": reviewer.strip(),
            "review_basis": "single_user_listening_review_with_midi_note_audit",
            "overall_decision": "reject_all",
            "candidate_decision": "reject",
            "primary_failure": "midi_note_random_large_leaps",
            "timing": "unusable",
            "phrase": "not_musical",
            "vocabulary": "not_musical",
            "assessment": assessment.strip(),
            "notes": notes.strip(),
        },
        "midi_note_failure": {
            "all_reviewed_candidates_failed": all_failed,
            "candidate_count": len(reviewed_items),
            "failed_candidate_count": sum(
                1 for item in reviewed_items if bool(_dict(item.get("midi_note_audit")).get("failed", False))
            ),
            "primary_failure": "range_interval_guard_missing",
            "repair_targets": [
                "constrain_solo_pitch_range",
                "limit_max_adjacent_interval",
                "penalize_severe_register_jumps",
                "require_step_or_small_leap_contour_support",
            ],
        },
        "claim_boundary": {
            "single_user_review_completed": True,
            "midi_note_audit_completed": True,
            "human_audio_keep_claimed": False,
            "human_audio_preference_claimed": False,
            "human_audio_reject_all_recorded": True,
            "audio_render_used": True,
            "audio_rendered_quality_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "boundary": "generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_reject_all",
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_user_listening_review_input",
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "repair_target": "range_interval_guard_missing",
        },
        "proven": [
            "single_user_audio_review_completed",
            "midi_note_audit_completed",
            "phrase_continuation_candidate_rejected",
        ],
        "not_proven": [
            "human_audio_keep",
            "audio_rendered_quality",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic tiny checkpoint repair phrase continuation range interval guard decision",
    }


def validate_failure_review(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_primary_failure: str | None,
    expected_file_count: int,
    min_max_interval: int,
    require_no_keep_claim: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    review = _dict(report.get("user_listening_review"))
    failure = _dict(report.get("midi_note_failure"))
    claim = _dict(report.get("claim_boundary"))
    boundary = str(claim.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    primary_failure = str(review.get("primary_failure") or "")
    if expected_primary_failure and primary_failure != expected_primary_failure:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            f"expected primary failure {expected_primary_failure}, got {primary_failure}"
        )
    reviewed = _list(report.get("reviewed_audio_files"))
    if len(reviewed) != expected_file_count:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            f"expected reviewed file count {expected_file_count}, got {len(reviewed)}"
        )
    max_intervals = [_int(_dict(item.get("midi_note_audit")).get("max_abs_interval")) for item in reviewed]
    if max(max_intervals or [0]) < min_max_interval:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "max interval evidence below expected failure threshold"
        )
    if not bool(failure.get("all_reviewed_candidates_failed", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "all reviewed candidates must fail MIDI note audit"
        )
    if require_no_keep_claim and bool(claim.get("human_audio_keep_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
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
            raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
                "quality claims must not be set"
            )
    decision = _dict(report.get("decision"))
    first_audit = _dict(_dict(reviewed[0]).get("midi_note_audit")) if reviewed else {}
    return {
        "boundary": boundary,
        "overall_decision": str(review.get("overall_decision") or ""),
        "candidate_decision": str(review.get("candidate_decision") or ""),
        "primary_failure": primary_failure,
        "reviewed_audio_file_count": len(reviewed),
        "note_count": _int(first_audit.get("note_count")),
        "pitch_min": first_audit.get("pitch_min"),
        "pitch_max": first_audit.get("pitch_max"),
        "pitch_span": _int(first_audit.get("pitch_span")),
        "max_abs_interval": _int(first_audit.get("max_abs_interval")),
        "large_interval_ratio": _float(first_audit.get("large_interval_ratio")),
        "severe_interval_count": _int(first_audit.get("severe_interval_count")),
        "human_audio_keep_claimed": bool(claim.get("human_audio_keep_claimed", True)),
        "musical_quality_claimed": bool(claim.get("musical_quality_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    review = report["user_listening_review"]
    failure = report["midi_note_failure"]
    claim = report["claim_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation MIDI Note Failure Review",
        "",
        "## Summary",
        "",
        f"- boundary: `{claim['boundary']}`",
        f"- overall decision: `{review['overall_decision']}`",
        f"- candidate decision: `{review['candidate_decision']}`",
        f"- primary failure: `{review['primary_failure']}`",
        f"- failed candidate count: `{failure['failed_candidate_count']}/{failure['candidate_count']}`",
        f"- repair target: `{decision['repair_target']}`",
        f"- human/audio keep claimed: `{_bool_token(claim['human_audio_keep_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(claim['musical_quality_claimed'])}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        "",
        "## MIDI Note Audit",
        "",
        "| rank | seed | sample | notes | pitch range | span | max interval | large interval ratio | severe intervals | failure reasons |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|---|",
    ]
    for item in report.get("reviewed_audio_files", []):
        audit = _dict(item.get("midi_note_audit"))
        pitch_range = f"{audit.get('pitch_min')}-{audit.get('pitch_max')}"
        reasons = ", ".join(_list(audit.get("failure_reasons"))) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item.get("review_rank") or ""),
                    str(item.get("sample_seed") or ""),
                    str(item.get("sample_index") or ""),
                    str(audit.get("note_count") or ""),
                    pitch_range,
                    str(audit.get("pitch_span") or ""),
                    str(audit.get("max_abs_interval") or ""),
                    f"{_float(audit.get('large_interval_ratio')):.3f}",
                    str(audit.get("severe_interval_count") or ""),
                    reasons,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Note Sequence", ""])
    for item in report.get("reviewed_audio_files", []):
        audit = _dict(item.get("midi_note_audit"))
        lines.append(f"### Rank {item.get('review_rank')}")
        lines.append("")
        lines.append("| index | start | duration | pitch | name |")
        lines.append("|---:|---:|---:|---:|---|")
        for note in _list(audit.get("note_sequence")):
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(note.get("index") or ""),
                        f"{_float(note.get('start_beats')):.3f}",
                        f"{_float(note.get('duration_beats')):.3f}",
                        str(note.get("pitch") or ""),
                        str(note.get("pitch_name") or ""),
                    ]
                )
                + " |"
            )
        lines.append("")
        lines.append(f"- intervals: `{audit.get('intervals')}`")
        lines.append("")
    lines.extend(["## Repair Targets", ""])
    for item in failure.get("repair_targets", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record phrase-continuation MIDI note failure review")
    parser.add_argument(
        "--audio_render_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_local_audio_render_attempt.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--reviewer", type=str, default="user")
    parser.add_argument("--assessment", type=str, default="candidate is not musical and fails MIDI note audit")
    parser.add_argument("--notes", type=str, default="")
    parser.add_argument("--max_pitch_span", type=int, default=24)
    parser.add_argument("--max_abs_interval", type=int, default=12)
    parser.add_argument("--max_large_interval_ratio", type=float, default=0.35)
    parser.add_argument("--large_interval_threshold", type=int, default=12)
    parser.add_argument("--severe_interval_threshold", type=int, default=24)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_primary_failure", type=str, default="")
    parser.add_argument("--expected_file_count", type=int, default=1)
    parser.add_argument("--min_max_interval", type=int, default=24)
    parser.add_argument("--require_no_keep_claim", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    audio_render_report_path = Path(args.audio_render_report)
    if not audio_render_report_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationMidiNoteFailureReviewError(
            "audio render report required"
        )
    report = build_failure_review(
        read_json(audio_render_report_path),
        output_dir=output_dir,
        reviewer=str(args.reviewer or ""),
        assessment=str(args.assessment or ""),
        notes=str(args.notes or ""),
        max_pitch_span=int(args.max_pitch_span),
        max_abs_interval=int(args.max_abs_interval),
        max_large_interval_ratio=float(args.max_large_interval_ratio),
        large_interval_threshold=int(args.large_interval_threshold),
        severe_interval_threshold=int(args.severe_interval_threshold),
    )
    summary = validate_failure_review(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_primary_failure=str(args.expected_primary_failure or ""),
        expected_file_count=int(args.expected_file_count),
        min_max_interval=int(args.min_max_interval),
        require_no_keep_claim=bool(args.require_no_keep_claim),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
