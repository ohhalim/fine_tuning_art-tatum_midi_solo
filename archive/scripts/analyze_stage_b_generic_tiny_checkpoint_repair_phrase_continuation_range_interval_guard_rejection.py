"""Analyze range/interval guard listening rejection from MIDI evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.review_midi_note_objectives import (  # noqa: E402
    active_polyphony_summary,
    duration_summary,
    grid_summary,
    pitch_name,
    read_midi_notes,
)
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
    ValueError
):
    pass


SCHEMA_VERSION = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis_v1"
)
SOURCE_BOUNDARY = (
    "generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_review_reject_all"
)
BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis"
)
NEXT_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision"
)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def ratio(part: int | float, total: int | float) -> float:
    if total <= 0:
        return 0.0
    return float(part / total)


def validate_user_listening_review_report(
    report: dict[str, Any],
    *,
    expected_file_count: int,
) -> list[dict[str, Any]]:
    claim = _dict(report.get("claim_boundary"))
    review = _dict(report.get("user_listening_review"))
    decision = _dict(report.get("decision"))
    boundary = str(claim.get("boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"unexpected source boundary: {boundary}"
        )
    if str(review.get("status") or "") != "reviewed":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            "source listening review must be reviewed"
        )
    if str(review.get("overall_decision") or "") != "reject_all":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            "source listening review must be reject_all"
        )
    if str(review.get("candidate_decision") or "") != "reject":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            "source candidate decision must be reject"
        )
    if not bool(claim.get("human_audio_reject_all_recorded", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            "source reject_all boundary must be recorded"
        )
    forbidden_claims = [
        "human_audio_keep_claimed",
        "human_audio_preference_claimed",
        "audio_rendered_quality_claimed",
        "musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in forbidden_claims if bool(claim.get(name, False))]
    if claimed:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"source report contains unsupported claims: {', '.join(claimed)}"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            "source report is not routed to rejection analysis"
        )
    reviewed = [dict(item) for item in _list(report.get("reviewed_audio_files")) if isinstance(item, dict)]
    if len(reviewed) != expected_file_count:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"expected {expected_file_count} reviewed files, got {len(reviewed)}"
        )
    for item in reviewed:
        source_midi = Path(str(item.get("source_midi_path") or ""))
        if not source_midi.exists():
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
                f"source MIDI not found: {source_midi}"
            )
    return reviewed


def interval_values(pitches: list[int]) -> list[int]:
    return [pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1)]


def ngram_summary(sequence: list[int], size: int) -> dict[str, Any]:
    if len(sequence) < size:
        return {
            "size": int(size),
            "unique_count": 0,
            "repeated_unique_count": 0,
            "repeated_excess_count": 0,
            "top_repeated": [],
        }
    counts = Counter(tuple(sequence[index : index + size]) for index in range(len(sequence) - size + 1))
    repeated = [
        {"pattern": list(pattern), "count": count}
        for pattern, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if count > 1
    ]
    return {
        "size": int(size),
        "unique_count": int(len(counts)),
        "repeated_unique_count": int(len(repeated)),
        "repeated_excess_count": int(sum(item["count"] - 1 for item in repeated)),
        "top_repeated": repeated[:5],
    }


def two_note_oscillation_windows(pitches: list[int]) -> int:
    return sum(
        1
        for index in range(max(0, len(pitches) - 3))
        if pitches[index] == pitches[index + 2]
        and pitches[index + 1] == pitches[index + 3]
        and pitches[index] != pitches[index + 1]
    )


def gap_summary(notes: list[dict[str, int]], ticks_per_beat: int, phrase_window_beats: float) -> dict[str, Any]:
    if not notes:
        return {
            "phrase_window_beats": float(phrase_window_beats),
            "head_gap_beats": 0.0,
            "tail_gap_beats": 0.0,
            "internal_gap_count": 0,
            "max_internal_gap_beats": 0.0,
            "total_gap_beats": 0.0,
            "gap_ratio_to_window": 0.0,
            "active_duration_beats": 0.0,
            "phrase_span_beats": 0.0,
        }
    starts = [note["start"] / ticks_per_beat for note in notes]
    ends = [note["end"] / ticks_per_beat for note in notes]
    internal_gaps = [
        max(0.0, starts[index + 1] - ends[index])
        for index in range(len(notes) - 1)
    ]
    head_gap = max(0.0, starts[0])
    tail_gap = max(0.0, phrase_window_beats - ends[-1])
    total_gap = head_gap + tail_gap + sum(internal_gaps)
    return {
        "phrase_window_beats": float(phrase_window_beats),
        "head_gap_beats": round(head_gap, 4),
        "tail_gap_beats": round(tail_gap, 4),
        "internal_gap_count": int(sum(1 for gap in internal_gaps if gap > 0)),
        "max_internal_gap_beats": round(max(internal_gaps, default=0.0), 4),
        "total_gap_beats": round(total_gap, 4),
        "gap_ratio_to_window": round(total_gap / phrase_window_beats, 4) if phrase_window_beats else 0.0,
        "active_duration_beats": round(sum((note["end"] - note["start"]) / ticks_per_beat for note in notes), 4),
        "phrase_span_beats": round(max(ends) - min(starts), 4),
    }


def evidence_flags(metrics: dict[str, Any], *, sparse_gap_ratio: float, long_gap_beats: float) -> list[str]:
    flags: list[str] = []
    if _float(metrics.get("gap_ratio_to_window")) >= sparse_gap_ratio:
        flags.append("high_dead_air_or_sparse_phrase")
    if _float(metrics.get("max_internal_gap_beats")) >= long_gap_beats:
        flags.append("long_internal_gap_present")
    if _int(metrics.get("adjacent_repeat_count")) > 0:
        flags.append("adjacent_pitch_repeat_present")
    if _int(metrics.get("two_note_oscillation_window_count")) > 0:
        flags.append("two_note_oscillation_present")
    pitch_bigram = _dict(metrics.get("pitch_bigram_repetition"))
    if _int(pitch_bigram.get("repeated_excess_count")) > 0:
        flags.append("pitch_cell_repetition_present")
    if _int(metrics.get("max_abs_interval")) >= 12:
        flags.append("octave_or_larger_interval_present")
    elif _int(metrics.get("max_abs_interval")) >= 9:
        flags.append("guard_edge_interval_present")
    if _float(metrics.get("unique_pitch_ratio")) <= 0.70:
        flags.append("compressed_pitch_vocabulary")
    duration = _dict(metrics.get("duration_summary"))
    if _float(duration.get("most_common_duration_ratio")) >= 0.50:
        flags.append("repetitive_duration_profile")
    return flags


def analyze_reviewed_candidate(
    item: dict[str, Any],
    *,
    phrase_window_beats: float,
    sparse_gap_ratio: float,
    long_gap_beats: float,
) -> dict[str, Any]:
    midi_path = Path(str(item.get("source_midi_path") or ""))
    ticks_per_beat, notes = read_midi_notes(midi_path)
    if not notes:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"empty MIDI note sequence: {midi_path}"
        )
    pitches = [int(note["pitch"]) for note in notes]
    intervals = interval_values(pitches)
    abs_intervals = [abs(interval) for interval in intervals]
    adjacent_repeat_count = sum(1 for interval in intervals if interval == 0)
    unique_pitch_count = len(set(pitches))
    metrics: dict[str, Any] = {
        "ticks_per_beat": int(ticks_per_beat),
        "note_count": int(len(notes)),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "pitch_span": max(pitches) - min(pitches),
        "unique_pitch_count": int(unique_pitch_count),
        "unique_pitch_ratio": round(ratio(unique_pitch_count, len(pitches)), 4),
        "pitch_sequence": pitches,
        "pitch_name_sequence": [pitch_name(pitch) for pitch in pitches],
        "intervals": intervals,
        "abs_intervals": abs_intervals,
        "max_abs_interval": max(abs_intervals) if abs_intervals else 0,
        "avg_abs_interval": round(sum(abs_intervals) / len(abs_intervals), 4) if abs_intervals else 0.0,
        "large_interval_count_ge_7": int(sum(1 for interval in abs_intervals if interval >= 7)),
        "large_interval_ratio_ge_7": round(ratio(sum(1 for interval in abs_intervals if interval >= 7), len(abs_intervals)), 4),
        "adjacent_repeat_count": int(adjacent_repeat_count),
        "adjacent_repeat_ratio": round(ratio(adjacent_repeat_count, len(intervals)), 4),
        "two_note_oscillation_window_count": int(two_note_oscillation_windows(pitches)),
        "pitch_bigram_repetition": ngram_summary(pitches, 2),
        "pitch_trigram_repetition": ngram_summary(pitches, 3),
        "pitch_class_bigram_repetition": ngram_summary([pitch % 12 for pitch in pitches], 2),
        "interval_bigram_repetition": ngram_summary(intervals, 2),
        "duration_summary": duration_summary(notes, ticks_per_beat),
        "grid_summary": grid_summary(notes, ticks_per_beat),
        "polyphony_summary": active_polyphony_summary(notes),
        "note_sequence": [
            {
                "index": index,
                "start_beats": round(note["start"] / ticks_per_beat, 4),
                "duration_beats": round((note["end"] - note["start"]) / ticks_per_beat, 4),
                "pitch": int(note["pitch"]),
                "pitch_name": pitch_name(int(note["pitch"])),
            }
            for index, note in enumerate(notes, start=1)
        ],
    }
    metrics.update(gap_summary(notes, ticks_per_beat, phrase_window_beats))
    flags = evidence_flags(metrics, sparse_gap_ratio=sparse_gap_ratio, long_gap_beats=long_gap_beats)
    return {
        "review_rank": _int(item.get("review_rank")),
        "interval_cap": _int(item.get("interval_cap")),
        "sample_seed": _int(item.get("sample_seed")),
        "sample_index": _int(item.get("sample_index")),
        "source_midi_path": str(midi_path),
        "wav_path": str(item.get("wav_path") or ""),
        "metrics": metrics,
        "evidence_flags": flags,
    }


def repair_target_candidates(flag_counts: Counter[str], candidate_count: int) -> list[str]:
    targets: list[str] = []
    if flag_counts.get("high_dead_air_or_sparse_phrase", 0) == candidate_count:
        targets.append("sparse_phrase_continuity_after_range_interval_guard")
    if flag_counts.get("long_internal_gap_present", 0) >= max(1, candidate_count - 1):
        targets.append("long_gap_reduction")
    if flag_counts.get("adjacent_pitch_repeat_present", 0) or flag_counts.get("pitch_cell_repetition_present", 0):
        targets.append("pitch_cell_repetition_control")
    if flag_counts.get("octave_or_larger_interval_present", 0):
        targets.append("octave_interval_soft_limit")
    if flag_counts.get("repetitive_duration_profile", 0):
        targets.append("duration_density_variation")
    return targets or ["manual_repair_target_review"]


def build_rejection_analysis(
    user_listening_review_report: dict[str, Any],
    *,
    output_dir: Path,
    expected_file_count: int,
    phrase_window_beats: float,
    sparse_gap_ratio: float,
    long_gap_beats: float,
) -> dict[str, Any]:
    reviewed = validate_user_listening_review_report(
        user_listening_review_report,
        expected_file_count=expected_file_count,
    )
    analyzed = [
        analyze_reviewed_candidate(
            item,
            phrase_window_beats=phrase_window_beats,
            sparse_gap_ratio=sparse_gap_ratio,
            long_gap_beats=long_gap_beats,
        )
        for item in reviewed
    ]
    flag_counts = Counter(flag for candidate in analyzed for flag in candidate["evidence_flags"])
    targets = repair_target_candidates(flag_counts, len(analyzed))
    common_flags = sorted(flag for flag, count in flag_counts.items() if count == len(analyzed))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schema": str(user_listening_review_report.get("schema_version") or ""),
        "source_review_boundary": SOURCE_BOUNDARY,
        "analysis_boundary": {
            "boundary": BOUNDARY,
            "input_reject_all_verified": True,
            "analyzed_candidate_count": len(analyzed),
            "human_audio_keep_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "quality_cause_claimed": False,
        },
        "analysis_parameters": {
            "phrase_window_beats": float(phrase_window_beats),
            "sparse_gap_ratio": float(sparse_gap_ratio),
            "long_gap_beats": float(long_gap_beats),
        },
        "rejection_analysis": {
            "candidate_count": len(analyzed),
            "evidence_flag_counts": dict(sorted(flag_counts.items())),
            "common_evidence_flags": common_flags,
            "repair_target_candidates": targets,
            "primary_next_repair_target": targets[0],
            "cause_claim": "not_claimed",
        },
        "candidates": analyzed,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "next_recommended_issue": (
                "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair decision"
            ),
        },
        "proven": [
            "single_user_reject_all_report_consumed",
            "reviewed_candidate_midi_sequences_analyzed",
            "post_guard_midi_evidence_flags_recorded",
        ],
        "not_proven": [
            "musical_quality",
            "quality_root_cause",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
    }


def validate_rejection_analysis(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_candidate_count: int,
    require_reject_all_source: bool,
    require_no_quality_claim: bool,
    min_common_evidence_flags: int,
) -> dict[str, Any]:
    boundary = _dict(report.get("analysis_boundary"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    candidate_count = len(_list(report.get("candidates")))
    if candidate_count != expected_candidate_count:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"expected {expected_candidate_count} candidates, got {candidate_count}"
        )
    if require_reject_all_source and not bool(boundary.get("input_reject_all_verified", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            "reject_all source verification required"
        )
    if require_no_quality_claim:
        claimed = [
            bool(boundary.get("human_audio_keep_claimed", True)),
            bool(boundary.get("human_audio_preference_claimed", True)),
            bool(boundary.get("musical_quality_claimed", True)),
            bool(boundary.get("quality_cause_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
                "analysis must not claim quality or quality cause"
            )
    rejection = _dict(report.get("rejection_analysis"))
    common_flags = _list(rejection.get("common_evidence_flags"))
    if len(common_flags) < min_common_evidence_flags:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardRejectionAnalysisError(
            f"expected at least {min_common_evidence_flags} common evidence flags"
        )
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_review_boundary": str(report.get("source_review_boundary") or ""),
        "candidate_count": candidate_count,
        "common_evidence_flags": common_flags,
        "evidence_flag_counts": dict(rejection.get("evidence_flag_counts") or {}),
        "primary_next_repair_target": str(rejection.get("primary_next_repair_target") or ""),
        "quality_cause_claimed": bool(boundary.get("quality_cause_claimed", True)),
        "musical_quality_claimed": bool(boundary.get("musical_quality_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(decision.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["analysis_boundary"]
    rejection = report["rejection_analysis"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Rejection Analysis",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{report['source_review_boundary']}`",
        f"- analyzed candidates: `{boundary['analyzed_candidate_count']}`",
        f"- common evidence flags: `{', '.join(rejection['common_evidence_flags'])}`",
        f"- primary next repair target: `{rejection['primary_next_repair_target']}`",
        f"- quality cause claim: `{rejection['cause_claim']}`",
        f"- musical quality claimed: `{_bool_token(boundary['musical_quality_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Evidence Flag Counts",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    for flag, count in rejection["evidence_flag_counts"].items():
        lines.append(f"| `{flag}` | {count} |")
    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| rank | notes | unique | gap ratio | max gap | max interval | adjacent repeat | two-note windows | flags |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in report["candidates"]:
        metrics = candidate["metrics"]
        flags = ", ".join(f"`{flag}`" for flag in candidate["evidence_flags"]) or "`none`"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(candidate["review_rank"]),
                    str(metrics["note_count"]),
                    str(metrics["unique_pitch_count"]),
                    f"{_float(metrics.get('gap_ratio_to_window')):.4f}",
                    f"{_float(metrics.get('max_internal_gap_beats')):.4f}",
                    str(metrics["max_abs_interval"]),
                    str(metrics["adjacent_repeat_count"]),
                    str(metrics["two_note_oscillation_window_count"]),
                    flags,
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze range/interval guard listening rejection from MIDI evidence"
    )
    parser.add_argument(
        "--user_listening_review_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_user_listening_review.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default=BOUNDARY)
    parser.add_argument("--expected_candidate_count", type=int, default=3)
    parser.add_argument("--phrase_window_beats", type=float, default=8.0)
    parser.add_argument("--sparse_gap_ratio", type=float, default=0.45)
    parser.add_argument("--long_gap_beats", type=float, default=1.0)
    parser.add_argument("--require_reject_all_source", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    parser.add_argument("--min_common_evidence_flags", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_rejection_analysis(
        read_json(Path(args.user_listening_review_report)),
        output_dir=output_dir,
        expected_file_count=int(args.expected_candidate_count),
        phrase_window_beats=float(args.phrase_window_beats),
        sparse_gap_ratio=float(args.sparse_gap_ratio),
        long_gap_beats=float(args.long_gap_beats),
    )
    summary = validate_rejection_analysis(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_candidate_count=int(args.expected_candidate_count),
        require_reject_all_source=bool(args.require_reject_all_source),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        min_common_evidence_flags=int(args.min_common_evidence_flags),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
