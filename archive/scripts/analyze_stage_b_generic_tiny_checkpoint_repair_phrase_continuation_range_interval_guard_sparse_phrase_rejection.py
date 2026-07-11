"""Analyze sparse phrase repair listening rejection from MIDI evidence."""

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

from scripts.analyze_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection import (  # noqa: E402
    analyze_reviewed_candidate,
)
from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
    ValueError
):
    pass


SCHEMA_VERSION = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_rejection_analysis_v1"
)
SOURCE_BOUNDARY = (
    "generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_audio_review_reject_all"
)
BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_rejection_analysis"
)
NEXT_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
    "sparse_phrase_model_core_review_decision"
)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            f"unexpected source boundary: {boundary}"
        )
    if str(review.get("status") or "") != "reviewed":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            "source listening review must be reviewed"
        )
    if str(review.get("overall_decision") or "") != "reject_all":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            "source listening review must be reject_all"
        )
    if str(review.get("candidate_decision") or "") != "reject":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            "source candidate decision must be reject"
        )
    if not bool(claim.get("human_audio_reject_all_recorded", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
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
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            f"source report contains unsupported claims: {', '.join(claimed)}"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            "source report is not routed to sparse phrase rejection analysis"
        )
    reviewed = [dict(item) for item in _list(report.get("reviewed_audio_files")) if isinstance(item, dict)]
    if len(reviewed) != expected_file_count:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            f"expected {expected_file_count} reviewed files, got {len(reviewed)}"
        )
    for item in reviewed:
        source_midi = Path(str(item.get("source_midi_path") or ""))
        if not source_midi.exists():
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
                f"source MIDI not found: {source_midi}"
            )
    return reviewed


def primary_next_review_target(
    *,
    common_flags: list[str],
    candidates_without_flags: list[int],
    candidate_count: int,
) -> str:
    if candidates_without_flags:
        return "model_core_review_after_objective_proxy_gap"
    if common_flags:
        return "shared_midi_flag_repair_decision"
    if candidate_count > 0:
        return "manual_midi_evidence_review"
    return "insufficient_candidate_evidence"


def build_sparse_phrase_rejection_analysis(
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
    common_flags = sorted(flag for flag, count in flag_counts.items() if count == len(analyzed))
    candidates_without_flags = [
        int(candidate["review_rank"]) for candidate in analyzed if not candidate["evidence_flags"]
    ]
    objective_proxy_gap = bool(candidates_without_flags)
    target = primary_next_review_target(
        common_flags=common_flags,
        candidates_without_flags=candidates_without_flags,
        candidate_count=len(analyzed),
    )
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
            "objective_proxy_gap_recorded": objective_proxy_gap,
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
            "candidates_without_evidence_flags": candidates_without_flags,
            "objective_proxy_gap": {
                "recorded": objective_proxy_gap,
                "all_candidates_rejected_by_listening_review": True,
                "candidate_without_flag_count": len(candidates_without_flags),
                "interpretation": (
                    "objective_midi_proxy_not_sufficient_for_listening_acceptance"
                    if objective_proxy_gap
                    else "objective_flags_present_for_all_rejected_candidates"
                ),
            },
            "primary_next_review_target": target,
            "cause_claim": "not_claimed",
        },
        "candidates": analyzed,
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "next_recommended_issue": (
                "Stage B generic tiny checkpoint repair phrase continuation range interval guard "
                "sparse phrase model core review decision"
            ),
        },
        "proven": [
            "single_user_reject_all_report_consumed",
            "reviewed_candidate_midi_sequences_analyzed",
            "objective_proxy_gap_recorded" if objective_proxy_gap else "all_candidates_have_objective_flags",
        ],
        "not_proven": [
            "musical_quality",
            "quality_root_cause",
            "multi_reviewer_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
    }


def validate_sparse_phrase_rejection_analysis(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_candidate_count: int,
    require_reject_all_source: bool,
    require_no_quality_claim: bool,
    require_proxy_gap: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("analysis_boundary"))
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    candidate_count = len(_list(report.get("candidates")))
    if candidate_count != expected_candidate_count:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            f"expected {expected_candidate_count} candidates, got {candidate_count}"
        )
    if require_reject_all_source and not bool(boundary.get("input_reject_all_verified", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
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
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
                "analysis must not claim quality or quality cause"
            )
    rejection = _dict(report.get("rejection_analysis"))
    proxy_gap = _dict(rejection.get("objective_proxy_gap"))
    if require_proxy_gap and not bool(proxy_gap.get("recorded", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRejectionAnalysisError(
            "objective proxy gap must be recorded"
        )
    decision = _dict(report.get("decision"))
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "source_review_boundary": str(report.get("source_review_boundary") or ""),
        "candidate_count": candidate_count,
        "common_evidence_flags": _list(rejection.get("common_evidence_flags")),
        "evidence_flag_counts": dict(rejection.get("evidence_flag_counts") or {}),
        "candidates_without_evidence_flags": _list(rejection.get("candidates_without_evidence_flags")),
        "objective_proxy_gap_recorded": bool(proxy_gap.get("recorded", False)),
        "primary_next_review_target": str(rejection.get("primary_next_review_target") or ""),
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
    proxy_gap = rejection["objective_proxy_gap"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase Rejection Analysis",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- source boundary: `{report['source_review_boundary']}`",
        f"- analyzed candidates: `{boundary['analyzed_candidate_count']}`",
        f"- common evidence flags: `{', '.join(rejection['common_evidence_flags'])}`",
        f"- candidates without evidence flags: `{', '.join(str(rank) for rank in rejection['candidates_without_evidence_flags'])}`",
        f"- objective proxy gap recorded: `{_bool_token(proxy_gap['recorded'])}`",
        f"- primary next review target: `{rejection['primary_next_review_target']}`",
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
            "| rank | notes | unique | gap ratio | max gap | max interval | adjacent repeat | duration common | flags |",
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
                    f"{_float(_dict(metrics.get('duration_summary')).get('most_common_duration_ratio')):.4f}",
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
        description="Analyze sparse phrase listening rejection from MIDI evidence"
    )
    parser.add_argument(
        "--user_listening_review_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_user_listening_review/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_user_listening_review/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
        "sparse_phrase_user_listening_review.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_"
            "sparse_phrase_rejection_analysis"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default=BOUNDARY)
    parser.add_argument("--expected_candidate_count", type=int, default=3)
    parser.add_argument("--phrase_window_beats", type=float, default=8.0)
    parser.add_argument("--sparse_gap_ratio", type=float, default=0.35)
    parser.add_argument("--long_gap_beats", type=float, default=1.0)
    parser.add_argument("--require_reject_all_source", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    parser.add_argument("--require_proxy_gap", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_sparse_phrase_rejection_analysis(
        read_json(Path(args.user_listening_review_report)),
        output_dir=output_dir,
        expected_file_count=int(args.expected_candidate_count),
        phrase_window_beats=float(args.phrase_window_beats),
        sparse_gap_ratio=float(args.sparse_gap_ratio),
        long_gap_beats=float(args.long_gap_beats),
    )
    summary = validate_sparse_phrase_rejection_analysis(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_candidate_count=int(args.expected_candidate_count),
        require_reject_all_source=bool(args.require_reject_all_source),
        require_no_quality_claim=bool(args.require_no_quality_claim),
        require_proxy_gap=bool(args.require_proxy_gap),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_rejection_analysis.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
