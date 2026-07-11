"""Decide sparse phrase repair targets after range/interval guard rejection analysis."""

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


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
    ValueError
):
    pass


SOURCE_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis"
)
BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision"
)
NEXT_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep"
)
PRIMARY_TARGET = "sparse_phrase_continuity_after_range_interval_guard"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _avg(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def validate_rejection_analysis(report: dict[str, Any]) -> list[dict[str, Any]]:
    boundary = _dict(report.get("analysis_boundary"))
    rejection = _dict(report.get("rejection_analysis"))
    decision = _dict(report.get("decision"))
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "unexpected rejection analysis boundary"
        )
    if not bool(boundary.get("input_reject_all_verified", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "reject_all input verification required"
        )
    if bool(boundary.get("human_audio_keep_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "human/audio keep must not be claimed"
        )
    if bool(boundary.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "musical quality must not be claimed"
        )
    if bool(boundary.get("quality_cause_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "quality root cause must not be claimed"
        )
    if str(rejection.get("primary_next_repair_target") or "") != PRIMARY_TARGET:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "sparse phrase primary repair target required"
        )
    common_flags = set(str(flag) for flag in _list(rejection.get("common_evidence_flags")))
    if "high_dead_air_or_sparse_phrase" not in common_flags:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "high_dead_air_or_sparse_phrase common flag required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "unexpected rejection analysis next boundary"
        )
    candidates = [dict(item) for item in _list(report.get("candidates")) if isinstance(item, dict)]
    if not candidates:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "candidate evidence required"
        )
    return candidates


def summarize_candidate_evidence(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    gap_ratios = [_float(_dict(candidate.get("metrics")).get("gap_ratio_to_window")) for candidate in candidates]
    max_gaps = [_float(_dict(candidate.get("metrics")).get("max_internal_gap_beats")) for candidate in candidates]
    max_intervals = [_int(_dict(candidate.get("metrics")).get("max_abs_interval")) for candidate in candidates]
    note_counts = [_int(_dict(candidate.get("metrics")).get("note_count")) for candidate in candidates]
    adjacent_repeats = [_int(_dict(candidate.get("metrics")).get("adjacent_repeat_count")) for candidate in candidates]
    return {
        "candidate_count": len(candidates),
        "gap_ratio_min": round(min(gap_ratios), 4),
        "gap_ratio_max": round(max(gap_ratios), 4),
        "gap_ratio_avg": round(_avg(gap_ratios), 4),
        "max_internal_gap_beats_min": round(min(max_gaps), 4),
        "max_internal_gap_beats_max": round(max(max_gaps), 4),
        "max_internal_gap_beats_avg": round(_avg(max_gaps), 4),
        "note_count_min": min(note_counts),
        "note_count_max": max(note_counts),
        "max_abs_interval_max": max(max_intervals),
        "adjacent_repeat_candidate_count": sum(1 for value in adjacent_repeats if value > 0),
        "octave_or_larger_interval_candidate_count": sum(1 for value in max_intervals if value >= 12),
    }


def build_sparse_phrase_repair_decision(
    rejection_analysis: dict[str, Any],
    *,
    output_dir: Path,
    target_max_gap_ratio_to_window: float,
    target_max_internal_gap_beats: float,
    target_min_note_count: int,
    target_min_phrase_coverage_ratio: float,
    target_max_tail_empty_steps: int,
    target_max_abs_interval: int,
) -> dict[str, Any]:
    candidates = validate_rejection_analysis(rejection_analysis)
    rejection = _dict(rejection_analysis.get("rejection_analysis"))
    evidence = summarize_candidate_evidence(candidates)
    repair_targets = [
        PRIMARY_TARGET,
        "long_gap_reduction",
        "adjacent_pitch_repeat_control",
        "octave_interval_soft_limit",
    ]
    target_thresholds = {
        "max_gap_ratio_to_window": float(target_max_gap_ratio_to_window),
        "max_internal_gap_beats": float(target_max_internal_gap_beats),
        "min_note_count": int(target_min_note_count),
        "min_phrase_coverage_ratio": float(target_min_phrase_coverage_ratio),
        "max_tail_empty_steps": int(target_max_tail_empty_steps),
        "max_abs_interval": int(target_max_abs_interval),
    }
    return {
        "schema_version": (
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision_v1"
        ),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_rejection_analysis_schema": str(rejection_analysis.get("schema_version") or ""),
        "source_boundary": SOURCE_BOUNDARY,
        "observed_evidence": {
            **evidence,
            "common_evidence_flags": _list(rejection.get("common_evidence_flags")),
            "evidence_flag_counts": dict(rejection.get("evidence_flag_counts") or {}),
            "source_primary_next_repair_target": str(rejection.get("primary_next_repair_target") or ""),
        },
        "repair_decision": {
            "boundary": BOUNDARY,
            "decision": "run_sparse_phrase_repair_sweep",
            "primary_repair_target": PRIMARY_TARGET,
            "repair_targets": repair_targets,
            "target_thresholds": target_thresholds,
            "planned_sweep_controls": {
                "interval_caps": [9, 7, 5],
                "coverage_aware_positions": True,
                "coverage_position_window": 0,
                "keep_range_interval_guard": True,
                "rank_by_gap_ratio_and_internal_gap": True,
                "reject_adjacent_pitch_repeats_when_possible": True,
            },
            "excluded_scope": [
                "broad_retraining",
                "brad_style_adaptation",
                "audio_quality_claim",
                "quality_root_cause_claim",
            ],
        },
        "readiness": {
            "sparse_phrase_repair_decision_recorded": True,
            "human_audio_keep_claimed": False,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "quality_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "next_recommended_issue": (
                "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase repair sweep"
            ),
        },
        "not_proven": [
            "sparse_phrase_repair_candidate_exists",
            "human_audio_keep",
            "musical_quality",
            "quality_root_cause",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
    }


def validate_sparse_phrase_repair_decision(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_primary_target: str | None,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    repair = _dict(report.get("repair_decision"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    evidence = _dict(report.get("observed_evidence"))
    thresholds = _dict(repair.get("target_thresholds"))
    boundary = str(repair.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    primary_target = str(repair.get("primary_repair_target") or "")
    if require_primary_target and primary_target != require_primary_target:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            f"expected primary target {require_primary_target}, got {primary_target}"
        )
    if _float(thresholds.get("max_gap_ratio_to_window")) >= _float(evidence.get("gap_ratio_max")):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "target gap ratio must be below observed max"
        )
    if _float(thresholds.get("max_internal_gap_beats")) >= _float(evidence.get("max_internal_gap_beats_max")):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
            "target internal gap must be below observed max"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("human_audio_keep_claimed", True)),
            bool(readiness.get("human_audio_preference_claimed", True)),
            bool(readiness.get("musical_quality_claimed", True)),
            bool(readiness.get("quality_cause_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairDecisionError(
                "quality claims must not be set"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": next_boundary,
        "candidate_count": _int(evidence.get("candidate_count")),
        "gap_ratio_max": _float(evidence.get("gap_ratio_max")),
        "max_internal_gap_beats_max": _float(evidence.get("max_internal_gap_beats_max")),
        "target_max_gap_ratio_to_window": _float(thresholds.get("max_gap_ratio_to_window")),
        "target_max_internal_gap_beats": _float(thresholds.get("max_internal_gap_beats")),
        "primary_repair_target": primary_target,
        "repair_target_count": len(_list(repair.get("repair_targets"))),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "quality_cause_claimed": bool(readiness.get("quality_cause_claimed", True)),
        "next_recommended_issue": str(decision.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    repair = report["repair_decision"]
    evidence = report["observed_evidence"]
    thresholds = repair["target_thresholds"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase Repair Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{repair['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- decision: `{repair['decision']}`",
        f"- primary repair target: `{repair['primary_repair_target']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        f"- quality cause claimed: `{_bool_token(readiness['quality_cause_claimed'])}`",
        "",
        "## Observed Evidence",
        "",
        f"- candidate count: `{evidence['candidate_count']}`",
        f"- gap ratio min / avg / max: `{evidence['gap_ratio_min']}` / `{evidence['gap_ratio_avg']}` / `{evidence['gap_ratio_max']}`",
        f"- max internal gap min / avg / max: `{evidence['max_internal_gap_beats_min']}` / `{evidence['max_internal_gap_beats_avg']}` / `{evidence['max_internal_gap_beats_max']}`",
        f"- adjacent repeat candidates: `{evidence['adjacent_repeat_candidate_count']}`",
        f"- octave-or-larger interval candidates: `{evidence['octave_or_larger_interval_candidate_count']}`",
        "",
        "## Target Thresholds",
        "",
        "| target | value |",
        "|---|---:|",
    ]
    for key, value in thresholds.items():
        lines.append(f"| `{key}` | {value} |")
    lines.extend(["", "## Planned Sweep Controls", "", "| control | value |", "|---|---|"])
    for key, value in repair["planned_sweep_controls"].items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide sparse phrase repair targets after range/interval guard rejection analysis"
    )
    parser.add_argument(
        "--rejection_analysis",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection_analysis.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--target_max_gap_ratio_to_window", type=float, default=0.40)
    parser.add_argument("--target_max_internal_gap_beats", type=float, default=0.75)
    parser.add_argument("--target_min_note_count", type=int, default=10)
    parser.add_argument("--target_min_phrase_coverage_ratio", type=float, default=0.90)
    parser.add_argument("--target_max_tail_empty_steps", type=int, default=0)
    parser.add_argument("--target_max_abs_interval", type=int, default=12)
    parser.add_argument("--expected_boundary", type=str, default=BOUNDARY)
    parser.add_argument("--expected_next_boundary", type=str, default=NEXT_BOUNDARY)
    parser.add_argument("--require_primary_target", type=str, default=PRIMARY_TARGET)
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_sparse_phrase_repair_decision(
        read_json(Path(args.rejection_analysis)),
        output_dir=output_dir,
        target_max_gap_ratio_to_window=float(args.target_max_gap_ratio_to_window),
        target_max_internal_gap_beats=float(args.target_max_internal_gap_beats),
        target_min_note_count=int(args.target_min_note_count),
        target_min_phrase_coverage_ratio=float(args.target_min_phrase_coverage_ratio),
        target_max_tail_empty_steps=int(args.target_max_tail_empty_steps),
        target_max_abs_interval=int(args.target_max_abs_interval),
    )
    summary = validate_sparse_phrase_repair_decision(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_primary_target=str(args.require_primary_target or ""),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
