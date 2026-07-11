"""Run sparse phrase repair sweep after range/interval guard rejection."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_rejection import (  # noqa: E402
    analyze_reviewed_candidate,
)
from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402
from scripts.fill_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_midi_note_failure_review import (  # noqa: E402
    note_audit,
)
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
    run_command,
)
from scripts.run_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep import (  # noqa: E402
    parse_int_list,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
    ValueError
):
    pass


SCHEMA_VERSION = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep_v1"
)
SOURCE_BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision"
)
BOUNDARY = (
    "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep"
)


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_sparse_phrase_repair_decision(report: dict[str, Any]) -> dict[str, Any]:
    repair = _dict(report.get("repair_decision"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    thresholds = _dict(repair.get("target_thresholds"))
    if str(repair.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "sparse phrase repair decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "unexpected sparse phrase repair next boundary"
        )
    if str(repair.get("primary_repair_target") or "") != "sparse_phrase_continuity_after_range_interval_guard":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "sparse phrase primary repair target required"
        )
    forbidden_claims = [
        "human_audio_keep_claimed",
        "human_audio_preference_claimed",
        "musical_quality_claimed",
        "quality_cause_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in forbidden_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            f"source decision contains unsupported claims: {', '.join(claimed)}"
        )
    required = (
        "max_gap_ratio_to_window",
        "max_internal_gap_beats",
        "min_note_count",
        "min_phrase_coverage_ratio",
        "max_tail_empty_steps",
        "max_abs_interval",
    )
    missing = [key for key in required if key not in thresholds]
    if missing:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            f"target thresholds missing: {missing}"
        )
    return thresholds


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
    interval_cap: int,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_stage_b_generation_probe.py",
        "--output_root",
        str(output_root),
        "--run_id",
        run_id,
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--skip_prepare",
        "--skip_train",
        "--issue_number",
        str(args.issue_number),
        "--max_sequence",
        str(args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.note_groups_per_bar),
        "--jazz_duration_tokens",
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        str(args.chord_pitch_mode),
        "--chord_pitch_repeat_window",
        str(args.chord_pitch_repeat_window),
        "--constrained_pitch_min",
        str(args.pitch_min),
        "--constrained_pitch_max",
        str(args.pitch_max),
        "--constrained_max_adjacent_interval",
        str(interval_cap),
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--require_all_grammar_samples",
    ]
    if bool(args.coverage_aware_positions):
        command.extend(
            [
                "--coverage_aware_positions",
                "--coverage_position_window",
                str(args.coverage_position_window),
            ]
        )
    return command


def target_failure_reasons(
    row: dict[str, Any],
    *,
    range_audit: dict[str, Any],
    sparse_metrics: dict[str, Any],
    thresholds: dict[str, Any],
    args: argparse.Namespace,
) -> list[str]:
    temporal = _dict(row.get("temporal_coverage"))
    collapse = _dict(row.get("collapse"))
    metrics = _dict(row.get("metrics"))
    reasons: list[str] = []
    if not bool(row.get("grammar_gate_passed", False)):
        reasons.append("grammar_gate_failed")
    if not bool(row.get("strict_valid", False)):
        reasons.append("strict_valid_failed")
    if _int(metrics.get("note_count")) < _int(thresholds.get("min_note_count")):
        reasons.append("note_count_below_target")
    if _float(metrics.get("phrase_coverage_ratio")) < _float(thresholds.get("min_phrase_coverage_ratio")):
        reasons.append("phrase_coverage_below_target")
    if _int(temporal.get("tail_empty_steps")) > int(args.applied_max_tail_empty_steps):
        reasons.append("tail_empty_above_applied_target")
    if _int(metrics.get("max_simultaneous_notes")) > int(args.max_simultaneous_notes):
        reasons.append("max_simultaneous_notes_above_target")
    if _float(collapse.get("postprocess_removal_ratio")) > float(args.max_postprocess_removal_ratio):
        reasons.append("postprocess_removal_above_target")
    if _int(range_audit.get("pitch_span")) > int(args.max_pitch_span):
        reasons.append("pitch_span_above_target")
    if _int(range_audit.get("max_abs_interval")) > _int(thresholds.get("max_abs_interval")):
        reasons.append("max_interval_above_target")
    if _float(sparse_metrics.get("gap_ratio_to_window")) > _float(thresholds.get("max_gap_ratio_to_window")):
        reasons.append("gap_ratio_above_target")
    if _float(sparse_metrics.get("max_internal_gap_beats")) > _float(thresholds.get("max_internal_gap_beats")):
        reasons.append("max_internal_gap_above_target")
    return reasons


def soft_failure_reasons(
    row: dict[str, Any],
    *,
    sparse_metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> list[str]:
    temporal = _dict(row.get("temporal_coverage"))
    reasons: list[str] = []
    if _int(temporal.get("tail_empty_steps")) > _int(thresholds.get("max_tail_empty_steps")):
        reasons.append("tail_empty_above_decision_target")
    if _int(sparse_metrics.get("adjacent_repeat_count")) > 0:
        reasons.append("adjacent_pitch_repeat_present")
    return reasons


def compact_candidate(
    row: dict[str, Any],
    *,
    interval_cap: int,
    thresholds: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    midi_path = Path(str(row.get("midi_path") or ""))
    range_audit = note_audit(
        midi_path,
        max_pitch_span=int(args.max_pitch_span),
        max_abs_interval=_int(thresholds.get("max_abs_interval")),
        max_large_interval_ratio=float(args.max_large_interval_ratio),
        large_interval_threshold=int(args.large_interval_threshold),
        severe_interval_threshold=int(args.severe_interval_threshold),
    )
    sparse = analyze_reviewed_candidate(
        {
            "review_rank": _int(row.get("sample_index")),
            "interval_cap": interval_cap,
            "sample_seed": _int(row.get("sample_seed")),
            "sample_index": _int(row.get("sample_index")),
            "source_midi_path": str(midi_path),
            "wav_path": "",
        },
        phrase_window_beats=float(args.phrase_window_beats),
        sparse_gap_ratio=_float(thresholds.get("max_gap_ratio_to_window")),
        long_gap_beats=_float(thresholds.get("max_internal_gap_beats")),
    )
    sparse_metrics = _dict(sparse.get("metrics"))
    reasons = target_failure_reasons(
        row,
        range_audit=range_audit,
        sparse_metrics=sparse_metrics,
        thresholds=thresholds,
        args=args,
    )
    soft_reasons = soft_failure_reasons(row, sparse_metrics=sparse_metrics, thresholds=thresholds)
    temporal = _dict(row.get("temporal_coverage"))
    collapse = _dict(row.get("collapse"))
    metrics = _dict(row.get("metrics"))
    return {
        "sample_index": _int(row.get("sample_index")),
        "sample_seed": _int(row.get("sample_seed")),
        "interval_cap": int(interval_cap),
        "midi_path": str(midi_path),
        "valid": bool(row.get("valid", False)),
        "strict_valid": bool(row.get("strict_valid", False)),
        "grammar_gate_passed": bool(row.get("grammar_gate_passed", False)),
        "target_qualified": not reasons,
        "target_failure_reasons": reasons,
        "soft_failure_reasons": soft_reasons,
        "note_count": _int(metrics.get("note_count")),
        "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
        "tail_empty_steps": _int(temporal.get("tail_empty_steps")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
        "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
        "sparse_phrase_metrics": {
            "gap_ratio_to_window": _float(sparse_metrics.get("gap_ratio_to_window")),
            "max_internal_gap_beats": _float(sparse_metrics.get("max_internal_gap_beats")),
            "adjacent_repeat_count": _int(sparse_metrics.get("adjacent_repeat_count")),
            "two_note_oscillation_window_count": _int(sparse_metrics.get("two_note_oscillation_window_count")),
            "evidence_flags": _list(sparse.get("evidence_flags")),
        },
        "midi_note_audit": {
            "note_count": _int(range_audit.get("note_count")),
            "pitch_min": range_audit.get("pitch_min"),
            "pitch_max": range_audit.get("pitch_max"),
            "pitch_span": _int(range_audit.get("pitch_span")),
            "pitch_sequence": _list(range_audit.get("pitch_sequence")),
            "pitch_name_sequence": _list(range_audit.get("pitch_name_sequence")),
            "intervals": _list(range_audit.get("intervals")),
            "max_abs_interval": _int(range_audit.get("max_abs_interval")),
            "large_interval_count": _int(range_audit.get("large_interval_count")),
            "large_interval_ratio": _float(range_audit.get("large_interval_ratio")),
            "severe_interval_count": _int(range_audit.get("severe_interval_count")),
            "failure_reasons": _list(range_audit.get("failure_reasons")),
        },
    }


def sort_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in candidates],
        key=lambda row: (
            not bool(row.get("target_qualified", False)),
            _float(_dict(row.get("sparse_phrase_metrics")).get("gap_ratio_to_window")),
            _float(_dict(row.get("sparse_phrase_metrics")).get("max_internal_gap_beats")),
            len(_list(row.get("soft_failure_reasons"))),
            _int(_dict(row.get("midi_note_audit")).get("max_abs_interval")),
            -_int(row.get("note_count")),
            _int(row.get("interval_cap")),
            _int(row.get("sample_index")),
        ),
    )


def build_sweep_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    decision_report_path: Path,
    decision_report: dict[str, Any],
    generation_runs: Sequence[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    thresholds = validate_sparse_phrase_repair_decision(decision_report)
    runs: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for generation_run in generation_runs:
        generation_report = _dict(generation_run.get("report"))
        summary = _dict(generation_report.get("summary"))
        interval_cap = _int(generation_run.get("interval_cap"))
        candidates = [
            compact_candidate(row, interval_cap=interval_cap, thresholds=thresholds, args=args)
            for row in _list(generation_report.get("samples"))
            if isinstance(row, dict)
        ]
        ranked = sort_candidates(candidates)
        all_candidates.extend(ranked)
        runs.append(
            {
                "interval_cap": interval_cap,
                "generation_result": _dict(generation_run.get("result")),
                "generation_report_path": str(generation_run.get("report_path") or ""),
                "summary": {
                    "sample_count": _int(summary.get("sample_count")),
                    "valid_sample_count": _int(summary.get("valid_sample_count")),
                    "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
                    "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
                    "passed_generation_gate": bool(summary.get("passed_generation_gate", False)),
                    "passed_strict_generation_gate": bool(summary.get("passed_strict_generation_gate", False)),
                    "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
                },
                "target_qualified_count": sum(1 for row in ranked if bool(row.get("target_qualified", False))),
                "top_candidates": ranked[:5],
            }
        )
    ranked_candidates = sort_candidates(all_candidates)
    target_qualified = [row for row in ranked_candidates if bool(row.get("target_qualified", False))]
    generation_success = all(_int(_dict(row.get("result")).get("returncode")) == 0 for row in generation_runs)
    grammar_all = all(
        _int(_dict(_dict(row.get("report")).get("summary")).get("grammar_gate_sample_count"))
        == _int(_dict(_dict(row.get("report")).get("summary")).get("sample_count"))
        for row in generation_runs
    )
    target_passed = bool(
        generation_success
        and grammar_all
        and len(target_qualified) >= int(args.min_target_qualified)
    )
    top = _dict(target_qualified[0]) if target_qualified else _dict(ranked_candidates[0]) if ranked_candidates else {}
    top_sparse = _dict(top.get("sparse_phrase_metrics"))
    observed_baseline = _dict(decision_report.get("observed_evidence"))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decision_report_path": str(decision_report_path),
        "source_decision_schema": str(decision_report.get("schema_version") or ""),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples_per_cap": int(args.num_samples),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "note_groups_per_bar": int(args.note_groups_per_bar),
            "interval_caps": parse_int_list(str(args.interval_caps)),
            "coverage_aware_positions": bool(args.coverage_aware_positions),
            "coverage_position_window": int(args.coverage_position_window),
            "pitch_range": [int(args.pitch_min), int(args.pitch_max)],
            "applied_max_tail_empty_steps": int(args.applied_max_tail_empty_steps),
            "decision_max_tail_empty_steps": _int(thresholds.get("max_tail_empty_steps")),
            "min_target_qualified": int(args.min_target_qualified),
        },
        "target_thresholds": dict(thresholds),
        "baseline_evidence": {
            "source_gap_ratio_max": _float(observed_baseline.get("gap_ratio_max")),
            "source_max_internal_gap_beats_max": _float(observed_baseline.get("max_internal_gap_beats_max")),
        },
        "generation_runs": runs,
        "sparse_phrase_repair": {
            "target_passed": target_passed,
            "target_qualified_count": len(target_qualified),
            "candidate_count": len(ranked_candidates),
            "generation_success": generation_success,
            "all_samples_grammar_valid": grammar_all,
            "top_gap_ratio_to_window": _float(top_sparse.get("gap_ratio_to_window")),
            "top_max_internal_gap_beats": _float(top_sparse.get("max_internal_gap_beats")),
            "gap_ratio_reduced_vs_source_max": (
                _float(top_sparse.get("gap_ratio_to_window")) < _float(observed_baseline.get("gap_ratio_max"))
            ),
            "max_internal_gap_reduced_vs_source_max": (
                _float(top_sparse.get("max_internal_gap_beats")) < _float(observed_baseline.get("max_internal_gap_beats_max"))
            ),
            "ranked_candidates": ranked_candidates,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "sparse_phrase_repair_target_passed": target_passed,
            "human_audio_preference_claimed": False,
            "musical_quality_claimed": False,
            "quality_cause_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_audio_render_package"
                if target_passed
                else "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_sweep_tuning"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "sparse phrase objective gate evaluated after range/interval guard",
        },
        "not_proven": [
            "human_audio_keep",
            "human_audio_preference",
            "musical_quality",
            "quality_root_cause",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase audio render package"
            if target_passed
            else "Stage B generic tiny checkpoint repair phrase continuation range interval guard sparse phrase sweep tuning"
        ),
    }


def validate_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_generation_runs: int,
    min_candidate_count: int,
    min_target_qualified: int,
    require_gap_reduction: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("sparse_phrase_repair"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if len(_list(report.get("generation_runs"))) < int(min_generation_runs):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "generation run count below target"
        )
    if _int(repair.get("candidate_count")) < int(min_candidate_count):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "candidate count below target"
        )
    if _int(repair.get("target_qualified_count")) < int(min_target_qualified):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "target-qualified count below target"
        )
    if not bool(repair.get("generation_success", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "generation command failed"
        )
    if not bool(repair.get("all_samples_grammar_valid", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "not all samples passed grammar gate"
        )
    if require_gap_reduction and not bool(repair.get("gap_ratio_reduced_vs_source_max", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "gap ratio reduction required"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("human_audio_preference_claimed", True)),
            bool(readiness.get("musical_quality_claimed", True)),
            bool(readiness.get("quality_cause_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
                "quality claims must not be set"
            )
    top = _dict(_list(repair.get("ranked_candidates"))[0]) if _list(repair.get("ranked_candidates")) else {}
    top_sparse = _dict(top.get("sparse_phrase_metrics"))
    top_audit = _dict(top.get("midi_note_audit"))
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "target_passed": bool(repair.get("target_passed", False)),
        "target_qualified_count": _int(repair.get("target_qualified_count")),
        "candidate_count": _int(repair.get("candidate_count")),
        "top_interval_cap": _int(top.get("interval_cap")),
        "top_sample_seed": _int(top.get("sample_seed")),
        "top_sample_index": _int(top.get("sample_index")),
        "top_note_count": _int(top.get("note_count")),
        "top_gap_ratio_to_window": _float(top_sparse.get("gap_ratio_to_window")),
        "top_max_internal_gap_beats": _float(top_sparse.get("max_internal_gap_beats")),
        "top_max_abs_interval": _int(top_audit.get("max_abs_interval")),
        "top_target_qualified": bool(top.get("target_qualified", False)),
        "gap_ratio_reduced_vs_source_max": bool(repair.get("gap_ratio_reduced_vs_source_max", False)),
        "max_internal_gap_reduced_vs_source_max": bool(repair.get("max_internal_gap_reduced_vs_source_max", False)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    repair = report["sparse_phrase_repair"]
    baseline = report["baseline_evidence"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sparse Phrase Repair Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- target passed: `{_bool_token(repair['target_passed'])}`",
        f"- target qualified count: `{repair['target_qualified_count']}`",
        f"- candidate count: `{repair['candidate_count']}`",
        f"- top gap ratio / source max: `{repair['top_gap_ratio_to_window']}` / `{baseline['source_gap_ratio_max']}`",
        f"- top max internal gap / source max: `{repair['top_max_internal_gap_beats']}` / `{baseline['source_max_internal_gap_beats_max']}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
        "",
        "## Generation Runs",
        "",
        "| interval cap | samples | valid | strict | grammar | target qualified | collapse warning rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in report.get("generation_runs", []):
        summary = _dict(run.get("summary"))
        lines.append(
            "| "
            + " | ".join(
                [
                    str(run.get("interval_cap")),
                    str(summary.get("sample_count")),
                    str(summary.get("valid_sample_count")),
                    str(summary.get("strict_valid_sample_count")),
                    str(summary.get("grammar_gate_sample_count")),
                    str(run.get("target_qualified_count")),
                    f"{_float(summary.get('collapse_warning_sample_rate')):.3f}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Top Candidates",
            "",
            "| rank | cap | seed | sample | target | notes | gap ratio | max gap | tail | max interval | hard failures | soft failures | midi |",
            "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for rank, candidate in enumerate(repair.get("ranked_candidates", [])[:8], start=1):
        sparse = _dict(candidate.get("sparse_phrase_metrics"))
        audit = _dict(candidate.get("midi_note_audit"))
        hard = ", ".join(_list(candidate.get("target_failure_reasons"))) or "none"
        soft = ", ".join(_list(candidate.get("soft_failure_reasons"))) or "none"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(rank),
                    str(candidate.get("interval_cap")),
                    str(candidate.get("sample_seed")),
                    str(candidate.get("sample_index")),
                    _bool_token(bool(candidate.get("target_qualified", False))),
                    str(candidate.get("note_count")),
                    f"{_float(sparse.get('gap_ratio_to_window')):.4f}",
                    f"{_float(sparse.get('max_internal_gap_beats')):.4f}",
                    str(candidate.get("tail_empty_steps")),
                    str(audit.get("max_abs_interval")),
                    hard,
                    soft,
                    str(candidate.get("midi_path")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run sparse phrase repair sweep")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_decision.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/"
            "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=435)
    parser.add_argument("--max_sequence", type=int, default=180)
    parser.add_argument("--num_samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=80)
    parser.add_argument("--temperature", type=float, default=0.74)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--note_groups_per_bar", type=int, default=10)
    parser.add_argument("--chord_pitch_mode", type=str, default="tones_tensions")
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--interval_caps", type=str, default="9,7,5")
    parser.add_argument("--pitch_min", type=int, default=48)
    parser.add_argument("--pitch_max", type=int, default=84)
    parser.add_argument("--coverage_aware_positions", action="store_true")
    parser.add_argument("--coverage_position_window", type=int, default=0)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--max_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--max_pitch_span", type=int, default=24)
    parser.add_argument("--max_large_interval_ratio", type=float, default=0.35)
    parser.add_argument("--large_interval_threshold", type=int, default=12)
    parser.add_argument("--severe_interval_threshold", type=int, default=24)
    parser.add_argument("--phrase_window_beats", type=float, default=8.0)
    parser.add_argument("--applied_max_tail_empty_steps", type=int, default=1)
    parser.add_argument("--min_target_qualified", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_gap_reduction", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "checkpoint required"
        )
    decision_report_path = Path(args.decision_report)
    if not decision_report_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
            "sparse phrase repair decision report required"
        )
    decision_report = read_json(decision_report_path)
    validate_sparse_phrase_repair_decision(decision_report)
    probe_output_root = run_dir / "generation_probe"
    generation_runs: list[dict[str, Any]] = []
    for interval_cap in parse_int_list(str(args.interval_caps)):
        probe_run_id = f"cap_{interval_cap}_seed_{int(args.seed)}"
        command = build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            output_root=probe_output_root,
            run_id=probe_run_id,
            interval_cap=interval_cap,
        )
        result = run_command(command)
        report_path = probe_output_root / probe_run_id / "report.json"
        if not report_path.exists():
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSparsePhraseRepairSweepError(
                f"generation report not produced: {report_path}"
            )
        generation_runs.append(
            {
                "interval_cap": int(interval_cap),
                "result": result,
                "report_path": str(report_path),
                "report": read_json(report_path),
            }
        )
    report = build_sweep_report(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        decision_report_path=decision_report_path,
        decision_report=decision_report,
        generation_runs=generation_runs,
        args=args,
    )
    summary = validate_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_generation_runs=len(parse_int_list(str(args.interval_caps))),
        min_candidate_count=len(parse_int_list(str(args.interval_caps))) * int(args.num_samples),
        min_target_qualified=int(args.min_target_qualified),
        require_gap_reduction=bool(args.require_gap_reduction),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        run_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep.json",
        report,
    )
    write_json(
        run_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        run_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sparse_phrase_repair_sweep.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
