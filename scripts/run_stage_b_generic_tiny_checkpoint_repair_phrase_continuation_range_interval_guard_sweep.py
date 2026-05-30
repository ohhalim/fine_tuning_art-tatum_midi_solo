"""Run range/interval guard sweep for phrase-continuation repair candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

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


class StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(ValueError):
    pass


SCHEMA_VERSION = "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep_v1"
BOUNDARY = "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep"


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in str(value or "").split(",") if item.strip()]
    if not items:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "at least one interval cap is required"
        )
    return [int(item) for item in items]


def validate_guard_decision(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    targets = _dict(report.get("guard_targets"))
    if str(readiness.get("boundary") or "") != (
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision"
    ):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "range/interval guard decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "unexpected range/interval guard next boundary"
        )
    if bool(readiness.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "source decision must not claim musical quality"
        )
    required = (
        "max_pitch_span",
        "max_abs_interval",
        "max_large_interval_ratio",
        "max_severe_interval_count",
        "preferred_pitch_floor",
        "preferred_pitch_ceiling",
        "large_interval_threshold",
        "severe_interval_threshold",
    )
    missing = [key for key in required if key not in targets]
    if missing:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            f"guard targets missing: {missing}"
        )
    return targets


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
    interval_cap: int,
    targets: dict[str, Any],
) -> list[str]:
    return [
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
        str(_int(targets.get("preferred_pitch_floor"))),
        "--constrained_pitch_max",
        str(_int(targets.get("preferred_pitch_ceiling"))),
        "--constrained_max_adjacent_interval",
        str(interval_cap),
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--require_all_grammar_samples",
    ]


def target_failure_reasons(row: dict[str, Any], audit: dict[str, Any], *, args: argparse.Namespace) -> list[str]:
    temporal = _dict(row.get("temporal_coverage"))
    collapse = _dict(row.get("collapse"))
    metrics = _dict(row.get("metrics"))
    targets = _dict(audit.get("targets"))
    reasons: list[str] = []
    if not bool(row.get("grammar_gate_passed", False)):
        reasons.append("grammar_gate_failed")
    if not bool(row.get("strict_valid", False)):
        reasons.append("strict_valid_failed")
    if _int(metrics.get("note_count")) < int(args.min_note_count):
        reasons.append("note_count_below_target")
    if _float(metrics.get("phrase_coverage_ratio")) < float(args.min_phrase_coverage_ratio):
        reasons.append("phrase_coverage_below_target")
    if _int(temporal.get("tail_empty_steps")) > int(args.max_tail_empty_steps):
        reasons.append("tail_empty_above_target")
    if _int(metrics.get("max_simultaneous_notes")) > int(args.max_simultaneous_notes):
        reasons.append("max_simultaneous_notes_above_target")
    if _float(collapse.get("postprocess_removal_ratio")) > float(args.max_postprocess_removal_ratio):
        reasons.append("postprocess_removal_above_target")
    if _int(audit.get("pitch_span")) > _int(targets.get("max_pitch_span")):
        reasons.append("pitch_span_above_target")
    if _int(audit.get("max_abs_interval")) > _int(targets.get("max_abs_interval")):
        reasons.append("max_interval_above_target")
    if _float(audit.get("large_interval_ratio")) > _float(targets.get("max_large_interval_ratio")):
        reasons.append("large_interval_ratio_above_target")
    if _int(audit.get("severe_interval_count")) > int(args.max_severe_interval_count):
        reasons.append("severe_interval_present")
    return reasons


def compact_candidate(
    row: dict[str, Any],
    *,
    interval_cap: int,
    args: argparse.Namespace,
    targets: dict[str, Any],
) -> dict[str, Any]:
    midi_path = Path(str(row.get("midi_path") or ""))
    audit = note_audit(
        midi_path,
        max_pitch_span=_int(targets.get("max_pitch_span")),
        max_abs_interval=_int(targets.get("max_abs_interval")),
        max_large_interval_ratio=_float(targets.get("max_large_interval_ratio")),
        large_interval_threshold=_int(targets.get("large_interval_threshold")),
        severe_interval_threshold=_int(targets.get("severe_interval_threshold")),
    )
    reasons = target_failure_reasons(row, audit, args=args)
    temporal = _dict(row.get("temporal_coverage"))
    collapse = _dict(row.get("collapse"))
    pitch_roles = _dict(row.get("pitch_roles"))
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
        "note_count": _int(metrics.get("note_count")),
        "phrase_coverage_ratio": _float(metrics.get("phrase_coverage_ratio")),
        "dead_air_ratio": _float(metrics.get("dead_air_ratio")),
        "tail_empty_steps": _int(temporal.get("tail_empty_steps")),
        "position_span_ratio": _float(temporal.get("position_span_ratio")),
        "postprocess_removal_ratio": _float(collapse.get("postprocess_removal_ratio")),
        "max_simultaneous_notes": _int(metrics.get("max_simultaneous_notes")),
        "pitch_role_chord_tone_ratio": _float(pitch_roles.get("chord_tone_ratio")),
        "midi_note_audit": {
            "note_count": _int(audit.get("note_count")),
            "pitch_min": audit.get("pitch_min"),
            "pitch_max": audit.get("pitch_max"),
            "pitch_span": _int(audit.get("pitch_span")),
            "pitch_sequence": _list(audit.get("pitch_sequence")),
            "pitch_name_sequence": _list(audit.get("pitch_name_sequence")),
            "intervals": _list(audit.get("intervals")),
            "max_abs_interval": _int(audit.get("max_abs_interval")),
            "large_interval_count": _int(audit.get("large_interval_count")),
            "large_interval_ratio": _float(audit.get("large_interval_ratio")),
            "severe_interval_count": _int(audit.get("severe_interval_count")),
            "failure_reasons": _list(audit.get("failure_reasons")),
        },
    }


def sort_candidates(candidates: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [dict(row) for row in candidates],
        key=lambda row: (
            not bool(row.get("target_qualified", False)),
            _int(_dict(row.get("midi_note_audit")).get("max_abs_interval")),
            _int(_dict(row.get("midi_note_audit")).get("pitch_span")),
            -_float(row.get("phrase_coverage_ratio")),
            _float(row.get("postprocess_removal_ratio")),
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
    targets = validate_guard_decision(decision_report)
    runs: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for generation_run in generation_runs:
        generation_report = _dict(generation_run.get("report"))
        summary = _dict(generation_report.get("summary"))
        interval_cap = _int(generation_run.get("interval_cap"))
        candidates = [
            compact_candidate(row, interval_cap=interval_cap, args=args, targets=targets)
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
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "decision_report_path": str(decision_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples_per_cap": int(args.num_samples),
            "seed": int(args.seed),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "note_groups_per_bar": int(args.note_groups_per_bar),
            "interval_caps": parse_int_list(str(args.interval_caps)),
            "pitch_range": [
                _int(targets.get("preferred_pitch_floor")),
                _int(targets.get("preferred_pitch_ceiling")),
            ],
            "min_note_count": int(args.min_note_count),
            "min_phrase_coverage_ratio": float(args.min_phrase_coverage_ratio),
            "max_tail_empty_steps": int(args.max_tail_empty_steps),
            "max_postprocess_removal_ratio": float(args.max_postprocess_removal_ratio),
            "min_target_qualified": int(args.min_target_qualified),
        },
        "guard_targets": targets,
        "generation_runs": runs,
        "range_interval_guard": {
            "target_passed": target_passed,
            "target_qualified_count": len(target_qualified),
            "candidate_count": len(ranked_candidates),
            "generation_success": generation_success,
            "all_samples_grammar_valid": grammar_all,
            "ranked_candidates": ranked_candidates,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "range_interval_guard_target_passed": target_passed,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_audio_render_package"
                if target_passed
                else "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep_tuning"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "actual MIDI note range/interval guard evaluated after overlap postprocess",
        },
        "not_proven": [
            "audio_rendered_quality",
            "human_audio_keep",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair phrase continuation range interval guard audio render package"
            if target_passed
            else "Stage B generic tiny checkpoint repair phrase continuation range interval guard sweep tuning"
        ),
    }


def validate_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_generation_runs: int,
    min_candidate_count: int,
    min_target_qualified: int,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("range_interval_guard"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if len(_list(report.get("generation_runs"))) < int(min_generation_runs):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "generation run count below target"
        )
    if _int(guard.get("candidate_count")) < int(min_candidate_count):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "candidate count below target"
        )
    if _int(guard.get("target_qualified_count")) < int(min_target_qualified):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "target-qualified count below target"
        )
    if not bool(guard.get("generation_success", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "generation command failed"
        )
    if not bool(guard.get("all_samples_grammar_valid", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "not all samples passed grammar gate"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("musical_quality_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
                "quality claims must not be set"
            )
    top = _dict(_list(guard.get("ranked_candidates"))[0]) if _list(guard.get("ranked_candidates")) else {}
    top_audit = _dict(top.get("midi_note_audit"))
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "target_passed": bool(guard.get("target_passed", False)),
        "target_qualified_count": _int(guard.get("target_qualified_count")),
        "candidate_count": _int(guard.get("candidate_count")),
        "top_interval_cap": _int(top.get("interval_cap")),
        "top_sample_seed": _int(top.get("sample_seed")),
        "top_note_count": _int(top.get("note_count")),
        "top_phrase_coverage_ratio": _float(top.get("phrase_coverage_ratio")),
        "top_pitch_span": _int(top_audit.get("pitch_span")),
        "top_max_abs_interval": _int(top_audit.get("max_abs_interval")),
        "top_large_interval_ratio": _float(top_audit.get("large_interval_ratio")),
        "top_target_qualified": bool(top.get("target_qualified", False)),
        "musical_quality_claimed": bool(readiness.get("musical_quality_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    guard = report["range_interval_guard"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Range Interval Guard Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- target passed: `{_bool_token(guard['target_passed'])}`",
        f"- target qualified count: `{guard['target_qualified_count']}`",
        f"- candidate count: `{guard['candidate_count']}`",
        f"- generation success: `{_bool_token(guard['generation_success'])}`",
        f"- all samples grammar valid: `{_bool_token(guard['all_samples_grammar_valid'])}`",
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
            "| rank | cap | seed | sample | target | notes | coverage | tail | postprocess removal | pitch range | span | max interval | large ratio | failures | midi |",
            "|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---|---:|---:|---:|---|---|",
        ]
    )
    for rank, candidate in enumerate(guard.get("ranked_candidates", [])[:8], start=1):
        audit = _dict(candidate.get("midi_note_audit"))
        pitch_range = f"{audit.get('pitch_min')}-{audit.get('pitch_max')}"
        failures = ", ".join(_list(candidate.get("target_failure_reasons"))) or "none"
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
                    f"{_float(candidate.get('phrase_coverage_ratio')):.3f}",
                    str(candidate.get("tail_empty_steps")),
                    f"{_float(candidate.get('postprocess_removal_ratio')):.3f}",
                    pitch_range,
                    str(audit.get("pitch_span")),
                    str(audit.get("max_abs_interval")),
                    f"{_float(audit.get('large_interval_ratio')):.3f}",
                    failures,
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
    parser = argparse.ArgumentParser(description="Run phrase-continuation range/interval guard sweep")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision/"
        "harness_stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision/"
        "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_decision.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=423)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--num_samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=62)
    parser.add_argument("--temperature", type=float, default=0.78)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--note_groups_per_bar", type=int, default=8)
    parser.add_argument("--chord_pitch_mode", type=str, default="tones_tensions")
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--interval_caps", type=str, default="12,9,7,5")
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_note_count", type=int, default=8)
    parser.add_argument("--min_phrase_coverage_ratio", type=float, default=0.85)
    parser.add_argument("--max_tail_empty_steps", type=int, default=2)
    parser.add_argument("--max_postprocess_removal_ratio", type=float, default=0.49)
    parser.add_argument("--max_severe_interval_count", type=int, default=0)
    parser.add_argument("--min_target_qualified", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "checkpoint required"
        )
    decision_report_path = Path(args.decision_report)
    if not decision_report_path.exists():
        raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
            "range/interval guard decision report required"
        )
    decision_report = read_json(decision_report_path)
    targets = validate_guard_decision(decision_report)
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
            targets=targets,
        )
        result = run_command(command)
        report_path = probe_output_root / probe_run_id / "report.json"
        if not report_path.exists():
            raise StageBGenericTinyCheckpointRepairPhraseContinuationRangeIntervalGuardSweepError(
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
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep.json", report)
    write_json(
        run_dir
        / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_range_interval_guard_sweep.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
