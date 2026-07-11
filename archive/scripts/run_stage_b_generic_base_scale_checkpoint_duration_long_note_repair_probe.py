"""Run duration/long-note repair probe for the generic-base scale checkpoint."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import read_json, write_json, write_text  # noqa: E402


class StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision"
BASELINE_REPAIR_BOUNDARY = "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe"
NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_duration_long_note_remaining_blocker_decision"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def run_command(command: Sequence[str]) -> dict[str, Any]:
    completed = subprocess.run(
        list(command),
        cwd=str(ROOT_DIR),
        check=False,
        text=True,
        capture_output=True,
    )
    return {
        "cmd": list(command),
        "returncode": int(completed.returncode),
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def long_note_failure_count(summary: dict[str, Any]) -> int:
    diagnostic = _dict(summary.get("diagnostic_failure_reasons"))
    return sum(
        _int(count)
        for reason, count in diagnostic.items()
        if str(reason).startswith("too many long notes:")
    )


def validate_remaining_blocker_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    if str(decision.get("current_boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "remaining blocker decision boundary required"
        )
    if str(decision.get("selected_target") or "") != "duration_long_note_ratio_repair":
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "duration long-note repair target required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "decision must route to duration long-note repair probe"
        )
    if bool(decision.get("audio_review_selected", True)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "audio review must not be selected"
        )
    if bool(decision.get("quality_root_cause_claimed", True)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "quality root cause must not be claimed"
        )
    if bool(claim.get("musical_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "musical quality must not be claimed"
        )
    if _int(evidence.get("long_note_failure_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "long-note failure evidence required"
        )
    return {
        "selected_target": str(decision.get("selected_target") or ""),
        "source_long_note_failure_count": _int(evidence.get("long_note_failure_count")),
        "source_sample_count": _int(evidence.get("sample_count")),
        "source_valid_sample_count": _int(evidence.get("valid_sample_count")),
        "source_strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "source_grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "remaining_blocker": str(evidence.get("remaining_blocker") or ""),
    }


def validate_density_coverage_repair(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("repair_summary"))
    baseline = _dict(report.get("baseline_summary"))
    comparison = _dict(report.get("comparison"))
    if str(readiness.get("boundary") or "") != BASELINE_REPAIR_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "density/coverage repair boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "density/coverage repair must route to remaining blocker decision"
        )
    if not bool(readiness.get("density_coverage_target_qualified", False)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "density/coverage target must be qualified"
        )
    if bool(readiness.get("raw_generation_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "raw generation quality must not be claimed"
        )
    if bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "broad trained-model quality must not be claimed"
        )
    if bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "Brad style adaptation must not be claimed"
        )
    source_long_note_failures = long_note_failure_count(repair)
    if source_long_note_failures <= 0:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "source long-note failure evidence required"
        )
    checkpoint_dir = Path(str(baseline.get("checkpoint_dir") or ""))
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "scale checkpoint required"
        )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "sample_count": _int(repair.get("sample_count")),
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "long_note_failure_count": int(source_long_note_failures),
        "failure_reasons": _dict(repair.get("failure_reasons")),
        "diagnostic_failure_reasons": _dict(repair.get("diagnostic_failure_reasons")),
        "avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(repair.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            repair.get("max_longest_sustained_empty_run_steps")
        ),
        "note_count_failure_delta": _int(comparison.get("note_count_failure_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
    }


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    probe_output_root: Path,
    probe_run_id: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_stage_b_generation_probe.py",
        "--output_root",
        str(probe_output_root),
        "--run_id",
        probe_run_id,
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
        str(args.constrained_note_groups_per_bar),
        "--coverage_aware_positions",
        "--coverage_position_window",
        str(args.coverage_position_window),
        "--jazz_duration_tokens",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
    ]


def build_repair_report(
    *,
    run_dir: Path,
    decision_summary: dict[str, Any],
    source_repair_summary: dict[str, Any],
    generation_report_path: Path,
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    duration_summary = _dict(generation_report.get("summary"))
    duration_long_note_failures = long_note_failure_count(duration_summary)
    source_long_note_failures = _int(source_repair_summary.get("long_note_failure_count"))
    long_note_failure_delta = source_long_note_failures - duration_long_note_failures
    valid_sample_delta = _int(duration_summary.get("valid_sample_count")) - _int(
        source_repair_summary.get("valid_sample_count")
    )
    strict_sample_delta = _int(duration_summary.get("strict_valid_sample_count")) - _int(
        source_repair_summary.get("strict_valid_sample_count")
    )
    grammar_gate_delta = _int(duration_summary.get("grammar_gate_sample_count")) - _int(
        source_repair_summary.get("grammar_gate_sample_count")
    )
    onset_delta = _float(duration_summary.get("avg_onset_coverage_ratio")) - _float(
        source_repair_summary.get("avg_onset_coverage_ratio")
    )
    sustained_delta = _float(duration_summary.get("avg_sustained_coverage_ratio")) - _float(
        source_repair_summary.get("avg_sustained_coverage_ratio")
    )
    longest_sustained_empty_run_delta = _int(
        source_repair_summary.get("max_longest_sustained_empty_run_steps")
    ) - _int(duration_summary.get("max_longest_sustained_empty_run_steps"))
    target_qualified = (
        _int(generation_result.get("returncode")) == 0
        and _int(duration_summary.get("sample_count")) > 0
        and bool(generation_report.get("passed_strict_review_gate", False))
        and long_note_failure_delta > 0
        and duration_long_note_failures == 0
    )
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "input_boundary": SOURCE_BOUNDARY,
        "baseline_repair_boundary": BASELINE_REPAIR_BOUNDARY,
        "decision_summary": decision_summary,
        "source_repair_summary": source_repair_summary,
        "generation_report_path": str(generation_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "max_sequence": int(args.max_sequence),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "generation_mode": "constrained",
            "constrained_note_groups_per_bar": int(args.constrained_note_groups_per_bar),
            "coverage_aware_positions": True,
            "coverage_position_window": int(args.coverage_position_window),
            "jazz_duration_tokens": True,
            "postprocess_overlap": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "generation_command": generation_result,
        "duration_repair_summary": {
            "sample_count": _int(duration_summary.get("sample_count")),
            "valid_sample_count": _int(duration_summary.get("valid_sample_count")),
            "strict_valid_sample_count": _int(duration_summary.get("strict_valid_sample_count")),
            "grammar_gate_sample_count": _int(duration_summary.get("grammar_gate_sample_count")),
            "passed_generation_gate": bool(generation_report.get("passed_generation_gate", False)),
            "passed_grammar_gate": bool(generation_report.get("passed_grammar_gate", False)),
            "passed_strict_review_gate": bool(generation_report.get("passed_strict_review_gate", False)),
            "long_note_failure_count": int(duration_long_note_failures),
            "failure_reasons": _dict(duration_summary.get("failure_reasons")),
            "diagnostic_failure_reasons": _dict(duration_summary.get("diagnostic_failure_reasons")),
            "avg_onset_coverage_ratio": _float(duration_summary.get("avg_onset_coverage_ratio")),
            "avg_sustained_coverage_ratio": _float(duration_summary.get("avg_sustained_coverage_ratio")),
            "max_longest_sustained_empty_run_steps": _int(
                duration_summary.get("max_longest_sustained_empty_run_steps")
            ),
            "collapse_warning_sample_rate": _float(duration_summary.get("collapse_warning_sample_rate")),
            "avg_duration_diversity_ratio": _float(duration_summary.get("avg_duration_diversity_ratio")),
            "avg_most_common_duration_ratio": _float(duration_summary.get("avg_most_common_duration_ratio")),
        },
        "comparison": {
            "long_note_failure_delta": int(long_note_failure_delta),
            "valid_sample_delta": int(valid_sample_delta),
            "strict_valid_sample_delta": int(strict_sample_delta),
            "grammar_gate_sample_delta": int(grammar_gate_delta),
            "onset_coverage_delta": float(onset_delta),
            "sustained_coverage_delta": float(sustained_delta),
            "longest_sustained_empty_run_delta": int(longest_sustained_empty_run_delta),
            "coverage_regression_observed": bool(onset_delta < 0.0 or sustained_delta < 0.0),
            "target_qualified": bool(target_qualified),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "duration_long_note_repair_probe_completed": _int(generation_result.get("returncode")) == 0,
            "duration_long_note_target_qualified": bool(target_qualified),
            "raw_generation_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "duration-token constraint removed the long-note failure count; remaining failure reasons "
                "and coverage regression require a separate boundary before any listening or quality claim"
            ),
        },
        "not_proven": [
            "raw_generation_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base scale checkpoint duration long-note remaining blocker decision",
    }


def validate_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_target_qualified: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("generation_command"))
    duration = _dict(report.get("duration_repair_summary"))
    comparison = _dict(report.get("comparison"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if _int(generation.get("returncode")) != 0:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "generation command must succeed"
        )
    if _int(duration.get("sample_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "duration repair sample count required"
        )
    if require_target_qualified and not bool(
        readiness.get("duration_long_note_target_qualified", False)
    ):
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "duration long-note target should qualify"
        )
    if _int(comparison.get("long_note_failure_delta")) <= 0:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "long-note failure should improve"
        )
    if _int(duration.get("long_note_failure_count")) != 0:
        raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
            "long-note failure should be removed"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("raw_generation_quality_claimed", True)),
            bool(readiness.get("human_audio_preference_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
            bool(readiness.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericBaseScaleCheckpointDurationLongNoteRepairProbeError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "duration_long_note_repair_probe_completed": bool(
            readiness.get("duration_long_note_repair_probe_completed", False)
        ),
        "duration_long_note_target_qualified": bool(
            readiness.get("duration_long_note_target_qualified", False)
        ),
        "sample_count": _int(duration.get("sample_count")),
        "valid_sample_count": _int(duration.get("valid_sample_count")),
        "strict_valid_sample_count": _int(duration.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(duration.get("grammar_gate_sample_count")),
        "long_note_failure_count": _int(duration.get("long_note_failure_count")),
        "long_note_failure_delta": _int(comparison.get("long_note_failure_delta")),
        "valid_sample_delta": _int(comparison.get("valid_sample_delta")),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
        "coverage_regression_observed": bool(comparison.get("coverage_regression_observed", False)),
        "raw_generation_quality_claimed": bool(readiness.get("raw_generation_quality_claimed", True)),
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
    source = report["source_repair_summary"]
    duration = report["duration_repair_summary"]
    comparison = report["comparison"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Duration Long-Note Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- duration long-note target qualified: `{_bool_token(readiness['duration_long_note_target_qualified'])}`",
        f"- raw generation quality claimed: `{_bool_token(readiness['raw_generation_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Source Repair",
        "",
        f"- sample count: `{source['sample_count']}`",
        f"- valid / strict / grammar gate: `{source['valid_sample_count']}` / `{source['strict_valid_sample_count']}` / `{source['grammar_gate_sample_count']}`",
        f"- long-note failure count: `{source['long_note_failure_count']}`",
        f"- avg onset / sustained coverage: `{source['avg_onset_coverage_ratio']}` / `{source['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{source['max_longest_sustained_empty_run_steps']}`",
        "",
        "## Duration Repair",
        "",
        f"- sample count: `{duration['sample_count']}`",
        f"- valid / strict / grammar gate: `{duration['valid_sample_count']}` / `{duration['strict_valid_sample_count']}` / `{duration['grammar_gate_sample_count']}`",
        f"- long-note failure count: `{duration['long_note_failure_count']}`",
        f"- avg onset / sustained coverage: `{duration['avg_onset_coverage_ratio']}` / `{duration['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{duration['max_longest_sustained_empty_run_steps']}`",
        f"- avg duration diversity / most common duration ratio: `{duration['avg_duration_diversity_ratio']}` / `{duration['avg_most_common_duration_ratio']}`",
        "",
        "## Delta",
        "",
        f"- long-note failure delta: `{comparison['long_note_failure_delta']}`",
        f"- valid / strict sample delta: `{comparison['valid_sample_delta']}` / `{comparison['strict_valid_sample_delta']}`",
        f"- onset / sustained coverage delta: `{comparison['onset_coverage_delta']}` / `{comparison['sustained_coverage_delta']}`",
        f"- coverage regression observed: `{_bool_token(comparison['coverage_regression_observed'])}`",
        "",
        "## Remaining Failure Reasons",
        "",
    ]
    for reason, count in duration["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Stage B generic base scale checkpoint duration/long-note repair"
    )
    parser.add_argument(
        "--remaining_blocker_decision",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision/"
        "harness_stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision/"
        "stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision.json",
    )
    parser.add_argument(
        "--density_coverage_repair",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe/"
        "harness_stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe/"
        "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=461)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--constrained_note_groups_per_bar", type=int, default=4)
    parser.add_argument("--coverage_position_window", type=int, default=1)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_target_qualified", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    decision_summary = validate_remaining_blocker_decision(
        read_json(Path(args.remaining_blocker_decision))
    )
    source_repair_summary = validate_density_coverage_repair(
        read_json(Path(args.density_coverage_repair))
    )
    checkpoint_dir = Path(str(source_repair_summary["checkpoint_dir"]))
    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "duration_long_note_repair"
    generation_result = run_command(
        build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            probe_output_root=probe_output_root,
            probe_run_id=probe_run_id,
        )
    )
    generation_report_path = probe_output_root / probe_run_id / "report.json"
    generation_report = read_json(generation_report_path) if generation_report_path.exists() else {}
    report = build_repair_report(
        run_dir=run_dir,
        decision_summary=decision_summary,
        source_repair_summary=source_repair_summary,
        generation_report_path=generation_report_path,
        generation_result=generation_result,
        generation_report=generation_report,
        args=args,
    )
    summary = validate_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_target_qualified=bool(args.require_target_qualified),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(run_dir / "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe.json", report)
    write_json(
        run_dir / "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_base_scale_checkpoint_duration_long_note_repair_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
