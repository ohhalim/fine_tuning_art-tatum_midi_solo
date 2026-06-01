"""Run density/coverage repair probe for the generic-base scale checkpoint."""

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


class StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_grammar_representation_decision"
BASELINE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_generation_probe"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe"
NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_density_coverage_remaining_blocker_decision"


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


def note_count_failure_count(summary: dict[str, Any]) -> int:
    diagnostic = _dict(summary.get("diagnostic_failure_reasons"))
    return sum(
        _int(count)
        for reason, count in diagnostic.items()
        if str(reason).startswith("note count too low:")
    )


def validate_decision_report(report: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    if str(decision.get("current_boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("decision boundary required")
    if str(decision.get("selected_target") or "") != "target_density_coverage_repair":
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("density/coverage target required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("decision must route to repair probe")
    if bool(decision.get("postprocess_only_repair_selected", True)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("postprocess-only repair must be false")
    if bool(decision.get("audio_review_selected", True)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("audio review must not be selected")
    if bool(claim.get("quality_root_cause_claimed", True)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("quality root cause must not be claimed")
    if not bool(evidence.get("all_samples_note_count_failed", False)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("note-count failure evidence required")
    if not bool(evidence.get("low_coverage_observed", False)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("low coverage evidence required")
    return {
        "selected_target": str(decision.get("selected_target") or ""),
        "baseline_sample_count": _int(evidence.get("sample_count")),
        "baseline_note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "baseline_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "baseline_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
    }


def validate_baseline_generation_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    summary = _dict(report.get("generation_summary"))
    if str(readiness.get("boundary") or "") != BASELINE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("baseline generation boundary required")
    if not bool(readiness.get("generation_path_executable", False)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("baseline generation path required")
    if bool(readiness.get("raw_generation_quality_ready", True)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("baseline raw generation should fail")
    checkpoint_dir = Path(str(report.get("checkpoint_dir") or ""))
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("baseline checkpoint required")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "note_count_failure_count": note_count_failure_count(summary),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(summary.get("max_longest_sustained_empty_run_steps")),
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
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
    ]


def build_repair_report(
    *,
    run_dir: Path,
    decision_summary: dict[str, Any],
    baseline_summary: dict[str, Any],
    generation_report_path: Path,
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    repair_summary = _dict(generation_report.get("summary"))
    repair_note_count_failures = note_count_failure_count(repair_summary)
    onset_delta = _float(repair_summary.get("avg_onset_coverage_ratio")) - _float(
        baseline_summary.get("avg_onset_coverage_ratio")
    )
    sustained_delta = _float(repair_summary.get("avg_sustained_coverage_ratio")) - _float(
        baseline_summary.get("avg_sustained_coverage_ratio")
    )
    note_count_failure_delta = _int(baseline_summary.get("note_count_failure_count")) - repair_note_count_failures
    target_qualified = (
        _int(generation_result.get("returncode")) == 0
        and _int(repair_summary.get("sample_count")) > 0
        and bool(generation_report.get("passed_strict_review_gate", False))
        and note_count_failure_delta > 0
        and onset_delta > 0.0
        and sustained_delta > 0.0
    )
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "input_boundary": SOURCE_BOUNDARY,
        "baseline_boundary": BASELINE_BOUNDARY,
        "decision_summary": decision_summary,
        "baseline_summary": baseline_summary,
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
            "postprocess_overlap": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "generation_command": generation_result,
        "repair_summary": {
            "sample_count": _int(repair_summary.get("sample_count")),
            "valid_sample_count": _int(repair_summary.get("valid_sample_count")),
            "strict_valid_sample_count": _int(repair_summary.get("strict_valid_sample_count")),
            "grammar_gate_sample_count": _int(repair_summary.get("grammar_gate_sample_count")),
            "passed_generation_gate": bool(generation_report.get("passed_generation_gate", False)),
            "passed_grammar_gate": bool(generation_report.get("passed_grammar_gate", False)),
            "passed_strict_review_gate": bool(generation_report.get("passed_strict_review_gate", False)),
            "note_count_failure_count": int(repair_note_count_failures),
            "failure_reasons": _dict(repair_summary.get("failure_reasons")),
            "diagnostic_failure_reasons": _dict(repair_summary.get("diagnostic_failure_reasons")),
            "avg_onset_coverage_ratio": _float(repair_summary.get("avg_onset_coverage_ratio")),
            "avg_sustained_coverage_ratio": _float(repair_summary.get("avg_sustained_coverage_ratio")),
            "max_longest_sustained_empty_run_steps": _int(
                repair_summary.get("max_longest_sustained_empty_run_steps")
            ),
            "collapse_warning_sample_rate": _float(repair_summary.get("collapse_warning_sample_rate")),
        },
        "comparison": {
            "note_count_failure_delta": int(note_count_failure_delta),
            "onset_coverage_delta": float(onset_delta),
            "sustained_coverage_delta": float(sustained_delta),
            "target_qualified": bool(target_qualified),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "density_coverage_repair_probe_completed": _int(generation_result.get("returncode")) == 0,
            "density_coverage_target_qualified": bool(target_qualified),
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
                "density/coverage target improved gate and coverage metrics, but remaining failure reasons "
                "must be separated before any listening or quality claim"
            ),
        },
        "not_proven": [
            "raw_generation_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base scale checkpoint density coverage remaining blocker decision",
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
    repair = _dict(report.get("repair_summary"))
    comparison = _dict(report.get("comparison"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if _int(generation.get("returncode")) != 0:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("generation command must succeed")
    if _int(repair.get("sample_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("repair sample count required")
    if require_target_qualified and not bool(readiness.get("density_coverage_target_qualified", False)):
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("density/coverage target should qualify")
    if _int(comparison.get("note_count_failure_delta")) <= 0:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("note-count failure should improve")
    if _float(comparison.get("onset_coverage_delta")) <= 0.0:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("onset coverage should improve")
    if _float(comparison.get("sustained_coverage_delta")) <= 0.0:
        raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError("sustained coverage should improve")
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("raw_generation_quality_claimed", True)),
            bool(readiness.get("human_audio_preference_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
            bool(readiness.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericBaseScaleCheckpointDensityCoverageRepairProbeError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "density_coverage_repair_probe_completed": bool(
            readiness.get("density_coverage_repair_probe_completed", False)
        ),
        "density_coverage_target_qualified": bool(
            readiness.get("density_coverage_target_qualified", False)
        ),
        "sample_count": _int(repair.get("sample_count")),
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "note_count_failure_delta": _int(comparison.get("note_count_failure_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
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
    baseline = report["baseline_summary"]
    repair = report["repair_summary"]
    comparison = report["comparison"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Density Coverage Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- density/coverage target qualified: `{_bool_token(readiness['density_coverage_target_qualified'])}`",
        f"- raw generation quality claimed: `{_bool_token(readiness['raw_generation_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Baseline",
        "",
        f"- sample count: `{baseline['sample_count']}`",
        f"- valid / strict / grammar gate: `{baseline['valid_sample_count']}` / `{baseline['strict_valid_sample_count']}` / `{baseline['grammar_gate_sample_count']}`",
        f"- note count failure count: `{baseline['note_count_failure_count']}`",
        f"- avg onset / sustained coverage: `{baseline['avg_onset_coverage_ratio']}` / `{baseline['avg_sustained_coverage_ratio']}`",
        "",
        "## Repair",
        "",
        f"- sample count: `{repair['sample_count']}`",
        f"- valid / strict / grammar gate: `{repair['valid_sample_count']}` / `{repair['strict_valid_sample_count']}` / `{repair['grammar_gate_sample_count']}`",
        f"- note count failure count: `{repair['note_count_failure_count']}`",
        f"- avg onset / sustained coverage: `{repair['avg_onset_coverage_ratio']}` / `{repair['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{repair['max_longest_sustained_empty_run_steps']}`",
        "",
        "## Delta",
        "",
        f"- note count failure delta: `{comparison['note_count_failure_delta']}`",
        f"- onset coverage delta: `{comparison['onset_coverage_delta']}`",
        f"- sustained coverage delta: `{comparison['sustained_coverage_delta']}`",
        "",
        "## Remaining Failure Reasons",
        "",
    ]
    for reason, count in repair["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic base scale checkpoint density/coverage repair")
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_grammar_representation_decision/"
        "harness_stage_b_generic_base_scale_checkpoint_grammar_representation_decision/"
        "stage_b_generic_base_scale_checkpoint_grammar_representation_decision.json",
    )
    parser.add_argument(
        "--baseline_generation_probe",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_generation_probe/"
        "harness_stage_b_generic_base_scale_checkpoint_generation_probe/"
        "stage_b_generic_base_scale_checkpoint_generation_probe.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=457)
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

    decision_summary = validate_decision_report(read_json(Path(args.decision_report)))
    baseline_report = read_json(Path(args.baseline_generation_probe))
    baseline_summary = validate_baseline_generation_probe(baseline_report)
    checkpoint_dir = Path(str(baseline_summary["checkpoint_dir"]))
    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "density_coverage_repair"
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
        baseline_summary=baseline_summary,
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
    write_json(run_dir / "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe.json", report)
    write_json(
        run_dir / "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_base_scale_checkpoint_density_coverage_repair_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
