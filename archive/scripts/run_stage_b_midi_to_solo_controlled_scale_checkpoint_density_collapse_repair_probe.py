"""Run density/collapse repair probe for the controlled MIDI-to-solo checkpoint."""

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

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_repair import (  # noqa: E402
    BOUNDARY as DECISION_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe import (  # noqa: E402
    BOUNDARY as BASELINE_BOUNDARY,
    read_json,
)


class StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe"
PASS_NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_repeatability_probe"
FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe_v1"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def _count_prefixed(reasons: dict[str, Any], prefix: str) -> int:
    return sum(_int(count) for reason, count in reasons.items() if str(reason).startswith(prefix))


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


def validate_decision(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != DECISION_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("repair decision boundary required")
    decision = _dict(report.get("repair_decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("decision must route to repair probe")
    if str(decision.get("selected_target") or "") != "target_density_collapse_postprocess_repair":
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("density/collapse repair target required")
    if not bool(evidence.get("collapse_across_all_samples", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("collapse evidence required")
    blocked = [
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked if bool(claim.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError(f"unexpected quality claim: {claimed}")
    return {
        "selected_target": str(decision.get("selected_target") or ""),
        "sample_count": _int(evidence.get("sample_count")),
        "note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(evidence.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
    }


def validate_baseline(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != BASELINE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("baseline generation boundary required")
    generation = _dict(report.get("generation_summary"))
    readiness = _dict(report.get("readiness"))
    if not bool(readiness.get("generation_path_executable", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("baseline generation path required")
    if _int(generation.get("sample_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("baseline samples required")
    if _int(generation.get("note_count_failure_count")) <= 0 and not _dict(generation.get("diagnostic_failure_reasons")):
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("baseline failure evidence required")
    diagnostic = _dict(generation.get("diagnostic_failure_reasons"))
    note_count_failures = _count_prefixed(diagnostic, "note count too low:") or _int(
        generation.get("note_count_failure_count")
    )
    return {
        "sample_count": _int(generation.get("sample_count")),
        "valid_sample_count": _int(generation.get("valid_sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "note_count_failure_count": note_count_failures,
        "collapse_warning_sample_count": _int(generation.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(generation.get("collapse_warning_sample_rate")),
        "avg_postprocess_removal_ratio": _float(generation.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(generation.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(generation.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(generation.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            generation.get("max_longest_sustained_empty_run_steps")
        ),
    }


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
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
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.constrained_note_groups_per_bar),
        "--coverage_aware_positions",
        "--coverage_position_window",
        str(args.coverage_position_window),
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        args.chord_pitch_mode,
        "--chord_pitch_repeat_window",
        str(args.chord_pitch_repeat_window),
        "--jazz_rhythm_positions",
        "--jazz_duration_tokens",
        "--cap_duration_to_next_position",
        "--fill_duration_to_next_position",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
    ]


def _repair_metrics(summary: dict[str, Any]) -> dict[str, Any]:
    diagnostic = _dict(summary.get("diagnostic_failure_reasons"))
    return {
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "note_count_failure_count": _count_prefixed(diagnostic, "note count too low:"),
        "dead_air_failure_count": _count_prefixed(diagnostic, "dead-air ratio too high:"),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(summary.get("max_longest_sustained_empty_run_steps")),
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
    }


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
    repair_summary = _repair_metrics(_dict(generation_report.get("summary")))
    note_count_failure_delta = baseline_summary["note_count_failure_count"] - repair_summary["note_count_failure_count"]
    collapse_warning_delta = baseline_summary["collapse_warning_sample_count"] - repair_summary["collapse_warning_sample_count"]
    postprocess_removal_delta = (
        baseline_summary["avg_postprocess_removal_ratio"] - repair_summary["avg_postprocess_removal_ratio"]
    )
    onset_coverage_delta = repair_summary["avg_onset_coverage_ratio"] - baseline_summary["avg_onset_coverage_ratio"]
    sustained_coverage_delta = (
        repair_summary["avg_sustained_coverage_ratio"] - baseline_summary["avg_sustained_coverage_ratio"]
    )
    target_supported = (
        _int(generation_result.get("returncode")) == 0
        and repair_summary["grammar_gate_sample_count"] >= baseline_summary["sample_count"]
        and note_count_failure_delta > 0
        and collapse_warning_delta > 0
        and postprocess_removal_delta > 0.0
        and onset_coverage_delta > 0.0
        and sustained_coverage_delta > 0.0
    )
    strict_recovered = bool(generation_report.get("passed_strict_review_gate", False))
    next_boundary = PASS_NEXT_BOUNDARY if strict_recovered else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "boundary": BOUNDARY,
        "source_boundary": DECISION_BOUNDARY,
        "baseline_boundary": BASELINE_BOUNDARY,
        "decision_summary": decision_summary,
        "baseline_summary": baseline_summary,
        "generation_report_path": str(generation_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "max_sequence": int(args.max_sequence),
            "generation_mode": "constrained",
            "constrained_note_groups_per_bar": int(args.constrained_note_groups_per_bar),
            "coverage_position_window": int(args.coverage_position_window),
            "chord_pitch_mode": args.chord_pitch_mode,
            "jazz_rhythm_positions": True,
            "jazz_duration_tokens": True,
            "cap_duration_to_next_position": True,
            "fill_duration_to_next_position": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "generation_command": generation_result,
        "repair_summary": repair_summary,
        "comparison": {
            "note_count_failure_delta": int(note_count_failure_delta),
            "collapse_warning_delta": int(collapse_warning_delta),
            "postprocess_removal_delta": float(postprocess_removal_delta),
            "onset_coverage_delta": float(onset_coverage_delta),
            "sustained_coverage_delta": float(sustained_coverage_delta),
            "density_collapse_target_supported": bool(target_supported),
            "strict_gate_recovered": bool(strict_recovered),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "density_collapse_repair_probe_completed": _int(generation_result.get("returncode")) == 0,
            "density_collapse_target_supported": bool(target_supported),
            "strict_gate_recovered": bool(strict_recovered),
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "density/collapse target support is measured; remaining dead-air blocker is routed separately "
                "when strict gate is not recovered"
            ),
        },
        "not_proven": [
            "strict_gate_recovered" if not strict_recovered else "repeatability",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled checkpoint repeatability probe"
            if strict_recovered
            else "Stage B MIDI-to-solo controlled checkpoint dead-air remaining blocker decision"
        ),
    }


def validate_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_target_supported: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    repair = _dict(report.get("repair_summary"))
    comparison = _dict(report.get("comparison"))
    command = _dict(report.get("generation_command"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("unexpected boundary")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("unexpected next boundary")
    if _int(command.get("returncode")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("generation command must succeed")
    if require_target_supported and not bool(comparison.get("density_collapse_target_supported", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("repair target support required")
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "sample_count": _int(repair.get("sample_count")),
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(repair.get("note_count_failure_count")),
        "dead_air_failure_count": _int(repair.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(repair.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(repair.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(repair.get("avg_sustained_coverage_ratio")),
        "note_count_failure_delta": _int(comparison.get("note_count_failure_delta")),
        "collapse_warning_delta": _int(comparison.get("collapse_warning_delta")),
        "postprocess_removal_delta": _float(comparison.get("postprocess_removal_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
        "density_collapse_target_supported": bool(comparison.get("density_collapse_target_supported", False)),
        "strict_gate_recovered": bool(comparison.get("strict_gate_recovered", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(readiness.get("midi_to_solo_musical_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    repair = report["repair_summary"]
    comparison = report["comparison"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Density Collapse Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- density/collapse target supported: `{_bool_token(comparison['density_collapse_target_supported'])}`",
        f"- strict gate recovered: `{_bool_token(comparison['strict_gate_recovered'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Repair Result",
        "",
        f"- sample count: `{repair['sample_count']}`",
        f"- valid / strict / grammar: `{repair['valid_sample_count']}` / `{repair['strict_valid_sample_count']}` / `{repair['grammar_gate_sample_count']}`",
        f"- note-count / dead-air failure count: `{repair['note_count_failure_count']}` / `{repair['dead_air_failure_count']}`",
        f"- collapse warning count / rate: `{repair['collapse_warning_sample_count']}` / `{repair['collapse_warning_sample_rate']}`",
        f"- avg onset / sustained coverage ratio: `{repair['avg_onset_coverage_ratio']}` / `{repair['avg_sustained_coverage_ratio']}`",
        f"- avg / max postprocess removal ratio: `{repair['avg_postprocess_removal_ratio']}` / `{repair['max_postprocess_removal_ratio']}`",
        "",
        "## Comparison",
        "",
        f"- note count failure delta: `{comparison['note_count_failure_delta']}`",
        f"- collapse warning delta: `{comparison['collapse_warning_delta']}`",
        f"- postprocess removal delta: `{comparison['postprocess_removal_delta']}`",
        f"- onset / sustained coverage delta: `{comparison['onset_coverage_delta']}` / `{comparison['sustained_coverage_delta']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in repair["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run controlled checkpoint density/collapse repair probe")
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_repair_decision.json",
    )
    parser.add_argument(
        "--baseline_generation_probe",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_generation_probe.json",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=558)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--constrained_note_groups_per_bar", type=int, default=8)
    parser.add_argument("--coverage_position_window", type=int, default=1)
    parser.add_argument("--chord_pitch_mode", type=str, default="approach_tensions")
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_target_supported", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    decision_summary = validate_decision(read_json(Path(args.decision_report)))
    baseline_summary = validate_baseline(read_json(Path(args.baseline_generation_probe)))
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBMidiToSoloControlledScaleCheckpointDensityCollapseRepairProbeError("checkpoint_epoch1.pt required")

    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "density_collapse_repair"
    generation_result = run_command(
        build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            output_root=probe_output_root,
            run_id=probe_run_id,
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
        require_target_supported=bool(args.require_target_supported),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(run_dir / "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe.json", report)
    write_json(
        run_dir / "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
