"""Run selected-scale checkpoint dead-air repair probe."""

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
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker import (  # noqa: E402
    BOUNDARY as DECISION_BOUNDARY,
    SOURCE_BOUNDARY as SOURCE_REPEATABILITY_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repair_probe"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repair_repeatability_probe"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repair_probe_v1"
)


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


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


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
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    if str(report.get("input_boundary") or "") != SOURCE_REPEATABILITY_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "source repeatability boundary required"
        )
    if str(decision.get("current_boundary") or "") != DECISION_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "dead-air decision boundary required"
        )
    if str(decision.get("selected_target") or "") != (
        "selected_scale_dead_air_sustained_coverage_repair"
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "selected-scale dead-air repair target required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "decision must route to selected-scale dead-air repair probe"
        )
    if bool(decision.get("density_grammar_collapse_followup_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "density/grammar/collapse follow-up must not be selected"
        )
    if bool(decision.get("audio_review_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "audio review must not be selected"
        )
    if bool(decision.get("additional_training_scale_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "additional training scale must not be selected"
        )
    if _int(evidence.get("dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "dead-air evidence required"
        )
    if _int(evidence.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "note-count failure must be removed before dead-air repair"
        )
    if _int(evidence.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "grammar failure must be removed before dead-air repair"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "collapse warning must be removed before dead-air repair"
        )
    claimed = [
        name
        for name in [
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        if bool(claim.get(name, False))
    ]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            f"unexpected quality claim: {claimed}"
        )
    return {
        "seed_count": _int(evidence.get("seed_count")),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "grammar_failure_count": _int(evidence.get("grammar_failure_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(evidence.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
        "failure_reasons": _dict(evidence.get("failure_reasons")),
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
        "--jazz_rhythm_profile",
        args.jazz_rhythm_profile,
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
    strict = _dict(summary.get("strict_failure_reasons"))
    return {
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "note_count_failure_count": _count_prefixed(diagnostic, "note count too low:"),
        "grammar_failure_count": _count_prefixed(strict, "grammar_gate_failed"),
        "dead_air_failure_count": _count_prefixed(diagnostic, "dead-air ratio too high:"),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            summary.get("max_longest_sustained_empty_run_steps")
        ),
        "avg_adjacent_repeated_pitch_ratio": _float(
            summary.get("avg_adjacent_repeated_pitch_ratio")
        ),
        "avg_direction_change_ratio": _float(summary.get("avg_direction_change_ratio")),
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": strict,
    }


def build_repair_report(
    *,
    run_dir: Path,
    source_summary: dict[str, Any],
    generation_report_path: Path,
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    repair_summary = _repair_metrics(_dict(generation_report.get("summary")))
    source_sample_count = source_summary["sample_count"]
    repair_sample_count = repair_summary["sample_count"]
    source_valid_rate = _ratio(source_summary["valid_sample_count"], source_sample_count)
    source_strict_rate = _ratio(source_summary["strict_valid_sample_count"], source_sample_count)
    repair_valid_rate = _ratio(repair_summary["valid_sample_count"], repair_sample_count)
    repair_strict_rate = _ratio(
        repair_summary["strict_valid_sample_count"], repair_sample_count
    )
    dead_air_failure_delta = (
        source_summary["dead_air_failure_count"] - repair_summary["dead_air_failure_count"]
    )
    valid_sample_rate_delta = repair_valid_rate - source_valid_rate
    strict_valid_sample_rate_delta = repair_strict_rate - source_strict_rate
    onset_coverage_delta = (
        repair_summary["avg_onset_coverage_ratio"] - source_summary["avg_onset_coverage_ratio"]
    )
    sustained_coverage_delta = (
        repair_summary["avg_sustained_coverage_ratio"]
        - source_summary["avg_sustained_coverage_ratio"]
    )
    postprocess_removal_delta = (
        repair_summary["avg_postprocess_removal_ratio"]
        - source_summary["avg_postprocess_removal_ratio"]
    )
    target_qualified = (
        _int(generation_result.get("returncode")) == 0
        and bool(generation_report.get("passed_strict_review_gate", False))
        and repair_summary["dead_air_failure_count"] == 0
        and repair_summary["note_count_failure_count"] == 0
        and repair_summary["grammar_failure_count"] == 0
        and repair_summary["collapse_warning_sample_count"] == 0
        and dead_air_failure_delta > 0
        and valid_sample_rate_delta > 0.0
        and strict_valid_sample_rate_delta > 0.0
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "run_dir": str(run_dir),
        "boundary": BOUNDARY,
        "input_boundary": DECISION_BOUNDARY,
        "source_repeatability_boundary": SOURCE_REPEATABILITY_BOUNDARY,
        "source_summary": source_summary,
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
            "coverage_position_window": int(args.coverage_position_window),
            "chord_pitch_mode": args.chord_pitch_mode,
            "chord_pitch_repeat_window": int(args.chord_pitch_repeat_window),
            "jazz_rhythm_profile": args.jazz_rhythm_profile,
            "fill_duration_to_next_position": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "generation_command": generation_result,
        "repair_summary": repair_summary,
        "comparison": {
            "dead_air_failure_delta": int(dead_air_failure_delta),
            "source_valid_sample_rate": float(source_valid_rate),
            "repair_valid_sample_rate": float(repair_valid_rate),
            "valid_sample_rate_delta": float(valid_sample_rate_delta),
            "source_strict_valid_sample_rate": float(source_strict_rate),
            "repair_strict_valid_sample_rate": float(repair_strict_rate),
            "strict_valid_sample_rate_delta": float(strict_valid_sample_rate_delta),
            "onset_coverage_delta": float(onset_coverage_delta),
            "sustained_coverage_delta": float(sustained_coverage_delta),
            "postprocess_removal_delta": float(postprocess_removal_delta),
            "target_qualified": bool(target_qualified),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "dead_air_repair_probe_completed": _int(generation_result.get("returncode")) == 0,
            "selected_scale_dead_air_target_qualified": bool(target_qualified),
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "12 constrained note groups per bar remove the selected-scale dead-air "
                "blocker for the current seed set; repeatability remains unverified"
            ),
        },
        "not_proven": [
            "repeatability",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair repeatability probe"
        ),
    }


def validate_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_target_qualified: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    repair = _dict(report.get("repair_summary"))
    comparison = _dict(report.get("comparison"))
    command = _dict(report.get("generation_command"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "unexpected next boundary"
        )
    if _int(command.get("returncode")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "generation command must succeed"
        )
    if require_target_qualified and not bool(comparison.get("target_qualified", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "dead-air target qualification required"
        )
    if _int(repair.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "dead-air failure must be removed"
        )
    if _int(repair.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "note-count failure must remain removed"
        )
    if _int(repair.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "grammar failure must remain removed"
        )
    if _int(repair.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "collapse warning must remain removed"
        )
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "sample_count": _int(repair.get("sample_count")),
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(repair.get("note_count_failure_count")),
        "grammar_failure_count": _int(repair.get("grammar_failure_count")),
        "dead_air_failure_count": _int(repair.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(repair.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(repair.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(repair.get("avg_sustained_coverage_ratio")),
        "dead_air_failure_delta": _int(comparison.get("dead_air_failure_delta")),
        "valid_sample_rate_delta": _float(comparison.get("valid_sample_rate_delta")),
        "strict_valid_sample_rate_delta": _float(
            comparison.get("strict_valid_sample_rate_delta")
        ),
        "postprocess_removal_delta": _float(comparison.get("postprocess_removal_delta")),
        "selected_scale_dead_air_target_qualified": bool(
            comparison.get("target_qualified", False)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_summary"]
    repair = report["repair_summary"]
    comparison = report["comparison"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected-scale dead-air target qualified: `{_bool_token(comparison['target_qualified'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Source Repeatability",
        "",
        f"- seed count: `{source['seed_count']}`",
        f"- sample count: `{source['sample_count']}`",
        f"- valid / strict / grammar: `{source['valid_sample_count']}` / `{source['strict_valid_sample_count']}` / `{source['grammar_gate_sample_count']}`",
        f"- note-count / grammar / dead-air / collapse failure count: `{source['note_count_failure_count']}` / `{source['grammar_failure_count']}` / `{source['dead_air_failure_count']}` / `{source['collapse_warning_sample_count']}`",
        f"- avg onset / sustained coverage ratio: `{source['avg_onset_coverage_ratio']}` / `{source['avg_sustained_coverage_ratio']}`",
        "",
        "## Repair Result",
        "",
        f"- constrained note groups per bar: `{report['input']['constrained_note_groups_per_bar']}`",
        f"- sample count: `{repair['sample_count']}`",
        f"- valid / strict / grammar: `{repair['valid_sample_count']}` / `{repair['strict_valid_sample_count']}` / `{repair['grammar_gate_sample_count']}`",
        f"- note-count / grammar / dead-air / collapse failure count: `{repair['note_count_failure_count']}` / `{repair['grammar_failure_count']}` / `{repair['dead_air_failure_count']}` / `{repair['collapse_warning_sample_count']}`",
        f"- avg onset / sustained coverage ratio: `{repair['avg_onset_coverage_ratio']}` / `{repair['avg_sustained_coverage_ratio']}`",
        f"- avg / max postprocess removal ratio: `{repair['avg_postprocess_removal_ratio']}` / `{repair['max_postprocess_removal_ratio']}`",
        "",
        "## Delta",
        "",
        f"- dead-air failure delta: `{comparison['dead_air_failure_delta']}`",
        f"- valid sample rate delta: `{comparison['valid_sample_rate_delta']}`",
        f"- strict sample rate delta: `{comparison['strict_valid_sample_rate_delta']}`",
        f"- onset / sustained coverage delta: `{comparison['onset_coverage_delta']}` / `{comparison['sustained_coverage_delta']}`",
        f"- postprocess removal delta: `{comparison['postprocess_removal_delta']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    if repair["diagnostic_failure_reasons"]:
        for reason, count in repair["diagnostic_failure_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run selected-scale checkpoint dead-air repair probe")
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision.json",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=592)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--constrained_note_groups_per_bar", type=int, default=12)
    parser.add_argument("--coverage_position_window", type=int, default=1)
    parser.add_argument("--chord_pitch_mode", type=str, default="approach_tensions")
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--jazz_rhythm_profile", type=str, default="swing_motif")
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
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
    source_summary = validate_decision(read_json(Path(args.decision_report)))
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepairProbeError(
            "checkpoint_epoch1.pt required"
        )

    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "selected_scale_dead_air_repair"
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
        source_summary=source_summary,
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
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe.json",
        report,
    )
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
