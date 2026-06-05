"""Run postprocess-removal dead-air repair probe for selected-scale checkpoint."""

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


class StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
    ValueError
):
    pass


SOURCE_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_followup_decision"
)
BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_probe"
)
PASS_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_consolidation"
)
FAIL_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_residual_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_probe_v1"
)


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _count_prefixed(reasons: dict[str, Any], prefix: str) -> int:
    return sum(_int(count) for reason, count in reasons.items() if str(reason).startswith(prefix))


def merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = _int(target.get(str(key))) + _int(value)


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


def validate_followup_decision(report: dict[str, Any]) -> dict[str, Any]:
    evidence = _dict(report.get("evidence"))
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    repair = _dict(decision.get("repair_config"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "postprocess-removal follow-up decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "follow-up decision must route to postprocess-removal repair probe"
        )
    if str(decision.get("selected_target") or "") != "postprocess_removal_dead_air_repair":
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "postprocess-removal dead-air target required"
        )
    if not bool(decision.get("postprocess_removal_repair_selected", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "postprocess removal repair selection required"
        )
    if bool(decision.get("temperature_followup_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "temperature follow-up must be excluded before this repair probe"
        )
    blocked = [
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked if bool(claim.get(name, True))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            f"unexpected quality claim: {claimed}"
        )
    if _int(evidence.get("dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "source dead-air residual evidence required"
        )
    if _int(evidence.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "source note-count failure must remain excluded"
        )
    if _int(evidence.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "source grammar failure must remain excluded"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "source collapse warning must remain excluded"
        )
    seeds = [int(_int(seed)) for seed in _list(repair.get("seeds"))]
    if not seeds:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "repair seeds required"
        )
    return {
        "source_sample_count": _int(evidence.get("sample_count")),
        "source_valid_sample_count": _int(evidence.get("valid_sample_count")),
        "source_strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "source_grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "source_strict_sample_shortfall": _int(evidence.get("strict_sample_shortfall")),
        "source_dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "source_collapse_warning_sample_count": _int(
            evidence.get("collapse_warning_sample_count")
        ),
        "source_avg_postprocess_removal_ratio": _float(
            evidence.get("avg_postprocess_removal_ratio")
        ),
        "source_avg_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "source_avg_sustained_coverage_ratio": _float(
            evidence.get("avg_sustained_coverage_ratio")
        ),
        "temperature": _float(repair.get("source_temperature")),
        "top_k": _int(repair.get("top_k")),
        "seeds": seeds,
        "num_samples": _int(repair.get("num_samples")),
        "max_sequence": _int(repair.get("max_sequence")),
        "constrained_note_groups_per_bar": _int(
            repair.get("constrained_note_groups_per_bar")
        ),
        "coverage_position_window": _int(repair.get("coverage_position_window")),
        "chord_pitch_mode": str(repair.get("chord_pitch_mode") or "approach_tensions"),
        "jazz_rhythm_profile": str(repair.get("jazz_rhythm_profile") or "swing_motif"),
        "max_simultaneous_notes": _int(repair.get("max_simultaneous_notes")),
        "target_avg_postprocess_removal_ratio": _float(
            repair.get("target_avg_postprocess_removal_ratio")
        ),
        "target_dead_air_failure_count": _int(repair.get("target_dead_air_failure_count")),
    }


def build_generation_command(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
    repair_config: dict[str, Any],
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
        str(repair_config["max_sequence"] or args.max_sequence),
        "--num_samples",
        str(repair_config["num_samples"] or args.num_samples),
        "--seed",
        str(seed),
        "--temperature",
        str(repair_config["temperature"] or args.temperature),
        "--top_k",
        str(repair_config["top_k"] or args.top_k),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(repair_config["constrained_note_groups_per_bar"]),
        "--coverage_aware_positions",
        "--coverage_position_window",
        str(repair_config["coverage_position_window"]),
        "--chord_aware_pitches",
        "--chord_pitch_mode",
        str(repair_config["chord_pitch_mode"]),
        "--chord_pitch_repeat_window",
        str(args.chord_pitch_repeat_window),
        "--jazz_rhythm_positions",
        "--jazz_rhythm_profile",
        str(repair_config["jazz_rhythm_profile"]),
        "--jazz_duration_tokens",
        "--cap_duration_to_next_position",
        "--fill_duration_to_next_position",
        "--avoid_reused_positions",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(repair_config["max_simultaneous_notes"] or args.max_simultaneous_notes),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
    ]


def run_seed_probe(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_dir: Path,
    output_root: Path,
    repair_config: dict[str, Any],
) -> dict[str, Any]:
    run_id = f"seed_{seed}"
    command_result = run_command(
        build_generation_command(
            args,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            output_root=output_root,
            run_id=run_id,
            repair_config=repair_config,
        )
    )
    report_path = output_root / run_id / "report.json"
    generation_report = read_json(report_path) if report_path.exists() else {}
    summary = _dict(generation_report.get("summary"))
    diagnostic = _dict(summary.get("diagnostic_failure_reasons"))
    strict = _dict(summary.get("strict_failure_reasons"))
    return {
        "seed": int(seed),
        "generation_report_path": str(report_path),
        "generation_command": command_result,
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "note_count_failure_count": _count_prefixed(diagnostic, "note count too low:"),
        "grammar_failure_count": _count_prefixed(strict, "grammar_gate_failed"),
        "dead_air_failure_count": _count_prefixed(diagnostic, "dead-air ratio too high:"),
        "diagnostic_failure_reasons": diagnostic,
        "strict_failure_reasons": strict,
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "passed_strict_review_gate": bool(generation_report.get("passed_strict_review_gate", False)),
    }


def aggregate_seed_rows(seed_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_samples = sum(_int(row.get("sample_count")) for row in seed_rows)
    total_valid = sum(_int(row.get("valid_sample_count")) for row in seed_rows)
    total_strict = sum(_int(row.get("strict_valid_sample_count")) for row in seed_rows)
    total_grammar = sum(_int(row.get("grammar_gate_sample_count")) for row in seed_rows)
    diagnostics: dict[str, int] = {}
    strict_failures: dict[str, int] = {}
    for row in seed_rows:
        merge_counts(diagnostics, _dict(row.get("diagnostic_failure_reasons")))
        merge_counts(strict_failures, _dict(row.get("strict_failure_reasons")))
    weighted_postprocess = sum(
        _float(row.get("avg_postprocess_removal_ratio")) * _int(row.get("sample_count"))
        for row in seed_rows
    )
    weighted_onset = sum(
        _float(row.get("avg_onset_coverage_ratio")) * _int(row.get("sample_count"))
        for row in seed_rows
    )
    weighted_sustained = sum(
        _float(row.get("avg_sustained_coverage_ratio")) * _int(row.get("sample_count"))
        for row in seed_rows
    )
    return {
        "seed_count": len(seed_rows),
        "seeds": [int(row["seed"]) for row in seed_rows],
        "sample_count": int(total_samples),
        "valid_sample_count": int(total_valid),
        "strict_valid_sample_count": int(total_strict),
        "grammar_gate_sample_count": int(total_grammar),
        "valid_sample_rate": float(total_valid / total_samples) if total_samples else 0.0,
        "strict_valid_sample_rate": float(total_strict / total_samples) if total_samples else 0.0,
        "grammar_gate_sample_rate": float(total_grammar / total_samples) if total_samples else 0.0,
        "note_count_failure_count": sum(
            _int(row.get("note_count_failure_count")) for row in seed_rows
        ),
        "grammar_failure_count": sum(
            _int(row.get("grammar_failure_count")) for row in seed_rows
        ),
        "dead_air_failure_count": sum(
            _int(row.get("dead_air_failure_count")) for row in seed_rows
        ),
        "diagnostic_failure_reasons": diagnostics,
        "strict_failure_reasons": strict_failures,
        "collapse_warning_sample_count": sum(
            _int(row.get("collapse_warning_sample_count")) for row in seed_rows
        ),
        "avg_postprocess_removal_ratio": (
            float(weighted_postprocess / total_samples) if total_samples else 0.0
        ),
        "max_postprocess_removal_ratio": max(
            [_float(row.get("max_postprocess_removal_ratio")) for row in seed_rows] or [0.0]
        ),
        "avg_onset_coverage_ratio": float(weighted_onset / total_samples) if total_samples else 0.0,
        "avg_sustained_coverage_ratio": (
            float(weighted_sustained / total_samples) if total_samples else 0.0
        ),
        "all_seed_commands_succeeded": all(
            _int(_dict(row.get("generation_command")).get("returncode")) == 0 for row in seed_rows
        ),
        "all_seed_gate_passed": all(bool(row.get("passed_strict_review_gate", False)) for row in seed_rows),
        "all_samples_strict_valid": bool(total_samples > 0 and total_strict == total_samples),
    }


def build_repair_report(
    *,
    run_dir: Path,
    source_summary: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    issue_number: int,
) -> dict[str, Any]:
    aggregate = aggregate_seed_rows(seed_rows)
    target_qualified = (
        bool(aggregate["all_seed_commands_succeeded"])
        and bool(aggregate["all_samples_strict_valid"])
        and _int(aggregate["note_count_failure_count"]) == 0
        and _int(aggregate["grammar_failure_count"]) == 0
        and _int(aggregate["dead_air_failure_count"]) <= _int(
            source_summary["target_dead_air_failure_count"]
        )
        and _int(aggregate["collapse_warning_sample_count"]) == 0
        and _float(aggregate["avg_postprocess_removal_ratio"])
        <= _float(source_summary["target_avg_postprocess_removal_ratio"])
    )
    next_boundary = PASS_NEXT_BOUNDARY if target_qualified else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "run_dir": str(run_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_summary": source_summary,
        "input": {
            "issue_number": int(issue_number),
            "temperature": float(source_summary["temperature"]),
            "top_k": int(source_summary["top_k"]),
            "seeds": list(source_summary["seeds"]),
            "num_samples": int(source_summary["num_samples"]),
            "max_sequence": int(source_summary["max_sequence"]),
            "constrained_note_groups_per_bar": int(
                source_summary["constrained_note_groups_per_bar"]
            ),
            "coverage_position_window": int(source_summary["coverage_position_window"]),
            "chord_pitch_mode": str(source_summary["chord_pitch_mode"]),
            "jazz_rhythm_profile": str(source_summary["jazz_rhythm_profile"]),
            "max_simultaneous_notes": int(source_summary["max_simultaneous_notes"]),
            "avoid_reused_positions": True,
            "target_avg_postprocess_removal_ratio": float(
                source_summary["target_avg_postprocess_removal_ratio"]
            ),
            "target_dead_air_failure_count": int(
                source_summary["target_dead_air_failure_count"]
            ),
        },
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "comparison": {
            "source_sample_count": int(source_summary["source_sample_count"]),
            "repair_sample_count": int(aggregate["sample_count"]),
            "strict_valid_sample_delta": _int(aggregate["strict_valid_sample_count"])
            - _int(source_summary["source_strict_valid_sample_count"]),
            "strict_sample_shortfall_delta": (
                (_int(aggregate["sample_count"]) - _int(aggregate["strict_valid_sample_count"]))
                - _int(source_summary["source_strict_sample_shortfall"])
            ),
            "dead_air_failure_delta": _int(aggregate["dead_air_failure_count"])
            - _int(source_summary["source_dead_air_failure_count"]),
            "collapse_warning_delta": _int(aggregate["collapse_warning_sample_count"])
            - _int(source_summary["source_collapse_warning_sample_count"]),
            "postprocess_removal_delta": _float(aggregate["avg_postprocess_removal_ratio"])
            - _float(source_summary["source_avg_postprocess_removal_ratio"]),
            "onset_coverage_delta": _float(aggregate["avg_onset_coverage_ratio"])
            - _float(source_summary["source_avg_onset_coverage_ratio"]),
            "sustained_coverage_delta": _float(aggregate["avg_sustained_coverage_ratio"])
            - _float(source_summary["source_avg_sustained_coverage_ratio"]),
            "target_qualified": bool(target_qualified),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "postprocess_removal_dead_air_repair_probe_completed": True,
            "postprocess_removal_dead_air_repair_target_qualified": bool(target_qualified),
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
                "reused-position guard reduces postprocess removals before overlap limiting "
                "and verifies the residual dead-air boundary"
            ),
        },
        "not_proven": [
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair consolidation"
            if target_qualified
            else "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air residual decision"
        ),
    }


def validate_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_completed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    aggregate = _dict(report.get("aggregate"))
    comparison = _dict(report.get("comparison"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "unexpected next boundary"
        )
    if require_completed and not bool(
        readiness.get("postprocess_removal_dead_air_repair_probe_completed", False)
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "repair probe completion required"
        )
    if _int(aggregate.get("sample_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "repair samples required"
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "seed_count": _int(aggregate.get("seed_count")),
        "sample_count": _int(aggregate.get("sample_count")),
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(aggregate.get("note_count_failure_count")),
        "grammar_failure_count": _int(aggregate.get("grammar_failure_count")),
        "dead_air_failure_count": _int(aggregate.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(aggregate.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(
            aggregate.get("avg_postprocess_removal_ratio")
        ),
        "max_postprocess_removal_ratio": _float(
            aggregate.get("max_postprocess_removal_ratio")
        ),
        "avg_onset_coverage_ratio": _float(aggregate.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(
            aggregate.get("avg_sustained_coverage_ratio")
        ),
        "all_seed_gate_passed": bool(aggregate.get("all_seed_gate_passed", False)),
        "all_samples_strict_valid": bool(aggregate.get("all_samples_strict_valid", False)),
        "postprocess_removal_dead_air_repair_target_qualified": bool(
            readiness.get("postprocess_removal_dead_air_repair_target_qualified", False)
        ),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "dead_air_failure_delta": _int(comparison.get("dead_air_failure_delta")),
        "postprocess_removal_delta": _float(comparison.get("postprocess_removal_delta")),
        "avoid_reused_positions": bool(_dict(report.get("input")).get("avoid_reused_positions", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    readiness = report["readiness"]
    decision = report["decision"]
    comparison = report["comparison"]
    input_config = report["input"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Postprocess Removal Dead-Air Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- target qualified: `{_bool_token(readiness['postprocess_removal_dead_air_repair_target_qualified'])}`",
        f"- avoid reused positions: `{_bool_token(input_config['avoid_reused_positions'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Aggregate",
        "",
        f"- seed count: `{aggregate['seed_count']}`",
        f"- sample count: `{aggregate['sample_count']}`",
        f"- valid / strict / grammar: `{aggregate['valid_sample_count']}` / `{aggregate['strict_valid_sample_count']}` / `{aggregate['grammar_gate_sample_count']}`",
        f"- note-count / grammar / dead-air / collapse failure count: `{aggregate['note_count_failure_count']}` / `{aggregate['grammar_failure_count']}` / `{aggregate['dead_air_failure_count']}` / `{aggregate['collapse_warning_sample_count']}`",
        f"- avg / max postprocess removal ratio: `{aggregate['avg_postprocess_removal_ratio']}` / `{aggregate['max_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{aggregate['avg_onset_coverage_ratio']}` / `{aggregate['avg_sustained_coverage_ratio']}`",
        "",
        "## Delta",
        "",
        f"- strict valid sample delta: `{comparison['strict_valid_sample_delta']}`",
        f"- strict sample shortfall delta: `{comparison['strict_sample_shortfall_delta']}`",
        f"- dead-air failure delta: `{comparison['dead_air_failure_delta']}`",
        f"- collapse warning delta: `{comparison['collapse_warning_delta']}`",
        f"- postprocess removal delta: `{comparison['postprocess_removal_delta']}`",
        f"- onset / sustained coverage delta: `{comparison['onset_coverage_delta']}` / `{comparison['sustained_coverage_delta']}`",
        "",
        "## Seed Rows",
        "",
    ]
    for row in report["seed_rows"]:
        lines.append(
            "- seed `{seed}`: valid/strict/grammar `{valid}`/`{strict}`/`{grammar}`, dead-air `{dead_air}`, collapse `{collapse}`, avg removal `{removal}`".format(
                seed=row["seed"],
                valid=row["valid_sample_count"],
                strict=row["strict_valid_sample_count"],
                grammar=row["grammar_gate_sample_count"],
                dead_air=row["dead_air_failure_count"],
                collapse=row["collapse_warning_sample_count"],
                removal=row["avg_postprocess_removal_ratio"],
            )
        )
    lines.extend(["", "## Failure Reasons", ""])
    if aggregate["diagnostic_failure_reasons"]:
        for reason, count in aggregate["diagnostic_failure_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run selected-scale postprocess-removal dead-air repair probe"
    )
    parser.add_argument(
        "--followup_decision_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision.json",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=602)
    parser.add_argument("--max_sequence", type=int, default=160)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--chord_pitch_repeat_window", type=int, default=2)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    source_summary = validate_followup_decision(read_json(Path(args.followup_decision_report)))
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairProbeError(
            "checkpoint_epoch1.pt required"
        )
    probe_output_root = run_dir / "generation_probe"
    seed_rows = [
        run_seed_probe(
            args,
            seed=int(seed),
            checkpoint_dir=checkpoint_dir,
            output_root=probe_output_root,
            repair_config=source_summary,
        )
        for seed in source_summary["seeds"]
    ]
    report = build_repair_report(
        run_dir=run_dir,
        source_summary=source_summary,
        seed_rows=seed_rows,
        issue_number=int(args.issue_number),
    )
    summary = validate_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_completed=bool(args.require_completed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe.json",
        report,
    )
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
