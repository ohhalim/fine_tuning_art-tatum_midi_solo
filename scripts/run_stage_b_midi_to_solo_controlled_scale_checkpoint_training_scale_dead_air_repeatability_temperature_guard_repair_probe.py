"""Run lower-temperature guard repair probe for selected-scale dead-air repeatability."""

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
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_repeatability_probe import (  # noqa: E402
    aggregate_seed_rows,
    run_seed_probe,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_repair_probe"
)
PASS_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_repair_consolidation"
)
FAIL_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_followup_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_repair_probe_v1"
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


def _failure_count(reasons: dict[str, Any], token: str) -> int:
    return sum(_int(count) for reason, count in reasons.items() if token in str(reason))


def validate_guard_decision(report: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    guard = _dict(decision.get("guard_config"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "temperature guard decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "temperature guard decision must route to repair probe"
        )
    if str(decision.get("selected_target") or "") != "lower_temperature_repeatability_guard_repair":
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "lower-temperature guard target required"
        )
    if not bool(decision.get("temperature_change_selected", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "temperature change selection required"
        )
    if bool(decision.get("top_k_change_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "top_k must remain fixed for isolated repair probe"
        )
    if _float(guard.get("temperature")) >= _float(evidence.get("source_temperature")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "guard temperature must be lower than source temperature"
        )
    if _int(guard.get("top_k")) != _int(evidence.get("source_top_k")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "guard top_k must match source top_k"
        )
    if _int(evidence.get("strict_sample_shortfall")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "strict shortfall evidence required"
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
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            f"unexpected quality claim: {claimed}"
        )
    seeds = [int(_int(seed)) for seed in _list(guard.get("seeds"))]
    if not seeds:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "guard seeds required"
        )
    return {
        "source_sample_count": _int(evidence.get("sample_count")),
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
        "source_temperature": _float(evidence.get("source_temperature")),
        "source_top_k": _int(evidence.get("source_top_k")),
        "temperature": _float(guard.get("temperature")),
        "top_k": _int(guard.get("top_k")),
        "seeds": seeds,
        "num_samples": _int(guard.get("num_samples")),
        "max_sequence": _int(guard.get("max_sequence")),
        "constrained_note_groups_per_bar": _int(
            guard.get("constrained_note_groups_per_bar")
        ),
        "coverage_position_window": _int(guard.get("coverage_position_window")),
        "chord_pitch_mode": str(guard.get("chord_pitch_mode") or "approach_tensions"),
        "jazz_rhythm_profile": str(guard.get("jazz_rhythm_profile") or "swing_motif"),
        "max_simultaneous_notes": _int(guard.get("max_simultaneous_notes")),
    }


def summarize_failures(aggregate: dict[str, Any]) -> dict[str, int]:
    diagnostic = _dict(aggregate.get("diagnostic_failure_reasons"))
    strict = _dict(aggregate.get("strict_failure_reasons"))
    return {
        "dead_air_failure_count": _failure_count(diagnostic, "dead-air ratio too high"),
        "postprocess_collapse_failure_count": _failure_count(
            diagnostic, "collapse=postprocess_removed_majority"
        ),
        "postprocess_removal_failure_count": _failure_count(
            strict, "postprocess removal ratio too high"
        ),
    }


def build_repair_report(
    *,
    run_dir: Path,
    decision_summary: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    issue_number: int,
) -> dict[str, Any]:
    aggregate = aggregate_seed_rows(seed_rows)
    failure_summary = summarize_failures(aggregate)
    target_qualified = (
        bool(aggregate["all_seed_commands_succeeded"])
        and bool(aggregate["all_samples_strict_valid"])
        and _int(aggregate["note_count_failure_count"]) == 0
        and _int(aggregate["grammar_failure_count"]) == 0
        and _int(failure_summary["dead_air_failure_count"]) == 0
        and _int(aggregate["collapse_warning_sample_count"]) == 0
    )
    next_boundary = PASS_NEXT_BOUNDARY if target_qualified else FAIL_NEXT_BOUNDARY
    comparison = {
        "source_strict_valid_sample_count": int(
            decision_summary["source_strict_valid_sample_count"]
        ),
        "repair_strict_valid_sample_count": int(aggregate["strict_valid_sample_count"]),
        "strict_valid_sample_delta": _int(aggregate["strict_valid_sample_count"])
        - _int(decision_summary["source_strict_valid_sample_count"]),
        "source_strict_sample_shortfall": int(decision_summary["source_strict_sample_shortfall"]),
        "repair_strict_sample_shortfall": _int(aggregate["sample_count"])
        - _int(aggregate["strict_valid_sample_count"]),
        "source_dead_air_failure_count": int(decision_summary["source_dead_air_failure_count"]),
        "repair_dead_air_failure_count": int(failure_summary["dead_air_failure_count"]),
        "source_collapse_warning_sample_count": int(
            decision_summary["source_collapse_warning_sample_count"]
        ),
        "repair_collapse_warning_sample_count": _int(
            aggregate["collapse_warning_sample_count"]
        ),
        "postprocess_removal_delta": _float(aggregate["avg_postprocess_removal_ratio"])
        - _float(decision_summary["source_avg_postprocess_removal_ratio"]),
        "onset_coverage_delta": _float(aggregate["avg_onset_coverage_ratio"])
        - _float(decision_summary["source_avg_onset_coverage_ratio"]),
        "sustained_coverage_delta": _float(aggregate["avg_sustained_coverage_ratio"])
        - _float(decision_summary["source_avg_sustained_coverage_ratio"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "run_dir": str(run_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "input": {
            "issue_number": int(issue_number),
            "source_temperature": float(decision_summary["source_temperature"]),
            "temperature": float(decision_summary["temperature"]),
            "top_k": int(decision_summary["top_k"]),
            "seeds": list(decision_summary["seeds"]),
            "num_samples": int(decision_summary["num_samples"]),
            "max_sequence": int(decision_summary["max_sequence"]),
            "constrained_note_groups_per_bar": int(
                decision_summary["constrained_note_groups_per_bar"]
            ),
            "coverage_position_window": int(decision_summary["coverage_position_window"]),
            "chord_pitch_mode": str(decision_summary["chord_pitch_mode"]),
            "jazz_rhythm_profile": str(decision_summary["jazz_rhythm_profile"]),
            "max_simultaneous_notes": int(decision_summary["max_simultaneous_notes"]),
        },
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "failure_summary": failure_summary,
        "comparison": comparison,
        "readiness": {
            "boundary": BOUNDARY,
            "temperature_guard_repair_probe_completed": True,
            "selected_scale_temperature_guard_repair_target_qualified": bool(
                target_qualified
            ),
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
                "lower-temperature guard repair probe records whether selected guard config "
                "removes the selected-scale repeatability strict shortfall"
            ),
        },
        "not_proven": [
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air temperature guard repair consolidation"
            if target_qualified
            else "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air temperature guard follow-up decision"
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
    failure = _dict(report.get("failure_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "unexpected next boundary"
        )
    if require_completed and not bool(readiness.get("temperature_guard_repair_probe_completed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "repair probe completion required"
        )
    if _int(aggregate.get("sample_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "sample_count": _int(aggregate.get("sample_count")),
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(aggregate.get("note_count_failure_count")),
        "grammar_failure_count": _int(aggregate.get("grammar_failure_count")),
        "collapse_warning_sample_count": _int(aggregate.get("collapse_warning_sample_count")),
        "diagnostic_failure_reasons": _dict(aggregate.get("diagnostic_failure_reasons")),
        "dead_air_failure_count": _int(failure.get("dead_air_failure_count")),
        "postprocess_collapse_failure_count": _int(
            failure.get("postprocess_collapse_failure_count")
        ),
        "all_seed_gate_passed": bool(aggregate.get("all_seed_gate_passed", False)),
        "all_samples_strict_valid": bool(aggregate.get("all_samples_strict_valid", False)),
        "selected_scale_temperature_guard_repair_target_qualified": bool(
            readiness.get("selected_scale_temperature_guard_repair_target_qualified", False)
        ),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "repair_strict_sample_shortfall": _int(
            comparison.get("repair_strict_sample_shortfall")
        ),
        "source_temperature": _float(_dict(report.get("input")).get("source_temperature")),
        "temperature": _float(_dict(report.get("input")).get("temperature")),
        "top_k": _int(_dict(report.get("input")).get("top_k")),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    comparison = report["comparison"]
    failure = report["failure_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    input_config = report["input"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repeatability Temperature Guard Repair Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected-scale temperature guard repair target qualified: `{_bool_token(readiness['selected_scale_temperature_guard_repair_target_qualified'])}`",
        f"- source / repair temperature: `{input_config['source_temperature']}` / `{input_config['temperature']}`",
        f"- top_k: `{input_config['top_k']}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Aggregate",
        "",
        f"- seed count: `{aggregate['seed_count']}`",
        f"- sample count: `{aggregate['sample_count']}`",
        f"- valid / strict / grammar: `{aggregate['valid_sample_count']}` / `{aggregate['strict_valid_sample_count']}` / `{aggregate['grammar_gate_sample_count']}`",
        f"- note-count / grammar / dead-air / collapse failure count: `{aggregate['note_count_failure_count']}` / `{aggregate['grammar_failure_count']}` / `{failure['dead_air_failure_count']}` / `{aggregate['collapse_warning_sample_count']}`",
        f"- all seed gate passed: `{_bool_token(aggregate['all_seed_gate_passed'])}`",
        f"- all samples strict valid: `{_bool_token(aggregate['all_samples_strict_valid'])}`",
        f"- avg postprocess removal ratio: `{aggregate['avg_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{aggregate['avg_onset_coverage_ratio']}` / `{aggregate['avg_sustained_coverage_ratio']}`",
        "",
        "## Delta",
        "",
        f"- strict valid sample delta: `{comparison['strict_valid_sample_delta']}`",
        f"- strict sample shortfall: `{comparison['source_strict_sample_shortfall']}` -> `{comparison['repair_strict_sample_shortfall']}`",
        f"- dead-air failure count: `{comparison['source_dead_air_failure_count']}` -> `{comparison['repair_dead_air_failure_count']}`",
        f"- collapse warning sample count: `{comparison['source_collapse_warning_sample_count']}` -> `{comparison['repair_collapse_warning_sample_count']}`",
        f"- postprocess removal delta: `{comparison['postprocess_removal_delta']}`",
        f"- onset / sustained coverage delta: `{comparison['onset_coverage_delta']}` / `{comparison['sustained_coverage_delta']}`",
        "",
        "## Seed Rows",
        "",
    ]
    for row in report["seed_rows"]:
        lines.append(
            "- seed `{seed}`: valid/strict/grammar `{valid}`/`{strict}`/`{grammar}`, dead-air `{dead_air}`, collapse `{collapse}`".format(
                seed=row["seed"],
                valid=row["valid_sample_count"],
                strict=row["strict_valid_sample_count"],
                grammar=row["grammar_gate_sample_count"],
                dead_air=row["dead_air_failure_count"],
                collapse=row["collapse_warning_sample_count"],
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
        description="Run selected-scale dead-air repeatability temperature guard repair probe"
    )
    parser.add_argument(
        "--guard_decision_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_decision.json",
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=598)
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
    decision_summary = validate_guard_decision(read_json(Path(args.guard_decision_report)))
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardRepairProbeError(
            "checkpoint_epoch1.pt required"
        )
    args.num_samples = int(decision_summary["num_samples"])
    args.max_sequence = int(decision_summary["max_sequence"])
    args.temperature = float(decision_summary["temperature"])
    args.top_k = int(decision_summary["top_k"])
    args.max_simultaneous_notes = int(decision_summary["max_simultaneous_notes"])
    probe_output_root = run_dir / "generation_probe"
    seed_rows = [
        run_seed_probe(
            args,
            seed=int(seed),
            checkpoint_dir=checkpoint_dir,
            output_root=probe_output_root,
            repair_config=decision_summary,
        )
        for seed in decision_summary["seeds"]
    ]
    report = build_repair_report(
        run_dir=run_dir,
        decision_summary=decision_summary,
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
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe.json",
        report,
    )
    write_json(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        run_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
