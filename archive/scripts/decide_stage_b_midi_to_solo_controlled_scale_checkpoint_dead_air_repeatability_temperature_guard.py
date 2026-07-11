"""Decide temperature guard after controlled dead-air repeatability failure."""

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


class StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
    ValueError
):
    pass


SOURCE_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe"
BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair_probe"
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_decision_v1"
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


def _count_reason_containing(reasons: dict[str, Any], token: str) -> int:
    return sum(_int(count) for reason, count in reasons.items() if token in str(reason))


def _failed_seed_rows(rows: list[Any]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for row_value in rows:
        row = _dict(row_value)
        if (
            _int(row.get("strict_valid_sample_count")) < _int(row.get("sample_count"))
            or _int(row.get("collapse_warning_sample_count")) > 0
            or bool(_dict(row.get("diagnostic_failure_reasons")))
        ):
            failed.append(row)
    return failed


def validate_repeatability_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    source = _dict(report.get("source_summary"))
    input_config = _dict(report.get("input"))
    seed_rows = _list(report.get("seed_rows"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "dead-air repeatability probe boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "repeatability probe must route to temperature guard decision"
        )
    if not bool(readiness.get("dead_air_repair_repeatability_probe_completed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "repeatability probe completion required"
        )
    if bool(readiness.get("dead_air_repair_repeatability_target_qualified", True)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "temperature guard decision requires partial repeatability failure"
        )
    blocked = [
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked if bool(readiness.get(name, True))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            f"unexpected quality claim: {claimed}"
        )

    sample_count = _int(aggregate.get("sample_count"))
    strict_count = _int(aggregate.get("strict_valid_sample_count"))
    grammar_count = _int(aggregate.get("grammar_gate_sample_count"))
    if sample_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "repeatability samples required"
        )
    if strict_count >= sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "strict shortfall evidence required"
        )
    if grammar_count != sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "grammar-valid sample coverage required before temperature guard"
        )
    if not bool(aggregate.get("all_seed_commands_succeeded", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "all seed commands must succeed before guard decision"
        )
    if not bool(aggregate.get("all_seed_gate_passed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "seed-level gate must pass before guard decision"
        )

    diagnostics = _dict(aggregate.get("diagnostic_failure_reasons"))
    dead_air_failure_count = _count_reason_containing(diagnostics, "dead-air ratio too high")
    postprocess_collapse_failure_count = _count_reason_containing(
        diagnostics, "collapse=postprocess_removed_majority"
    )
    collapse_warning_count = _int(aggregate.get("collapse_warning_sample_count"))
    if dead_air_failure_count <= 0 and collapse_warning_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "dead-air or collapse failure evidence required"
        )
    failed_rows = _failed_seed_rows(seed_rows)
    if not failed_rows:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "failed seed rows required"
        )

    return {
        "seed_count": _int(aggregate.get("seed_count")),
        "seeds": [int(seed) for seed in _list(aggregate.get("seeds"))],
        "sample_count": sample_count,
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": grammar_count,
        "strict_sample_shortfall": sample_count - strict_count,
        "collapse_warning_sample_count": collapse_warning_count,
        "dead_air_failure_count": dead_air_failure_count,
        "postprocess_collapse_failure_count": postprocess_collapse_failure_count,
        "diagnostic_failure_reasons": diagnostics,
        "avg_postprocess_removal_ratio": _float(aggregate.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(aggregate.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(aggregate.get("avg_sustained_coverage_ratio")),
        "all_seed_gate_passed": bool(aggregate.get("all_seed_gate_passed", False)),
        "all_samples_strict_valid": bool(aggregate.get("all_samples_strict_valid", False)),
        "failed_seed_count": len(failed_rows),
        "failed_seeds": [int(_int(row.get("seed"))) for row in failed_rows],
        "source_temperature": _float(source.get("temperature")),
        "source_top_k": _int(source.get("top_k")),
        "max_sequence": _int(source.get("max_sequence")),
        "constrained_note_groups_per_bar": _int(source.get("constrained_note_groups_per_bar")),
        "coverage_position_window": _int(source.get("coverage_position_window")),
        "chord_pitch_mode": str(source.get("chord_pitch_mode") or "approach_tensions"),
        "jazz_rhythm_profile": str(source.get("jazz_rhythm_profile") or "swing_motif"),
        "max_simultaneous_notes": _int(source.get("max_simultaneous_notes")),
        "num_samples": _int(input_config.get("num_samples")),
    }


def build_decision_report(
    repeatability_probe: dict[str, Any],
    *,
    output_dir: Path,
    issue_number: int,
    selected_temperature: float,
    selected_top_k: int | None,
) -> dict[str, Any]:
    evidence = validate_repeatability_probe(repeatability_probe)
    source_temperature = _float(evidence.get("source_temperature"))
    source_top_k = _int(evidence.get("source_top_k"))
    target_top_k = int(selected_top_k if selected_top_k is not None else source_top_k)
    if selected_temperature >= source_temperature:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "selected temperature must be lower than source temperature"
        )
    if target_top_k <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "selected top_k required"
        )
    guard_config = {
        "issue_number": int(issue_number),
        "temperature": float(selected_temperature),
        "top_k": int(target_top_k),
        "seeds": list(evidence["seeds"]),
        "num_samples": int(evidence["num_samples"]),
        "max_sequence": int(evidence["max_sequence"]),
        "constrained_note_groups_per_bar": int(evidence["constrained_note_groups_per_bar"]),
        "coverage_position_window": int(evidence["coverage_position_window"]),
        "chord_pitch_mode": str(evidence["chord_pitch_mode"]),
        "jazz_rhythm_profile": str(evidence["jazz_rhythm_profile"]),
        "max_simultaneous_notes": int(evidence["max_simultaneous_notes"]),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_schema": str(repeatability_probe.get("schema_version") or ""),
        "evidence": evidence,
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_lower_temperature_repeatability_guard_probe",
            "selected_target": "lower_temperature_repeatability_guard_repair",
            "source_temperature": source_temperature,
            "source_top_k": source_top_k,
            "temperature_change_selected": True,
            "top_k_change_selected": target_top_k != source_top_k,
            "audio_review_selected": False,
            "training_scale_change_selected": False,
            "quality_root_cause_claimed": False,
            "guard_config": guard_config,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "repeatability probe preserved grammar and seed-level gates, but all-sample "
                "strict validity failed under source temperature"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "guard_target_selected": True,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "repeatability_probe_completed",
            "grammar_validity_preserved",
            "seed_level_gate_passed",
            "strict_shortfall_recorded",
            "lower_temperature_guard_selected",
        ],
        "not_proven": [
            "guard_probe_result",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled checkpoint dead-air repeatability "
            "temperature guard repair probe"
        ),
    }


def validate_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_guard_target: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    evidence = _dict(report.get("evidence"))
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    guard = _dict(decision.get("guard_config"))
    boundary = str(report.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "unexpected boundary"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "unexpected next boundary"
        )
    if require_guard_target and str(decision.get("selected_target") or "") != (
        "lower_temperature_repeatability_guard_repair"
    ):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "lower-temperature guard target required"
        )
    if _int(evidence.get("strict_sample_shortfall")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "strict shortfall evidence required"
        )
    if not bool(evidence.get("all_seed_gate_passed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "seed-level gate evidence required"
        )
    if bool(evidence.get("all_samples_strict_valid", True)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "all-sample strict validity must remain false for guard decision"
        )
    if _float(guard.get("temperature")) >= _float(evidence.get("source_temperature")):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "guard temperature must be lower than source temperature"
        )
    if _int(guard.get("top_k")) != _int(evidence.get("source_top_k")):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
            "top_k should remain fixed for temperature guard isolation"
        )
    if require_no_quality_claim:
        blocked = [
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointDeadAirRepeatabilityTemperatureGuardDecisionError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": next_boundary,
        "selected_target": str(decision.get("selected_target") or ""),
        "sample_count": _int(evidence.get("sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "strict_sample_shortfall": _int(evidence.get("strict_sample_shortfall")),
        "failed_seed_count": _int(evidence.get("failed_seed_count")),
        "failed_seeds": _list(evidence.get("failed_seeds")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "source_temperature": _float(evidence.get("source_temperature")),
        "selected_temperature": _float(guard.get("temperature")),
        "selected_top_k": _int(guard.get("top_k")),
        "temperature_change_selected": bool(decision.get("temperature_change_selected", False)),
        "top_k_change_selected": bool(decision.get("top_k_change_selected", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            claim.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["decision"]
    guard = decision["guard_config"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Dead-Air Repeatability Temperature Guard Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- source temperature / top_k: `{decision['source_temperature']}` / `{decision['source_top_k']}`",
        f"- selected temperature / top_k: `{guard['temperature']}` / `{guard['top_k']}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(claim['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Evidence",
        "",
        f"- seed count: `{evidence['seed_count']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- valid / strict / grammar: `{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / `{evidence['grammar_gate_sample_count']}`",
        f"- strict sample shortfall: `{evidence['strict_sample_shortfall']}`",
        f"- failed seeds: `{evidence['failed_seeds']}`",
        f"- dead-air failure count: `{evidence['dead_air_failure_count']}`",
        f"- collapse warning sample count: `{evidence['collapse_warning_sample_count']}`",
        f"- avg postprocess removal ratio: `{evidence['avg_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    if evidence["diagnostic_failure_reasons"]:
        for reason, count in evidence["diagnostic_failure_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- temperature change selected: `{}`".format(
                _bool_token(decision["temperature_change_selected"])
            ),
            "- top_k change selected: `{}`".format(_bool_token(decision["top_k_change_selected"])),
            "- audio review selected: `{}`".format(_bool_token(decision["audio_review_selected"])),
            "- training scale change selected: `{}`".format(
                _bool_token(decision["training_scale_change_selected"])
            ),
            "- critical user input required: `{}`".format(
                _bool_token(decision["critical_user_input_required"])
            ),
            "",
            "## Guard Config",
            "",
        ]
    )
    for key in [
        "temperature",
        "top_k",
        "seeds",
        "num_samples",
        "max_sequence",
        "constrained_note_groups_per_bar",
        "coverage_position_window",
        "chord_pitch_mode",
        "jazz_rhythm_profile",
        "max_simultaneous_notes",
    ]:
        lines.append(f"- {key}: `{guard[key]}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide controlled dead-air repeatability temperature guard"
    )
    parser.add_argument(
        "--repeatability_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_repeatability_probe.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=566)
    parser.add_argument("--selected_temperature", type=float, default=0.75)
    parser.add_argument("--selected_top_k", type=int, default=0)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_guard_target", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    selected_top_k = int(args.selected_top_k) if int(args.selected_top_k) > 0 else None
    report = build_decision_report(
        read_json(Path(args.repeatability_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        selected_temperature=float(args.selected_temperature),
        selected_top_k=selected_top_k,
    )
    summary = validate_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_guard_target=bool(args.require_guard_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
