"""Decide follow-up after selected-scale temperature guard partial repair."""

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


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
    ValueError
):
    pass


SOURCE_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_repair_probe"
)
BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_followup_decision"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_probe"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_repeatability_temperature_guard_followup_decision_v1"
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
            or _dict(row.get("diagnostic_failure_reasons"))
            or _int(row.get("collapse_warning_sample_count")) > 0
        ):
            failed.append(row)
    return failed


def _max_failed_row_postprocess_removal(rows: list[dict[str, Any]]) -> float:
    ratios = [_float(row.get("avg_postprocess_removal_ratio")) for row in rows]
    return max(ratios) if ratios else 0.0


def validate_temperature_guard_repair_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    comparison = _dict(report.get("comparison"))
    failure = _dict(report.get("failure_summary"))
    input_config = _dict(report.get("input"))
    seed_rows = _list(report.get("seed_rows"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")

    if boundary != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "temperature guard repair probe boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "temperature guard repair probe must route to follow-up decision"
        )
    if not bool(readiness.get("temperature_guard_repair_probe_completed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "temperature guard repair probe completion required"
        )
    if bool(readiness.get("selected_scale_temperature_guard_repair_target_qualified", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "follow-up decision requires partial temperature guard repair"
        )

    blocked = [
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            f"unexpected quality claim: {claimed}"
        )

    sample_count = _int(aggregate.get("sample_count"))
    strict_count = _int(aggregate.get("strict_valid_sample_count"))
    grammar_count = _int(aggregate.get("grammar_gate_sample_count"))
    note_count_failures = _int(aggregate.get("note_count_failure_count"))
    grammar_failures = _int(aggregate.get("grammar_failure_count"))
    collapse_count = _int(aggregate.get("collapse_warning_sample_count"))
    dead_air_count = _int(failure.get("dead_air_failure_count"))
    diagnostics = _dict(aggregate.get("diagnostic_failure_reasons"))
    dead_air_reason_count = _count_reason_containing(diagnostics, "dead-air ratio too high")
    if sample_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "repair samples required"
        )
    if strict_count >= sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "strict shortfall evidence required"
        )
    if grammar_count != sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "grammar-valid sample coverage required before follow-up decision"
        )
    if note_count_failures != 0 or grammar_failures != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "note-count and grammar failures must remain removed"
        )
    if collapse_count != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "collapse must be separated before postprocess-removal dead-air repair"
        )
    if dead_air_count <= 0 or dead_air_reason_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "dead-air residual evidence required"
        )
    if not bool(aggregate.get("all_seed_commands_succeeded", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "all seed commands must succeed before follow-up decision"
        )
    if not bool(aggregate.get("all_seed_gate_passed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "seed-level gate evidence required"
        )

    failed_rows = _failed_seed_rows(seed_rows)
    if not failed_rows:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "failed seed rows required"
        )
    avg_removal = _float(aggregate.get("avg_postprocess_removal_ratio"))
    if avg_removal <= 0.0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "postprocess removal evidence required"
        )

    return {
        "seed_count": _int(aggregate.get("seed_count")),
        "seeds": [int(_int(seed)) for seed in _list(aggregate.get("seeds"))],
        "sample_count": sample_count,
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": strict_count,
        "grammar_gate_sample_count": grammar_count,
        "strict_sample_shortfall": sample_count - strict_count,
        "note_count_failure_count": note_count_failures,
        "grammar_failure_count": grammar_failures,
        "dead_air_failure_count": dead_air_count,
        "dead_air_reason_count": dead_air_reason_count,
        "collapse_warning_sample_count": collapse_count,
        "diagnostic_failure_reasons": diagnostics,
        "strict_failure_reasons": _dict(aggregate.get("strict_failure_reasons")),
        "avg_postprocess_removal_ratio": avg_removal,
        "max_failed_seed_avg_postprocess_removal_ratio": _max_failed_row_postprocess_removal(
            failed_rows
        ),
        "avg_onset_coverage_ratio": _float(aggregate.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(
            aggregate.get("avg_sustained_coverage_ratio")
        ),
        "all_seed_gate_passed": bool(aggregate.get("all_seed_gate_passed", False)),
        "all_samples_strict_valid": bool(aggregate.get("all_samples_strict_valid", False)),
        "failed_seed_count": len(failed_rows),
        "failed_seeds": [int(_int(row.get("seed"))) for row in failed_rows],
        "source_strict_sample_shortfall": _int(
            comparison.get("source_strict_sample_shortfall")
        ),
        "repair_strict_sample_shortfall": _int(
            comparison.get("repair_strict_sample_shortfall")
        ),
        "source_dead_air_failure_count": _int(
            comparison.get("source_dead_air_failure_count")
        ),
        "repair_dead_air_failure_count": _int(
            comparison.get("repair_dead_air_failure_count")
        ),
        "source_collapse_warning_sample_count": _int(
            comparison.get("source_collapse_warning_sample_count")
        ),
        "repair_collapse_warning_sample_count": _int(
            comparison.get("repair_collapse_warning_sample_count")
        ),
        "source_temperature": _float(input_config.get("source_temperature")),
        "temperature": _float(input_config.get("temperature")),
        "top_k": _int(input_config.get("top_k")),
        "num_samples": _int(input_config.get("num_samples")),
        "max_sequence": _int(input_config.get("max_sequence")),
        "constrained_note_groups_per_bar": _int(
            input_config.get("constrained_note_groups_per_bar")
        ),
        "coverage_position_window": _int(input_config.get("coverage_position_window")),
        "chord_pitch_mode": str(input_config.get("chord_pitch_mode") or "approach_tensions"),
        "jazz_rhythm_profile": str(input_config.get("jazz_rhythm_profile") or "swing_motif"),
        "max_simultaneous_notes": _int(input_config.get("max_simultaneous_notes")),
    }


def build_decision_report(
    temperature_guard_repair_probe: dict[str, Any],
    *,
    output_dir: Path,
    issue_number: int,
    target_avg_postprocess_removal_ratio: float,
    target_dead_air_failure_count: int,
) -> dict[str, Any]:
    evidence = validate_temperature_guard_repair_probe(temperature_guard_repair_probe)
    if target_avg_postprocess_removal_ratio <= 0.0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "target postprocess removal ratio required"
        )
    if target_dead_air_failure_count < 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "target dead-air failure count must be non-negative"
        )

    repair_config = {
        "issue_number": int(issue_number),
        "source_temperature": float(evidence["temperature"]),
        "top_k": int(evidence["top_k"]),
        "seeds": list(evidence["seeds"]),
        "num_samples": int(evidence["num_samples"]),
        "max_sequence": int(evidence["max_sequence"]),
        "constrained_note_groups_per_bar": int(
            evidence["constrained_note_groups_per_bar"]
        ),
        "coverage_position_window": int(evidence["coverage_position_window"]),
        "chord_pitch_mode": str(evidence["chord_pitch_mode"]),
        "jazz_rhythm_profile": str(evidence["jazz_rhythm_profile"]),
        "max_simultaneous_notes": int(evidence["max_simultaneous_notes"]),
        "target_avg_postprocess_removal_ratio": float(
            target_avg_postprocess_removal_ratio
        ),
        "target_dead_air_failure_count": int(target_dead_air_failure_count),
        "strategy": "reduce_overlap_before_postprocess_then_verify_dead_air",
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_schema": str(temperature_guard_repair_probe.get("schema_version") or ""),
        "evidence": evidence,
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_postprocess_removal_dead_air_repair_probe",
            "selected_target": "postprocess_removal_dead_air_repair",
            "temperature_followup_selected": False,
            "top_k_followup_selected": False,
            "postprocess_removal_repair_selected": True,
            "coverage_repair_selected": True,
            "audio_review_selected": False,
            "additional_training_scale_selected": False,
            "quality_root_cause_claimed": False,
            "repair_config": repair_config,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "temperature guard removed collapse but left one dead-air strict shortfall "
                "with postprocess removal evidence"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "repair_target_selected": True,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "temperature_guard_repair_probe_completed",
            "grammar_validity_preserved",
            "note_count_failure_removed",
            "collapse_warning_removed",
            "residual_dead_air_recorded",
            "postprocess_removal_evidence_recorded",
        ],
        "not_proven": [
            "postprocess_removal_repair_result",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale "
            "postprocess removal dead-air repair probe"
        ),
    }


def validate_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repair_target: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    evidence = _dict(report.get("evidence"))
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    repair = _dict(decision.get("repair_config"))
    boundary = str(report.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "unexpected boundary"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "unexpected next boundary"
        )
    if require_repair_target and str(decision.get("selected_target") or "") != (
        "postprocess_removal_dead_air_repair"
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "postprocess-removal dead-air repair target required"
        )
    if bool(decision.get("temperature_followup_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "additional temperature follow-up must remain excluded"
        )
    if not bool(decision.get("postprocess_removal_repair_selected", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "postprocess removal repair selection required"
        )
    if _int(evidence.get("dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "dead-air residual evidence required"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "collapse must remain separated"
        )
    if _int(evidence.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "note-count failure must remain removed"
        )
    if _int(evidence.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "grammar failure must remain removed"
        )
    if _float(evidence.get("avg_postprocess_removal_ratio")) <= 0.0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "postprocess removal evidence required"
        )
    if _int(repair.get("target_dead_air_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
            "dead-air repair target must be zero failures"
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRepeatabilityTemperatureGuardFollowupDecisionError(
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
        "avg_postprocess_removal_ratio": _float(
            evidence.get("avg_postprocess_removal_ratio")
        ),
        "max_failed_seed_avg_postprocess_removal_ratio": _float(
            evidence.get("max_failed_seed_avg_postprocess_removal_ratio")
        ),
        "temperature": _float(evidence.get("temperature")),
        "top_k": _int(evidence.get("top_k")),
        "temperature_followup_selected": bool(
            decision.get("temperature_followup_selected", True)
        ),
        "postprocess_removal_repair_selected": bool(
            decision.get("postprocess_removal_repair_selected", False)
        ),
        "target_avg_postprocess_removal_ratio": _float(
            repair.get("target_avg_postprocess_removal_ratio")
        ),
        "target_dead_air_failure_count": _int(repair.get("target_dead_air_failure_count")),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            claim.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["decision"]
    repair = decision["repair_config"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Repeatability Temperature Guard Follow-Up Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- source / repair strict shortfall: `{evidence['source_strict_sample_shortfall']}` -> `{evidence['repair_strict_sample_shortfall']}`",
        f"- source / repair dead-air failure: `{evidence['source_dead_air_failure_count']}` -> `{evidence['repair_dead_air_failure_count']}`",
        f"- source / repair collapse warning: `{evidence['source_collapse_warning_sample_count']}` -> `{evidence['repair_collapse_warning_sample_count']}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(claim['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Evidence",
        "",
        f"- seed count: `{evidence['seed_count']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- valid / strict / grammar: `{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / `{evidence['grammar_gate_sample_count']}`",
        f"- note-count / grammar / dead-air / collapse failure count: `{evidence['note_count_failure_count']}` / `{evidence['grammar_failure_count']}` / `{evidence['dead_air_failure_count']}` / `{evidence['collapse_warning_sample_count']}`",
        f"- failed seeds: `{evidence['failed_seeds']}`",
        f"- avg postprocess removal ratio: `{evidence['avg_postprocess_removal_ratio']}`",
        f"- max failed-seed avg postprocess removal ratio: `{evidence['max_failed_seed_avg_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        f"- temperature / top_k: `{evidence['temperature']}` / `{evidence['top_k']}`",
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
            "- temperature follow-up selected: `{}`".format(
                _bool_token(decision["temperature_followup_selected"])
            ),
            "- top_k follow-up selected: `{}`".format(
                _bool_token(decision["top_k_followup_selected"])
            ),
            "- postprocess removal repair selected: `{}`".format(
                _bool_token(decision["postprocess_removal_repair_selected"])
            ),
            "- coverage repair selected: `{}`".format(
                _bool_token(decision["coverage_repair_selected"])
            ),
            "- audio review selected: `{}`".format(_bool_token(decision["audio_review_selected"])),
            "- additional training scale selected: `{}`".format(
                _bool_token(decision["additional_training_scale_selected"])
            ),
            "- critical user input required: `{}`".format(
                _bool_token(decision["critical_user_input_required"])
            ),
            "",
            "## Repair Config",
            "",
        ]
    )
    for key in [
        "source_temperature",
        "top_k",
        "seeds",
        "num_samples",
        "max_sequence",
        "constrained_note_groups_per_bar",
        "coverage_position_window",
        "chord_pitch_mode",
        "jazz_rhythm_profile",
        "max_simultaneous_notes",
        "target_avg_postprocess_removal_ratio",
        "target_dead_air_failure_count",
        "strategy",
    ]:
        lines.append(f"- {key}: `{repair[key]}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide selected-scale temperature guard follow-up target"
    )
    parser.add_argument(
        "--temperature_guard_repair_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_repair_probe.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=600)
    parser.add_argument("--target_avg_postprocess_removal_ratio", type=float, default=0.3)
    parser.add_argument("--target_dead_air_failure_count", type=int, default=0)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repair_target", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_decision_report(
        read_json(Path(args.temperature_guard_repair_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        target_avg_postprocess_removal_ratio=float(
            args.target_avg_postprocess_removal_ratio
        ),
        target_dead_air_failure_count=int(args.target_dead_air_failure_count),
    )
    summary = validate_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repair_target=bool(args.require_repair_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repeatability_temperature_guard_followup_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
