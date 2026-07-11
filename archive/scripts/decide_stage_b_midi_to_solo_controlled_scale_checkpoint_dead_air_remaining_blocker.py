"""Decide remaining blocker after controlled checkpoint density/collapse repair."""

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


class StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
    ValueError
):
    pass


SOURCE_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe"
BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repair_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision_v1"


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


def validate_density_collapse_repair(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("repair_summary"))
    comparison = _dict(report.get("comparison"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "density/collapse repair boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "repair probe must route to dead-air remaining blocker decision"
        )
    if not bool(readiness.get("density_collapse_target_supported", False)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "density/collapse target support required"
        )
    if bool(readiness.get("strict_gate_recovered", True)):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "strict gate must remain unrecovered for blocker decision"
        )
    if _int(repair.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "note-count failure must be removed before dead-air decision"
        )
    if _int(repair.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "collapse warning must be removed before dead-air decision"
        )
    dead_air_failures = _int(repair.get("dead_air_failure_count"))
    if dead_air_failures <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "dead-air failure evidence required"
        )
    blocked = [
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked if bool(readiness.get(name, True))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            f"unexpected quality claim: {claimed}"
        )
    return {
        "sample_count": _int(repair.get("sample_count")),
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(repair.get("note_count_failure_count")),
        "dead_air_failure_count": dead_air_failures,
        "collapse_warning_sample_count": _int(repair.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(repair.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(repair.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            repair.get("max_longest_sustained_empty_run_steps")
        ),
        "note_count_failure_delta": _int(comparison.get("note_count_failure_delta")),
        "collapse_warning_delta": _int(comparison.get("collapse_warning_delta")),
        "postprocess_removal_delta": _float(comparison.get("postprocess_removal_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
        "failure_reasons": _dict(repair.get("diagnostic_failure_reasons")),
    }


def build_decision_report(
    density_collapse_repair: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_density_collapse_repair(density_collapse_repair)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "source_schema": str(density_collapse_repair.get("schema_version") or ""),
        "input_boundary": SOURCE_BOUNDARY,
        "evidence": {
            **evidence,
            "density_collapse_target_supported": True,
            "remaining_blocker": "dead_air_sustained_coverage",
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_dead_air_repair_probe",
            "selected_target": "dead_air_sustained_coverage_repair",
            "audio_review_selected": False,
            "training_scale_change_selected": False,
            "quality_root_cause_claimed": False,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "density/collapse target support measured, but all samples still fail the MIDI gate "
                "because of dead-air"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "remaining_blocker_classified": True,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "density_collapse_target_supported",
            "note_count_failure_removed",
            "collapse_warning_removed",
            "dead_air_remaining_blocker_recorded",
            "dead_air_repair_target_selected",
        ],
        "not_proven": [
            "strict_gate_recovered",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo controlled checkpoint dead-air repair probe",
    }


def validate_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_dead_air_target: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    boundary = str(decision.get("current_boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_dead_air_target and str(decision.get("selected_target") or "") != (
        "dead_air_sustained_coverage_repair"
    ):
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "dead-air repair target required"
        )
    if _int(evidence.get("dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "dead-air failure evidence required"
        )
    if _int(evidence.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "note-count failure must remain removed"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
            "collapse warning must remain removed"
        )
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("midi_to_solo_musical_quality_claimed", True)),
            bool(claim.get("human_audio_preference_claimed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
            bool(claim.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBMidiToSoloControlledScaleCheckpointDeadAirRemainingBlockerDecisionError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "input_boundary": str(report.get("input_boundary") or ""),
        "decision": str(decision.get("decision") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "audio_review_selected": bool(decision.get("audio_review_selected", True)),
        "training_scale_change_selected": bool(
            decision.get("training_scale_change_selected", True)
        ),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "avg_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
        "remaining_blocker": str(evidence.get("remaining_blocker") or ""),
        "midi_to_solo_musical_quality_claimed": bool(
            claim.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "broad_trained_model_quality_claimed": bool(
            claim.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(
            claim.get("brad_style_adaptation_claimed", True)
        ),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "next_boundary": next_boundary,
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["decision"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Dead-Air Remaining Blocker Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{decision['current_boundary']}`",
        f"- decision: `{decision['decision']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- remaining blocker: `{evidence['remaining_blocker']}`",
        f"- audio review selected: `{_bool_token(decision['audio_review_selected'])}`",
        f"- training scale change selected: `{_bool_token(decision['training_scale_change_selected'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(claim['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(claim['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(claim['brad_style_adaptation_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Evidence",
        "",
        f"- sample count: `{evidence['sample_count']}`",
        (
            "- valid / strict / grammar gate sample count: "
            f"`{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / "
            f"`{evidence['grammar_gate_sample_count']}`"
        ),
        f"- note-count failure count: `{evidence['note_count_failure_count']}`",
        f"- collapse warning sample count: `{evidence['collapse_warning_sample_count']}`",
        f"- dead-air failure count: `{evidence['dead_air_failure_count']}`",
        f"- avg postprocess removal ratio: `{evidence['avg_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{evidence['max_longest_sustained_empty_run_steps']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in evidence["failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide controlled checkpoint dead-air remaining blocker"
    )
    parser.add_argument(
        "--density_collapse_repair",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_density_collapse_repair_probe.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_dead_air_target", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_decision_report(
        read_json(Path(args.density_collapse_repair)),
        output_dir=output_dir,
    )
    summary = validate_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_dead_air_target=bool(args.require_dead_air_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_remaining_blocker_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
