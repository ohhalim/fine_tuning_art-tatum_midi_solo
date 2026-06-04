"""Decide remaining blocker after selected-scale repair repeatability."""

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


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
    ValueError
):
    pass


SOURCE_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "density_grammar_collapse_repeatability_probe"
)
BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "density_grammar_collapse_dead_air_remaining_blocker_decision"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_repair_probe"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "dead_air_remaining_blocker_decision_v1"
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


def validate_repeatability_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    comparison = _dict(report.get("comparison"))
    if str(report.get("boundary") or readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "density/grammar/collapse repeatability boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "repeatability probe must route to dead-air remaining blocker decision"
        )
    if not bool(readiness.get("density_grammar_collapse_repeatability_target_supported", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "density/grammar/collapse repeatability target support required"
        )
    if bool(readiness.get("strict_gate_stable", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "strict gate must remain unstable for dead-air decision"
        )
    if _int(aggregate.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "note-count failure must be removed before dead-air decision"
        )
    if _int(aggregate.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "grammar failure must be removed before dead-air decision"
        )
    if _int(aggregate.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "collapse warning must be removed before dead-air decision"
        )
    dead_air_failures = _int(aggregate.get("dead_air_failure_count"))
    if dead_air_failures <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "dead-air failure evidence required"
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
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            f"unexpected quality claim: {claimed}"
        )
    return {
        "seed_count": _int(aggregate.get("seed_count")),
        "sample_count": _int(aggregate.get("sample_count")),
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(aggregate.get("note_count_failure_count")),
        "grammar_failure_count": _int(aggregate.get("grammar_failure_count")),
        "dead_air_failure_count": dead_air_failures,
        "collapse_warning_sample_count": _int(aggregate.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(aggregate.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(aggregate.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(aggregate.get("avg_sustained_coverage_ratio")),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "postprocess_removal_delta": _float(comparison.get("postprocess_removal_delta")),
        "failure_reasons": _dict(aggregate.get("diagnostic_failure_reasons")),
    }


def build_decision_report(
    repeatability_probe: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_repeatability_probe(repeatability_probe)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "source_schema": str(repeatability_probe.get("schema_version") or ""),
        "input_boundary": SOURCE_BOUNDARY,
        "evidence": {
            **evidence,
            "density_grammar_collapse_repeatability_target_supported": True,
            "remaining_blocker": "dead_air_sustained_coverage",
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_dead_air_repair_probe",
            "selected_target": "selected_scale_dead_air_sustained_coverage_repair",
            "density_grammar_collapse_followup_selected": False,
            "audio_review_selected": False,
            "additional_training_scale_selected": False,
            "quality_root_cause_claimed": False,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "density/grammar/collapse repeatability target support measured, "
                "but strict gate remains unstable because of dead-air"
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
            "density_grammar_collapse_repeatability_target_supported",
            "note_count_failure_removed",
            "grammar_failure_removed",
            "collapse_warning_removed",
            "dead_air_remaining_blocker_recorded",
            "dead_air_repair_target_selected",
        ],
        "not_proven": [
            "dead_air_repair_result",
            "strict_gate_stability",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale dead-air repair probe"
        ),
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
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_dead_air_target and str(decision.get("selected_target") or "") != (
        "selected_scale_dead_air_sustained_coverage_repair"
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "dead-air repair target required"
        )
    if _int(evidence.get("dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "dead-air failure evidence required"
        )
    if _int(evidence.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "note-count failure must remain removed"
        )
    if _int(evidence.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
            "grammar failure must remain removed"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleDeadAirRemainingBlockerDecisionError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "input_boundary": str(report.get("input_boundary") or ""),
        "decision": str(decision.get("decision") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "density_grammar_collapse_followup_selected": bool(
            decision.get("density_grammar_collapse_followup_selected", True)
        ),
        "audio_review_selected": bool(decision.get("audio_review_selected", True)),
        "additional_training_scale_selected": bool(
            decision.get("additional_training_scale_selected", True)
        ),
        "seed_count": _int(evidence.get("seed_count")),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "note_count_failure_count": _int(evidence.get("note_count_failure_count")),
        "grammar_failure_count": _int(evidence.get("grammar_failure_count")),
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
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Dead-Air Remaining Blocker Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{decision['current_boundary']}`",
        f"- decision: `{decision['decision']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- remaining blocker: `{evidence['remaining_blocker']}`",
        f"- density/grammar/collapse follow-up selected: `{_bool_token(decision['density_grammar_collapse_followup_selected'])}`",
        f"- audio review selected: `{_bool_token(decision['audio_review_selected'])}`",
        f"- additional training scale selected: `{_bool_token(decision['additional_training_scale_selected'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(claim['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(claim['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(claim['brad_style_adaptation_claimed'])}`",
        f"- next boundary: `{decision['next_boundary']}`",
        "",
        "## Evidence",
        "",
        f"- seed count: `{evidence['seed_count']}`",
        f"- sample count: `{evidence['sample_count']}`",
        (
            "- valid / strict / grammar gate sample count: "
            f"`{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / "
            f"`{evidence['grammar_gate_sample_count']}`"
        ),
        f"- note-count failure count: `{evidence['note_count_failure_count']}`",
        f"- grammar failure count: `{evidence['grammar_failure_count']}`",
        f"- collapse warning sample count: `{evidence['collapse_warning_sample_count']}`",
        f"- dead-air failure count: `{evidence['dead_air_failure_count']}`",
        f"- avg postprocess removal ratio: `{evidence['avg_postprocess_removal_ratio']}`",
        f"- avg onset / sustained coverage ratio: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
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
        description="Decide selected-scale dead-air remaining blocker"
    )
    parser.add_argument(
        "--repeatability_probe",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
            "density_grammar_collapse_repeatability_probe/"
            "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
            "density_grammar_collapse_repeatability_probe/"
            "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
            "density_grammar_collapse_repeatability_probe.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
            "dead_air_remaining_blocker_decision"
        ),
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
        read_json(Path(args.repeatability_probe)),
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
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_dead_air_remaining_blocker_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
