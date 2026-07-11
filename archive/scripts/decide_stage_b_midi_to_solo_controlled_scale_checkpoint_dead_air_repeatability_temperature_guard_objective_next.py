"""Decide the objective-only next boundary after controlled temperature guard review."""

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
from scripts.build_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_listening_review import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)
from scripts.consolidate_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_repair import (  # noqa: E402
    BOUNDARY as CONSOLIDATION_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_objective_only_next_decision"
)
FINAL_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_objective_path_complete"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
    "temperature_guard_objective_next_v1"
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


def _count_or_len(summary: dict[str, Any], count_key: str, list_key: str) -> int:
    count = _int(summary.get(count_key))
    if count:
        return count
    return len(_list(summary.get(list_key)))


def validate_listening_review_boundary(report: dict[str, Any]) -> dict[str, Any]:
    boundary = _dict(report.get("listening_review_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review_summary = _dict(report.get("review_input_summary"))
    if str(boundary.get("boundary") or report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "temperature guard listening review boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "listening review must route to objective-only next decision"
        )
    if not bool(readiness.get("listening_review_boundary_prepared", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "listening review boundary preparation required"
        )
    if bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "validated review input should be absent for objective-only decision"
        )
    blocked_claims = [
        "listening_review_completed",
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            f"unexpected listening review claim: {claimed}"
        )
    candidates = [_dict(item) for item in _list(report.get("review_candidates")) if isinstance(item, dict)]
    if not candidates:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "review candidates required"
        )
    return {
        "candidate_count": len(candidates),
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "review_input_template_written": bool(boundary.get("review_input_template_written", False)),
        "validated_review_input_present": bool(readiness.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "pending_status_field_count": _count_or_len(
            review_summary,
            "pending_status_field_count",
            "pending_status_fields",
        ),
        "pending_candidate_decision_count": _count_or_len(
            review_summary,
            "pending_candidate_decision_count",
            "pending_candidate_decisions",
        ),
        "pending_candidate_field_count": _count_or_len(
            review_summary,
            "pending_candidate_field_count",
            "pending_candidate_fields",
        ),
    }


def validate_temperature_guard_consolidation(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != CONSOLIDATION_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "temperature guard repair consolidation boundary required"
        )
    evidence = _dict(report.get("evidence_summary"))
    result = _dict(report.get("consolidation_result"))
    readiness = _dict(report.get("readiness"))
    if not bool(result.get("objective_temperature_guard_support", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "objective temperature guard support required"
        )
    if bool(result.get("additional_repair_required", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "additional repair should be false"
        )
    if not bool(result.get("audio_review_package_required", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "audio review package should be required before listening boundary"
        )
    sample_count = _int(evidence.get("sample_count"))
    if sample_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "sample count required"
        )
    if _int(evidence.get("strict_valid_sample_count")) != sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "all samples should be strict valid"
        )
    if _int(evidence.get("grammar_gate_sample_count")) != sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "all samples should pass grammar gate"
        )
    if _int(evidence.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "dead-air failures should be zero"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "collapse warnings should be zero"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            f"unexpected consolidation claim: {claimed}"
        )
    return {
        "sample_count": sample_count,
        "seed_count": _int(evidence.get("seed_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "postprocess_collapse_failure_count": _int(
            evidence.get("postprocess_collapse_failure_count")
        ),
        "strict_valid_sample_delta": _int(evidence.get("strict_valid_sample_delta")),
        "source_strict_sample_shortfall": _int(evidence.get("source_strict_sample_shortfall")),
        "repair_strict_sample_shortfall": _int(evidence.get("repair_strict_sample_shortfall")),
        "source_dead_air_failure_count": _int(evidence.get("source_dead_air_failure_count")),
        "repair_dead_air_failure_count": _int(evidence.get("repair_dead_air_failure_count")),
        "source_temperature": _float(evidence.get("source_temperature")),
        "temperature": _float(evidence.get("temperature")),
        "top_k": _int(evidence.get("top_k")),
        "avg_postprocess_removal_ratio": _float(evidence.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
        "objective_temperature_guard_support": bool(
            result.get("objective_temperature_guard_support", False)
        ),
    }


def build_objective_next_decision_report(
    listening_review: dict[str, Any],
    consolidation: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    review = validate_listening_review_boundary(listening_review)
    evidence = validate_temperature_guard_consolidation(consolidation)
    objective_path_supported = (
        bool(evidence.get("objective_temperature_guard_support", False))
        and _int(evidence.get("strict_valid_sample_count")) == _int(evidence.get("sample_count"))
        and _int(evidence.get("dead_air_failure_count")) == 0
        and _int(evidence.get("collapse_warning_sample_count")) == 0
        and not bool(review.get("validated_review_input_present", True))
    )
    if not objective_path_supported:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "objective temperature guard path support required"
        )
    boundary = {
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "consolidation_boundary": CONSOLIDATION_BOUNDARY,
        "final_boundary": FINAL_BOUNDARY,
        "objective_only_decision_completed": True,
        "objective_temperature_guard_path_supported": True,
        "selected_next_boundary": NEXT_BOUNDARY,
        "validated_review_input_present": False,
        "preference_fill_allowed": False,
        "human_audio_preference_claimed": False,
        "midi_to_solo_musical_quality_claimed": False,
        "broad_trained_model_quality_claimed": False,
        "brad_style_adaptation_claimed": False,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "final_boundary": FINAL_BOUNDARY,
        "review_boundary_summary": review,
        "temperature_guard_summary": evidence,
        "objective_next_decision_boundary": boundary,
        "readiness": {
            "boundary": BOUNDARY,
            "objective_only_decision_completed": True,
            "objective_temperature_guard_path_supported": True,
            "validated_review_input_present": False,
            "preference_fill_allowed": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "final_boundary": FINAL_BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "temperature guard objective MIDI evidence is supported; listening review input remains pending, "
                "so close this objective path and route the next boundary to controlled training scale expansion"
            ),
        },
        "proven": [
            "temperature_guard_strict_valid_9_of_9",
            "dead_air_and_collapse_failures_zero",
            "technical_audio_review_package_prepared",
            "human_audio_preference_claim_blocked_without_review_input",
        ],
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale expansion decision"
        ),
    }


def validate_objective_next_decision_report(
    report: dict[str, Any],
    *,
    expected_final_boundary: str | None,
    expected_next_boundary: str | None,
    min_sample_count: int,
    min_candidate_count: int,
    require_objective_support: bool,
    require_pending_review: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = _dict(report.get("objective_next_decision_boundary"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review = _dict(report.get("review_boundary_summary"))
    evidence = _dict(report.get("temperature_guard_summary"))
    if expected_final_boundary and str(decision.get("final_boundary") or "") != expected_final_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "unexpected final boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "unexpected next boundary"
        )
    if _int(evidence.get("sample_count")) < int(min_sample_count):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "sample count below requirement"
        )
    if _int(review.get("candidate_count")) < int(min_candidate_count):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "candidate count below requirement"
        )
    if require_objective_support and not bool(
        readiness.get("objective_temperature_guard_path_supported", False)
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "objective temperature guard path support required"
        )
    if require_pending_review and bool(readiness.get("validated_review_input_present", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
            "review input should remain pending"
        )
    if require_no_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointTemperatureGuardObjectiveNextError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": str(boundary.get("boundary") or ""),
        "final_boundary": str(decision.get("final_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "candidate_count": _int(review.get("candidate_count")),
        "rendered_audio_file_count": _int(review.get("rendered_audio_file_count")),
        "sample_count": _int(evidence.get("sample_count")),
        "seed_count": _int(evidence.get("seed_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "strict_valid_sample_delta": _int(evidence.get("strict_valid_sample_delta")),
        "source_temperature": _float(evidence.get("source_temperature")),
        "temperature": _float(evidence.get("temperature")),
        "top_k": _int(evidence.get("top_k")),
        "pending_status_field_count": _int(review.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(review.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(review.get("pending_candidate_field_count")),
        "objective_temperature_guard_path_supported": bool(
            readiness.get("objective_temperature_guard_path_supported", False)
        ),
        "validated_review_input_present": bool(readiness.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(readiness.get("preference_fill_allowed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["objective_next_decision_boundary"]
    decision = report["decision"]
    review = report["review_boundary_summary"]
    evidence = report["temperature_guard_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Temperature Guard Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{boundary['boundary']}`",
        f"- final boundary: `{decision['final_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective-only decision completed: `{_bool_token(boundary['objective_only_decision_completed'])}`",
        f"- objective temperature guard path supported: `{_bool_token(boundary['objective_temperature_guard_path_supported'])}`",
        f"- candidate count: `{review['candidate_count']}`",
        f"- rendered audio file count: `{review['rendered_audio_file_count']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- seed count: `{evidence['seed_count']}`",
        f"- valid / strict / grammar: `{evidence['valid_sample_count']} / {evidence['strict_valid_sample_count']} / {evidence['grammar_gate_sample_count']}`",
        f"- dead-air / collapse failure count: `{evidence['dead_air_failure_count']} / {evidence['collapse_warning_sample_count']}`",
        f"- strict valid sample delta: `{evidence['strict_valid_sample_delta']}`",
        f"- source / selected temperature: `{evidence['source_temperature']:.2f} / {evidence['temperature']:.2f}`",
        f"- top_k: `{evidence['top_k']}`",
        f"- validated review input present: `{_bool_token(boundary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(boundary['preference_fill_allowed'])}`",
        f"- human/audio preference claimed: `{_bool_token(boundary['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(boundary['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Pending Review",
        "",
        f"- pending status fields: `{review['pending_status_field_count']}`",
        f"- pending candidate decisions: `{review['pending_candidate_decision_count']}`",
        f"- pending candidate fields: `{review['pending_candidate_field_count']}`",
        "",
        "## Proven",
        "",
    ]
    for item in report.get("proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide controlled temperature guard objective-only next boundary")
    parser.add_argument(
        "--listening_review",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
            "temperature_guard_listening_review/harness_stage_b_midi_to_solo_controlled_scale_"
            "checkpoint_dead_air_repeatability_temperature_guard_listening_review/"
            "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
            "temperature_guard_listening_review.json"
        ),
    )
    parser.add_argument(
        "--consolidation_report",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
            "temperature_guard_repair_consolidation/harness_stage_b_midi_to_solo_controlled_"
            "scale_checkpoint_dead_air_repeatability_temperature_guard_repair_consolidation/"
            "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
            "temperature_guard_repair_consolidation.json"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
            "temperature_guard_objective_next"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_final_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_sample_count", type=int, default=1)
    parser.add_argument("--min_candidate_count", type=int, default=1)
    parser.add_argument("--require_objective_support", action="store_true")
    parser.add_argument("--require_pending_review", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_objective_next_decision_report(
        read_json(Path(args.listening_review)),
        read_json(Path(args.consolidation_report)),
        output_dir=output_dir,
    )
    summary = validate_objective_next_decision_report(
        report,
        expected_final_boundary=str(args.expected_final_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_sample_count=int(args.min_sample_count),
        min_candidate_count=int(args.min_candidate_count),
        require_objective_support=bool(args.require_objective_support),
        require_pending_review=bool(args.require_pending_review),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
