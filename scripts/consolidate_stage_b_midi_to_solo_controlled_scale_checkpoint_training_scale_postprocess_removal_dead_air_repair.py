"""Consolidate selected-scale postprocess-removal dead-air repair evidence."""

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
from scripts.run_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_consolidation"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_audio_review_package"
)
FAILURE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_residual_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_"
    "postprocess_removal_dead_air_repair_consolidation_v1"
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


def validate_repair_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    comparison = _dict(report.get("comparison"))
    input_config = _dict(report.get("input"))
    source = _dict(report.get("source_summary"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "postprocess-removal repair probe boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "postprocess-removal repair probe must route to consolidation"
        )
    if not bool(readiness.get("postprocess_removal_dead_air_repair_probe_completed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "repair probe completion required"
        )
    if not bool(readiness.get("postprocess_removal_dead_air_repair_target_qualified", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "qualified postprocess-removal repair target required"
        )
    if not bool(aggregate.get("all_seed_gate_passed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "all seed gate support required"
        )
    if not bool(aggregate.get("all_samples_strict_valid", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "all-sample strict support required"
        )
    sample_count = _int(aggregate.get("sample_count"))
    if sample_count <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "sample count required"
        )
    if _int(aggregate.get("strict_valid_sample_count")) != sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "strict sample count must equal sample count"
        )
    if _int(aggregate.get("grammar_gate_sample_count")) != sample_count:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "grammar sample count must equal sample count"
        )
    if _int(aggregate.get("note_count_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "note-count failure must be zero"
        )
    if _int(aggregate.get("grammar_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "grammar failure must be zero"
        )
    if _int(aggregate.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "dead-air failure must be zero"
        )
    if _int(aggregate.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "collapse warning must be zero"
        )
    if _dict(aggregate.get("diagnostic_failure_reasons")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "diagnostic failures must be empty"
        )
    blocked = [
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked if bool(readiness.get(name, True))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            f"unexpected quality claim: {claimed}"
        )
    seed_rows = [_dict(row) for row in _list(report.get("seed_rows"))]
    return {
        "sample_count": sample_count,
        "seed_count": _int(aggregate.get("seed_count")),
        "seeds": [int(seed) for seed in _list(aggregate.get("seeds"))],
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
        "target_avg_postprocess_removal_ratio": _float(
            input_config.get("target_avg_postprocess_removal_ratio")
        ),
        "avg_onset_coverage_ratio": _float(aggregate.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(
            aggregate.get("avg_sustained_coverage_ratio")
        ),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "dead_air_failure_delta": _int(comparison.get("dead_air_failure_delta")),
        "postprocess_removal_delta": _float(comparison.get("postprocess_removal_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
        "source_strict_valid_sample_count": _int(
            source.get("source_strict_valid_sample_count")
        ),
        "source_dead_air_failure_count": _int(source.get("source_dead_air_failure_count")),
        "source_avg_postprocess_removal_ratio": _float(
            source.get("source_avg_postprocess_removal_ratio")
        ),
        "temperature": _float(input_config.get("temperature")),
        "top_k": _int(input_config.get("top_k")),
        "avoid_reused_positions": bool(input_config.get("avoid_reused_positions", False)),
        "generation_report_paths": [
            str(row.get("generation_report_path") or "") for row in seed_rows
        ],
    }


def build_consolidation_report(
    repair_probe: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_repair_probe(repair_probe)
    objective_support = (
        _int(evidence["sample_count"]) > 0
        and _int(evidence["strict_valid_sample_count"]) == _int(evidence["sample_count"])
        and _int(evidence["dead_air_failure_count"]) == 0
        and _int(evidence["collapse_warning_sample_count"]) == 0
        and _float(evidence["avg_postprocess_removal_ratio"])
        <= _float(evidence["target_avg_postprocess_removal_ratio"])
    )
    next_boundary = NEXT_BOUNDARY if objective_support else FAILURE_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "evidence_summary": evidence,
        "consolidation_result": {
            "objective_midi_support": objective_support,
            "additional_repair_required": not objective_support,
            "audio_review_package_required": objective_support,
            "support_scope": "objective_midi_only",
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "postprocess_removal_dead_air_repair_consolidation_completed": True,
            "objective_midi_support": objective_support,
            "audio_review_package_required": objective_support,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "postprocess-removal repair passed objective MIDI gates; route candidates "
                "to audio review package without claiming musical quality"
            ),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air repair audio review package"
            if objective_support
            else "Stage B MIDI-to-solo controlled scale checkpoint training scale postprocess removal dead-air residual decision"
        ),
    }


def validate_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_sample_count: int,
    require_objective_support: bool,
    require_audio_review_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    evidence = _dict(report.get("evidence_summary"))
    result = _dict(report.get("consolidation_result"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "unexpected next boundary"
        )
    if _int(evidence.get("sample_count")) < int(min_sample_count):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "minimum sample count not met"
        )
    if require_objective_support and not bool(result.get("objective_midi_support", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "objective support required"
        )
    if require_audio_review_required and not bool(
        result.get("audio_review_package_required", False)
    ):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
            "audio review package requirement expected"
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScalePostprocessRemovalDeadAirRepairConsolidationError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "sample_count": _int(evidence.get("sample_count")),
        "seed_count": _int(evidence.get("seed_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "avg_postprocess_removal_ratio": _float(
            evidence.get("avg_postprocess_removal_ratio")
        ),
        "max_postprocess_removal_ratio": _float(
            evidence.get("max_postprocess_removal_ratio")
        ),
        "target_avg_postprocess_removal_ratio": _float(
            evidence.get("target_avg_postprocess_removal_ratio")
        ),
        "strict_valid_sample_delta": _int(evidence.get("strict_valid_sample_delta")),
        "dead_air_failure_delta": _int(evidence.get("dead_air_failure_delta")),
        "postprocess_removal_delta": _float(evidence.get("postprocess_removal_delta")),
        "avoid_reused_positions": bool(evidence.get("avoid_reused_positions", False)),
        "objective_midi_support": bool(result.get("objective_midi_support", False)),
        "audio_review_package_required": bool(
            result.get("audio_review_package_required", False)
        ),
        "additional_repair_required": bool(result.get("additional_repair_required", True)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence_summary"]
    result = report["consolidation_result"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Postprocess Removal Dead-Air Repair Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective MIDI support: `{_bool_token(result['objective_midi_support'])}`",
        f"- audio review package required: `{_bool_token(result['audio_review_package_required'])}`",
        f"- additional repair required: `{_bool_token(result['additional_repair_required'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(result['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Evidence",
        "",
        f"- seed count: `{evidence['seed_count']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- valid / strict / grammar: `{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / `{evidence['grammar_gate_sample_count']}`",
        f"- dead-air / collapse failure count: `{evidence['dead_air_failure_count']}` / `{evidence['collapse_warning_sample_count']}`",
        f"- avg / max postprocess removal ratio: `{evidence['avg_postprocess_removal_ratio']}` / `{evidence['max_postprocess_removal_ratio']}`",
        f"- target avg postprocess removal ratio: `{evidence['target_avg_postprocess_removal_ratio']}`",
        f"- avoid reused positions: `{_bool_token(evidence['avoid_reused_positions'])}`",
        "",
        "## Delta",
        "",
        f"- strict valid sample delta: `{evidence['strict_valid_sample_delta']}`",
        f"- dead-air failure delta: `{evidence['dead_air_failure_delta']}`",
        f"- postprocess removal delta: `{evidence['postprocess_removal_delta']}`",
        f"- onset / sustained coverage delta: `{evidence['onset_coverage_delta']}` / `{evidence['sustained_coverage_delta']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate selected-scale postprocess-removal dead-air repair evidence"
    )
    parser.add_argument(
        "--repair_probe_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_probe.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_sample_count", type=int, default=9)
    parser.add_argument("--require_objective_support", action="store_true")
    parser.add_argument("--require_audio_review_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_consolidation_report(
        read_json(Path(args.repair_probe_report)),
        output_dir=output_dir,
    )
    summary = validate_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_sample_count=int(args.min_sample_count),
        require_objective_support=bool(args.require_objective_support),
        require_audio_review_required=bool(args.require_audio_review_required),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_postprocess_removal_dead_air_repair_consolidation.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
