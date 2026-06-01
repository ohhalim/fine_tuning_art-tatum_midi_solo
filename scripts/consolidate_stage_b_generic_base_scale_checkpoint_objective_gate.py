"""Consolidate objective-gate evidence for the generic-base scale checkpoint."""

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


class StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation"
NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep"


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


def validate_repair_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repair = _dict(report.get("repair_summary"))
    comparison = _dict(report.get("comparison"))
    input_config = _dict(report.get("input"))
    if str(readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "sustained coverage dead-air repair boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "repair probe must route to objective gate consolidation"
        )
    if not bool(readiness.get("sustained_coverage_dead_air_target_qualified", False)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "sustained coverage dead-air target must be qualified"
        )
    if bool(readiness.get("raw_generation_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "raw generation quality must not be claimed"
        )
    if bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "broad trained-model quality must not be claimed"
        )
    if bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "Brad style adaptation must not be claimed"
        )
    sample_count = _int(repair.get("sample_count"))
    if sample_count <= 0:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError("sample count required")
    if _int(repair.get("valid_sample_count")) != sample_count:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError("all samples must pass valid gate")
    if _int(repair.get("strict_valid_sample_count")) != sample_count:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError("all samples must pass strict gate")
    if _int(repair.get("grammar_gate_sample_count")) != sample_count:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError("all samples must pass grammar gate")
    if _int(repair.get("dead_air_failure_count")) != 0:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "dead-air failures must be absent"
        )
    if _int(repair.get("long_note_failure_count")) != 0:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "long-note failures must be absent"
        )
    if _dict(repair.get("diagnostic_failure_reasons")):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "diagnostic failure reasons must be empty"
        )
    return {
        "sample_count": sample_count,
        "valid_sample_count": _int(repair.get("valid_sample_count")),
        "strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "dead_air_failure_count": _int(repair.get("dead_air_failure_count")),
        "long_note_failure_count": _int(repair.get("long_note_failure_count")),
        "avg_onset_coverage_ratio": _float(repair.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(repair.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            repair.get("max_longest_sustained_empty_run_steps")
        ),
        "dead_air_failure_delta": _int(comparison.get("dead_air_failure_delta")),
        "valid_sample_delta": _int(comparison.get("valid_sample_delta")),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "onset_coverage_delta": _float(comparison.get("onset_coverage_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
        "long_note_failure_reintroduced": bool(
            comparison.get("long_note_failure_reintroduced", True)
        ),
        "constrained_note_groups_per_bar": _int(input_config.get("constrained_note_groups_per_bar")),
        "jazz_duration_tokens": bool(input_config.get("jazz_duration_tokens", False)),
        "coverage_aware_positions": bool(input_config.get("coverage_aware_positions", False)),
    }


def build_consolidation_report(
    repair_probe: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_repair_probe(repair_probe)
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schema": str(repair_probe.get("schema_version") or ""),
        "input_boundary": SOURCE_BOUNDARY,
        "evidence": {
            **evidence,
            "objective_gate_support": True,
            "single_seed_set_only": True,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_objective_gate_repeatability_sweep",
            "selected_target": "objective_gate_repeatability_sweep",
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "current seed set passes objective MIDI gates, but repeatability and listening quality "
                "remain unproven"
            ),
        },
        "claim_boundary": {
            "boundary": BOUNDARY,
            "objective_gate_support_consolidated": True,
            "repeatability_claimed": False,
            "musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "objective_gate_support_for_current_seed_set",
            "dead_air_failure_removed_for_current_seed_set",
            "long_note_guardrail_preserved_for_current_seed_set",
        ],
        "not_proven": [
            "repeatability",
            "musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base scale checkpoint objective gate repeatability sweep",
    }


def validate_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_repeatability_target: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    boundary = str(decision.get("current_boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_repeatability_target and str(decision.get("selected_target") or "") != (
        "objective_gate_repeatability_sweep"
    ):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "objective gate repeatability target required"
        )
    if not bool(evidence.get("objective_gate_support", False)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "objective gate support required"
        )
    if _int(evidence.get("dead_air_failure_count")) != 0:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "dead-air failures must remain absent"
        )
    if _int(evidence.get("long_note_failure_count")) != 0:
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "long-note failures must remain absent"
        )
    if bool(evidence.get("long_note_failure_reintroduced", True)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
            "long-note failure must not be reintroduced"
        )
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("repeatability_claimed", True)),
            bool(claim.get("musical_quality_claimed", True)),
            bool(claim.get("human_audio_preference_claimed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
            bool(claim.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericBaseScaleCheckpointObjectiveGateConsolidationError(
                "quality and repeatability claims must remain false"
            )
    return {
        "boundary": boundary,
        "input_boundary": str(report.get("input_boundary") or ""),
        "decision": str(decision.get("decision") or ""),
        "selected_target": str(decision.get("selected_target") or ""),
        "objective_gate_support": bool(evidence.get("objective_gate_support", False)),
        "single_seed_set_only": bool(evidence.get("single_seed_set_only", False)),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "long_note_failure_count": _int(evidence.get("long_note_failure_count")),
        "avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
        "repeatability_claimed": bool(claim.get("repeatability_claimed", True)),
        "musical_quality_claimed": bool(claim.get("musical_quality_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            claim.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(claim.get("brad_style_adaptation_claimed", True)),
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": next_boundary,
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    decision = report["decision"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Objective Gate Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{decision['current_boundary']}`",
        f"- decision: `{decision['decision']}`",
        f"- selected target: `{decision['selected_target']}`",
        f"- objective gate support: `{_bool_token(evidence['objective_gate_support'])}`",
        f"- single seed set only: `{_bool_token(evidence['single_seed_set_only'])}`",
        f"- repeatability claimed: `{_bool_token(claim['repeatability_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(claim['musical_quality_claimed'])}`",
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
        f"- dead-air / long-note failure count: `{evidence['dead_air_failure_count']}` / `{evidence['long_note_failure_count']}`",
        f"- avg onset / sustained coverage: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        f"- constrained note groups per bar: `{evidence['constrained_note_groups_per_bar']}`",
        f"- jazz duration tokens: `{_bool_token(evidence['jazz_duration_tokens'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate Stage B generic base scale checkpoint objective gate evidence"
    )
    parser.add_argument(
        "--repair_probe",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe/"
        "harness_stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe/"
        "stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_objective_gate_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_repeatability_target", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    report = build_consolidation_report(
        read_json(Path(args.repair_probe)),
        output_dir=output_dir,
    )
    summary = validate_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_repeatability_target=bool(args.require_repeatability_target),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir / "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation.json",
        report,
    )
    write_json(
        output_dir / "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
