"""Consolidate generic-base scale checkpoint repeatability evidence."""

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


class StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_repeatability_consolidation"
NEXT_BOUNDARY = "stage_b_model_core_evidence_readme_refresh"


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


def validate_repeatability_sweep(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    comparison = _dict(report.get("comparison"))
    input_config = _dict(report.get("input"))

    if str(readiness.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "objective gate repeatability sweep boundary required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "repeatability sweep must route to consolidation"
        )
    if not bool(readiness.get("objective_gate_repeatability_sweep_completed", False)):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "repeatability sweep completion required"
        )
    if not bool(readiness.get("objective_gate_repeatability_target_qualified", False)):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "qualified repeatability target required"
        )
    if not bool(readiness.get("repeatability_claimed", False)):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "objective repeatability claim required"
        )

    sample_count = _int(aggregate.get("sample_count"))
    if sample_count <= 0:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError("sample count required")
    if _int(aggregate.get("seed_count")) < 3:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError("at least 3 seeds required")
    if _int(aggregate.get("valid_sample_count")) != sample_count:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "all samples must pass valid gate"
        )
    if _int(aggregate.get("strict_valid_sample_count")) != sample_count:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "all samples must pass strict gate"
        )
    if _int(aggregate.get("grammar_gate_sample_count")) != sample_count:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "all samples must pass grammar gate"
        )
    if _dict(aggregate.get("failure_reasons")):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "failure reasons must be absent"
        )
    if _dict(aggregate.get("diagnostic_failure_reasons")):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "diagnostic failure reasons must be absent"
        )
    if _dict(aggregate.get("strict_failure_reasons")):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "strict failure reasons must be absent"
        )

    blocked_claims = [
        "raw_generation_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, True))]
    if claimed:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            f"unexpected quality claim: {claimed}"
        )

    return {
        "seeds": list(aggregate.get("seeds") or []),
        "seed_count": _int(aggregate.get("seed_count")),
        "sample_count": sample_count,
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "valid_sample_rate": _float(aggregate.get("valid_sample_rate")),
        "strict_valid_sample_rate": _float(aggregate.get("strict_valid_sample_rate")),
        "grammar_gate_sample_rate": _float(aggregate.get("grammar_gate_sample_rate")),
        "avg_onset_coverage_ratio": _float(aggregate.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(aggregate.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            aggregate.get("max_longest_sustained_empty_run_steps")
        ),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "sustained_coverage_delta": _float(comparison.get("sustained_coverage_delta")),
        "generation_mode": str(input_config.get("generation_mode") or ""),
        "constrained_note_groups_per_bar": _int(input_config.get("constrained_note_groups_per_bar")),
        "coverage_aware_positions": bool(input_config.get("coverage_aware_positions", False)),
        "coverage_position_window": _int(input_config.get("coverage_position_window")),
        "jazz_duration_tokens": bool(input_config.get("jazz_duration_tokens", False)),
        "postprocess_overlap": bool(input_config.get("postprocess_overlap", False)),
        "max_simultaneous_notes": _int(input_config.get("max_simultaneous_notes")),
    }


def build_repeatability_consolidation_report(
    repeatability_sweep: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    evidence = validate_repeatability_sweep(repeatability_sweep)
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_repeatability_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_schema": str(repeatability_sweep.get("schema_version") or ""),
        "input_boundary": SOURCE_BOUNDARY,
        "evidence": evidence,
        "claim_boundary": {
            "boundary": BOUNDARY,
            "objective_midi_gate_repeatability_claimed": True,
            "configured_seed_sweep_repeatability_claimed": True,
            "raw_generation_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "decision": "select_model_core_evidence_readme_refresh",
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "configured seed sweep passes objective MIDI gates; README evidence can be refreshed "
                "without claiming listening quality"
            ),
        },
        "proven": [
            "configured_seed_sweep_valid_gate_repeatability",
            "configured_seed_sweep_strict_gate_repeatability",
            "configured_seed_sweep_grammar_gate_repeatability",
        ],
        "not_proven": [
            "musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B model-core evidence README refresh",
    }


def validate_repeatability_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    claim = _dict(report.get("claim_boundary"))
    decision = _dict(report.get("decision"))
    evidence = _dict(report.get("evidence"))
    boundary = str(claim.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if not bool(claim.get("objective_midi_gate_repeatability_claimed", False)):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "objective MIDI gate repeatability claim required"
        )
    if _int(evidence.get("sample_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError("sample count required")
    if _int(evidence.get("strict_valid_sample_count")) != _int(evidence.get("sample_count")):
        raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
            "strict sample count must equal sample count"
        )
    if require_no_quality_claim:
        blocked = [
            "raw_generation_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(claim.get(name, True))]
        if claimed:
            raise StageBGenericBaseScaleCheckpointRepeatabilityConsolidationError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "objective_midi_gate_repeatability_claimed": bool(
            claim.get("objective_midi_gate_repeatability_claimed", False)
        ),
        "configured_seed_sweep_repeatability_claimed": bool(
            claim.get("configured_seed_sweep_repeatability_claimed", False)
        ),
        "seed_count": _int(evidence.get("seed_count")),
        "sample_count": _int(evidence.get("sample_count")),
        "valid_sample_count": _int(evidence.get("valid_sample_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "avg_onset_coverage_ratio": _float(evidence.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            evidence.get("max_longest_sustained_empty_run_steps")
        ),
        "raw_generation_quality_claimed": bool(claim.get("raw_generation_quality_claimed", True)),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            claim.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(claim.get("brad_style_adaptation_claimed", True)),
        "production_ready_improviser_claimed": bool(
            claim.get("production_ready_improviser_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    evidence = report["evidence"]
    claim = report["claim_boundary"]
    decision = report["decision"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Repeatability Consolidation",
        "",
        "## Summary",
        "",
        f"- boundary: `{claim['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective MIDI gate repeatability claimed: `{_bool_token(claim['objective_midi_gate_repeatability_claimed'])}`",
        f"- configured seed sweep repeatability claimed: `{_bool_token(claim['configured_seed_sweep_repeatability_claimed'])}`",
        f"- raw generation quality claimed: `{_bool_token(claim['raw_generation_quality_claimed'])}`",
        f"- human/audio preference claimed: `{_bool_token(claim['human_audio_preference_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(claim['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(claim['brad_style_adaptation_claimed'])}`",
        f"- production-ready improviser claimed: `{_bool_token(claim['production_ready_improviser_claimed'])}`",
        "",
        "## Evidence",
        "",
        f"- seeds: `{evidence['seeds']}`",
        f"- seed count: `{evidence['seed_count']}`",
        f"- sample count: `{evidence['sample_count']}`",
        f"- valid / strict / grammar gate sample count: `{evidence['valid_sample_count']}` / `{evidence['strict_valid_sample_count']}` / `{evidence['grammar_gate_sample_count']}`",
        f"- valid / strict / grammar gate sample rate: `{evidence['valid_sample_rate']}` / `{evidence['strict_valid_sample_rate']}` / `{evidence['grammar_gate_sample_rate']}`",
        f"- avg onset / sustained coverage: `{evidence['avg_onset_coverage_ratio']}` / `{evidence['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{evidence['max_longest_sustained_empty_run_steps']}`",
        f"- strict valid sample delta: `{evidence['strict_valid_sample_delta']}`",
        f"- sustained coverage delta: `{evidence['sustained_coverage_delta']}`",
        "",
        "## Generation Condition",
        "",
        f"- generation mode: `{evidence['generation_mode']}`",
        f"- constrained note groups per bar: `{evidence['constrained_note_groups_per_bar']}`",
        f"- coverage-aware positions: `{_bool_token(evidence['coverage_aware_positions'])}`",
        f"- coverage position window: `{evidence['coverage_position_window']}`",
        f"- jazz duration tokens: `{_bool_token(evidence['jazz_duration_tokens'])}`",
        f"- postprocess overlap: `{_bool_token(evidence['postprocess_overlap'])}`",
        f"- max simultaneous notes: `{evidence['max_simultaneous_notes']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Consolidate generic-base scale checkpoint repeatability evidence"
    )
    parser.add_argument(
        "--repeatability_sweep",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep/"
        "harness_stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep/"
        "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_repeatability_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_repeatability_consolidation_report(
        read_json(Path(args.repeatability_sweep)),
        output_dir=output_dir,
    )
    summary = validate_repeatability_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir / "stage_b_generic_base_scale_checkpoint_repeatability_consolidation.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_generic_base_scale_checkpoint_repeatability_consolidation_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_generic_base_scale_checkpoint_repeatability_consolidation.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
