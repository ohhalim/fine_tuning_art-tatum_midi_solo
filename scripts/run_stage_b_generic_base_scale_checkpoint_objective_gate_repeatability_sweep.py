"""Run objective-gate repeatability sweep for the generic-base scale checkpoint."""

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


class StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(ValueError):
    pass


SOURCE_BOUNDARY = "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation"
REPAIR_BOUNDARY = "stage_b_generic_base_scale_checkpoint_sustained_coverage_dead_air_repair_probe"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep"
NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_repeatability_consolidation"


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


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in str(raw).split(","):
        stripped = part.strip()
        if not stripped:
            continue
        seeds.append(int(stripped))
    if not seeds:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError("seed list required")
    return seeds


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


def merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = _int(target.get(str(key))) + _int(value)


def validate_consolidation(report: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    evidence = _dict(report.get("evidence"))
    if str(decision.get("current_boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "objective gate consolidation boundary required"
        )
    if str(decision.get("selected_target") or "") != "objective_gate_repeatability_sweep":
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "objective gate repeatability target required"
        )
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "consolidation must route to repeatability sweep"
        )
    if not bool(evidence.get("objective_gate_support", False)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "objective gate support required"
        )
    if not bool(evidence.get("single_seed_set_only", False)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "single seed set boundary required"
        )
    if bool(claim.get("repeatability_claimed", True)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "repeatability must not already be claimed"
        )
    if bool(claim.get("musical_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "musical quality must not be claimed"
        )
    return {
        "source_sample_count": _int(evidence.get("sample_count")),
        "source_valid_sample_count": _int(evidence.get("valid_sample_count")),
        "source_strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "source_grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "source_avg_sustained_coverage_ratio": _float(evidence.get("avg_sustained_coverage_ratio")),
        "constrained_note_groups_per_bar": _int(evidence.get("constrained_note_groups_per_bar")),
        "jazz_duration_tokens": bool(evidence.get("jazz_duration_tokens", False)),
        "coverage_aware_positions": bool(evidence.get("coverage_aware_positions", False)),
    }


def validate_repair_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    baseline = _dict(report.get("baseline_repair_summary"))
    input_config = _dict(report.get("input"))
    if str(readiness.get("boundary") or "") != REPAIR_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "sustained coverage dead-air repair boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "repair probe must route to objective gate consolidation"
        )
    if not bool(readiness.get("sustained_coverage_dead_air_target_qualified", False)):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "sustained coverage dead-air target must be qualified"
        )
    checkpoint_dir = Path(str(baseline.get("checkpoint_dir") or ""))
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "scale checkpoint required"
        )
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "constrained_note_groups_per_bar": _int(input_config.get("constrained_note_groups_per_bar")),
        "coverage_position_window": _int(input_config.get("coverage_position_window")),
        "max_simultaneous_notes": _int(input_config.get("max_simultaneous_notes")),
    }


def build_generation_command(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_dir: Path,
    probe_output_root: Path,
    probe_run_id: str,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_stage_b_generation_probe.py",
        "--output_root",
        str(probe_output_root),
        "--run_id",
        probe_run_id,
        "--checkpoint_dir",
        str(checkpoint_dir),
        "--skip_prepare",
        "--skip_train",
        "--issue_number",
        str(args.issue_number),
        "--max_sequence",
        str(args.max_sequence),
        "--num_samples",
        str(args.num_samples),
        "--seed",
        str(seed),
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.constrained_note_groups_per_bar),
        "--coverage_aware_positions",
        "--coverage_position_window",
        str(args.coverage_position_window),
        "--jazz_duration_tokens",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
    ]


def run_seed_probe(
    args: argparse.Namespace,
    *,
    seed: int,
    checkpoint_dir: Path,
    probe_output_root: Path,
) -> dict[str, Any]:
    probe_run_id = f"seed_{seed}"
    command_result = run_command(
        build_generation_command(
            args,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            probe_output_root=probe_output_root,
            probe_run_id=probe_run_id,
        )
    )
    report_path = probe_output_root / probe_run_id / "report.json"
    generation_report = read_json(report_path) if report_path.exists() else {}
    summary = _dict(generation_report.get("summary"))
    return {
        "seed": int(seed),
        "generation_report_path": str(report_path),
        "generation_command": command_result,
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "failure_reasons": _dict(summary.get("failure_reasons")),
        "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            summary.get("max_longest_sustained_empty_run_steps")
        ),
        "passed_strict_review_gate": bool(generation_report.get("passed_strict_review_gate", False)),
        "passed_grammar_gate": bool(generation_report.get("passed_grammar_gate", False)),
    }


def aggregate_seed_rows(seed_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total_sample_count = sum(_int(row.get("sample_count")) for row in seed_rows)
    total_valid = sum(_int(row.get("valid_sample_count")) for row in seed_rows)
    total_strict = sum(_int(row.get("strict_valid_sample_count")) for row in seed_rows)
    total_grammar = sum(_int(row.get("grammar_gate_sample_count")) for row in seed_rows)
    failure_reasons: dict[str, int] = {}
    diagnostic_failure_reasons: dict[str, int] = {}
    strict_failure_reasons: dict[str, int] = {}
    for row in seed_rows:
        merge_counts(failure_reasons, _dict(row.get("failure_reasons")))
        merge_counts(diagnostic_failure_reasons, _dict(row.get("diagnostic_failure_reasons")))
        merge_counts(strict_failure_reasons, _dict(row.get("strict_failure_reasons")))
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
        "sample_count": int(total_sample_count),
        "valid_sample_count": int(total_valid),
        "strict_valid_sample_count": int(total_strict),
        "grammar_gate_sample_count": int(total_grammar),
        "valid_sample_rate": float(total_valid / total_sample_count) if total_sample_count else 0.0,
        "strict_valid_sample_rate": float(total_strict / total_sample_count) if total_sample_count else 0.0,
        "grammar_gate_sample_rate": float(total_grammar / total_sample_count) if total_sample_count else 0.0,
        "failure_reasons": failure_reasons,
        "diagnostic_failure_reasons": diagnostic_failure_reasons,
        "strict_failure_reasons": strict_failure_reasons,
        "collapse_warning_sample_count": sum(
            _int(row.get("collapse_warning_sample_count")) for row in seed_rows
        ),
        "avg_onset_coverage_ratio": (
            float(weighted_onset / total_sample_count) if total_sample_count else 0.0
        ),
        "avg_sustained_coverage_ratio": (
            float(weighted_sustained / total_sample_count) if total_sample_count else 0.0
        ),
        "max_longest_sustained_empty_run_steps": max(
            [_int(row.get("max_longest_sustained_empty_run_steps")) for row in seed_rows] or [0]
        ),
        "all_seed_commands_succeeded": all(
            _int(_dict(row.get("generation_command")).get("returncode")) == 0 for row in seed_rows
        ),
        "all_seed_strict_review_gate_passed": all(
            bool(row.get("passed_strict_review_gate", False)) for row in seed_rows
        ),
    }


def build_sweep_report(
    *,
    run_dir: Path,
    consolidation_summary: dict[str, Any],
    repair_config: dict[str, Any],
    seed_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    aggregate = aggregate_seed_rows(seed_rows)
    target_qualified = (
        bool(aggregate["all_seed_commands_succeeded"])
        and bool(aggregate["all_seed_strict_review_gate_passed"])
        and _int(aggregate["sample_count"]) > 0
        and _int(aggregate["strict_valid_sample_count"]) == _int(aggregate["sample_count"])
        and _int(aggregate["grammar_gate_sample_count"]) == _int(aggregate["sample_count"])
        and not _dict(aggregate.get("diagnostic_failure_reasons"))
    )
    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "input_boundary": SOURCE_BOUNDARY,
        "repair_boundary": REPAIR_BOUNDARY,
        "consolidation_summary": consolidation_summary,
        "repair_config": repair_config,
        "input": {
            "issue_number": int(args.issue_number),
            "seeds": parse_seeds(args.seeds),
            "num_samples": int(args.num_samples),
            "max_sequence": int(args.max_sequence),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "generation_mode": "constrained",
            "constrained_note_groups_per_bar": int(args.constrained_note_groups_per_bar),
            "coverage_aware_positions": True,
            "coverage_position_window": int(args.coverage_position_window),
            "jazz_duration_tokens": True,
            "postprocess_overlap": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "seed_rows": seed_rows,
        "aggregate": aggregate,
        "comparison": {
            "source_sample_count": int(consolidation_summary["source_sample_count"]),
            "repeatability_sample_count": int(aggregate["sample_count"]),
            "strict_valid_sample_delta": _int(aggregate["strict_valid_sample_count"])
            - _int(consolidation_summary["source_strict_valid_sample_count"]),
            "sustained_coverage_delta": _float(aggregate["avg_sustained_coverage_ratio"])
            - _float(consolidation_summary["source_avg_sustained_coverage_ratio"]),
            "target_qualified": bool(target_qualified),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "objective_gate_repeatability_sweep_completed": True,
            "objective_gate_repeatability_target_qualified": bool(target_qualified),
            "repeatability_claimed": bool(target_qualified),
            "raw_generation_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "objective gates held across the configured seed sweep; consolidate repeatability "
                "before any listening or quality claim"
            ),
        },
        "not_proven": [
            "musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic base scale checkpoint repeatability consolidation",
    }


def validate_sweep_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_target_qualified: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    aggregate = _dict(report.get("aggregate"))
    comparison = _dict(report.get("comparison"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_target_qualified and not bool(
        readiness.get("objective_gate_repeatability_target_qualified", False)
    ):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "objective gate repeatability target should qualify"
        )
    if _int(aggregate.get("sample_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError("sample count required")
    if _int(aggregate.get("strict_valid_sample_count")) != _int(aggregate.get("sample_count")):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "all repeatability samples must pass strict gate"
        )
    if _dict(aggregate.get("diagnostic_failure_reasons")):
        raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
            "diagnostic failures must be absent"
        )
    if require_no_quality_claim:
        claimed = [
            bool(readiness.get("raw_generation_quality_claimed", True)),
            bool(readiness.get("human_audio_preference_claimed", True)),
            bool(readiness.get("broad_trained_model_quality_claimed", True)),
            bool(readiness.get("brad_style_adaptation_claimed", True)),
            bool(readiness.get("production_ready_improviser_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericBaseScaleCheckpointObjectiveGateRepeatabilitySweepError(
                "quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "objective_gate_repeatability_sweep_completed": bool(
            readiness.get("objective_gate_repeatability_sweep_completed", False)
        ),
        "objective_gate_repeatability_target_qualified": bool(
            readiness.get("objective_gate_repeatability_target_qualified", False)
        ),
        "repeatability_claimed": bool(readiness.get("repeatability_claimed", False)),
        "seed_count": _int(aggregate.get("seed_count")),
        "sample_count": _int(aggregate.get("sample_count")),
        "valid_sample_count": _int(aggregate.get("valid_sample_count")),
        "strict_valid_sample_count": _int(aggregate.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(aggregate.get("grammar_gate_sample_count")),
        "avg_sustained_coverage_ratio": _float(aggregate.get("avg_sustained_coverage_ratio")),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "raw_generation_quality_claimed": bool(readiness.get("raw_generation_quality_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    aggregate = report["aggregate"]
    comparison = report["comparison"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Objective Gate Repeatability Sweep",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- objective gate repeatability target qualified: `{_bool_token(readiness['objective_gate_repeatability_target_qualified'])}`",
        f"- repeatability claimed: `{_bool_token(readiness['repeatability_claimed'])}`",
        f"- raw generation quality claimed: `{_bool_token(readiness['raw_generation_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Aggregate",
        "",
        f"- seeds: `{aggregate['seeds']}`",
        f"- seed count: `{aggregate['seed_count']}`",
        f"- sample count: `{aggregate['sample_count']}`",
        f"- valid / strict / grammar gate sample count: `{aggregate['valid_sample_count']}` / `{aggregate['strict_valid_sample_count']}` / `{aggregate['grammar_gate_sample_count']}`",
        f"- avg onset / sustained coverage: `{aggregate['avg_onset_coverage_ratio']}` / `{aggregate['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{aggregate['max_longest_sustained_empty_run_steps']}`",
        "",
        "## Delta",
        "",
        f"- source sample count: `{comparison['source_sample_count']}`",
        f"- repeatability sample count: `{comparison['repeatability_sample_count']}`",
        f"- strict valid sample delta: `{comparison['strict_valid_sample_delta']}`",
        f"- sustained coverage delta: `{comparison['sustained_coverage_delta']}`",
        "",
        "## Failure Reasons",
        "",
    ]
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
        description="Run Stage B generic base scale checkpoint objective gate repeatability sweep"
    )
    parser.add_argument(
        "--consolidation",
        type=str,
        default="outputs/stage_b_generic_base_scale_checkpoint_objective_gate_consolidation/"
        "harness_stage_b_generic_base_scale_checkpoint_objective_gate_consolidation/"
        "stage_b_generic_base_scale_checkpoint_objective_gate_consolidation.json",
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
        default="outputs/stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=469)
    parser.add_argument("--seeds", type=str, default="44,52,60")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--constrained_note_groups_per_bar", type=int, default=8)
    parser.add_argument("--coverage_position_window", type=int, default=1)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_target_qualified", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    consolidation_summary = validate_consolidation(read_json(Path(args.consolidation)))
    repair_config = validate_repair_probe(read_json(Path(args.repair_probe)))
    checkpoint_dir = Path(str(repair_config["checkpoint_dir"]))
    probe_output_root = run_dir / "generation_probe"
    seed_rows = [
        run_seed_probe(
            args,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            probe_output_root=probe_output_root,
        )
        for seed in parse_seeds(args.seeds)
    ]
    report = build_sweep_report(
        run_dir=run_dir,
        consolidation_summary=consolidation_summary,
        repair_config=repair_config,
        seed_rows=seed_rows,
        args=args,
    )
    summary = validate_sweep_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_target_qualified=bool(args.require_target_qualified),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        run_dir / "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep.json",
        report,
    )
    write_json(
        run_dir / "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        run_dir / "stage_b_generic_base_scale_checkpoint_objective_gate_repeatability_sweep.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
