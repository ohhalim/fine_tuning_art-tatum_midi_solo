"""Compare raw and grammar-repaired generation from the generic tiny checkpoint."""

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
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
    _float,
    _int,
    run_command,
)


class StageBGenericTinyCheckpointGrammarRepairError(ValueError):
    pass


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
    mode: str,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_stage_b_generation_probe.py",
        "--output_root",
        str(output_root),
        "--run_id",
        run_id,
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
        str(args.seed),
        "--temperature",
        str(args.temperature),
        "--top_k",
        str(args.top_k),
        "--min_valid_samples",
        str(args.min_valid_samples),
        "--min_strict_valid_samples",
        str(args.min_strict_valid_samples),
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
    ]
    if mode == "repair":
        command.extend(
            [
                "--generation_mode",
                "constrained",
                "--constrained_note_groups_per_bar",
                str(args.repair_note_groups_per_bar),
                "--jazz_duration_tokens",
                "--require_note_groups",
                "--require_valid_sample",
                "--require_strict_valid_sample",
            ]
        )
    return command


def summarize_generation_report(report: dict[str, Any]) -> dict[str, Any]:
    summary = _dict(report.get("summary"))
    return {
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "valid_sample_rate": _float(summary.get("valid_sample_rate")),
        "strict_valid_sample_rate": _float(summary.get("strict_valid_sample_rate")),
        "grammar_gate_sample_rate": _float(summary.get("grammar_gate_sample_rate")),
        "passed_generation_gate": bool(report.get("passed_generation_gate", False)),
        "passed_grammar_gate": bool(report.get("passed_grammar_gate", False)),
        "passed_strict_review_gate": bool(report.get("passed_strict_review_gate", False)),
        "failure_reasons": _dict(summary.get("failure_reasons")),
        "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(summary.get("max_longest_sustained_empty_run_steps")),
        "avg_adjacent_repeated_pitch_ratio": _float(summary.get("avg_adjacent_repeated_pitch_ratio")),
        "avg_root_tone_ratio": _float(summary.get("avg_root_tone_ratio")),
        "avg_tension_ratio": _float(summary.get("avg_tension_ratio")),
    }


def build_repair_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    baseline_result: dict[str, Any],
    baseline_report_path: Path,
    baseline_report: dict[str, Any],
    repair_result: dict[str, Any],
    repair_report_path: Path,
    repair_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    baseline = summarize_generation_report(baseline_report)
    repair = summarize_generation_report(repair_report)
    grammar_gate_recovered = repair["grammar_gate_sample_count"] > baseline["grammar_gate_sample_count"]
    review_gate_recovered = repair["valid_sample_count"] > baseline["valid_sample_count"]
    strict_gate_recovered = repair["strict_valid_sample_count"] > baseline["strict_valid_sample_count"]
    repair_passed = bool(
        _int(repair_result.get("returncode")) == 0
        and repair["passed_strict_review_gate"]
        and repair["grammar_gate_sample_count"] == repair["sample_count"]
        and grammar_gate_recovered
        and review_gate_recovered
        and strict_gate_recovered
    )
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_grammar_repair_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file_exists": (checkpoint_dir / "checkpoint_epoch1.pt").exists(),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "max_sequence": int(args.max_sequence),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "repair_generation_mode": "constrained",
            "repair_note_groups_per_bar": int(args.repair_note_groups_per_bar),
            "repair_jazz_duration_tokens": True,
            "postprocess_overlap": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "baseline": {
            "label": "unconstrained_postprocess",
            "report_path": str(baseline_report_path),
            "command": baseline_result,
            "summary": baseline,
        },
        "repair": {
            "label": "constrained_jazz_duration_postprocess",
            "report_path": str(repair_report_path),
            "command": repair_result,
            "summary": repair,
        },
        "comparison": {
            "grammar_gate_delta": int(repair["grammar_gate_sample_count"] - baseline["grammar_gate_sample_count"]),
            "valid_sample_delta": int(repair["valid_sample_count"] - baseline["valid_sample_count"]),
            "strict_valid_sample_delta": int(
                repair["strict_valid_sample_count"] - baseline["strict_valid_sample_count"]
            ),
            "grammar_gate_recovered": grammar_gate_recovered,
            "review_gate_recovered": review_gate_recovered,
            "strict_gate_recovered": strict_gate_recovered,
            "repair_passed": repair_passed,
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_grammar_repair",
            "grammar_repair_passed": repair_passed,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_grammar_repair",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_repeatability_probe"
                if repair_passed
                else "stage_b_generic_tiny_checkpoint_training_grammar_objective_repair"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "constrained Stage B grammar and jazz-duration token selection are evaluated "
                "as a repair boundary, not as raw model quality evidence"
            ),
        },
        "not_proven": [
            "unconstrained_raw_generation_quality",
            "generic_base_multi_seed_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair repeatability probe"
            if repair_passed
            else "Stage B generic tiny checkpoint training grammar objective repair"
        ),
    }


def validate_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_repair_passed: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    comparison = _dict(report.get("comparison"))
    baseline = _dict(_dict(report.get("baseline")).get("summary"))
    repair = _dict(_dict(report.get("repair")).get("summary"))
    baseline_command = _dict(_dict(report.get("baseline")).get("command"))
    repair_command = _dict(_dict(report.get("repair")).get("command"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointGrammarRepairError(f"expected boundary {expected_boundary}, got {boundary}")
    if _int(baseline_command.get("returncode")) != 0:
        raise StageBGenericTinyCheckpointGrammarRepairError("baseline generation command must succeed")
    if _int(repair_command.get("returncode")) != 0:
        raise StageBGenericTinyCheckpointGrammarRepairError("repair generation command must succeed")
    if require_repair_passed and not bool(readiness.get("grammar_repair_passed", False)):
        raise StageBGenericTinyCheckpointGrammarRepairError("grammar repair should pass")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointGrammarRepairError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointGrammarRepairError("Brad style adaptation must not be claimed")
    if _int(baseline.get("sample_count")) <= 0 or _int(repair.get("sample_count")) <= 0:
        raise StageBGenericTinyCheckpointGrammarRepairError("both probe sample counts must be positive")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "baseline_sample_count": _int(baseline.get("sample_count")),
        "baseline_valid_sample_count": _int(baseline.get("valid_sample_count")),
        "baseline_strict_valid_sample_count": _int(baseline.get("strict_valid_sample_count")),
        "baseline_grammar_gate_sample_count": _int(baseline.get("grammar_gate_sample_count")),
        "repair_sample_count": _int(repair.get("sample_count")),
        "repair_valid_sample_count": _int(repair.get("valid_sample_count")),
        "repair_strict_valid_sample_count": _int(repair.get("strict_valid_sample_count")),
        "repair_grammar_gate_sample_count": _int(repair.get("grammar_gate_sample_count")),
        "grammar_gate_delta": _int(comparison.get("grammar_gate_delta")),
        "valid_sample_delta": _int(comparison.get("valid_sample_delta")),
        "strict_valid_sample_delta": _int(comparison.get("strict_valid_sample_delta")),
        "grammar_repair_passed": bool(readiness.get("grammar_repair_passed", False)),
        "raw_generation_quality_claimed": bool(readiness.get("raw_generation_quality_claimed", True)),
        "constrained_generation_quality_claimed": bool(
            readiness.get("constrained_generation_quality_claimed", True)
        ),
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
    comparison = report["comparison"]
    baseline = report["baseline"]["summary"]
    repair = report["repair"]["summary"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Grammar Repair",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- grammar repair passed: `{_bool_token(readiness['grammar_repair_passed'])}`",
        f"- raw generation quality claimed: `{_bool_token(readiness['raw_generation_quality_claimed'])}`",
        f"- constrained generation quality claimed: `{_bool_token(readiness['constrained_generation_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Comparison",
        "",
        f"- baseline valid/strict/grammar: `{baseline['valid_sample_count']}/"
        f"{baseline['strict_valid_sample_count']}/{baseline['grammar_gate_sample_count']}`",
        f"- repair valid/strict/grammar: `{repair['valid_sample_count']}/"
        f"{repair['strict_valid_sample_count']}/{repair['grammar_gate_sample_count']}`",
        f"- grammar gate delta: `{comparison['grammar_gate_delta']}`",
        f"- valid sample delta: `{comparison['valid_sample_delta']}`",
        f"- strict valid sample delta: `{comparison['strict_valid_sample_delta']}`",
        f"- repair collapse warning sample rate: `{repair['collapse_warning_sample_rate']}`",
        f"- repair avg postprocess removal ratio: `{repair['avg_postprocess_removal_ratio']}`",
        f"- repair avg onset coverage ratio: `{repair['avg_onset_coverage_ratio']}`",
        f"- repair avg sustained coverage ratio: `{repair['avg_sustained_coverage_ratio']}`",
        "",
        "## Baseline Failure Reasons",
        "",
    ]
    for reason, count in baseline["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Repair Failure Reasons", ""])
    if repair["diagnostic_failure_reasons"]:
        for reason, count in repair["diagnostic_failure_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic tiny checkpoint grammar repair")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/stage_b_generic_base_tiny_training_smoke/"
        "harness_stage_b_generic_base_tiny_training_smoke/checkpoints",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_grammar_repair")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=395)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--repair_note_groups_per_bar", type=int, default=4)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_repair_passed", action="store_true")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    parser.add_argument("--require_no_brad_style_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir)
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericTinyCheckpointGrammarRepairError("checkpoint_epoch1.pt required")

    baseline_output_root = run_dir / "baseline_generation_probe"
    repair_output_root = run_dir / "repair_generation_probe"
    baseline_result = run_command(
        build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            output_root=baseline_output_root,
            run_id="baseline_unconstrained",
            mode="baseline",
        )
    )
    baseline_report_path = baseline_output_root / "baseline_unconstrained" / "report.json"
    baseline_report = read_json(baseline_report_path) if baseline_report_path.exists() else {}

    repair_result = run_command(
        build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            output_root=repair_output_root,
            run_id="repair_constrained_jazz_duration",
            mode="repair",
        )
    )
    repair_report_path = repair_output_root / "repair_constrained_jazz_duration" / "report.json"
    repair_report = read_json(repair_report_path) if repair_report_path.exists() else {}

    report = build_repair_report(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        baseline_result=baseline_result,
        baseline_report_path=baseline_report_path,
        baseline_report=baseline_report,
        repair_result=repair_result,
        repair_report_path=repair_report_path,
        repair_report=repair_report,
        args=args,
    )
    summary = validate_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_repair_passed=bool(args.require_repair_passed),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_grammar_repair.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_grammar_repair_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_grammar_repair.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
