"""Run a repeatability probe for the generic tiny checkpoint repair path."""

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


class StageBGenericTinyCheckpointRepairRepeatabilityError(ValueError):
    pass


def build_generation_command(
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
    output_root: Path,
    run_id: str,
) -> list[str]:
    return [
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
        "--generation_mode",
        "constrained",
        "--constrained_note_groups_per_bar",
        str(args.repair_note_groups_per_bar),
        "--jazz_duration_tokens",
        "--postprocess_overlap",
        "--max_simultaneous_notes",
        str(args.max_simultaneous_notes),
        "--require_all_grammar_samples",
        "--require_note_groups",
        "--require_valid_sample",
        "--require_strict_valid_sample",
    ]


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
        "valid_sample_indices": summary.get("valid_sample_indices") or [],
        "strict_valid_sample_indices": summary.get("strict_valid_sample_indices") or [],
        "grammar_gate_sample_indices": summary.get("grammar_gate_sample_indices") or [],
        "passed_generation_gate": bool(report.get("passed_generation_gate", False)),
        "passed_grammar_gate": bool(report.get("passed_grammar_gate", False)),
        "passed_strict_review_gate": bool(report.get("passed_strict_review_gate", False)),
        "failure_reasons": _dict(summary.get("failure_reasons")),
        "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(summary.get("max_longest_sustained_empty_run_steps")),
        "avg_adjacent_repeated_pitch_ratio": _float(summary.get("avg_adjacent_repeated_pitch_ratio")),
        "max_longest_same_pitch_run": _int(summary.get("max_longest_same_pitch_run")),
        "avg_root_tone_ratio": _float(summary.get("avg_root_tone_ratio")),
        "avg_tension_ratio": _float(summary.get("avg_tension_ratio")),
    }


def build_repeatability_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    generation_result: dict[str, Any],
    generation_report_path: Path,
    generation_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = summarize_generation_report(generation_report)
    all_grammar = summary["grammar_gate_sample_count"] == summary["sample_count"]
    min_strict_met = summary["strict_valid_sample_count"] >= int(args.min_strict_valid_samples)
    min_valid_met = summary["valid_sample_count"] >= int(args.min_valid_samples)
    collapse_rate_met = summary["collapse_warning_sample_rate"] <= float(args.max_collapse_warning_sample_rate)
    repeatability_passed = bool(
        _int(generation_result.get("returncode")) == 0
        and summary["sample_count"] == int(args.num_samples)
        and all_grammar
        and min_valid_met
        and min_strict_met
        and collapse_rate_met
    )
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_repeatability_v1",
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
            "min_valid_samples": int(args.min_valid_samples),
            "min_strict_valid_samples": int(args.min_strict_valid_samples),
            "repair_generation_mode": "constrained",
            "repair_note_groups_per_bar": int(args.repair_note_groups_per_bar),
            "repair_jazz_duration_tokens": True,
            "postprocess_overlap": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
            "max_collapse_warning_sample_rate": float(args.max_collapse_warning_sample_rate),
        },
        "generation": {
            "label": "constrained_jazz_duration_repeatability",
            "report_path": str(generation_report_path),
            "command": generation_result,
            "summary": summary,
        },
        "repeatability": {
            "repeatability_passed": repeatability_passed,
            "all_grammar_samples": all_grammar,
            "min_valid_met": min_valid_met,
            "min_strict_met": min_strict_met,
            "collapse_rate_met": collapse_rate_met,
            "strict_valid_margin": int(summary["strict_valid_sample_count"] - int(args.min_strict_valid_samples)),
            "valid_sample_margin": int(summary["valid_sample_count"] - int(args.min_valid_samples)),
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
            "repair_repeatability_passed": repeatability_passed,
            "raw_generation_quality_claimed": False,
            "constrained_generation_quality_claimed": False,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_repair_repeatability_probe",
            "next_boundary": (
                "stage_b_generic_tiny_checkpoint_repair_review_package"
                if repeatability_passed
                else "stage_b_generic_tiny_checkpoint_repair_failure_diagnostics"
            ),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "repeatability is assessed on constrained repair outputs; raw trained-model and musical quality "
                "claims remain out of scope"
            ),
        },
        "not_proven": [
            "unconstrained_raw_generation_quality",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repair review package"
            if repeatability_passed
            else "Stage B generic tiny checkpoint repair failure diagnostics"
        ),
    }


def validate_repeatability_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_repeatability_passed: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    repeatability = _dict(report.get("repeatability"))
    generation = _dict(report.get("generation"))
    command = _dict(generation.get("command"))
    summary = _dict(generation.get("summary"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointRepairRepeatabilityError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if _int(command.get("returncode")) != 0:
        raise StageBGenericTinyCheckpointRepairRepeatabilityError("repeatability generation command must succeed")
    if require_repeatability_passed and not bool(readiness.get("repair_repeatability_passed", False)):
        raise StageBGenericTinyCheckpointRepairRepeatabilityError("repair repeatability should pass")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairRepeatabilityError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointRepairRepeatabilityError("Brad style adaptation must not be claimed")
    if _int(summary.get("sample_count")) <= 0:
        raise StageBGenericTinyCheckpointRepairRepeatabilityError("generated sample count must be positive")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "valid_sample_rate": _float(summary.get("valid_sample_rate")),
        "strict_valid_sample_rate": _float(summary.get("strict_valid_sample_rate")),
        "grammar_gate_sample_rate": _float(summary.get("grammar_gate_sample_rate")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "repeatability_passed": bool(repeatability.get("repeatability_passed", False)),
        "all_grammar_samples": bool(repeatability.get("all_grammar_samples", False)),
        "min_valid_met": bool(repeatability.get("min_valid_met", False)),
        "min_strict_met": bool(repeatability.get("min_strict_met", False)),
        "collapse_rate_met": bool(repeatability.get("collapse_rate_met", False)),
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
    summary = report["generation"]["summary"]
    repeatability = report["repeatability"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Repeatability",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- repair repeatability passed: `{_bool_token(readiness['repair_repeatability_passed'])}`",
        f"- raw generation quality claimed: `{_bool_token(readiness['raw_generation_quality_claimed'])}`",
        f"- constrained generation quality claimed: `{_bool_token(readiness['constrained_generation_quality_claimed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Repeatability",
        "",
        f"- sample count: `{summary['sample_count']}`",
        f"- valid sample count: `{summary['valid_sample_count']}`",
        f"- strict valid sample count: `{summary['strict_valid_sample_count']}`",
        f"- grammar gate sample count: `{summary['grammar_gate_sample_count']}`",
        f"- valid / strict / grammar rate: `{summary['valid_sample_rate']}/"
        f"{summary['strict_valid_sample_rate']}/{summary['grammar_gate_sample_rate']}`",
        f"- collapse warning sample rate: `{summary['collapse_warning_sample_rate']}`",
        f"- avg postprocess removal ratio: `{summary['avg_postprocess_removal_ratio']}`",
        f"- avg onset coverage ratio: `{summary['avg_onset_coverage_ratio']}`",
        f"- avg sustained coverage ratio: `{summary['avg_sustained_coverage_ratio']}`",
        f"- all grammar samples: `{_bool_token(repeatability['all_grammar_samples'])}`",
        f"- min strict met: `{_bool_token(repeatability['min_strict_met'])}`",
        "",
        "## Failure Reasons",
        "",
    ]
    if summary["diagnostic_failure_reasons"]:
        for reason, count in summary["diagnostic_failure_reasons"].items():
            lines.append(f"- `{reason}`: `{count}`")
    else:
        lines.append("- none")
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic tiny checkpoint repair repeatability")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/stage_b_generic_base_tiny_training_smoke/"
        "harness_stage_b_generic_base_tiny_training_smoke/checkpoints",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_repair_repeatability")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=397)
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_valid_samples", type=int, default=5)
    parser.add_argument("--min_strict_valid_samples", type=int, default=5)
    parser.add_argument("--max_collapse_warning_sample_rate", type=float, default=0.34)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--repair_note_groups_per_bar", type=int, default=4)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_repeatability_passed", action="store_true")
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
        raise StageBGenericTinyCheckpointRepairRepeatabilityError("checkpoint_epoch1.pt required")

    probe_output_root = run_dir / "repair_repeatability_probe"
    generation_result = run_command(
        build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            output_root=probe_output_root,
            run_id="repair_repeatability",
        )
    )
    generation_report_path = probe_output_root / "repair_repeatability" / "report.json"
    generation_report = read_json(generation_report_path) if generation_report_path.exists() else {}
    report = build_repeatability_report(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        generation_result=generation_result,
        generation_report_path=generation_report_path,
        generation_report=generation_report,
        args=args,
    )
    summary = validate_repeatability_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_repeatability_passed=bool(args.require_repeatability_passed),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_repeatability.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_repair_repeatability_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_repair_repeatability.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
