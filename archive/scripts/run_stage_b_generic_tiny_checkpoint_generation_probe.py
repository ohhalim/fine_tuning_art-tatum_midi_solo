"""Probe generation/decode behavior from the generic tiny checkpoint."""

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


class StageBGenericTinyCheckpointGenerationProbeError(ValueError):
    pass


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


def build_generation_command(
    args: argparse.Namespace,
    *,
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


def build_probe_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    generation_report_path: Path,
    generation_result: dict[str, Any],
    generation_report: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    summary = _dict(generation_report.get("summary"))
    sample_count = _int(summary.get("sample_count"))
    valid_sample_count = _int(summary.get("valid_sample_count"))
    strict_valid_sample_count = _int(summary.get("strict_valid_sample_count"))
    grammar_gate_sample_count = _int(summary.get("grammar_gate_sample_count"))
    checkpoint_file = checkpoint_dir / "checkpoint_epoch1.pt"
    lora_file = checkpoint_dir / "lora_weights.pt"
    generation_command_succeeded = _int(generation_result.get("returncode")) == 0
    generation_report_loaded = bool(generation_report)
    generation_probe_completed = bool(generation_command_succeeded and generation_report_loaded and sample_count > 0)
    passed_generation_gate = bool(generation_report.get("passed_generation_gate", False))
    passed_grammar_gate = bool(generation_report.get("passed_grammar_gate", False))
    passed_strict_review_gate = bool(generation_report.get("passed_strict_review_gate", False))
    next_boundary = (
        "stage_b_generic_tiny_checkpoint_repeatability_probe"
        if passed_strict_review_gate
        else "stage_b_generic_tiny_checkpoint_grammar_repair"
    )

    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_generation_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file_exists": checkpoint_file.exists(),
        "lora_file_exists": lora_file.exists(),
        "generation_report_path": str(generation_report_path),
        "input": {
            "issue_number": int(args.issue_number),
            "num_samples": int(args.num_samples),
            "seed": int(args.seed),
            "max_sequence": int(args.max_sequence),
            "temperature": float(args.temperature),
            "top_k": int(args.top_k),
            "postprocess_overlap": True,
            "max_simultaneous_notes": int(args.max_simultaneous_notes),
        },
        "generation_command": generation_result,
        "generation_summary": {
            "sample_count": sample_count,
            "valid_sample_count": valid_sample_count,
            "strict_valid_sample_count": strict_valid_sample_count,
            "grammar_gate_sample_count": grammar_gate_sample_count,
            "valid_sample_rate": _float(summary.get("valid_sample_rate")),
            "strict_valid_sample_rate": _float(summary.get("strict_valid_sample_rate")),
            "grammar_gate_sample_rate": _float(summary.get("grammar_gate_sample_rate")),
            "passed_generation_gate": passed_generation_gate,
            "passed_grammar_gate": passed_grammar_gate,
            "passed_strict_review_gate": passed_strict_review_gate,
            "failure_reasons": _dict(summary.get("failure_reasons")),
            "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
            "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
            "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
            "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
            "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
            "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
            "max_longest_sustained_empty_run_steps": _int(
                summary.get("max_longest_sustained_empty_run_steps")
            ),
            "avg_adjacent_repeated_pitch_ratio": _float(summary.get("avg_adjacent_repeated_pitch_ratio")),
            "avg_tension_ratio": _float(summary.get("avg_tension_ratio")),
        },
        "readiness": {
            "boundary": "stage_b_generic_tiny_checkpoint_generation_probe",
            "generic_tiny_checkpoint_loaded": generation_probe_completed,
            "generation_path_executable": generation_probe_completed,
            "midi_outputs_written": sample_count > 0,
            "raw_generation_quality_ready": passed_strict_review_gate,
            "broad_training_execution_ready": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": "stage_b_generic_tiny_checkpoint_generation_probe",
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "tiny checkpoint generation/decode path is measured separately from "
                "broad model quality and Brad style adaptation"
            ),
        },
        "not_proven": [
            "generic_base_generation_quality",
            "generic_base_multi_seed_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic tiny checkpoint repeatability probe"
            if passed_strict_review_gate
            else "Stage B generic tiny checkpoint grammar repair"
        ),
    }


def validate_probe_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    require_probe_completed: bool,
    require_no_broad_quality_claim: bool,
    require_no_brad_style_claim: bool,
) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("generation_command"))
    summary = _dict(report.get("generation_summary"))
    boundary = str(readiness.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBGenericTinyCheckpointGenerationProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_probe_completed and not bool(readiness.get("generation_path_executable", False)):
        raise StageBGenericTinyCheckpointGenerationProbeError("generation probe should complete")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointGenerationProbeError("broad trained-model quality must not be claimed")
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointGenerationProbeError("Brad style adaptation must not be claimed")
    if _int(generation.get("returncode")) != 0:
        raise StageBGenericTinyCheckpointGenerationProbeError("generation command must succeed")
    if _int(summary.get("sample_count")) <= 0:
        raise StageBGenericTinyCheckpointGenerationProbeError("generated sample count must be positive")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "generation_path_executable": bool(readiness.get("generation_path_executable", False)),
        "midi_outputs_written": bool(readiness.get("midi_outputs_written", False)),
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "passed_generation_gate": bool(summary.get("passed_generation_gate", False)),
        "passed_grammar_gate": bool(summary.get("passed_grammar_gate", False)),
        "passed_strict_review_gate": bool(summary.get("passed_strict_review_gate", False)),
        "raw_generation_quality_ready": bool(readiness.get("raw_generation_quality_ready", True)),
        "broad_training_execution_ready": bool(readiness.get("broad_training_execution_ready", True)),
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
    generation = report["generation_summary"]
    command = report["generation_command"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Generation Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generation path executable: `{_bool_token(readiness['generation_path_executable'])}`",
        f"- raw generation quality ready: `{_bool_token(readiness['raw_generation_quality_ready'])}`",
        f"- broad training execution ready: `{_bool_token(readiness['broad_training_execution_ready'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Generation",
        "",
        f"- command returncode: `{command['returncode']}`",
        f"- sample count: `{generation['sample_count']}`",
        f"- valid sample count: `{generation['valid_sample_count']}`",
        f"- strict valid sample count: `{generation['strict_valid_sample_count']}`",
        f"- grammar gate sample count: `{generation['grammar_gate_sample_count']}`",
        f"- collapse warning sample rate: `{generation['collapse_warning_sample_rate']}`",
        f"- avg onset coverage ratio: `{generation['avg_onset_coverage_ratio']}`",
        f"- avg sustained coverage ratio: `{generation['avg_sustained_coverage_ratio']}`",
        f"- max longest sustained empty run steps: `{generation['max_longest_sustained_empty_run_steps']}`",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in generation["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(
        [
            "",
            "## Not Proven",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic tiny checkpoint generation probe")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="outputs/stage_b_generic_base_tiny_training_smoke/"
        "harness_stage_b_generic_base_tiny_training_smoke/checkpoints",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_tiny_checkpoint_generation_probe")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=393)
    parser.add_argument("--num_samples", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_sequence", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=4)
    parser.add_argument("--min_valid_samples", type=int, default=1)
    parser.add_argument("--min_strict_valid_samples", type=int, default=1)
    parser.add_argument("--max_simultaneous_notes", type=int, default=2)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_probe_completed", action="store_true")
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
        raise StageBGenericTinyCheckpointGenerationProbeError("checkpoint_epoch1.pt required")

    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "tiny_checkpoint"
    generation_result = run_command(
        build_generation_command(
            args,
            checkpoint_dir=checkpoint_dir,
            probe_output_root=probe_output_root,
            probe_run_id=probe_run_id,
        )
    )
    generation_report_path = probe_output_root / probe_run_id / "report.json"
    generation_report = read_json(generation_report_path) if generation_report_path.exists() else {}
    report = build_probe_report(
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        generation_report_path=generation_report_path,
        generation_result=generation_result,
        generation_report=generation_report,
        args=args,
    )
    summary = validate_probe_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        require_probe_completed=bool(args.require_probe_completed),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
        require_no_brad_style_claim=bool(args.require_no_brad_style_claim),
    )
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_generation_probe.json", report)
    write_json(run_dir / "stage_b_generic_tiny_checkpoint_generation_probe_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_tiny_checkpoint_generation_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
