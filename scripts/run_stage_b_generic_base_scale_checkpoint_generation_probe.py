"""Probe generation/decode behavior from the generic-base scale checkpoint."""

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
    build_generation_command,
    run_command,
)


class StageBGenericBaseScaleCheckpointGenerationProbeError(ValueError):
    pass


SCALE_TRAINING_BOUNDARY = "stage_b_generic_base_training_scale_smoke"
BOUNDARY = "stage_b_generic_base_scale_checkpoint_generation_probe"
PASS_NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_repeatability_probe"
FAIL_NEXT_BOUNDARY = "stage_b_generic_base_scale_checkpoint_grammar_representation_decision"


def validate_training_scale_smoke(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    inputs = _dict(report.get("input"))
    token = _dict(report.get("token_stats"))
    training = _dict(report.get("training"))
    artifacts = _dict(report.get("artifacts"))
    source = _dict(report.get("source_window_summary"))
    boundary = str(readiness.get("boundary") or "")
    next_boundary = str(decision.get("next_boundary") or "")
    if boundary != SCALE_TRAINING_BOUNDARY:
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("scale training smoke boundary required")
    if next_boundary != BOUNDARY:
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("scale training smoke must route to generation probe")
    if not bool(readiness.get("training_scale_smoke_passed", False)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("scale training smoke must pass")
    if bool(readiness.get("full_generic_training_executed", True)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("full generic training must not be claimed")
    if bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("broad trained-model quality must not be claimed")
    if bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("Brad style adaptation must not be claimed")
    if not bool(token.get("fits_vocab", False)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("scale training token ids must fit vocab")
    if _int(training.get("returncode")) != 0:
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("scale training command must succeed")
    if training.get("best_validation_loss") is None:
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("best validation loss required")
    checkpoint_dir = Path(str(report.get("checkpoint_dir") or ""))
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("checkpoint_epoch1.pt required")
    return {
        "checkpoint_dir": str(checkpoint_dir),
        "source_tokenized_train_files": _int(source.get("source_tokenized_train_files")),
        "source_tokenized_val_files": _int(source.get("source_tokenized_val_files")),
        "selected_train_records": _int(inputs.get("selected_train_records")),
        "selected_val_records": _int(inputs.get("selected_val_records")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "best_validation_loss": training.get("best_validation_loss"),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
    }


def build_probe_report(
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    training_scale_summary: dict[str, Any],
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
    next_boundary = PASS_NEXT_BOUNDARY if passed_strict_review_gate else FAIL_NEXT_BOUNDARY

    return {
        "schema_version": "stage_b_generic_base_scale_checkpoint_generation_probe_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_file_exists": checkpoint_file.exists(),
        "lora_file_exists": lora_file.exists(),
        "training_scale_summary": training_scale_summary,
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
            "boundary": BOUNDARY,
            "scale_checkpoint_loaded": generation_probe_completed,
            "generation_path_executable": generation_probe_completed,
            "midi_outputs_written": sample_count > 0,
            "raw_generation_quality_ready": passed_strict_review_gate,
            "full_generic_training_executed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "human_audio_preference_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "scale checkpoint generation/decode path is measured separately from "
                "full generic training quality and Brad style adaptation"
            ),
        },
        "not_proven": [
            "full_generic_training_run",
            "generic_base_generation_quality",
            "generic_base_multi_seed_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B generic base scale checkpoint repeatability probe"
            if passed_strict_review_gate
            else "Stage B generic base scale checkpoint grammar representation decision"
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
        raise StageBGenericBaseScaleCheckpointGenerationProbeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if require_probe_completed and not bool(readiness.get("generation_path_executable", False)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("generation probe should complete")
    if _int(generation.get("returncode")) != 0:
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("generation command must succeed")
    if _int(summary.get("sample_count")) <= 0:
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("generated sample count must be positive")
    if bool(readiness.get("full_generic_training_executed", True)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("full generic training must not be claimed")
    if require_no_broad_quality_claim and bool(readiness.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError(
            "broad trained-model quality must not be claimed"
        )
    if require_no_brad_style_claim and bool(readiness.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("Brad style adaptation must not be claimed")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "scale_checkpoint_loaded": bool(readiness.get("scale_checkpoint_loaded", False)),
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
        "full_generic_training_executed": bool(readiness.get("full_generic_training_executed", True)),
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
    training = report["training_scale_summary"]
    generation = report["generation_summary"]
    command = report["generation_command"]
    lines = [
        "# Stage B Generic Base Scale Checkpoint Generation Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{readiness['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- generation path executable: `{_bool_token(readiness['generation_path_executable'])}`",
        f"- raw generation quality ready: `{_bool_token(readiness['raw_generation_quality_ready'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Training Source",
        "",
        (
            "- source tokenized train / val records: "
            f"`{training['source_tokenized_train_files']}` / `{training['source_tokenized_val_files']}`"
        ),
        (
            "- selected train / val records: "
            f"`{training['selected_train_records']}` / `{training['selected_val_records']}`"
        ),
        f"- best validation loss: `{training['best_validation_loss']}`",
        f"- checkpoint count: `{training['checkpoint_count']}`",
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
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Stage B generic base scale checkpoint generation probe")
    parser.add_argument(
        "--training_scale_smoke",
        type=str,
        default="outputs/stage_b_generic_base_training_scale_smoke/"
        "harness_stage_b_generic_base_training_scale_smoke/"
        "stage_b_generic_base_training_scale_smoke.json",
    )
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_generic_base_scale_checkpoint_generation_probe")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=453)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=43)
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

    training_scale_report = read_json(Path(args.training_scale_smoke))
    training_scale_summary = validate_training_scale_smoke(training_scale_report)
    checkpoint_dir = Path(args.checkpoint_dir or training_scale_summary["checkpoint_dir"])
    if not (checkpoint_dir / "checkpoint_epoch1.pt").exists():
        raise StageBGenericBaseScaleCheckpointGenerationProbeError("checkpoint_epoch1.pt required")

    probe_output_root = run_dir / "generation_probe"
    probe_run_id = "scale_checkpoint"
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
        training_scale_summary=training_scale_summary,
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
    write_json(run_dir / "stage_b_generic_base_scale_checkpoint_generation_probe.json", report)
    write_json(run_dir / "stage_b_generic_base_scale_checkpoint_generation_probe_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(run_dir / "stage_b_generic_base_scale_checkpoint_generation_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
