"""Summarize generation from the selected-scale controlled checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402
from scripts.run_stage_b_generic_base_scale_checkpoint_generation_probe import (  # noqa: E402
    BOUNDARY as GENERIC_GENERATION_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke import (  # noqa: E402
    BOUNDARY as SELECTED_TRAINING_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
    ValueError
):
    pass


BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe"
PASS_NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repeatability_probe"
FAIL_NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_repair_decision"
SCHEMA_VERSION = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            f"report missing: {path}"
        )
    return json.loads(path.read_text(encoding="utf-8"))


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


def validate_selected_training(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != SELECTED_TRAINING_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "selected training smoke boundary required"
        )
    readiness = _dict(report.get("readiness"))
    training = _dict(report.get("training_result"))
    if not bool(readiness.get("checkpoint_generation_probe_ready", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "checkpoint generation readiness required"
        )
    if _int(training.get("checkpoint_count")) < 1:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "checkpoint count required"
        )
    return {
        "selected_train_records": _int(training.get("selected_train_records")),
        "selected_val_records": _int(training.get("selected_val_records")),
        "max_sequence": _int(training.get("max_sequence")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(training.get("checkpoint_count")),
    }


def validate_generation_probe(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    command = _dict(report.get("generation_command"))
    summary = _dict(report.get("generation_summary"))
    if str(readiness.get("boundary") or "") != GENERIC_GENERATION_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "generic generation probe boundary required"
        )
    if not bool(readiness.get("generation_path_executable", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "generation path should be executable"
        )
    if _int(command.get("returncode")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "generation command must succeed"
        )
    if _int(summary.get("sample_count")) <= 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "sample count required"
        )
    blocked_claims = [
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "human_audio_preference_claimed",
        "production_ready_improviser_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            f"unexpected generation claim: {claimed}"
        )
    return {
        "source_next_boundary": str(decision.get("next_boundary") or ""),
        "command_returncode": _int(command.get("returncode")),
        "sample_count": _int(summary.get("sample_count")),
        "valid_sample_count": _int(summary.get("valid_sample_count")),
        "strict_valid_sample_count": _int(summary.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(summary.get("grammar_gate_sample_count")),
        "passed_generation_gate": bool(summary.get("passed_generation_gate", False)),
        "passed_grammar_gate": bool(summary.get("passed_grammar_gate", False)),
        "passed_strict_review_gate": bool(summary.get("passed_strict_review_gate", False)),
        "collapse_warning_sample_count": _int(summary.get("collapse_warning_sample_count")),
        "collapse_warning_sample_rate": _float(summary.get("collapse_warning_sample_rate")),
        "avg_onset_coverage_ratio": _float(summary.get("avg_onset_coverage_ratio")),
        "avg_sustained_coverage_ratio": _float(summary.get("avg_sustained_coverage_ratio")),
        "max_longest_sustained_empty_run_steps": _int(
            summary.get("max_longest_sustained_empty_run_steps")
        ),
        "avg_postprocess_removal_ratio": _float(summary.get("avg_postprocess_removal_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "failure_reasons": _dict(summary.get("failure_reasons")),
        "strict_failure_reasons": _dict(summary.get("strict_failure_reasons")),
        "diagnostic_failure_reasons": _dict(summary.get("diagnostic_failure_reasons")),
    }


def build_generation_probe_summary(
    selected_training: dict[str, Any],
    generation_probe: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    training = validate_selected_training(selected_training)
    generation = validate_generation_probe(generation_probe)
    passed_strict = bool(generation.get("passed_strict_review_gate", False))
    next_boundary = PASS_NEXT_BOUNDARY if passed_strict else FAIL_NEXT_BOUNDARY
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "selected_training": SELECTED_TRAINING_BOUNDARY,
            "generation_probe": GENERIC_GENERATION_BOUNDARY,
        },
        "training_source": training,
        "generation_summary": generation,
        "readiness": {
            "boundary": BOUNDARY,
            "controlled_scale_checkpoint_generation_probe_completed": True,
            "generation_path_executable": True,
            "raw_generation_quality_ready": passed_strict,
            "model_direct_generation_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "selected-scale checkpoint generation probe completed; next boundary selected by strict gate outcome"
            ),
        },
        "not_proven": [
            "selected_scale_checkpoint_generation_repeatability",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale repeatability probe"
            if passed_strict
            else "Stage B MIDI-to-solo controlled scale checkpoint training scale repair decision"
        ),
    }


def validate_generation_probe_summary(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    min_sample_count: int,
    require_generation_executable: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    generation = _dict(report.get("generation_summary"))
    training = _dict(report.get("training_source"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "unexpected boundary"
        )
    if _int(generation.get("sample_count")) < int(min_sample_count):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "sample count below requirement"
        )
    if require_generation_executable and not bool(readiness.get("generation_path_executable", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
            "generation executable required"
        )
    if require_no_quality_claim:
        blocked = [
            "model_direct_generation_quality_claimed",
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleGenerationProbeError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_train_records": _int(training.get("selected_train_records")),
        "selected_val_records": _int(training.get("selected_val_records")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "sample_count": _int(generation.get("sample_count")),
        "valid_sample_count": _int(generation.get("valid_sample_count")),
        "strict_valid_sample_count": _int(generation.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(generation.get("grammar_gate_sample_count")),
        "collapse_warning_sample_count": _int(generation.get("collapse_warning_sample_count")),
        "raw_generation_quality_ready": bool(readiness.get("raw_generation_quality_ready", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    training = report["training_source"]
    generation = report["generation_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Generation Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- train / val records: `{training['selected_train_records']}` / `{training['selected_val_records']}`",
        f"- best validation loss: `{training['best_validation_loss']}`",
        f"- sample count: `{generation['sample_count']}`",
        f"- valid / strict / grammar: `{generation['valid_sample_count']}` / `{generation['strict_valid_sample_count']}` / `{generation['grammar_gate_sample_count']}`",
        f"- collapse warning sample count / rate: `{generation['collapse_warning_sample_count']}` / `{generation['collapse_warning_sample_rate']}`",
        f"- avg onset / sustained coverage ratio: `{generation['avg_onset_coverage_ratio']}` / `{generation['avg_sustained_coverage_ratio']}`",
        f"- avg / max postprocess removal ratio: `{generation['avg_postprocess_removal_ratio']}` / `{generation['max_postprocess_removal_ratio']}`",
        f"- raw generation quality ready: `{_bool_token(readiness['raw_generation_quality_ready'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Failure Reasons",
        "",
    ]
    for reason, count in generation["diagnostic_failure_reasons"].items():
        lines.append(f"- `{reason}`: `{count}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize selected-scale checkpoint generation probe")
    parser.add_argument("--selected_training", type=str, required=True)
    parser.add_argument("--generation_probe", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--min_sample_count", type=int, default=1)
    parser.add_argument("--require_generation_executable", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_generation_probe_summary(
        read_json(Path(args.selected_training)),
        read_json(Path(args.generation_probe)),
        output_dir=output_dir,
    )
    summary = validate_generation_probe_summary(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        min_sample_count=int(args.min_sample_count),
        require_generation_executable=bool(args.require_generation_executable),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
