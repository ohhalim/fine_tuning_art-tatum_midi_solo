"""Decide the next bounded controlled checkpoint training scale."""

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
from scripts.check_stage_b_midi_to_solo_training_resource_probe import (  # noqa: E402
    BOUNDARY as TRAINING_RESOURCE_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next import (  # noqa: E402
    BOUNDARY as OBJECTIVE_PATH_BOUNDARY,
    FINAL_BOUNDARY as OBJECTIVE_PATH_FINAL_BOUNDARY,
)
from scripts.summarize_stage_b_midi_to_solo_controlled_training_scale_smoke import (  # noqa: E402
    BOUNDARY as CONTROLLED_TRAINING_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke"
SCHEMA_VERSION = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
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


def validate_training_resource(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != TRAINING_RESOURCE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "training resource probe boundary required"
        )
    readiness = _dict(report.get("readiness"))
    full = _dict(report.get("full_window_resource"))
    if not bool(readiness.get("midi_to_solo_training_resource_ready", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "training resource readiness required"
        )
    if not bool(readiness.get("conditioned_generation_probe_ready", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "conditioned generation readiness required"
        )
    if not bool(full.get("fits_vocab", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "full window tokens must fit vocab"
        )
    blocked_claims = [
        "midi_to_solo_mvp_claimed",
        "broad_training_executed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "musical_quality_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            f"unexpected training resource claim: {claimed}"
        )
    return {
        "full_tokenized_train_records": _int(full.get("tokenized_train_files")),
        "full_tokenized_val_records": _int(full.get("tokenized_val_files")),
        "max_token_id": _int(full.get("max_token_id")),
        "vocab_size": _int(full.get("vocab_size")),
        "fits_vocab": bool(full.get("fits_vocab", False)),
    }


def validate_current_controlled_training(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != CONTROLLED_TRAINING_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "current controlled training smoke boundary required"
        )
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    training = _dict(report.get("training_result"))
    if not bool(readiness.get("controlled_training_scale_smoke_completed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "current controlled training smoke completion required"
        )
    if not bool(readiness.get("checkpoint_generation_probe_ready", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "checkpoint generation readiness required"
        )
    if _int(training.get("training_returncode")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "current training returncode must be zero"
        )
    if _int(training.get("checkpoint_count")) < 1:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "current checkpoint required"
        )
    if _float(training.get("best_validation_loss")) <= 0.0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "current validation loss required"
        )
    blocked_claims = [
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            f"unexpected controlled training claim: {claimed}"
        )
    return {
        "current_boundary": CONTROLLED_TRAINING_BOUNDARY,
        "current_next_boundary": str(decision.get("next_boundary") or ""),
        "current_train_records": _int(training.get("selected_train_records")),
        "current_val_records": _int(training.get("selected_val_records")),
        "max_sequence": _int(training.get("max_sequence")),
        "epochs": _int(training.get("epochs")),
        "batch_size": _int(training.get("batch_size")),
        "lr": _float(training.get("lr")),
        "training_returncode": _int(training.get("training_returncode")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(training.get("checkpoint_count")),
        "fits_vocab": bool(training.get("fits_vocab", False)),
    }


def validate_objective_path(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != OBJECTIVE_PATH_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "controlled objective path decision boundary required"
        )
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    evidence = _dict(report.get("temperature_guard_summary"))
    review = _dict(report.get("review_boundary_summary"))
    if str(decision.get("final_boundary") or "") != OBJECTIVE_PATH_FINAL_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "controlled objective final boundary required"
        )
    if not bool(readiness.get("objective_temperature_guard_path_supported", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "objective temperature guard path support required"
        )
    if _int(evidence.get("strict_valid_sample_count")) != _int(evidence.get("sample_count")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "all objective samples should be strict valid"
        )
    if _int(evidence.get("dead_air_failure_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "dead-air failure count should be zero"
        )
    if _int(evidence.get("collapse_warning_sample_count")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "collapse warning count should be zero"
        )
    if bool(review.get("validated_review_input_present", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "listening review should remain pending"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            f"unexpected objective path claim: {claimed}"
        )
    return {
        "final_boundary": str(decision.get("final_boundary") or ""),
        "sample_count": _int(evidence.get("sample_count")),
        "seed_count": _int(evidence.get("seed_count")),
        "strict_valid_sample_count": _int(evidence.get("strict_valid_sample_count")),
        "grammar_gate_sample_count": _int(evidence.get("grammar_gate_sample_count")),
        "dead_air_failure_count": _int(evidence.get("dead_air_failure_count")),
        "collapse_warning_sample_count": _int(evidence.get("collapse_warning_sample_count")),
        "rendered_audio_file_count": _int(review.get("rendered_audio_file_count")),
        "pending_status_field_count": _int(review.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(review.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(review.get("pending_candidate_field_count")),
    }


def selected_scale_plan(
    resource: dict[str, Any],
    current_training: dict[str, Any],
    *,
    target_train_records: int,
    target_val_records: int,
) -> dict[str, Any]:
    current_train = _int(current_training.get("current_train_records"))
    current_val = _int(current_training.get("current_val_records"))
    target_train = int(target_train_records)
    target_val = int(target_val_records)
    if _int(resource.get("full_tokenized_train_records")) < target_train:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "full train resource below selected target"
        )
    if _int(resource.get("full_tokenized_val_records")) < target_val:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "full val resource below selected target"
        )
    if target_train <= current_train or target_val <= current_val:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "selected scale must exceed current controlled smoke"
        )
    return {
        "selected_train_records": target_train,
        "selected_val_records": target_val,
        "current_train_records": current_train,
        "current_val_records": current_val,
        "scale_multiplier_train": target_train / max(1, current_train),
        "scale_multiplier_val": target_val / max(1, current_val),
        "max_sequence": _int(current_training.get("max_sequence")),
        "epochs": 1,
        "batch_size": 16,
        "lr": _float(current_training.get("lr")) or 8e-4,
        "seed": 47,
        "n_layers": 1,
        "d_model": 64,
        "num_heads": 4,
        "dim_feedforward": 128,
        "lora_r": 4,
        "lora_alpha": 8,
        "execution_scope": "local_bounded_training_smoke",
    }


def build_training_scale_expansion_decision(
    training_resource: dict[str, Any],
    current_controlled_training: dict[str, Any],
    objective_path: dict[str, Any],
    *,
    output_dir: Path,
    target_train_records: int,
    target_val_records: int,
) -> dict[str, Any]:
    resource = validate_training_resource(training_resource)
    current = validate_current_controlled_training(current_controlled_training)
    objective = validate_objective_path(objective_path)
    scale_plan = selected_scale_plan(
        resource,
        current,
        target_train_records=target_train_records,
        target_val_records=target_val_records,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "training_resource": TRAINING_RESOURCE_BOUNDARY,
            "current_controlled_training": CONTROLLED_TRAINING_BOUNDARY,
            "objective_path": OBJECTIVE_PATH_BOUNDARY,
        },
        "training_resource_summary": resource,
        "current_training_summary": current,
        "objective_path_summary": objective,
        "selected_scale_plan": scale_plan,
        "readiness": {
            "boundary": BOUNDARY,
            "training_scale_expansion_decision_completed": True,
            "controlled_training_scale_smoke_ready": True,
            "cloud_or_gpu_spend_required": False,
            "full_training_selected": False,
            "broad_training_executed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "human_audio_preference_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "current 512/128 controlled smoke and objective temperature guard path passed; "
                "select next local bounded 2048/512 training smoke without full-training claim"
            ),
        },
        "not_proven": [
            "selected_scale_training_result",
            "checkpoint_generation_quality",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale smoke"
        ),
    }


def validate_training_scale_expansion_decision(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_selected_train_records: int,
    min_selected_val_records: int,
    require_scale_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    plan = _dict(report.get("selected_scale_plan"))
    current = _dict(report.get("current_training_summary"))
    objective = _dict(report.get("objective_path_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "unexpected next boundary"
        )
    if _int(plan.get("selected_train_records")) < int(min_selected_train_records):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "selected train records below requirement"
        )
    if _int(plan.get("selected_val_records")) < int(min_selected_val_records):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "selected val records below requirement"
        )
    if _int(plan.get("selected_train_records")) <= _int(current.get("current_train_records")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "selected train records should exceed current records"
        )
    if require_scale_ready and not bool(readiness.get("controlled_training_scale_smoke_ready", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
            "controlled scale smoke readiness required"
        )
    if require_no_quality_claim:
        blocked = [
            "midi_to_solo_musical_quality_claimed",
            "human_audio_preference_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleExpansionError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_train_records": _int(plan.get("selected_train_records")),
        "selected_val_records": _int(plan.get("selected_val_records")),
        "current_train_records": _int(current.get("current_train_records")),
        "current_val_records": _int(current.get("current_val_records")),
        "scale_multiplier_train": _float(plan.get("scale_multiplier_train")),
        "scale_multiplier_val": _float(plan.get("scale_multiplier_val")),
        "max_sequence": _int(plan.get("max_sequence")),
        "epochs": _int(plan.get("epochs")),
        "batch_size": _int(plan.get("batch_size")),
        "seed": _int(plan.get("seed")),
        "current_best_validation_loss": _float(current.get("best_validation_loss")),
        "objective_sample_count": _int(objective.get("sample_count")),
        "objective_strict_valid_sample_count": _int(
            objective.get("strict_valid_sample_count")
        ),
        "rendered_audio_file_count": _int(objective.get("rendered_audio_file_count")),
        "controlled_training_scale_smoke_ready": bool(
            readiness.get("controlled_training_scale_smoke_ready", False)
        ),
        "cloud_or_gpu_spend_required": bool(readiness.get("cloud_or_gpu_spend_required", True)),
        "full_training_selected": bool(readiness.get("full_training_selected", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    plan = report["selected_scale_plan"]
    current = report["current_training_summary"]
    objective = report["objective_path_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Expansion Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- current train / val records: `{current['current_train_records']}` / `{current['current_val_records']}`",
        f"- selected train / val records: `{plan['selected_train_records']}` / `{plan['selected_val_records']}`",
        f"- scale multiplier train / val: `{plan['scale_multiplier_train']:.1f}` / `{plan['scale_multiplier_val']:.1f}`",
        f"- max sequence: `{plan['max_sequence']}`",
        f"- epochs: `{plan['epochs']}`",
        f"- batch size: `{plan['batch_size']}`",
        f"- seed: `{plan['seed']}`",
        f"- current best validation loss: `{current['best_validation_loss']}`",
        f"- objective sample / strict: `{objective['sample_count']}` / `{objective['strict_valid_sample_count']}`",
        f"- rendered audio file count: `{objective['rendered_audio_file_count']}`",
        f"- controlled training scale smoke ready: `{_bool_token(readiness['controlled_training_scale_smoke_ready'])}`",
        f"- cloud or GPU spend required: `{_bool_token(readiness['cloud_or_gpu_spend_required'])}`",
        f"- full training selected: `{_bool_token(readiness['full_training_selected'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide controlled checkpoint training scale expansion")
    parser.add_argument(
        "--training_resource",
        type=str,
        default="outputs/stage_b_midi_to_solo_training_resource_probe/"
        "harness_stage_b_midi_to_solo_training_resource_probe/"
        "stage_b_midi_to_solo_training_resource_probe.json",
    )
    parser.add_argument(
        "--current_controlled_training",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_training_scale_smoke/"
        "harness_stage_b_midi_to_solo_controlled_training_scale_smoke/"
        "stage_b_midi_to_solo_controlled_training_scale_smoke.json",
    )
    parser.add_argument(
        "--objective_path",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_"
        "temperature_guard_objective_next/harness_stage_b_midi_to_solo_controlled_scale_checkpoint_"
        "dead_air_repeatability_temperature_guard_objective_next/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_dead_air_repeatability_temperature_guard_objective_next.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--target_train_records", type=int, default=2048)
    parser.add_argument("--target_val_records", type=int, default=512)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_selected_train_records", type=int, default=1)
    parser.add_argument("--min_selected_val_records", type=int, default=1)
    parser.add_argument("--require_scale_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_training_scale_expansion_decision(
        read_json(Path(args.training_resource)),
        read_json(Path(args.current_controlled_training)),
        read_json(Path(args.objective_path)),
        output_dir=output_dir,
        target_train_records=int(args.target_train_records),
        target_val_records=int(args.target_val_records),
    )
    summary = validate_training_scale_expansion_decision(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_selected_train_records=int(args.min_selected_train_records),
        min_selected_val_records=int(args.min_selected_val_records),
        require_scale_ready=bool(args.require_scale_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
