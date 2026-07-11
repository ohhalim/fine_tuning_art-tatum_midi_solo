"""Summarize the selected controlled checkpoint training scale smoke."""

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
from scripts.decide_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion import (  # noqa: E402
    BOUNDARY as DECISION_BOUNDARY,
    NEXT_BOUNDARY as DECISION_NEXT_BOUNDARY,
)
from scripts.run_stage_b_generic_base_training_scale_smoke import (  # noqa: E402
    BOUNDARY as TRAINING_SMOKE_BOUNDARY,
)


class StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke"
NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_generation_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
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


def validate_decision(decision_report: dict[str, Any]) -> dict[str, Any]:
    if str(decision_report.get("boundary") or "") != DECISION_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "controlled scale checkpoint training decision required"
        )
    readiness = _dict(decision_report.get("readiness"))
    decision = _dict(decision_report.get("decision"))
    plan = _dict(decision_report.get("selected_scale_plan"))
    if str(decision.get("next_boundary") or "") != DECISION_NEXT_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "decision must route to selected-scale smoke"
        )
    if not bool(readiness.get("controlled_training_scale_smoke_ready", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "controlled scale smoke readiness required"
        )
    if bool(readiness.get("cloud_or_gpu_spend_required", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "cloud/GPU spend should not be required"
        )
    if bool(readiness.get("full_training_selected", True)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "full training should not be selected"
        )
    return {
        "selected_train_records": _int(plan.get("selected_train_records")),
        "selected_val_records": _int(plan.get("selected_val_records")),
        "max_sequence": _int(plan.get("max_sequence")),
        "epochs": _int(plan.get("epochs")),
        "batch_size": _int(plan.get("batch_size")),
        "lr": _float(plan.get("lr")),
        "seed": _int(plan.get("seed")),
    }


def validate_training_smoke(training_smoke: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(training_smoke.get("readiness"))
    if str(training_smoke.get("boundary") or readiness.get("boundary") or "") != TRAINING_SMOKE_BOUNDARY:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "generic training scale smoke boundary required"
        )
    inputs = _dict(training_smoke.get("input"))
    config = _dict(training_smoke.get("training_config"))
    training = _dict(training_smoke.get("training"))
    token = _dict(training_smoke.get("token_stats"))
    artifacts = _dict(training_smoke.get("artifacts"))
    if not bool(readiness.get("training_scale_smoke_passed", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "training scale smoke should pass"
        )
    if _int(inputs.get("selected_train_records")) != _int(decision.get("selected_train_records")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "selected train record mismatch"
        )
    if _int(inputs.get("selected_val_records")) != _int(decision.get("selected_val_records")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "selected val record mismatch"
        )
    if _int(config.get("max_sequence")) != _int(decision.get("max_sequence")):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "max_sequence mismatch"
        )
    if _int(training.get("returncode")) != 0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "training returncode should be zero"
        )
    if _float(training.get("best_validation_loss")) <= 0.0:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "best validation loss required"
        )
    if _int(artifacts.get("checkpoint_count")) < 1:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "checkpoint artifact required"
        )
    if not bool(artifacts.get("lora_weights_exists", False)):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "lora weights artifact required"
        )
    blocked_claims = [
        "full_generic_training_executed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            f"unexpected training claim: {claimed}"
        )
    return {
        "selected_train_records": _int(inputs.get("selected_train_records")),
        "selected_val_records": _int(inputs.get("selected_val_records")),
        "max_sequence": _int(config.get("max_sequence")),
        "epochs": _int(config.get("epochs")),
        "batch_size": _int(config.get("batch_size")),
        "lr": _float(config.get("lr")),
        "training_returncode": _int(training.get("returncode")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
        "checkpoint_files": [str(path) for path in artifacts.get("checkpoint_files", [])],
        "lora_weights_exists": bool(artifacts.get("lora_weights_exists", False)),
    }


def build_selected_scale_smoke_summary(
    decision_report: dict[str, Any],
    training_smoke: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    decision = validate_decision(decision_report)
    training = validate_training_smoke(training_smoke, decision)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "decision": DECISION_BOUNDARY,
            "training_smoke": TRAINING_SMOKE_BOUNDARY,
        },
        "decision_plan": decision,
        "training_result": training,
        "readiness": {
            "boundary": BOUNDARY,
            "controlled_scale_checkpoint_training_scale_smoke_completed": True,
            "checkpoint_generation_probe_ready": True,
            "model_direct_generation_quality_claimed": False,
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
                "selected 2048/512 max_sequence 160 training smoke completed; "
                "next step is generation probe from the selected-scale checkpoint"
            ),
        },
        "not_proven": [
            "selected_scale_checkpoint_generation_result",
            "model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo controlled scale checkpoint training scale generation probe"
        ),
    }


def validate_selected_scale_smoke_summary(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    min_train_records: int,
    min_val_records: int,
    require_checkpoint: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    training = _dict(report.get("training_result"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "unexpected boundary"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "unexpected next boundary"
        )
    if _int(training.get("selected_train_records")) < int(min_train_records):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "train records below requirement"
        )
    if _int(training.get("selected_val_records")) < int(min_val_records):
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "val records below requirement"
        )
    if require_checkpoint and _int(training.get("checkpoint_count")) < 1:
        raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
            "checkpoint required"
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
            raise StageBMidiToSoloControlledScaleCheckpointTrainingScaleSmokeError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_train_records": _int(training.get("selected_train_records")),
        "selected_val_records": _int(training.get("selected_val_records")),
        "max_sequence": _int(training.get("max_sequence")),
        "epochs": _int(training.get("epochs")),
        "batch_size": _int(training.get("batch_size")),
        "training_returncode": _int(training.get("training_returncode")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(training.get("checkpoint_count")),
        "fits_vocab": bool(training.get("fits_vocab", False)),
        "checkpoint_generation_probe_ready": bool(
            readiness.get("checkpoint_generation_probe_ready", False)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    training = report["training_result"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Controlled Scale Checkpoint Training Scale Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- train / val records: `{training['selected_train_records']}` / `{training['selected_val_records']}`",
        f"- max sequence: `{training['max_sequence']}`",
        f"- epochs: `{training['epochs']}`",
        f"- batch size: `{training['batch_size']}`",
        f"- training returncode: `{training['training_returncode']}`",
        f"- best validation loss: `{training['best_validation_loss']}`",
        f"- checkpoint count: `{training['checkpoint_count']}`",
        f"- fits vocab: `{_bool_token(training['fits_vocab'])}`",
        f"- checkpoint generation probe ready: `{_bool_token(readiness['checkpoint_generation_probe_ready'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize selected controlled checkpoint training scale smoke")
    parser.add_argument(
        "--decision_report",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision/"
        "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_expansion_decision.json",
    )
    parser.add_argument(
        "--training_smoke",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/"
        "harness_stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke/training_smoke/"
        "controlled_2048_512_maxseq160/stage_b_generic_base_training_scale_smoke.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--min_train_records", type=int, default=1)
    parser.add_argument("--min_val_records", type=int, default=1)
    parser.add_argument("--require_checkpoint", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_selected_scale_smoke_summary(
        read_json(Path(args.decision_report)),
        read_json(Path(args.training_smoke)),
        output_dir=output_dir,
    )
    summary = validate_selected_scale_smoke_summary(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        min_train_records=int(args.min_train_records),
        min_val_records=int(args.min_val_records),
        require_checkpoint=bool(args.require_checkpoint),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_controlled_scale_checkpoint_training_scale_smoke.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
