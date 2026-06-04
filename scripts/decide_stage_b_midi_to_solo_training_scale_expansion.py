"""Decide the next controlled training scale expansion for MIDI-to-solo."""

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
from scripts.consolidate_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke import (  # noqa: E402
    BOUNDARY as SEQUENCE_BUDGET_BOUNDARY,
)
from scripts.decide_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_objective_next import (  # noqa: E402
    BOUNDARY as OBJECTIVE_PATH_BOUNDARY,
    FINAL_BOUNDARY as OBJECTIVE_PATH_FINAL_BOUNDARY,
)


class StageBMidiToSoloTrainingScaleExpansionDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_training_scale_expansion_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_controlled_training_scale_smoke"
SCHEMA_VERSION = "stage_b_midi_to_solo_training_scale_expansion_decision_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_training_resource(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != TRAINING_RESOURCE_BOUNDARY:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("training resource probe boundary required")
    readiness = _dict(report.get("readiness"))
    full = _dict(report.get("full_window_resource"))
    scale = _dict(report.get("scale_smoke_resource"))
    if not bool(readiness.get("midi_to_solo_training_resource_ready", False)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("MIDI-to-solo training resource readiness required")
    if not bool(readiness.get("conditioned_generation_probe_ready", False)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("conditioned generation probe readiness required")
    if _int(full.get("tokenized_train_files")) < 100_000:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("full tokenized train resource below requirement")
    if _int(full.get("tokenized_val_files")) < 10_000:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("full tokenized val resource below requirement")
    if _int(scale.get("selected_train_records")) < 128:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("prior scale train records below baseline")
    if _int(scale.get("selected_val_records")) < 32:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("prior scale val records below baseline")
    if _int(scale.get("checkpoint_count")) < 1:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("prior scale checkpoint required")
    blocked_claims = [
        "midi_to_solo_mvp_claimed",
        "broad_training_executed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "musical_quality_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError(f"unexpected training resource claim: {claimed}")
    return {
        "full_tokenized_train_files": _int(full.get("tokenized_train_files")),
        "full_tokenized_val_files": _int(full.get("tokenized_val_files")),
        "max_token_id": _int(full.get("max_token_id")),
        "vocab_size": _int(full.get("vocab_size")),
        "fits_vocab": bool(full.get("fits_vocab", False)),
        "prior_scale_train_records": _int(scale.get("selected_train_records")),
        "prior_scale_val_records": _int(scale.get("selected_val_records")),
        "prior_scale_best_validation_loss": _float(scale.get("best_validation_loss")),
        "prior_scale_checkpoint_count": _int(scale.get("checkpoint_count")),
    }


def validate_sequence_budget(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != SEQUENCE_BUDGET_BOUNDARY:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("sequence budget repair boundary required")
    readiness = _dict(report.get("readiness"))
    result = _dict(report.get("repair_result"))
    if not bool(result.get("sequence_budget_repaired", False)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("sequence budget repair support required")
    if not bool(readiness.get("model_direct_8bar_generation_probe_ready", False)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("direct 8-bar generation probe readiness required")
    if _int(result.get("repaired_max_sequence")) < 160:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("max_sequence 160 repair required")
    if _int(result.get("repaired_direct_note_capacity")) < _int(result.get("target_min_note_count")):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("repaired note capacity below target")
    blocked_claims = [
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "human_audio_preference_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError(f"unexpected sequence budget claim: {claimed}")
    return {
        "previous_max_sequence": _int(result.get("previous_max_sequence")),
        "repaired_max_sequence": _int(result.get("repaired_max_sequence")),
        "previous_direct_note_capacity": _int(result.get("previous_direct_note_capacity")),
        "repaired_direct_note_capacity": _int(result.get("repaired_direct_note_capacity")),
        "target_min_note_count": _int(result.get("target_min_note_count")),
        "minimum_contract_tokens": _int(result.get("minimum_contract_tokens")),
    }


def validate_objective_path(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != OBJECTIVE_PATH_BOUNDARY:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("objective path decision boundary required")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    objective = _dict(report.get("objective_repeatability_summary"))
    review = _dict(report.get("review_boundary_summary"))
    if str(decision.get("final_boundary") or "") != OBJECTIVE_PATH_FINAL_BOUNDARY:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("objective path final boundary required")
    if not bool(readiness.get("objective_repeatability_path_supported", False)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("objective repeatability support required")
    if _int(objective.get("qualified_candidate_count")) != _int(objective.get("sample_count")):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("all objective repeatability samples should qualify")
    if _int(objective.get("current_analysis_flag_count")) != 0:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("objective path flags should be zero")
    if _int(objective.get("overlap_detected_count")) != 0:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("objective path overlap count should be zero")
    if bool(review.get("validated_review_input_present", True)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("review input should remain pending")
    blocked_claims = [
        "human_audio_preference_claimed",
        "model_direct_generation_quality_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError(f"unexpected objective path claim: {claimed}")
    return {
        "final_boundary": str(decision.get("final_boundary") or ""),
        "sample_count": _int(objective.get("sample_count")),
        "qualified_candidate_count": _int(objective.get("qualified_candidate_count")),
        "objective_clean_pass_rate": _float(objective.get("objective_clean_pass_rate")),
        "current_analysis_flag_count": _int(objective.get("current_analysis_flag_count")),
        "overlap_detected_count": _int(objective.get("overlap_detected_count")),
        "rendered_audio_file_count": _int(review.get("rendered_audio_file_count")),
        "pending_status_field_count": _int(review.get("pending_status_field_count")),
        "pending_candidate_decision_count": _int(review.get("pending_candidate_decision_count")),
        "pending_candidate_field_count": _int(review.get("pending_candidate_field_count")),
    }


def selected_scale_plan(
    training: dict[str, Any],
    sequence: dict[str, Any],
    *,
    target_train_records: int,
    target_val_records: int,
) -> dict[str, Any]:
    train_records = min(int(target_train_records), _int(training.get("full_tokenized_train_files")))
    val_records = min(int(target_val_records), _int(training.get("full_tokenized_val_files")))
    prior_train = _int(training.get("prior_scale_train_records"))
    prior_val = _int(training.get("prior_scale_val_records"))
    if train_records <= prior_train or val_records <= prior_val:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("selected scale must exceed prior smoke size")
    return {
        "selected_train_records": train_records,
        "selected_val_records": val_records,
        "prior_train_records": prior_train,
        "prior_val_records": prior_val,
        "scale_multiplier_train": train_records / max(1, prior_train),
        "scale_multiplier_val": val_records / max(1, prior_val),
        "max_sequence": _int(sequence.get("repaired_max_sequence")),
        "epochs": 1,
        "batch_size": 16,
        "lr": 8e-4,
        "seed": 43,
        "n_layers": 1,
        "d_model": 64,
        "num_heads": 4,
        "dim_feedforward": 128,
        "lora_r": 4,
        "lora_alpha": 8,
    }


def build_training_scale_expansion_decision(
    training_resource: dict[str, Any],
    sequence_budget: dict[str, Any],
    objective_path: dict[str, Any],
    *,
    output_dir: Path,
    target_train_records: int,
    target_val_records: int,
) -> dict[str, Any]:
    training = validate_training_resource(training_resource)
    sequence = validate_sequence_budget(sequence_budget)
    objective = validate_objective_path(objective_path)
    scale_plan = selected_scale_plan(
        training,
        sequence,
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
            "sequence_budget": SEQUENCE_BUDGET_BOUNDARY,
            "objective_path": OBJECTIVE_PATH_BOUNDARY,
        },
        "training_resource_summary": training,
        "sequence_budget_summary": sequence,
        "objective_path_summary": objective,
        "selected_scale_plan": scale_plan,
        "readiness": {
            "boundary": BOUNDARY,
            "training_scale_expansion_decision_completed": True,
            "controlled_training_scale_smoke_ready": True,
            "cloud_or_gpu_spend_required": False,
            "broad_training_executed": False,
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
                "full tokenized resource, max_sequence 160 repair, and objective repeatability support are present; "
                "next step is a bounded local training smoke, not broad training"
            ),
        },
        "not_proven": [
            "controlled_training_scale_smoke_result",
            "improved_validation_loss",
            "improved_model_direct_generation_quality",
            "midi_to_solo_musical_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo controlled training scale smoke",
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
    decision = _dict(report.get("decision"))
    readiness = _dict(report.get("readiness"))
    plan = _dict(report.get("selected_scale_plan"))
    objective = _dict(report.get("objective_path_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("unexpected boundary")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("unexpected next boundary")
    if _int(plan.get("selected_train_records")) < int(min_selected_train_records):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("selected train records below requirement")
    if _int(plan.get("selected_val_records")) < int(min_selected_val_records):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("selected val records below requirement")
    if require_scale_ready and not bool(readiness.get("controlled_training_scale_smoke_ready", False)):
        raise StageBMidiToSoloTrainingScaleExpansionDecisionError("controlled scale smoke readiness required")
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
            raise StageBMidiToSoloTrainingScaleExpansionDecisionError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_train_records": _int(plan.get("selected_train_records")),
        "selected_val_records": _int(plan.get("selected_val_records")),
        "prior_train_records": _int(plan.get("prior_train_records")),
        "prior_val_records": _int(plan.get("prior_val_records")),
        "scale_multiplier_train": _float(plan.get("scale_multiplier_train")),
        "scale_multiplier_val": _float(plan.get("scale_multiplier_val")),
        "max_sequence": _int(plan.get("max_sequence")),
        "objective_sample_count": _int(objective.get("sample_count")),
        "objective_qualified_candidate_count": _int(objective.get("qualified_candidate_count")),
        "objective_clean_pass_rate": _float(objective.get("objective_clean_pass_rate")),
        "rendered_audio_file_count": _int(objective.get("rendered_audio_file_count")),
        "controlled_training_scale_smoke_ready": bool(
            readiness.get("controlled_training_scale_smoke_ready", False)
        ),
        "cloud_or_gpu_spend_required": bool(readiness.get("cloud_or_gpu_spend_required", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    plan = report["selected_scale_plan"]
    objective = report["objective_path_summary"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Training Scale Expansion Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- controlled training scale smoke ready: `{_bool_token(readiness['controlled_training_scale_smoke_ready'])}`",
        f"- cloud or GPU spend required: `{_bool_token(readiness['cloud_or_gpu_spend_required'])}`",
        f"- selected train / val records: `{plan['selected_train_records']}` / `{plan['selected_val_records']}`",
        f"- prior train / val records: `{plan['prior_train_records']}` / `{plan['prior_val_records']}`",
        f"- max sequence: `{plan['max_sequence']}`",
        f"- objective generated / qualified: `{objective['sample_count']}` / `{objective['qualified_candidate_count']}`",
        f"- objective clean pass rate: `{objective['objective_clean_pass_rate']:.4f}`",
        f"- rendered audio files: `{objective['rendered_audio_file_count']}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Selected Training Config",
        "",
        f"- epochs: `{plan['epochs']}`",
        f"- batch size: `{plan['batch_size']}`",
        f"- lr: `{plan['lr']}`",
        f"- seed: `{plan['seed']}`",
        f"- n layers / d model / heads: `{plan['n_layers']}` / `{plan['d_model']}` / `{plan['num_heads']}`",
        f"- LoRA r / alpha: `{plan['lora_r']}` / `{plan['lora_alpha']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide MIDI-to-solo training scale expansion")
    parser.add_argument(
        "--training_resource",
        type=str,
        default="outputs/stage_b_midi_to_solo_training_resource_probe/"
        "harness_stage_b_midi_to_solo_training_resource_probe/"
        "stage_b_midi_to_solo_training_resource_probe.json",
    )
    parser.add_argument(
        "--sequence_budget",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/"
        "harness_stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke/"
        "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json",
    )
    parser.add_argument(
        "--objective_path",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_"
        "repeatability_objective_next/harness_stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_"
        "contour_phrase_shape_repeatability_objective_next/"
        "stage_b_midi_to_solo_model_direct_jazz_phrase_vocabulary_contour_phrase_shape_repeatability_"
        "objective_next.json",
    )
    parser.add_argument("--output_root", type=str, default="outputs/stage_b_midi_to_solo_training_scale_expansion_decision")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--target_train_records", type=int, default=512)
    parser.add_argument("--target_val_records", type=int, default=128)
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
        read_json(Path(args.sequence_budget)),
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
    write_json(output_dir / "stage_b_midi_to_solo_training_scale_expansion_decision.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_training_scale_expansion_decision_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_training_scale_expansion_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
