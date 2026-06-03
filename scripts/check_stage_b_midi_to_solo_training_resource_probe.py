"""Check training/eval resource readiness for the Stage B MIDI-to-solo path."""

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


class StageBMidiToSoloTrainingResourceProbeError(ValueError):
    pass


CONTEXT_BOUNDARY = "stage_b_midi_to_solo_context_extraction_mvp"
FULL_WINDOW_BOUNDARY = "stage_b_generic_full_manifest_window_preparation"
SCALE_SMOKE_BOUNDARY = "stage_b_generic_base_training_scale_smoke"
BOUNDARY = "stage_b_midi_to_solo_training_resource_probe"
NEXT_BOUNDARY = "stage_b_midi_to_solo_conditioned_generation_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_training_resource_probe_v1"

REQUIRED_CONTEXT_FIELDS = {
    "bar_index",
    "position_index",
    "tempo",
    "chord_root",
    "chord_quality",
    "next_chord_root",
    "next_chord_quality",
    "bass_note",
    "chord_confidence",
    "chord_source",
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloTrainingResourceProbeError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def context_summary(context_report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(context_report.get("readiness"))
    summary = _dict(context_report.get("summary"))
    context = _dict(context_report.get("context"))
    events = _list(context.get("context_events"))
    missing_fields: set[str] = set()
    for event in events:
        if isinstance(event, dict):
            missing_fields.update(REQUIRED_CONTEXT_FIELDS - set(event))
        else:
            missing_fields.update(REQUIRED_CONTEXT_FIELDS)
    boundary = str(context_report.get("boundary") or readiness.get("boundary") or "")
    return {
        "boundary": boundary,
        "context_extraction_completed": bool(readiness.get("context_extraction_completed", False)),
        "required_context_fields_present": bool(readiness.get("required_context_fields_present", False))
        and not missing_fields,
        "context_bars": _int(summary.get("context_bars")),
        "positions_per_bar": _int(summary.get("positions_per_bar")),
        "context_event_count": _int(summary.get("context_event_count")),
        "inferred_chord_bar_count": _int(summary.get("inferred_chord_bar_count")),
        "carry_forward_chord_bar_count": _int(summary.get("carry_forward_chord_bar_count")),
        "unknown_chord_bar_count": _int(summary.get("unknown_chord_bar_count")),
        "low_confidence_bar_count": _int(summary.get("low_confidence_bar_count")),
        "bass_note_bar_count": _int(summary.get("bass_note_bar_count")),
        "missing_context_fields": sorted(missing_fields),
        "midi_to_solo_mvp_claimed": bool(readiness.get("midi_to_solo_mvp_claimed", True)),
        "harmony_analysis_quality_claimed": bool(
            readiness.get("harmony_analysis_quality_claimed", True)
        ),
    }


def full_window_summary(full_window_report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(full_window_report.get("readiness"))
    token = _dict(full_window_report.get("token_stats"))
    inputs = _dict(full_window_report.get("input"))
    dataset = _dict(full_window_report.get("dataset_summary"))
    return {
        "boundary": str(readiness.get("boundary") or ""),
        "full_manifest_window_preparation_ready": bool(
            readiness.get("full_manifest_window_preparation_ready", False)
        ),
        "sequence_format": str(dataset.get("sequence_format") or ""),
        "train_file_count": _int(inputs.get("train_file_count")),
        "val_file_count": _int(inputs.get("val_file_count")),
        "window_bars": _int(inputs.get("window_bars")),
        "window_stride_bars": _int(inputs.get("window_stride_bars")),
        "min_window_target_notes": _int(inputs.get("min_window_target_notes")),
        "tokenized_train_files": _int(token.get("tokenized_train_files")),
        "tokenized_val_files": _int(token.get("tokenized_val_files")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "full_training_executed": bool(readiness.get("full_training_executed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def scale_smoke_summary(scale_smoke_report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(scale_smoke_report.get("readiness"))
    inputs = _dict(scale_smoke_report.get("input"))
    token = _dict(scale_smoke_report.get("token_stats"))
    training = _dict(scale_smoke_report.get("training"))
    artifacts = _dict(scale_smoke_report.get("artifacts"))
    return {
        "boundary": str(readiness.get("boundary") or ""),
        "training_scale_smoke_passed": bool(readiness.get("training_scale_smoke_passed", False)),
        "generic_base_scale_checkpoint_generation_probe_ready": bool(
            readiness.get("generic_base_scale_checkpoint_generation_probe_ready", False)
        ),
        "selected_train_records": _int(inputs.get("selected_train_records")),
        "selected_val_records": _int(inputs.get("selected_val_records")),
        "token_records": _int(token.get("files")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "training_returncode": _int(training.get("returncode")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
        "lora_weights_exists": bool(artifacts.get("lora_weights_exists", False)),
        "full_generic_training_executed": bool(readiness.get("full_generic_training_executed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def build_resource_probe_report(
    *,
    context_report: dict[str, Any],
    full_window_report: dict[str, Any],
    scale_smoke_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    context = context_summary(context_report)
    full_window = full_window_summary(full_window_report)
    scale_smoke = scale_smoke_summary(scale_smoke_report)
    compatible = (
        context["boundary"] == CONTEXT_BOUNDARY
        and context["context_extraction_completed"]
        and context["required_context_fields_present"]
        and context["context_event_count"] > 0
        and context["positions_per_bar"] == 16
        and full_window["boundary"] == FULL_WINDOW_BOUNDARY
        and full_window["full_manifest_window_preparation_ready"]
        and full_window["sequence_format"] == "stage_b_v1"
        and full_window["fits_vocab"]
        and scale_smoke["boundary"] == SCALE_SMOKE_BOUNDARY
        and scale_smoke["training_scale_smoke_passed"]
        and scale_smoke["generic_base_scale_checkpoint_generation_probe_ready"]
        and scale_smoke["fits_vocab"]
        and scale_smoke["checkpoint_count"] > 0
        and scale_smoke["best_validation_loss"] is not None
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "context": CONTEXT_BOUNDARY,
            "full_window": FULL_WINDOW_BOUNDARY,
            "scale_smoke": SCALE_SMOKE_BOUNDARY,
        },
        "context_resource": context,
        "full_window_resource": full_window,
        "scale_smoke_resource": scale_smoke,
        "compatibility": {
            "context_fields_match_generation_contract": bool(
                context["required_context_fields_present"] and context["positions_per_bar"] == 16
            ),
            "stage_b_window_resource_available": bool(
                full_window["full_manifest_window_preparation_ready"] and full_window["fits_vocab"]
            ),
            "scale_checkpoint_resource_available": bool(
                scale_smoke["training_scale_smoke_passed"] and scale_smoke["checkpoint_count"] > 0
            ),
            "conditioned_generation_probe_ready": bool(compatible),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "training_resource_probe_completed": True,
            "midi_to_solo_training_resource_ready": bool(compatible),
            "conditioned_generation_probe_ready": bool(compatible),
            "midi_to_solo_mvp_claimed": False,
            "conditioned_generation_completed": False,
            "broad_training_executed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "musical_quality_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "MIDI context rows, full Stage B window resources, and scale-smoke checkpoint "
                "evidence are available for a conditioned generation probe"
            ),
        },
        "not_proven": [
            "conditioned_generation_output",
            "ranked_solo_midi_candidates",
            "midi_to_solo_mvp_completion",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "human_audio_preference",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo conditioned generation probe",
    }


def validate_resource_probe_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_ready: bool,
    require_no_final_claim: bool,
    min_context_events: int,
    min_full_train_records: int,
    min_full_val_records: int,
    min_scale_train_records: int,
    min_scale_val_records: int,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    context = _dict(report.get("context_resource"))
    full_window = _dict(report.get("full_window_resource"))
    scale_smoke = _dict(report.get("scale_smoke_resource"))
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloTrainingResourceProbeError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBMidiToSoloTrainingResourceProbeError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_ready and not bool(readiness.get("midi_to_solo_training_resource_ready", False)):
        raise StageBMidiToSoloTrainingResourceProbeError("training resource must be ready")
    if _int(context.get("context_event_count")) < int(min_context_events):
        raise StageBMidiToSoloTrainingResourceProbeError("context event count below threshold")
    if _list(context.get("missing_context_fields")):
        raise StageBMidiToSoloTrainingResourceProbeError("context fields missing")
    if _int(full_window.get("tokenized_train_files")) < int(min_full_train_records):
        raise StageBMidiToSoloTrainingResourceProbeError("full train token records below threshold")
    if _int(full_window.get("tokenized_val_files")) < int(min_full_val_records):
        raise StageBMidiToSoloTrainingResourceProbeError("full val token records below threshold")
    if not bool(full_window.get("fits_vocab", False)):
        raise StageBMidiToSoloTrainingResourceProbeError("full window token vocab guard failed")
    if _int(scale_smoke.get("selected_train_records")) < int(min_scale_train_records):
        raise StageBMidiToSoloTrainingResourceProbeError("scale-smoke train records below threshold")
    if _int(scale_smoke.get("selected_val_records")) < int(min_scale_val_records):
        raise StageBMidiToSoloTrainingResourceProbeError("scale-smoke val records below threshold")
    if not bool(scale_smoke.get("fits_vocab", False)):
        raise StageBMidiToSoloTrainingResourceProbeError("scale-smoke token vocab guard failed")
    if _int(scale_smoke.get("checkpoint_count")) <= 0:
        raise StageBMidiToSoloTrainingResourceProbeError("scale-smoke checkpoint missing")
    if scale_smoke.get("best_validation_loss") is None:
        raise StageBMidiToSoloTrainingResourceProbeError("scale-smoke validation loss missing")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloTrainingResourceProbeError("critical user input should not be required")
    if require_no_final_claim:
        blocked = [
            "midi_to_solo_mvp_claimed",
            "conditioned_generation_completed",
            "broad_training_executed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "musical_quality_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloTrainingResourceProbeError(f"unexpected final claim: {claimed}")
        upstream_claims = [
            bool(context.get("midi_to_solo_mvp_claimed", True)),
            bool(context.get("harmony_analysis_quality_claimed", True)),
            bool(full_window.get("broad_trained_model_quality_claimed", True)),
            bool(full_window.get("brad_style_adaptation_claimed", True)),
            bool(scale_smoke.get("broad_trained_model_quality_claimed", True)),
            bool(scale_smoke.get("brad_style_adaptation_claimed", True)),
        ]
        if any(upstream_claims):
            raise StageBMidiToSoloTrainingResourceProbeError("upstream quality claims must remain false")
    return {
        "boundary": boundary,
        "next_boundary": next_boundary,
        "midi_to_solo_training_resource_ready": bool(
            readiness.get("midi_to_solo_training_resource_ready", False)
        ),
        "conditioned_generation_probe_ready": bool(
            readiness.get("conditioned_generation_probe_ready", False)
        ),
        "context_event_count": _int(context.get("context_event_count")),
        "context_bars": _int(context.get("context_bars")),
        "full_tokenized_train_files": _int(full_window.get("tokenized_train_files")),
        "full_tokenized_val_files": _int(full_window.get("tokenized_val_files")),
        "scale_selected_train_records": _int(scale_smoke.get("selected_train_records")),
        "scale_selected_val_records": _int(scale_smoke.get("selected_val_records")),
        "scale_best_validation_loss": scale_smoke.get("best_validation_loss"),
        "scale_checkpoint_count": _int(scale_smoke.get("checkpoint_count")),
        "midi_to_solo_mvp_claimed": bool(readiness.get("midi_to_solo_mvp_claimed", True)),
        "conditioned_generation_completed": bool(
            readiness.get("conditioned_generation_completed", True)
        ),
        "broad_training_executed": bool(readiness.get("broad_training_executed", True)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    context = report["context_resource"]
    full_window = report["full_window_resource"]
    scale_smoke = report["scale_smoke_resource"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Training Resource Probe",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- training resource ready: `{_bool_token(readiness['midi_to_solo_training_resource_ready'])}`",
        f"- conditioned generation probe ready: `{_bool_token(readiness['conditioned_generation_probe_ready'])}`",
        f"- MIDI-to-solo MVP claimed: `{_bool_token(readiness['midi_to_solo_mvp_claimed'])}`",
        f"- conditioned generation completed: `{_bool_token(readiness['conditioned_generation_completed'])}`",
        "",
        "## Context Resource",
        "",
        f"- context bars: `{context['context_bars']}`",
        f"- context event count: `{context['context_event_count']}`",
        f"- inferred / carried / unknown chord bars: `{context['inferred_chord_bar_count']}` / `{context['carry_forward_chord_bar_count']}` / `{context['unknown_chord_bar_count']}`",
        f"- missing context fields: `{len(context['missing_context_fields'])}`",
        "",
        "## Stage B Window Resource",
        "",
        f"- sequence format: `{full_window['sequence_format']}`",
        f"- train / val manifest files: `{full_window['train_file_count']}` / `{full_window['val_file_count']}`",
        f"- tokenized train / val files: `{full_window['tokenized_train_files']}` / `{full_window['tokenized_val_files']}`",
        f"- max token id / vocab size: `{full_window['max_token_id']}` / `{full_window['vocab_size']}`",
        f"- fits vocab: `{_bool_token(full_window['fits_vocab'])}`",
        "",
        "## Scale-Smoke Resource",
        "",
        f"- selected train / val records: `{scale_smoke['selected_train_records']}` / `{scale_smoke['selected_val_records']}`",
        f"- best validation loss: `{scale_smoke['best_validation_loss']}`",
        f"- checkpoint count: `{scale_smoke['checkpoint_count']}`",
        f"- lora weights exists: `{_bool_token(scale_smoke['lora_weights_exists'])}`",
        "",
        "## Claim Boundary",
        "",
        f"- broad training executed: `{_bool_token(readiness['broad_training_executed'])}`",
        f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(readiness['musical_quality_claimed'])}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check MIDI-to-solo training resource readiness")
    parser.add_argument("--context_report", type=str, required=True)
    parser.add_argument("--full_window_preparation", type=str, required=True)
    parser.add_argument("--training_scale_smoke", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_training_resource_probe",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=485)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_ready", action="store_true")
    parser.add_argument("--require_no_final_claim", action="store_true")
    parser.add_argument("--min_context_events", type=int, default=128)
    parser.add_argument("--min_full_train_records", type=int, default=100000)
    parser.add_argument("--min_full_val_records", type=int, default=10000)
    parser.add_argument("--min_scale_train_records", type=int, default=64)
    parser.add_argument("--min_scale_val_records", type=int, default=16)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_resource_probe_report(
        context_report=read_json(Path(args.context_report)),
        full_window_report=read_json(Path(args.full_window_preparation)),
        scale_smoke_report=read_json(Path(args.training_scale_smoke)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_resource_probe_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_ready=bool(args.require_ready),
        require_no_final_claim=bool(args.require_no_final_claim),
        min_context_events=int(args.min_context_events),
        min_full_train_records=int(args.min_full_train_records),
        min_full_val_records=int(args.min_full_val_records),
        min_scale_train_records=int(args.min_scale_train_records),
        min_scale_val_records=int(args.min_scale_val_records),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_training_resource_probe.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_training_resource_probe_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_training_resource_probe.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
