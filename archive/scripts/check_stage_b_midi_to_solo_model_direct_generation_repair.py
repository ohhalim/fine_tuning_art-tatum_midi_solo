"""Define the Stage B MIDI-to-solo model-direct generation repair boundary."""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.assess_stage_b_generic_base_readiness import write_json, write_text  # noqa: E402


class StageBMidiToSoloModelDirectGenerationRepairError(ValueError):
    pass


MVP_BOUNDARY = "stage_b_midi_to_solo_mvp_execution_consolidation"
SCALE_SMOKE_BOUNDARY = "stage_b_generic_base_training_scale_smoke"
BOUNDARY = "stage_b_midi_to_solo_model_direct_generation_repair"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_generation_repair_v1"

CONTROL_PREFIX_TOKENS = 2
BAR_CHORD_TOKENS_PER_BAR = 3
NOTE_TOKENS_PER_NOTE = 4
END_TOKENS = 1
MIN_REPAIR_MAX_SEQUENCE = 160


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectGenerationRepairError(f"report missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def estimate_direct_stage_b_token_budget(
    *,
    target_bars: int,
    min_note_count: int,
    current_max_sequence: int,
) -> dict[str, Any]:
    bars = max(1, int(target_bars))
    min_notes = max(1, int(min_note_count))
    max_sequence = max(0, int(current_max_sequence))
    min_note_groups_per_bar = int(math.ceil(min_notes / bars))
    evenly_distributed_min_notes = min_note_groups_per_bar * bars
    overhead_tokens = CONTROL_PREFIX_TOKENS + BAR_CHORD_TOKENS_PER_BAR * bars + END_TOKENS
    min_contract_tokens = overhead_tokens + NOTE_TOKENS_PER_NOTE * min_notes
    even_bar_contract_tokens = overhead_tokens + NOTE_TOKENS_PER_NOTE * evenly_distributed_min_notes
    note_capacity = max(0, (max_sequence - overhead_tokens) // NOTE_TOKENS_PER_NOTE)
    even_note_groups_per_bar_capacity = note_capacity // bars
    recommended_max_sequence = max(
        MIN_REPAIR_MAX_SEQUENCE,
        int(math.ceil(max(min_contract_tokens, even_bar_contract_tokens) / 32.0) * 32),
    )
    return {
        "target_bars": bars,
        "min_note_count": min_notes,
        "min_note_groups_per_bar": min_note_groups_per_bar,
        "evenly_distributed_min_notes": evenly_distributed_min_notes,
        "token_accounting": {
            "control_prefix_tokens": CONTROL_PREFIX_TOKENS,
            "bar_chord_tokens_per_bar": BAR_CHORD_TOKENS_PER_BAR,
            "note_tokens_per_note": NOTE_TOKENS_PER_NOTE,
            "end_tokens": END_TOKENS,
            "overhead_tokens": overhead_tokens,
            "min_contract_tokens": min_contract_tokens,
            "even_bar_contract_tokens": even_bar_contract_tokens,
        },
        "current_checkpoint": {
            "max_sequence": max_sequence,
            "direct_note_capacity_under_budget": note_capacity,
            "even_note_groups_per_bar_capacity": even_note_groups_per_bar_capacity,
            "sequence_budget_sufficient_for_contract": bool(
                max_sequence >= min_contract_tokens and note_capacity >= min_notes
            ),
            "sequence_budget_sufficient_for_even_bar_distribution": bool(
                max_sequence >= even_bar_contract_tokens
                and even_note_groups_per_bar_capacity >= min_note_groups_per_bar
            ),
        },
        "repair_target": {
            "recommended_max_sequence": recommended_max_sequence,
            "target_note_groups_per_bar": min_note_groups_per_bar,
            "target_min_note_count": min_notes,
            "target_bars": bars,
        },
    }


def summarize_mvp_execution(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    execution = _dict(report.get("execution_path"))
    contract = _dict(report.get("contract"))
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != MVP_BOUNDARY:
        raise StageBMidiToSoloModelDirectGenerationRepairError("MVP execution consolidation boundary required")
    if not bool(readiness.get("midi_to_solo_technical_mvp_completed", False)):
        raise StageBMidiToSoloModelDirectGenerationRepairError("technical MVP completion required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectGenerationRepairError("MVP consolidation must route to model-direct repair")
    return {
        "boundary": boundary,
        "technical_execution_path_completed": bool(readiness.get("technical_execution_path_completed", False)),
        "midi_to_solo_technical_mvp_completed": bool(
            readiness.get("midi_to_solo_technical_mvp_completed", False)
        ),
        "input_to_ranked_midi_completed": bool(readiness.get("input_to_ranked_midi_completed", False)),
        "input_to_rendered_audio_completed": bool(readiness.get("input_to_rendered_audio_completed", False)),
        "generation_source": str(execution.get("generation_source") or ""),
        "exported_candidate_count": _int(execution.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(execution.get("rendered_audio_file_count")),
        "target_solo_bars": _int(contract.get("target_solo_bars")),
        "min_note_count": _int(contract.get("min_note_count")),
        "min_unique_pitch_count": _int(contract.get("min_unique_pitch_count")),
        "max_simultaneous_notes": _int(contract.get("max_simultaneous_notes")),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "model_checkpoint_direct_generation_quality_claimed": bool(
            readiness.get("model_checkpoint_direct_generation_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def summarize_scale_smoke(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    inputs = _dict(report.get("input"))
    training_config = _dict(report.get("training_config"))
    training = _dict(report.get("training"))
    token = _dict(report.get("token_stats"))
    artifacts = _dict(report.get("artifacts"))
    boundary = str(readiness.get("boundary") or report.get("boundary") or "")
    if boundary != SCALE_SMOKE_BOUNDARY:
        raise StageBMidiToSoloModelDirectGenerationRepairError("scale-smoke training boundary required")
    if not bool(readiness.get("training_scale_smoke_passed", False)):
        raise StageBMidiToSoloModelDirectGenerationRepairError("scale-smoke training must pass")
    if not bool(readiness.get("generic_base_scale_checkpoint_generation_probe_ready", False)):
        raise StageBMidiToSoloModelDirectGenerationRepairError("scale checkpoint generation probe readiness required")
    if _int(artifacts.get("checkpoint_count")) <= 0:
        raise StageBMidiToSoloModelDirectGenerationRepairError("scale-smoke checkpoint missing")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_train_records": _int(inputs.get("selected_train_records")),
        "selected_val_records": _int(inputs.get("selected_val_records")),
        "max_sequence": _int(training_config.get("max_sequence")),
        "epochs": _int(training_config.get("epochs")),
        "batch_size": _int(training_config.get("batch_size")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "full_generic_training_executed": bool(readiness.get("full_generic_training_executed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def build_model_direct_generation_repair_report(
    *,
    mvp_execution_report: dict[str, Any],
    training_scale_smoke: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    mvp = summarize_mvp_execution(mvp_execution_report)
    scale = summarize_scale_smoke(training_scale_smoke)
    budget = estimate_direct_stage_b_token_budget(
        target_bars=mvp["target_solo_bars"],
        min_note_count=mvp["min_note_count"],
        current_max_sequence=scale["max_sequence"],
    )
    current = _dict(budget.get("current_checkpoint"))
    token_accounting = _dict(budget.get("token_accounting"))
    budget_sufficient = bool(current.get("sequence_budget_sufficient_for_contract", False))
    direct_repair_required = not budget_sufficient
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "mvp_execution": mvp["boundary"],
            "training_scale_smoke": scale["boundary"],
        },
        "mvp_execution_summary": mvp,
        "scale_smoke_summary": scale,
        "direct_generation_contract": {
            "target_solo_bars": int(mvp["target_solo_bars"]),
            "min_note_count": int(mvp["min_note_count"]),
            "min_unique_pitch_count": int(mvp["min_unique_pitch_count"]),
            "max_simultaneous_notes": int(mvp["max_simultaneous_notes"]),
            "required_generation_source": "model_checkpoint_direct",
            "current_generation_source": str(mvp["generation_source"]),
        },
        "sequence_budget_analysis": budget,
        "repair_scope": {
            "direct_repair_required": bool(direct_repair_required),
            "primary_blocker": "scale_smoke_sequence_budget"
            if direct_repair_required
            else "sequence_budget_available_direct_probe_required",
            "current_skip_reason": (
                "scale-smoke max_sequence "
                f"{scale['max_sequence']} supports {current['direct_note_capacity_under_budget']} "
                "direct Stage B notes after 8-bar bar/chord overhead, below MVP minimum "
                f"{mvp['min_note_count']}"
            )
            if direct_repair_required
            else "current checkpoint sequence budget can enter a direct generation probe",
            "next_validation_target": NEXT_BOUNDARY,
            "recommended_max_sequence": int(_dict(budget.get("repair_target")).get("recommended_max_sequence", 0)),
            "minimum_contract_tokens": int(token_accounting.get("min_contract_tokens", 0)),
            "direct_note_capacity_under_current_budget": int(current.get("direct_note_capacity_under_budget", 0)),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "technical_mvp_preserved": bool(mvp["midi_to_solo_technical_mvp_completed"]),
            "model_direct_repair_boundary_defined": True,
            "current_checkpoint_sequence_budget_sufficient": bool(budget_sufficient),
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
                "direct Stage B checkpoint generation must first repair sequence budget against "
                "the 8-bar minimum-note MVP contract"
            ),
        },
        "not_proven": [
            "model_checkpoint_direct_8bar_generation",
            "model_checkpoint_direct_8bar_generation_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct sequence budget repair smoke",
    }


def validate_model_direct_generation_repair_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_technical_mvp: bool,
    require_sequence_budget_gap: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    mvp = _dict(report.get("mvp_execution_summary"))
    scale = _dict(report.get("scale_smoke_summary"))
    budget = _dict(report.get("sequence_budget_analysis"))
    current = _dict(budget.get("current_checkpoint"))
    token_accounting = _dict(budget.get("token_accounting"))
    repair = _dict(report.get("repair_scope"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectGenerationRepairError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectGenerationRepairError("unexpected next boundary")
    if require_technical_mvp and not bool(readiness.get("technical_mvp_preserved", False)):
        raise StageBMidiToSoloModelDirectGenerationRepairError("technical MVP must be preserved")
    if _int(scale.get("max_sequence")) <= 0:
        raise StageBMidiToSoloModelDirectGenerationRepairError("current max_sequence required")
    if _int(token_accounting.get("min_contract_tokens")) <= 0:
        raise StageBMidiToSoloModelDirectGenerationRepairError("minimum contract token estimate required")
    if require_sequence_budget_gap:
        if bool(readiness.get("current_checkpoint_sequence_budget_sufficient", True)):
            raise StageBMidiToSoloModelDirectGenerationRepairError("current checkpoint budget should remain insufficient")
        if _int(current.get("direct_note_capacity_under_budget")) >= _int(mvp.get("min_note_count")):
            raise StageBMidiToSoloModelDirectGenerationRepairError("direct note capacity should be below MVP minimum")
        if not bool(repair.get("direct_repair_required", False)):
            raise StageBMidiToSoloModelDirectGenerationRepairError("direct repair should be required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirectGenerationRepairError("critical user input should not be required")
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
            raise StageBMidiToSoloModelDirectGenerationRepairError(f"unexpected quality claim: {claimed}")
        upstream_claims = [
            bool(mvp.get("midi_to_solo_musical_quality_claimed", True)),
            bool(mvp.get("model_checkpoint_direct_generation_quality_claimed", True)),
            bool(mvp.get("human_audio_preference_claimed", True)),
            bool(mvp.get("broad_trained_model_quality_claimed", True)),
            bool(mvp.get("brad_style_adaptation_claimed", True)),
            bool(scale.get("broad_trained_model_quality_claimed", True)),
            bool(scale.get("brad_style_adaptation_claimed", True)),
        ]
        if any(upstream_claims):
            raise StageBMidiToSoloModelDirectGenerationRepairError("upstream quality claims must remain false")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "technical_mvp_preserved": bool(readiness.get("technical_mvp_preserved", False)),
        "current_generation_source": str(_dict(report.get("direct_generation_contract")).get("current_generation_source") or ""),
        "required_generation_source": str(
            _dict(report.get("direct_generation_contract")).get("required_generation_source") or ""
        ),
        "current_checkpoint_max_sequence": _int(scale.get("max_sequence")),
        "minimum_contract_tokens": _int(token_accounting.get("min_contract_tokens")),
        "direct_note_capacity_under_current_budget": _int(current.get("direct_note_capacity_under_budget")),
        "target_min_note_count": _int(mvp.get("min_note_count")),
        "current_checkpoint_sequence_budget_sufficient": bool(
            readiness.get("current_checkpoint_sequence_budget_sufficient", True)
        ),
        "direct_repair_required": bool(repair.get("direct_repair_required", False)),
        "recommended_max_sequence": _int(repair.get("recommended_max_sequence")),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    contract = report["direct_generation_contract"]
    budget = report["sequence_budget_analysis"]
    token_accounting = budget["token_accounting"]
    current = budget["current_checkpoint"]
    repair = report["repair_scope"]
    scale = report["scale_smoke_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Generation Repair",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- technical MVP preserved: `{_bool_token(readiness['technical_mvp_preserved'])}`",
        f"- current checkpoint sequence budget sufficient: `{_bool_token(readiness['current_checkpoint_sequence_budget_sufficient'])}`",
        f"- direct repair required: `{_bool_token(repair['direct_repair_required'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        "",
        "## Contract",
        "",
        f"- current generation source: `{contract['current_generation_source']}`",
        f"- required generation source: `{contract['required_generation_source']}`",
        f"- target solo bars: `{contract['target_solo_bars']}`",
        f"- min note count: `{contract['min_note_count']}`",
        f"- min unique pitch count: `{contract['min_unique_pitch_count']}`",
        f"- max simultaneous notes: `{contract['max_simultaneous_notes']}`",
        "",
        "## Sequence Budget",
        "",
        f"- current max sequence: `{scale['max_sequence']}`",
        f"- overhead tokens: `{token_accounting['overhead_tokens']}`",
        f"- minimum contract tokens: `{token_accounting['min_contract_tokens']}`",
        f"- direct note capacity under current budget: `{current['direct_note_capacity_under_budget']}`",
        f"- even note groups per bar capacity: `{current['even_note_groups_per_bar_capacity']}`",
        f"- recommended max sequence: `{repair['recommended_max_sequence']}`",
        "",
        "## Repair Scope",
        "",
        f"- primary blocker: `{repair['primary_blocker']}`",
        f"- current skip reason: `{repair['current_skip_reason']}`",
        f"- next validation target: `{repair['next_validation_target']}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Define MIDI-to-solo model-direct generation repair boundary")
    parser.add_argument("--mvp_execution", type=str, required=True)
    parser.add_argument("--training_scale_smoke", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_generation_repair",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=493)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_technical_mvp", action="store_true")
    parser.add_argument("--require_sequence_budget_gap", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_model_direct_generation_repair_report(
        mvp_execution_report=read_json(Path(args.mvp_execution)),
        training_scale_smoke=read_json(Path(args.training_scale_smoke)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_model_direct_generation_repair_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_technical_mvp=bool(args.require_technical_mvp),
        require_sequence_budget_gap=bool(args.require_sequence_budget_gap),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_generation_repair.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_generation_repair_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_generation_repair.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
