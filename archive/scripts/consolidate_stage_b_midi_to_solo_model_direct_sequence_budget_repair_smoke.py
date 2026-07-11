"""Consolidate the Stage B MIDI-to-solo model-direct sequence-budget repair smoke."""

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
from scripts.check_stage_b_midi_to_solo_model_direct_generation_repair import (  # noqa: E402
    BOUNDARY as PREVIOUS_BOUNDARY,
)
from scripts.check_stage_b_midi_to_solo_model_direct_generation_repair import (  # noqa: E402
    NEXT_BOUNDARY as BOUNDARY,
)
from scripts.check_stage_b_midi_to_solo_model_direct_generation_repair import (  # noqa: E402
    SCALE_SMOKE_BOUNDARY,
    estimate_direct_stage_b_token_budget,
)


class StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError(ValueError):
    pass


NEXT_BOUNDARY = "stage_b_midi_to_solo_model_direct_8bar_generation_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke_v1"


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError(f"report missing: {path}")
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


def summarize_previous_repair(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    contract = _dict(report.get("direct_generation_contract"))
    repair_scope = _dict(report.get("repair_scope"))
    sequence = _dict(report.get("sequence_budget_analysis"))
    current = _dict(sequence.get("current_checkpoint"))
    token_accounting = _dict(sequence.get("token_accounting"))
    boundary = str(report.get("boundary") or "")
    if boundary != PREVIOUS_BOUNDARY:
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("previous repair boundary required")
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("previous repair must route to this smoke")
    if not bool(readiness.get("technical_mvp_preserved", False)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("technical MVP preservation required")
    if not bool(repair_scope.get("direct_repair_required", False)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("previous repair must require direct repair")
    if bool(readiness.get("current_checkpoint_sequence_budget_sufficient", True)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("previous checkpoint should be insufficient")
    return {
        "boundary": boundary,
        "target_solo_bars": _int(contract.get("target_solo_bars")),
        "min_note_count": _int(contract.get("min_note_count")),
        "min_unique_pitch_count": _int(contract.get("min_unique_pitch_count")),
        "max_simultaneous_notes": _int(contract.get("max_simultaneous_notes")),
        "current_generation_source": str(contract.get("current_generation_source") or ""),
        "required_generation_source": str(contract.get("required_generation_source") or ""),
        "previous_max_sequence": _int(current.get("max_sequence")),
        "previous_minimum_contract_tokens": _int(token_accounting.get("min_contract_tokens")),
        "previous_direct_note_capacity": _int(current.get("direct_note_capacity_under_budget")),
        "previous_recommended_max_sequence": _int(repair_scope.get("recommended_max_sequence")),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def summarize_repaired_scale_smoke(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    inputs = _dict(report.get("input"))
    training_config = _dict(report.get("training_config"))
    token = _dict(report.get("token_stats"))
    training = _dict(report.get("training"))
    artifacts = _dict(report.get("artifacts"))
    boundary = str(readiness.get("boundary") or report.get("boundary") or "")
    if boundary != SCALE_SMOKE_BOUNDARY:
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("scale-smoke boundary required")
    if not bool(readiness.get("training_scale_smoke_passed", False)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("repaired scale-smoke must pass")
    if not bool(readiness.get("generic_base_scale_checkpoint_generation_probe_ready", False)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("generation probe readiness required")
    if _int(artifacts.get("checkpoint_count")) <= 0:
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("checkpoint artifact required")
    if not bool(artifacts.get("lora_weights_exists", False)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("lora weights required")
    return {
        "boundary": boundary,
        "selected_train_records": _int(inputs.get("selected_train_records")),
        "selected_val_records": _int(inputs.get("selected_val_records")),
        "max_sequence": _int(training_config.get("max_sequence")),
        "epochs": _int(training_config.get("epochs")),
        "batch_size": _int(training_config.get("batch_size")),
        "best_validation_loss": _float(training.get("best_validation_loss")),
        "training_returncode": _int(training.get("returncode")),
        "checkpoint_count": _int(artifacts.get("checkpoint_count")),
        "lora_weights_exists": bool(artifacts.get("lora_weights_exists", False)),
        "checkpoint_files": artifacts.get("checkpoint_files") if isinstance(artifacts.get("checkpoint_files"), list) else [],
        "max_token_id": _int(token.get("max_token_id")),
        "vocab_size": _int(token.get("vocab_size")),
        "fits_vocab": bool(token.get("fits_vocab", False)),
        "full_generic_training_executed": bool(readiness.get("full_generic_training_executed", True)),
        "broad_trained_model_quality_claimed": bool(
            readiness.get("broad_trained_model_quality_claimed", True)
        ),
        "brad_style_adaptation_claimed": bool(readiness.get("brad_style_adaptation_claimed", True)),
    }


def build_sequence_budget_repair_smoke_report(
    *,
    previous_repair: dict[str, Any],
    repaired_training_scale_smoke: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    previous = summarize_previous_repair(previous_repair)
    repaired = summarize_repaired_scale_smoke(repaired_training_scale_smoke)
    budget = estimate_direct_stage_b_token_budget(
        target_bars=previous["target_solo_bars"],
        min_note_count=previous["min_note_count"],
        current_max_sequence=repaired["max_sequence"],
    )
    current = _dict(budget.get("current_checkpoint"))
    budget_sufficient = bool(current.get("sequence_budget_sufficient_for_contract", False))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "previous_repair": previous["boundary"],
            "repaired_training_scale_smoke": repaired["boundary"],
        },
        "previous_repair_summary": previous,
        "repaired_scale_smoke_summary": repaired,
        "sequence_budget_analysis": budget,
        "repair_result": {
            "previous_max_sequence": previous["previous_max_sequence"],
            "repaired_max_sequence": repaired["max_sequence"],
            "previous_direct_note_capacity": previous["previous_direct_note_capacity"],
            "repaired_direct_note_capacity": _int(current.get("direct_note_capacity_under_budget")),
            "target_min_note_count": previous["min_note_count"],
            "minimum_contract_tokens": _int(_dict(budget.get("token_accounting")).get("min_contract_tokens")),
            "sequence_budget_repaired": bool(budget_sufficient),
            "model_direct_8bar_generation_probe_ready": bool(
                budget_sufficient
                and repaired["training_returncode"] == 0
                and repaired["checkpoint_count"] > 0
                and repaired["fits_vocab"]
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "sequence_budget_repair_smoked": True,
            "repaired_checkpoint_sequence_budget_sufficient": bool(budget_sufficient),
            "model_direct_8bar_generation_probe_ready": bool(
                budget_sufficient
                and repaired["training_returncode"] == 0
                and repaired["checkpoint_count"] > 0
                and repaired["fits_vocab"]
            ),
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
                "max_sequence 160 smoke checkpoint satisfies the minimum 8-bar / 24-note "
                "Stage B token budget; next step is direct 8-bar generation probe"
            ),
        },
        "not_proven": [
            "model_checkpoint_direct_8bar_generated_midi",
            "model_checkpoint_direct_8bar_generation_quality",
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-direct 8-bar generation probe",
    }


def validate_sequence_budget_repair_smoke_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_sequence_budget_sufficient: bool,
    require_probe_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    previous = _dict(report.get("previous_repair_summary"))
    repaired = _dict(report.get("repaired_scale_smoke_summary"))
    result = _dict(report.get("repair_result"))
    budget = _dict(report.get("sequence_budget_analysis"))
    current = _dict(budget.get("current_checkpoint"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("unexpected next boundary")
    if require_sequence_budget_sufficient and not bool(
        readiness.get("repaired_checkpoint_sequence_budget_sufficient", False)
    ):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("repaired checkpoint budget must be sufficient")
    if _int(result.get("repaired_direct_note_capacity")) < _int(result.get("target_min_note_count")):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("repaired direct note capacity below target")
    if _int(repaired.get("max_sequence")) < _int(result.get("minimum_contract_tokens")):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("repaired max_sequence below minimum contract")
    if require_probe_ready and not bool(readiness.get("model_direct_8bar_generation_probe_ready", False)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("direct 8-bar generation probe should be ready")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError("critical user input should not be required")
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
            raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError(f"unexpected quality claim: {claimed}")
        upstream_claims = [
            bool(previous.get("model_direct_generation_quality_claimed", True)),
            bool(previous.get("midi_to_solo_musical_quality_claimed", True)),
            bool(previous.get("human_audio_preference_claimed", True)),
            bool(previous.get("broad_trained_model_quality_claimed", True)),
            bool(previous.get("brad_style_adaptation_claimed", True)),
            bool(repaired.get("broad_trained_model_quality_claimed", True)),
            bool(repaired.get("brad_style_adaptation_claimed", True)),
        ]
        if any(upstream_claims):
            raise StageBMidiToSoloModelDirectSequenceBudgetRepairSmokeError(
                "upstream quality claims must remain false"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "previous_max_sequence": _int(result.get("previous_max_sequence")),
        "repaired_max_sequence": _int(result.get("repaired_max_sequence")),
        "minimum_contract_tokens": _int(result.get("minimum_contract_tokens")),
        "previous_direct_note_capacity": _int(result.get("previous_direct_note_capacity")),
        "repaired_direct_note_capacity": _int(result.get("repaired_direct_note_capacity")),
        "target_min_note_count": _int(result.get("target_min_note_count")),
        "sequence_budget_repaired": bool(result.get("sequence_budget_repaired", False)),
        "repaired_checkpoint_sequence_budget_sufficient": bool(
            readiness.get("repaired_checkpoint_sequence_budget_sufficient", False)
        ),
        "model_direct_8bar_generation_probe_ready": bool(
            readiness.get("model_direct_8bar_generation_probe_ready", False)
        ),
        "even_note_groups_per_bar_capacity": _int(current.get("even_note_groups_per_bar_capacity")),
        "best_validation_loss": repaired.get("best_validation_loss"),
        "checkpoint_count": _int(repaired.get("checkpoint_count")),
        "model_direct_generation_quality_claimed": bool(
            readiness.get("model_direct_generation_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    result = report["repair_result"]
    repaired = report["repaired_scale_smoke_summary"]
    budget = report["sequence_budget_analysis"]
    current = budget["current_checkpoint"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Direct Sequence Budget Repair Smoke",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- sequence budget repaired: `{_bool_token(result['sequence_budget_repaired'])}`",
        f"- model-direct 8-bar generation probe ready: `{_bool_token(readiness['model_direct_8bar_generation_probe_ready'])}`",
        f"- model-direct generation quality claimed: `{_bool_token(readiness['model_direct_generation_quality_claimed'])}`",
        "",
        "## Sequence Budget",
        "",
        f"- previous max sequence: `{result['previous_max_sequence']}`",
        f"- repaired max sequence: `{result['repaired_max_sequence']}`",
        f"- minimum contract tokens: `{result['minimum_contract_tokens']}`",
        f"- previous direct note capacity: `{result['previous_direct_note_capacity']}`",
        f"- repaired direct note capacity: `{result['repaired_direct_note_capacity']}`",
        f"- target min note count: `{result['target_min_note_count']}`",
        f"- even note groups per bar capacity: `{current['even_note_groups_per_bar_capacity']}`",
        "",
        "## Training Smoke",
        "",
        f"- selected train / val records: `{repaired['selected_train_records']}` / `{repaired['selected_val_records']}`",
        f"- best validation loss: `{repaired['best_validation_loss']}`",
        f"- checkpoint count: `{repaired['checkpoint_count']}`",
        f"- fits vocab: `{_bool_token(repaired['fits_vocab'])}`",
        "",
        "## Not Proven",
        "",
    ]
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate MIDI-to-solo model-direct sequence budget repair smoke")
    parser.add_argument("--previous_repair", type=str, required=True)
    parser.add_argument("--repaired_training_scale_smoke", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=495)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_sequence_budget_sufficient", action="store_true")
    parser.add_argument("--require_probe_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_sequence_budget_repair_smoke_report(
        previous_repair=read_json(Path(args.previous_repair)),
        repaired_training_scale_smoke=read_json(Path(args.repaired_training_scale_smoke)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_sequence_budget_repair_smoke_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_sequence_budget_sufficient=bool(args.require_sequence_budget_sufficient),
        require_probe_ready=bool(args.require_probe_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_direct_sequence_budget_repair_smoke.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
