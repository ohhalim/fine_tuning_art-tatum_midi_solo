"""Decide model-conditioned input-path dead-air/timing repair target."""

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
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_objective_next import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    REPAIR_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_source_objective_next(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "objective-only next decision boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "objective-only decision must route to dead-air timing repair decision"
        )
    if not bool(readiness.get("objective_only_next_decision_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "objective-only decision completion required"
        )
    if not bool(readiness.get("dead_air_timing_repair_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "dead-air timing repair should be required"
        )
    if bool(readiness.get("current_evidence_consolidation_ready", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "current evidence consolidation should remain blocked"
        )
    if _int(summary.get("dead_air_failure_count")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "dead-air failure count required"
        )
    if not bool(summary.get("all_candidates_dead_air_failure", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "all selected candidates should fail the current dead-air threshold"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "preference fill must remain blocked"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="objective-next readiness")
    return {
        "boundary": SOURCE_BOUNDARY,
        "candidate_count": _int(summary.get("candidate_count")),
        "exported_candidate_count": _int(summary.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "dead_air_threshold": _float(summary.get("dead_air_threshold")),
        "dead_air_failure_count": _int(summary.get("dead_air_failure_count")),
        "dead_air_min": _float(summary.get("dead_air_min")),
        "dead_air_max": _float(summary.get("dead_air_max")),
        "best_note_count": _int(summary.get("best_note_count")),
        "best_unique_pitch_count": _int(summary.get("best_unique_pitch_count")),
        "validated_review_input_present": bool(summary.get("validated_review_input_present", True)),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
        "candidate_reviews": [_dict(item) for item in _list(report.get("candidate_reviews"))],
    }


def build_dead_air_timing_repair_decision_report(
    *,
    objective_next_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    target_dead_air_max: float,
    max_postprocess_removal_ratio: float,
) -> dict[str, Any]:
    source = validate_source_objective_next(objective_next_report)
    source_dead_air_max = _float(source["dead_air_max"])
    required_gain = max(0.0, source_dead_air_max - float(target_dead_air_max))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "source_objective_summary": source,
        "repair_target": {
            "selected_target": "dead_air_timing_continuity",
            "source_dead_air_failure_count": _int(source["dead_air_failure_count"]),
            "source_dead_air_min": _float(source["dead_air_min"]),
            "source_dead_air_max": source_dead_air_max,
            "target_dead_air_max": float(target_dead_air_max),
            "required_dead_air_gain_min": float(required_gain),
            "strategy": "timing_gap_fill_and_duration_compaction",
            "repair_probe_required": True,
        },
        "guardrails": {
            "min_note_count": 24,
            "min_unique_pitch_count": 8,
            "max_simultaneous_notes": 1,
            "max_postprocess_removal_ratio": float(max_postprocess_removal_ratio),
            "require_ranked_midi_export": True,
            "require_technical_wav_validation": True,
            "require_preference_fill_blocked": True,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "dead_air_timing_repair_decision_completed": True,
            "repair_probe_required": True,
            "model_conditioned_technical_path_ready": True,
            "current_evidence_consolidation_ready": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "all selected model-conditioned candidates fail the dead-air threshold; repair probe target defined",
        },
        "not_proven": [
            "dead_air_repair_success",
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair probe",
    }


def validate_dead_air_timing_repair_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_decision_completed: bool,
    require_repair_probe: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    target = _dict(report.get("repair_target"))
    guardrails = _dict(report.get("guardrails"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "unexpected next boundary"
        )
    if require_decision_completed and not bool(
        readiness.get("dead_air_timing_repair_decision_completed", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "repair decision completion required"
        )
    if require_repair_probe and not bool(target.get("repair_probe_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "repair probe should be required"
        )
    if _float(target.get("required_dead_air_gain_min")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "required dead-air gain should be positive"
        )
    if not bool(guardrails.get("require_preference_fill_blocked", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "preference fill block guardrail required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairDecisionError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="repair decision readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "dead_air_timing_repair_decision_completed": bool(
            readiness.get("dead_air_timing_repair_decision_completed", False)
        ),
        "repair_probe_required": bool(target.get("repair_probe_required", False)),
        "selected_target": str(target.get("selected_target") or ""),
        "source_dead_air_failure_count": _int(target.get("source_dead_air_failure_count")),
        "source_dead_air_max": _float(target.get("source_dead_air_max")),
        "target_dead_air_max": _float(target.get("target_dead_air_max")),
        "required_dead_air_gain_min": _float(target.get("required_dead_air_gain_min")),
        "max_postprocess_removal_ratio": _float(guardrails.get("max_postprocess_removal_ratio")),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    target = report["repair_target"]
    guardrails = report["guardrails"]
    source = report["source_objective_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{target['selected_target']}`",
        f"- repair probe required: `{_bool_token(target['repair_probe_required'])}`",
        f"- source dead-air failure count: `{target['source_dead_air_failure_count']}`",
        f"- source dead-air min / max: `{target['source_dead_air_min']:.4f} / {target['source_dead_air_max']:.4f}`",
        f"- target dead-air max: `{target['target_dead_air_max']:.4f}`",
        f"- required dead-air gain min: `{target['required_dead_air_gain_min']:.4f}`",
        f"- strategy: `{target['strategy']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Source Objective Evidence",
        "",
        f"- candidate / exported / rendered: `{source['candidate_count']} / {source['exported_candidate_count']} / {source['rendered_audio_file_count']}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- validated review input present: `{_bool_token(source['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(source['preference_fill_allowed'])}`",
        "",
        "## Guardrails",
        "",
        f"- min note count: `{guardrails['min_note_count']}`",
        f"- min unique pitch count: `{guardrails['min_unique_pitch_count']}`",
        f"- max simultaneous notes: `{guardrails['max_simultaneous_notes']}`",
        f"- max postprocess removal ratio: `{guardrails['max_postprocess_removal_ratio']:.4f}`",
        f"- require ranked MIDI export: `{_bool_token(guardrails['require_ranked_midi_export'])}`",
        f"- require technical WAV validation: `{_bool_token(guardrails['require_technical_wav_validation'])}`",
        f"- require preference fill blocked: `{_bool_token(guardrails['require_preference_fill_blocked'])}`",
        "",
        "## Decision",
        "",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- reason: `{decision['reason']}`",
        f"- next recommended issue: `{report['next_recommended_issue']}`",
        "",
        "## Claim Boundary",
        "",
    ]
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide model-conditioned input-path dead-air timing repair target"
    )
    parser.add_argument("--objective_next_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=688)
    parser.add_argument("--target_dead_air_max", type=float, default=0.35)
    parser.add_argument("--max_postprocess_removal_ratio", type=float, default=0.25)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_decision_completed", action="store_true")
    parser.add_argument("--require_repair_probe", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_dead_air_timing_repair_decision_report(
        objective_next_report=read_json(Path(args.objective_next_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        target_dead_air_max=float(args.target_dead_air_max),
        max_postprocess_removal_ratio=float(args.max_postprocess_removal_ratio),
    )
    summary = validate_dead_air_timing_repair_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_decision_completed=bool(args.require_decision_completed),
        require_repair_probe=bool(args.require_repair_probe),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
