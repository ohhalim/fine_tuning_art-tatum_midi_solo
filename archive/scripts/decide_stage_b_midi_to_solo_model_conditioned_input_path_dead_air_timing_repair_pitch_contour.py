"""Define the pitch-contour repair target after dead-air timing repair evidence."""

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
from scripts.decide_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    PITCH_CONTOUR_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    validate_objective_next_report,
)


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_decision"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_probe"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_decision_v1"
)

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "audio_rendered_quality_claimed",
    "model_checkpoint_generation_quality_claimed",
    "model_direct_generation_quality_claimed",
    "broad_trained_model_quality_claimed",
    "brad_style_adaptation_claimed",
    "production_ready_claimed",
]


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


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_objective_next_source(
    report: dict[str, Any],
    *,
    target_max_interval: int,
) -> dict[str, Any]:
    try:
        summary = validate_objective_next_report(
            report,
            expected_boundary=SOURCE_BOUNDARY,
            expected_next_boundary=SOURCE_NEXT_BOUNDARY,
            require_objective_decision=True,
            require_wide_interval_followup=True,
            require_no_quality_claim=True,
        )
    except ValueError as exc:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            str(exc)
        ) from exc
    if not bool(summary["technical_wav_validation"]):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "technical WAV validation required"
        )
    if not bool(summary["dead_air_target_supported"]):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "dead-air target support required before pitch-contour decision"
        )
    if bool(summary["current_evidence_consolidation_ready"]):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "current evidence consolidation should remain blocked"
        )
    if _int(summary["max_repaired_interval"]) <= int(target_max_interval):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "pitch-contour repair target requires max interval above threshold"
        )
    return summary


def build_pitch_contour_decision_report(
    *,
    objective_next_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    target_max_interval: int,
    target_dead_air_max: float,
    min_repaired_candidate_count: int,
    max_simultaneous_notes: int,
    max_added_note_ratio_review_threshold: float,
) -> dict[str, Any]:
    source = validate_objective_next_source(
        objective_next_report,
        target_max_interval=int(target_max_interval),
    )
    source_max_interval = _int(source["max_repaired_interval"])
    required_interval_reduction_min = max(0, source_max_interval - int(target_max_interval))
    source_dead_air_max = _float(source["repaired_dead_air_max"])
    source_added_note_ratio = _float(source["max_added_note_ratio"])
    added_note_ratio_review_required = source_added_note_ratio >= float(
        max_added_note_ratio_review_threshold
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_objective_summary": {
            "rendered_audio_file_count": _int(source["rendered_audio_file_count"]),
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "dead_air_target_supported": bool(source["dead_air_target_supported"]),
            "repaired_dead_air_max": source_dead_air_max,
            "max_added_note_ratio": source_added_note_ratio,
            "added_note_ratio_review_required": bool(added_note_ratio_review_required),
            "max_repaired_interval": source_max_interval,
            "remaining_wide_interval_risk": bool(source["remaining_wide_interval_risk"]),
            "wide_interval_followup_required": bool(source["wide_interval_followup_required"]),
            "current_evidence_consolidation_ready": bool(
                source["current_evidence_consolidation_ready"]
            ),
        },
        "selected_repair_target": {
            "target": "wide_interval_pitch_contour_repair",
            "primary_metric": "max_repaired_interval",
            "source_max_interval": source_max_interval,
            "target_max_interval": int(target_max_interval),
            "required_interval_reduction_min": int(required_interval_reduction_min),
            "repair_probe_boundary": NEXT_BOUNDARY,
            "reason": "max repaired interval remains above objective contour threshold after dead-air timing repair",
        },
        "repair_guardrails": {
            "preserve_dead_air_target": True,
            "source_repaired_dead_air_max": source_dead_air_max,
            "target_dead_air_max": float(target_dead_air_max),
            "min_repaired_candidate_count": int(min_repaired_candidate_count),
            "max_simultaneous_notes": int(max_simultaneous_notes),
            "keep_note_count_and_unique_pitch_review": True,
            "max_added_note_ratio_review_threshold": float(
                max_added_note_ratio_review_threshold
            ),
            "source_max_added_note_ratio": source_added_note_ratio,
            "added_note_ratio_review_required": bool(added_note_ratio_review_required),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "pitch_contour_decision_completed": True,
            "repair_probe_required": True,
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "dead_air_target_supported": bool(source["dead_air_target_supported"]),
            "wide_interval_followup_required": bool(source["wide_interval_followup_required"]),
            "current_evidence_consolidation_ready": False,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
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
            "reason": "wide-interval pitch-contour repair target selected from objective evidence",
        },
        "not_proven": [
            "human_audio_preference",
            "audio_rendered_quality",
            "midi_to_solo_musical_quality",
            "model_checkpoint_generation_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo model-conditioned input path "
            "dead-air timing repair pitch contour probe"
        ),
    }


def validate_pitch_contour_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_pitch_contour_decision: bool,
    require_repair_probe: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    source = _dict(report.get("source_objective_summary"))
    target = _dict(report.get("selected_repair_target"))
    guardrails = _dict(report.get("repair_guardrails"))
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "unexpected next boundary"
        )
    if require_pitch_contour_decision and not bool(
        readiness.get("pitch_contour_decision_completed", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "pitch-contour decision completion required"
        )
    if require_repair_probe and not bool(readiness.get("repair_probe_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "repair probe should be required"
        )
    if _int(target.get("source_max_interval")) <= _int(target.get("target_max_interval")):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "source max interval must exceed target max interval"
        )
    if _int(target.get("required_interval_reduction_min")) <= 0:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "positive interval reduction target required"
        )
    if not bool(guardrails.get("preserve_dead_air_target", False)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "dead-air guardrail required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairPitchContourError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="pitch-contour decision readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "pitch_contour_decision_completed": bool(
            readiness.get("pitch_contour_decision_completed", False)
        ),
        "repair_probe_required": bool(readiness.get("repair_probe_required", False)),
        "technical_wav_validation": bool(source.get("technical_wav_validation", False)),
        "dead_air_target_supported": bool(source.get("dead_air_target_supported", False)),
        "source_repaired_dead_air_max": _float(source.get("repaired_dead_air_max")),
        "target_dead_air_max": _float(guardrails.get("target_dead_air_max")),
        "source_max_added_note_ratio": _float(guardrails.get("source_max_added_note_ratio")),
        "added_note_ratio_review_required": bool(
            guardrails.get("added_note_ratio_review_required", False)
        ),
        "source_max_interval": _int(target.get("source_max_interval")),
        "target_max_interval": _int(target.get("target_max_interval")),
        "required_interval_reduction_min": _int(
            target.get("required_interval_reduction_min")
        ),
        "current_evidence_consolidation_ready": bool(
            readiness.get("current_evidence_consolidation_ready", True)
        ),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["source_objective_summary"]
    target = report["selected_repair_target"]
    guardrails = report["repair_guardrails"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Pitch Contour Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{target['target']}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- dead-air target supported: `{_bool_token(source['dead_air_target_supported'])}`",
        f"- source repaired dead-air max: `{source['repaired_dead_air_max']:.4f}`",
        f"- target dead-air max: `{guardrails['target_dead_air_max']:.4f}`",
        f"- source max added-note ratio: `{guardrails['source_max_added_note_ratio']:.4f}`",
        f"- added-note ratio review required: `{_bool_token(guardrails['added_note_ratio_review_required'])}`",
        f"- source max interval: `{target['source_max_interval']}`",
        f"- target max interval: `{target['target_max_interval']}`",
        f"- required interval reduction min: `{target['required_interval_reduction_min']}`",
        f"- repair probe required: `{_bool_token(readiness['repair_probe_required'])}`",
        f"- current evidence consolidation ready: `{_bool_token(readiness['current_evidence_consolidation_ready'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Guardrails",
        "",
        f"- preserve dead-air target: `{_bool_token(guardrails['preserve_dead_air_target'])}`",
        f"- min repaired candidate count: `{guardrails['min_repaired_candidate_count']}`",
        f"- max simultaneous notes: `{guardrails['max_simultaneous_notes']}`",
        f"- keep note count and unique pitch review: `{_bool_token(guardrails['keep_note_count_and_unique_pitch_review'])}`",
        "",
        "## Decision",
        "",
        f"- reason: `{decision['reason']}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
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
        description="Define the pitch-contour repair target after dead-air timing repair evidence"
    )
    parser.add_argument("--objective_next_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_pitch_contour_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=696)
    parser.add_argument("--target_max_interval", type=int, default=12)
    parser.add_argument("--target_dead_air_max", type=float, default=0.35)
    parser.add_argument("--min_repaired_candidate_count", type=int, default=3)
    parser.add_argument("--max_simultaneous_notes", type=int, default=1)
    parser.add_argument("--max_added_note_ratio_review_threshold", type=float, default=0.75)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_pitch_contour_decision", action="store_true")
    parser.add_argument("--require_repair_probe", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_pitch_contour_decision_report(
        objective_next_report=read_json(Path(args.objective_next_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        target_max_interval=int(args.target_max_interval),
        target_dead_air_max=float(args.target_dead_air_max),
        min_repaired_candidate_count=int(args.min_repaired_candidate_count),
        max_simultaneous_notes=int(args.max_simultaneous_notes),
        max_added_note_ratio_review_threshold=float(
            args.max_added_note_ratio_review_threshold
        ),
    )
    summary = validate_pitch_contour_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_pitch_contour_decision=bool(args.require_pitch_contour_decision),
        require_repair_probe=bool(args.require_repair_probe),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_pitch_contour_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
