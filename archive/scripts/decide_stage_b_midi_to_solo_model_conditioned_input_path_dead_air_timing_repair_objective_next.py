"""Select the next boundary after repaired model-conditioned audio evidence."""

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
from scripts.render_stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_audio import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
    ValueError
):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_objective_next_decision"
)
PITCH_CONTOUR_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_pitch_contour_decision"
)
CURRENT_EVIDENCE_NEXT_BOUNDARY = "stage_b_midi_to_solo_mvp_current_evidence_consolidation"
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_model_conditioned_input_path_"
    "dead_air_timing_repair_objective_next_v1"
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
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_audio_package(report: dict[str, Any], *, expected_count: int) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "dead-air timing repair audio package boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "audio package must route to objective next decision"
        )
    required_true = [
        "render_attempted",
        "technical_wav_validation",
        "dead_air_timing_repair_audio_package_completed",
    ]
    missing = [name for name in required_true if not bool(boundary.get(name, False))]
    if missing:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            f"missing audio package readiness: {missing}"
        )
    if _int(boundary.get("rendered_audio_file_count")) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "rendered audio count below expected"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(boundary, label="audio package boundary")
    files = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(files) < int(expected_count):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "rendered audio rows below expected"
        )
    return {
        "boundary": SOURCE_BOUNDARY,
        "rendered_audio_file_count": _int(boundary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(boundary.get("technical_wav_validation", False)),
        "repaired_candidate_count": _int(summary.get("repaired_candidate_count")),
        "repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "max_added_note_ratio": _float(summary.get("max_added_note_ratio")),
        "max_postprocess_removal_ratio": _float(summary.get("max_postprocess_removal_ratio")),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
        "remaining_wide_interval_risk": bool(summary.get("remaining_wide_interval_risk", False)),
        "rendered_audio_files": files[: int(expected_count)],
    }


def build_objective_next_report(
    *,
    audio_package_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    expected_count: int,
    max_interval_threshold: int,
    max_added_note_ratio_review_threshold: float,
) -> dict[str, Any]:
    source = validate_audio_package(audio_package_report, expected_count=int(expected_count))
    wide_interval_followup_required = bool(
        source["remaining_wide_interval_risk"]
        or _int(source["max_repaired_interval"]) >= int(max_interval_threshold)
    )
    added_note_ratio_review_required = bool(
        _float(source["max_added_note_ratio"]) >= float(max_added_note_ratio_review_threshold)
    )
    next_boundary = (
        PITCH_CONTOUR_NEXT_BOUNDARY
        if wide_interval_followup_required
        else CURRENT_EVIDENCE_NEXT_BOUNDARY
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "objective_summary": {
            "rendered_audio_file_count": _int(source["rendered_audio_file_count"]),
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "repaired_candidate_count": _int(source["repaired_candidate_count"]),
            "repaired_dead_air_max": _float(source["repaired_dead_air_max"]),
            "max_added_note_ratio": _float(source["max_added_note_ratio"]),
            "max_postprocess_removal_ratio": _float(source["max_postprocess_removal_ratio"]),
            "max_repaired_interval": _int(source["max_repaired_interval"]),
            "max_interval_threshold": int(max_interval_threshold),
            "remaining_wide_interval_risk": bool(source["remaining_wide_interval_risk"]),
            "wide_interval_followup_required": bool(wide_interval_followup_required),
            "added_note_ratio_review_threshold": float(max_added_note_ratio_review_threshold),
            "added_note_ratio_review_required": bool(added_note_ratio_review_required),
        },
        "selected_next_target": {
            "target": (
                "wide_interval_pitch_contour_repair"
                if wide_interval_followup_required
                else "current_evidence_consolidation"
            ),
            "next_boundary": next_boundary,
            "reason": (
                "repaired dead-air target passed, but max repaired interval still exceeds objective contour threshold"
                if wide_interval_followup_required
                else "repaired dead-air audio evidence has no objective follow-up blocker"
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "objective_next_decision_completed": True,
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "dead_air_target_supported": _float(source["repaired_dead_air_max"]) <= 0.35,
            "wide_interval_followup_required": bool(wide_interval_followup_required),
            "current_evidence_consolidation_ready": not bool(wide_interval_followup_required),
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
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective MIDI/WAV evidence selected the next repair boundary without quality claim",
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
            "Stage B MIDI-to-solo model-conditioned input path dead-air timing repair pitch contour decision"
            if wide_interval_followup_required
            else "Stage B MIDI-to-solo MVP current evidence consolidation"
        ),
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_decision: bool,
    require_wide_interval_followup: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "unexpected next boundary"
        )
    if require_objective_decision and not bool(
        readiness.get("objective_next_decision_completed", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "objective next decision completion required"
        )
    if require_wide_interval_followup and not bool(
        readiness.get("wide_interval_followup_required", False)
    ):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "wide-interval follow-up should be required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathDeadAirTimingRepairObjectiveNextError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="objective next readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "objective_next_decision_completed": bool(
            readiness.get("objective_next_decision_completed", False)
        ),
        "technical_wav_validation": bool(readiness.get("technical_wav_validation", False)),
        "dead_air_target_supported": bool(readiness.get("dead_air_target_supported", False)),
        "wide_interval_followup_required": bool(
            readiness.get("wide_interval_followup_required", False)
        ),
        "current_evidence_consolidation_ready": bool(
            readiness.get("current_evidence_consolidation_ready", False)
        ),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "repaired_dead_air_max": _float(summary.get("repaired_dead_air_max")),
        "max_added_note_ratio": _float(summary.get("max_added_note_ratio")),
        "added_note_ratio_review_required": bool(
            summary.get("added_note_ratio_review_required", False)
        ),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
        "remaining_wide_interval_risk": bool(summary.get("remaining_wide_interval_risk", False)),
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
    readiness = report["readiness"]
    decision = report["decision"]
    summary = report["objective_summary"]
    target = report["selected_next_target"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Dead-Air Timing Repair Objective Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{target['target']}`",
        f"- technical WAV validation: `{_bool_token(summary['technical_wav_validation'])}`",
        f"- rendered audio file count: `{summary['rendered_audio_file_count']}`",
        f"- repaired dead-air max: `{summary['repaired_dead_air_max']:.4f}`",
        f"- max added-note ratio: `{summary['max_added_note_ratio']:.4f}`",
        f"- added-note ratio review required: `{_bool_token(summary['added_note_ratio_review_required'])}`",
        f"- max repaired interval: `{summary['max_repaired_interval']}`",
        f"- max interval threshold: `{summary['max_interval_threshold']}`",
        f"- wide-interval follow-up required: `{_bool_token(summary['wide_interval_followup_required'])}`",
        f"- current evidence consolidation ready: `{_bool_token(readiness['current_evidence_consolidation_ready'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Decision",
        "",
        f"- reason: `{target['reason']}`",
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
        description="Select the next boundary after repaired model-conditioned audio evidence"
    )
    parser.add_argument("--audio_package_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_conditioned_input_path_"
            "dead_air_timing_repair_objective_next_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=694)
    parser.add_argument("--expected_count", type=int, default=3)
    parser.add_argument("--max_interval_threshold", type=int, default=12)
    parser.add_argument("--max_added_note_ratio_review_threshold", type=float, default=0.75)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_wide_interval_followup", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_next_report(
        audio_package_report=read_json(Path(args.audio_package_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        expected_count=int(args.expected_count),
        max_interval_threshold=int(args.max_interval_threshold),
        max_added_note_ratio_review_threshold=float(args.max_added_note_ratio_review_threshold),
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_decision=bool(args.require_objective_decision),
        require_wide_interval_followup=bool(args.require_wide_interval_followup),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_input_path_dead_air_timing_repair_objective_next_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
