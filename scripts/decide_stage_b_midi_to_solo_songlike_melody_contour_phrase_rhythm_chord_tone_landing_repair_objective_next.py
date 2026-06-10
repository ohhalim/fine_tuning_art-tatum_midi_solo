"""Select the next boundary after chord-tone landing repair input guard evidence."""

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
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloChordToneLandingRepairObjectiveNextError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_only_next_decision"
)
FOLLOWUP_DECISION_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision"
)
SELECTED_TARGET = "songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_followup_decision"
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_next_v1"
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
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _dict(guard.get("source_summary"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "chord-tone landing repair listening review input guard boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "input guard must route to chord-tone landing objective-only next decision"
        )
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "input guard completion required"
        )
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "objective-only decision requires pending review input"
        )
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if not bool(source.get("technical_wav_validation", False)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "technical WAV validation required"
        )
    if _int(source.get("rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "rendered WAV count below 6"
        )
    if _int(source.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        source.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "outside-soloing objective and source counts must match"
        )
    if bool(source.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "outside-soloing repair target should remain false in objective next"
        )
    if not bool(source.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "outside-soloing residual risk context must be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="chord-tone landing input guard readiness")
    return {
        "boundary": SOURCE_BOUNDARY,
        "review_item_count": _int(guard.get("review_item_count")),
        "required_input_field_count": _int(guard.get("required_input_field_count")),
        "validated_review_input_present": bool(
            guard.get("validated_review_input_present", False)
        ),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", False)),
        "technical_wav_validation": bool(source.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(source.get("rendered_audio_file_count")),
        "sample_rate": _int(source.get("sample_rate")),
        "duration_min_seconds": _float(source.get("duration_min_seconds")),
        "duration_max_seconds": _float(source.get("duration_max_seconds")),
        "changed_note_total": _int(source.get("changed_note_total")),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            source.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "weak_chord_tone_landing_risk_delta": _int(
            source.get("weak_chord_tone_landing_risk_delta")
        ),
        "outside_soloing_pitch_role_risk_count_before": _int(
            source.get("outside_soloing_pitch_role_risk_count_before")
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            source.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            source.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            source.get("outside_soloing_repair_targeted", True)
        ),
        "outside_soloing_residual_risk_preserved": bool(
            source.get("outside_soloing_residual_risk_preserved", False)
        ),
        "final_landing_chord_tone_count_after": _int(
            source.get("final_landing_chord_tone_count_after")
        ),
        "audio_review_required": bool(source.get("audio_review_required", False)),
    }


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_input_guard_report(input_guard_report)
    residual_outside_risk = _int(source["outside_soloing_pitch_role_risk_count_after"])
    followup_required = bool(residual_outside_risk > 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "objective_summary": {
            "review_item_count": _int(source["review_item_count"]),
            "required_input_field_count": _int(source["required_input_field_count"]),
            "validated_review_input_present": bool(
                source["validated_review_input_present"]
            ),
            "preference_fill_allowed": bool(source["preference_fill_allowed"]),
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "rendered_audio_file_count": _int(source["rendered_audio_file_count"]),
            "sample_rate": _int(source["sample_rate"]),
            "duration_min_seconds": _float(source["duration_min_seconds"]),
            "duration_max_seconds": _float(source["duration_max_seconds"]),
            "changed_note_total": _int(source["changed_note_total"]),
            "objective_outside_soloing_pitch_role_risk_count": _int(
                source["objective_outside_soloing_pitch_role_risk_count"]
            ),
            "weak_chord_tone_landing_risk_delta": _int(
                source["weak_chord_tone_landing_risk_delta"]
            ),
            "outside_soloing_pitch_role_risk_count_before": _int(
                source["outside_soloing_pitch_role_risk_count_before"]
            ),
            "outside_soloing_pitch_role_risk_count_after": residual_outside_risk,
            "outside_soloing_pitch_role_risk_delta": _int(
                source["outside_soloing_pitch_role_risk_delta"]
            ),
            "outside_soloing_repair_targeted": bool(
                source["outside_soloing_repair_targeted"]
            ),
            "outside_soloing_residual_risk_preserved": bool(
                source["outside_soloing_residual_risk_preserved"]
            ),
            "final_landing_chord_tone_count_after": _int(
                source["final_landing_chord_tone_count_after"]
            ),
            "audio_review_required": bool(source["audio_review_required"]),
            "chord_tone_landing_followup_required": bool(followup_required),
            "current_quality_claim_ready": False,
        },
        "selected_next_target": {
            "target": SELECTED_TARGET,
            "next_boundary": FOLLOWUP_DECISION_NEXT_BOUNDARY,
            "reason": (
                "listening preference pending and residual outside-soloing pitch-role risk remains"
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "objective_next_decision_completed": True,
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "objective_outside_soloing_pitch_role_risk_count": _int(
                source["objective_outside_soloing_pitch_role_risk_count"]
            ),
            "outside_soloing_pitch_role_risk_count_before": _int(
                source["outside_soloing_pitch_role_risk_count_before"]
            ),
            "outside_soloing_pitch_role_risk_count_after": residual_outside_risk,
            "outside_soloing_repair_targeted": bool(
                source["outside_soloing_repair_targeted"]
            ),
            "outside_soloing_residual_risk_preserved": bool(
                source["outside_soloing_residual_risk_preserved"]
            ),
            "chord_tone_landing_followup_required": bool(followup_required),
            "current_quality_claim_ready": False,
            "human_audio_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": FOLLOWUP_DECISION_NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "objective-only evidence selected chord-tone landing follow-up decision without quality claim",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair follow-up decision"
        ),
    }


def validate_objective_next_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_objective_decision: bool,
    require_followup_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("objective_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "unexpected next boundary"
        )
    if require_objective_decision and not bool(
        readiness.get("objective_next_decision_completed", False)
    ):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "objective next decision completion required"
        )
    if require_followup_required and not bool(
        readiness.get("chord_tone_landing_followup_required", False)
    ):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "chord-tone landing follow-up requirement expected"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if bool(summary.get("current_quality_claim_ready", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "quality claim readiness must remain false"
        )
    if _int(summary.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        summary.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "outside-soloing objective and source counts must match"
        )
    if bool(summary.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "outside-soloing repair target should remain false in objective next"
        )
    if not bool(summary.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "outside-soloing residual risk context must be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairObjectiveNextError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="chord-tone landing objective next readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(_dict(report.get("selected_next_target")).get("target") or ""),
        "objective_next_decision_completed": bool(
            readiness.get("objective_next_decision_completed", False)
        ),
        "review_item_count": _int(summary.get("review_item_count")),
        "required_input_field_count": _int(summary.get("required_input_field_count")),
        "validated_review_input_present": bool(
            summary.get("validated_review_input_present", True)
        ),
        "preference_fill_allowed": bool(summary.get("preference_fill_allowed", True)),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "changed_note_total": _int(summary.get("changed_note_total")),
        "objective_outside_soloing_pitch_role_risk_count": _int(
            summary.get("objective_outside_soloing_pitch_role_risk_count")
        ),
        "weak_chord_tone_landing_risk_delta": _int(
            summary.get("weak_chord_tone_landing_risk_delta")
        ),
        "outside_soloing_pitch_role_risk_count_before": _int(
            summary.get("outside_soloing_pitch_role_risk_count_before")
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            summary.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            summary.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            summary.get("outside_soloing_repair_targeted", True)
        ),
        "outside_soloing_residual_risk_preserved": bool(
            summary.get("outside_soloing_residual_risk_preserved", False)
        ),
        "final_landing_chord_tone_count_after": _int(
            summary.get("final_landing_chord_tone_count_after")
        ),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
        "chord_tone_landing_followup_required": bool(
            summary.get("chord_tone_landing_followup_required", False)
        ),
        "current_quality_claim_ready": bool(summary.get("current_quality_claim_ready", True)),
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
        "# Stage B MIDI-to-Solo Chord-Tone Landing Repair Objective-Only Next Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{target['target']}`",
        f"- review item count: `{summary['review_item_count']}`",
        f"- required input field count: `{summary['required_input_field_count']}`",
        f"- validated review input present: `{_bool_token(summary['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(summary['preference_fill_allowed'])}`",
        f"- technical WAV validation: `{_bool_token(summary['technical_wav_validation'])}`",
        f"- rendered audio file count: `{summary['rendered_audio_file_count']}`",
        f"- changed note total: `{summary['changed_note_total']}`",
        f"- objective outside-soloing pitch-role risk count: `{summary['objective_outside_soloing_pitch_role_risk_count']}`",
        f"- weak chord-tone landing risk delta: `{summary['weak_chord_tone_landing_risk_delta']}`",
        f"- outside-soloing pitch-role risk count: `{summary['outside_soloing_pitch_role_risk_count_before']} -> {summary['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing repair targeted: `{_bool_token(summary['outside_soloing_repair_targeted'])}`",
        f"- outside-soloing residual risk preserved: `{_bool_token(summary['outside_soloing_residual_risk_preserved'])}`",
        f"- final landing chord-tone count after: `{summary['final_landing_chord_tone_count_after']}`",
        f"- chord-tone landing follow-up required: `{_bool_token(summary['chord_tone_landing_followup_required'])}`",
        f"- current quality claim ready: `{_bool_token(summary['current_quality_claim_ready'])}`",
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
        description="Decide chord-tone landing repair objective-only next step"
    )
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_objective_only_next_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=882)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_objective_decision", action="store_true")
    parser.add_argument("--require_followup_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_objective_next_report(
        input_guard_report=read_json(Path(args.input_guard_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_objective_next_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_objective_decision=bool(args.require_objective_decision),
        require_followup_required=bool(args.require_followup_required),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_objective_only_next_decision.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_objective_only_next_decision_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_objective_only_next_decision.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
