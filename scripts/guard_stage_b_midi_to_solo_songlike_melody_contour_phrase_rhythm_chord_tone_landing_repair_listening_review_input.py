"""Guard chord-tone landing repair listening review preference fill against missing input."""

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
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_package import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input_guard"
)
FILL_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_fill"
)
OBJECTIVE_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_objective_only_next_decision"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_repair_listening_review_input_guard_v1"
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
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_source_package(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    package = _dict(report.get("review_package"))
    source = _dict(report.get("source_summary"))
    review_items = [_dict(item) for item in _list(report.get("review_items"))]
    boundary = str(report.get("boundary") or readiness.get("boundary") or "")
    if boundary != SOURCE_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "chord-tone landing repair listening review package boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "listening package must route to input guard"
        )
    if not bool(readiness.get("listening_review_package_ready", False)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "listening review package readiness required"
        )
    if not bool(package.get("package_ready", False)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "review package ready flag required"
        )
    review_item_count = _int(readiness.get("review_item_count") or package.get("review_item_count"))
    if review_item_count <= 0 or len(review_items) < review_item_count:
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "review items required"
        )
    if not bool(source.get("technical_wav_validation", False)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "technical WAV validation required"
        )
    if _int(source.get("rendered_audio_file_count")) < review_item_count:
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "rendered audio count below review item count"
        )
    if not bool(source.get("audio_review_required", False)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "audio review requirement should remain recorded"
        )
    if _int(source.get("objective_outside_soloing_pitch_role_risk_count")) != _int(
        source.get("outside_soloing_pitch_role_risk_count_before")
    ):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "outside-soloing objective and source counts must match"
        )
    if bool(source.get("outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "outside-soloing repair target should remain false in input guard"
        )
    if not bool(source.get("outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "outside-soloing residual risk context must be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="chord-tone landing repair listening package readiness")
    return {
        "boundary": SOURCE_BOUNDARY,
        "review_item_count": review_item_count,
        "validated_review_input": bool(readiness.get("validated_review_input", False)),
        "human_review_required_now": bool(readiness.get("human_review_required_now", False)),
        "required_input_fields": [
            str(item) for item in _list(package.get("required_input_fields"))
        ],
        "wav_paths": [str(item.get("wav_path") or "") for item in review_items],
        "source_summary": {
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
        },
    }


def build_listening_review_input_guard_report(
    source_package: dict[str, Any],
    *,
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_source_package(source_package)
    validated_input = bool(source["validated_review_input"])
    next_boundary = FILL_BOUNDARY if validated_input else OBJECTIVE_NEXT_BOUNDARY
    reason = (
        "validated listening review input present; preference fill allowed"
        if validated_input
        else "listening review input pending; preference fill blocked"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "source_package_summary": source,
        "guard_result": {
            "validated_review_input_present": bool(validated_input),
            "preference_fill_allowed": bool(validated_input),
            "review_item_count": int(source["review_item_count"]),
            "required_input_field_count": len(source["required_input_fields"]),
            "missing_validated_input_reason": ""
            if validated_input
            else "validated_review_input=false",
            "source_summary": source["source_summary"],
        },
        "readiness": {
            "boundary": BOUNDARY,
            "listening_review_input_guard_completed": True,
            "validated_review_input_present": bool(validated_input),
            "preference_fill_allowed": bool(validated_input),
            "listening_review_completed": False,
            "human_review_required_now": bool(source["human_review_required_now"]),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
            **source["source_summary"],
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": reason,
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
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair listening review fill"
            if validated_input
            else "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing repair objective-only next decision"
        ),
    }


def validate_listening_review_input_guard_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_guard_completed: bool,
    require_preference_blocked: bool,
    require_pending_input: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _dict(guard.get("source_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "unexpected next boundary"
        )
    if require_guard_completed and not bool(
        readiness.get("listening_review_input_guard_completed", False)
    ):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "guard completion required"
        )
    if require_preference_blocked and bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "preference fill should remain blocked"
        )
    if require_pending_input and bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "validated input should remain pending"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingRepairListeningInputGuardError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="chord-tone landing repair input guard readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "validated_review_input_present": bool(
            guard.get("validated_review_input_present", True)
        ),
        "preference_fill_allowed": bool(guard.get("preference_fill_allowed", True)),
        "review_item_count": _int(guard.get("review_item_count")),
        "required_input_field_count": _int(guard.get("required_input_field_count")),
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
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_boundary": str(decision.get("next_boundary") or ""),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    guard = report["guard_result"]
    source = guard["source_summary"]
    package = report["source_package_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Chord-Tone Landing Repair Listening Review Input Guard",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- review item count: `{guard['review_item_count']}`",
        f"- required input field count: `{guard['required_input_field_count']}`",
        f"- validated review input present: `{_bool_token(guard['validated_review_input_present'])}`",
        f"- preference fill allowed: `{_bool_token(guard['preference_fill_allowed'])}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- rendered audio file count: `{source['rendered_audio_file_count']}`",
        f"- duration range: `{source['duration_min_seconds']:.3f}s-{source['duration_max_seconds']:.3f}s`",
        f"- changed note total: `{source['changed_note_total']}`",
        f"- objective outside-soloing pitch-role risk count: `{source['objective_outside_soloing_pitch_role_risk_count']}`",
        f"- weak chord-tone landing risk delta: `{source['weak_chord_tone_landing_risk_delta']}`",
        f"- outside-soloing pitch-role risk count: `{source['outside_soloing_pitch_role_risk_count_before']} -> {source['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing repair targeted: `{_bool_token(source['outside_soloing_repair_targeted'])}`",
        f"- outside-soloing residual risk preserved: `{_bool_token(source['outside_soloing_residual_risk_preserved'])}`",
        f"- final landing chord-tone count after: `{source['final_landing_chord_tone_count_after']}`",
        f"- audio review required: `{_bool_token(source['audio_review_required'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        "",
        "## Decision",
        "",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- reason: `{decision['reason']}`",
        f"- next recommended issue: `{report['next_recommended_issue']}`",
        "",
        "## Required Input Fields",
        "",
    ]
    for field in package["required_input_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Review WAV Paths", ""])
    for path in package["wav_paths"]:
        lines.append(f"- `{path}`")
    lines.extend(["", "## Claim Boundary", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Guard chord-tone landing repair listening review input"
    )
    parser.add_argument("--source_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_listening_review_input_guard"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=796)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_guard_completed", action="store_true")
    parser.add_argument("--require_preference_blocked", action="store_true")
    parser.add_argument("--require_pending_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_listening_review_input_guard_report(
        read_json(Path(args.source_package)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_listening_review_input_guard_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_guard_completed=bool(args.require_guard_completed),
        require_preference_blocked=bool(
            args.require_preference_blocked or args.require_pending_input
        ),
        require_pending_input=bool(args.require_pending_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_listening_review_input_guard.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_listening_review_input_guard_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_repair_listening_review_input_guard.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
