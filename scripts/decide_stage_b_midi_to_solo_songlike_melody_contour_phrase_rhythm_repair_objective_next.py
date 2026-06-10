"""Select the next boundary after songlike melody contour phrase/rhythm repair input guard evidence."""

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
from scripts.guard_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_listening_review_input import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    OBJECTIVE_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision"
FOLLOWUP_DECISION_NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_followup_decision"
)
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_next_v1"

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
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_input_guard_report(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    guard = _dict(report.get("guard_result"))
    source = _dict(guard.get("source_summary"))
    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "songlike melody contour phrase/rhythm repair listening review input guard boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "input guard must route to phrase/rhythm objective-only next decision"
        )
    if not bool(readiness.get("listening_review_input_guard_completed", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "input guard completion required"
        )
    if bool(guard.get("validated_review_input_present", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "objective-only decision requires pending review input"
        )
    if bool(guard.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if not bool(source.get("technical_wav_validation", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "technical WAV validation required"
        )
    if _int(source.get("rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "rendered WAV count below 6"
        )
    if not bool(source.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "outside-soloing repair evidence should be ready"
        )
    if _int(source.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "source outside-soloing repair pitch-role risk should be resolved"
        )
    if _int(source.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "source outside-soloing not-evaluable count should be preserved"
        )
    if _int(source.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "repaired outside-soloing not-evaluable count should be preserved"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="phrase/rhythm repair input guard readiness")
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
        "failure_label_delta": _int(source.get("failure_label_delta")),
        "source_phrase_rhythm_failure_count": _int(source.get("source_phrase_rhythm_failure_count")),
        "repaired_phrase_rhythm_failure_count": _int(
            source.get("repaired_phrase_rhythm_failure_count")
        ),
        "phrase_rhythm_failure_delta": _int(source.get("phrase_rhythm_failure_delta")),
        "source_outside_soloing_repair_evidence_ready": bool(
            source.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_not_evaluable_count": _int(
            source.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            source.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(source.get("repaired_not_evaluable_counts")),
        "audio_review_required": bool(source.get("audio_review_required", False)),
    }


def build_objective_next_report(
    *,
    input_guard_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_input_guard_report(input_guard_report)
    followup_required = True
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
            "failure_label_delta": _int(source["failure_label_delta"]),
            "source_phrase_rhythm_failure_count": _int(
                source["source_phrase_rhythm_failure_count"]
            ),
            "repaired_phrase_rhythm_failure_count": _int(
                source["repaired_phrase_rhythm_failure_count"]
            ),
            "phrase_rhythm_failure_delta": _int(source["phrase_rhythm_failure_delta"]),
            "source_outside_soloing_repair_evidence_ready": bool(
                source["source_outside_soloing_repair_evidence_ready"]
            ),
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                source["source_outside_soloing_repair_pitch_role_risk_count_after"]
            ),
            "source_outside_soloing_not_evaluable_count": _int(
                source["source_outside_soloing_not_evaluable_count"]
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                source["repaired_outside_soloing_not_evaluable_count"]
            ),
            "repaired_not_evaluable_counts": _dict(source["repaired_not_evaluable_counts"]),
            "audio_review_required": bool(source["audio_review_required"]),
            "phrase_rhythm_followup_required": bool(followup_required),
            "current_quality_claim_ready": False,
        },
        "selected_next_target": {
            "target": "songlike_melody_contour_phrase_rhythm_repair_followup_decision",
            "next_boundary": FOLLOWUP_DECISION_NEXT_BOUNDARY,
            "reason": (
                "listening preference pending and quality claim unavailable; route to follow-up repair decision"
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "objective_next_decision_completed": True,
            "technical_wav_validation": bool(source["technical_wav_validation"]),
            "phrase_rhythm_followup_required": bool(followup_required),
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
            "reason": "objective-only evidence selected follow-up repair decision without quality claim",
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
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm repair follow-up decision"
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
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "unexpected next boundary"
        )
    if require_objective_decision and not bool(
        readiness.get("objective_next_decision_completed", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "objective next decision completion required"
        )
    if require_followup_required and not bool(
        readiness.get("phrase_rhythm_followup_required", False)
    ):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "phrase/rhythm follow-up requirement expected"
        )
    if bool(summary.get("preference_fill_allowed", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "preference fill must remain blocked"
        )
    if bool(summary.get("current_quality_claim_ready", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "quality claim readiness must remain false"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourPhraseRhythmRepairObjectiveNextError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="phrase/rhythm repair objective next readiness")
    return {
        "boundary": boundary,
        "source_boundary": str(report.get("source_boundary") or ""),
        "next_boundary": str(decision.get("next_boundary") or ""),
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
        "sample_rate": _int(summary.get("sample_rate")),
        "duration_min_seconds": _float(summary.get("duration_min_seconds")),
        "duration_max_seconds": _float(summary.get("duration_max_seconds")),
        "failure_label_delta": _int(summary.get("failure_label_delta")),
        "phrase_rhythm_failure_delta": _int(summary.get("phrase_rhythm_failure_delta")),
        "source_outside_soloing_repair_evidence_ready": bool(
            summary.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_not_evaluable_count": _int(
            summary.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            summary.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(summary.get("repaired_not_evaluable_counts")),
        "audio_review_required": bool(summary.get("audio_review_required", False)),
        "phrase_rhythm_followup_required": bool(
            summary.get("phrase_rhythm_followup_required", False)
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
        "# Stage B MIDI-to-Solo Songlike Melody Contour Phrase/Rhythm Repair Objective-Only Next Decision",
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
        f"- duration range: `{summary['duration_min_seconds']:.3f}s-{summary['duration_max_seconds']:.3f}s`",
        f"- failure label delta: `{summary['failure_label_delta']}`",
        f"- phrase/rhythm failure count: `{summary['source_phrase_rhythm_failure_count']} -> {summary['repaired_phrase_rhythm_failure_count']}`",
        f"- phrase/rhythm failure delta: `{summary['phrase_rhythm_failure_delta']}`",
        f"- source outside-soloing repair evidence ready: `{_bool_token(summary['source_outside_soloing_repair_evidence_ready'])}`",
        f"- source outside-soloing repair pitch-role risk count after: `{summary['source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- source/repaired outside-soloing not evaluable count: `{summary['source_outside_soloing_not_evaluable_count']}/{summary['repaired_outside_soloing_not_evaluable_count']}`",
        f"- phrase/rhythm follow-up required: `{_bool_token(summary['phrase_rhythm_followup_required'])}`",
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
        "## Repaired Not Evaluable Counts",
        "",
    ]
    for label, count in sorted(summary["repaired_not_evaluable_counts"].items()):
        lines.append(f"- `{label}`: `{count}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
        ]
    )
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decide songlike melody contour phrase/rhythm repair objective-only next step"
    )
    parser.add_argument("--input_guard_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=866)
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
        output_dir / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir / "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_repair_objective_only_next_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
