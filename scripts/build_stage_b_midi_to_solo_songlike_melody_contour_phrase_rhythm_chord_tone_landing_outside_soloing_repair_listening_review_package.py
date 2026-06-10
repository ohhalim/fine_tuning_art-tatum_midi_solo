"""Build a listening review package for outside-soloing repaired MIDI-to-solo output."""

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
from scripts.build_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_context_pitch_role_bridge import (  # noqa: E402
    BRIDGE_SOURCE_CONTEXT_KEYS,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_audio import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
)


class StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(ValueError):
    pass


BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package"
)
NEXT_BOUNDARY = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_input_guard"
)
SCHEMA_VERSION = (
    "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_chord_tone_landing_outside_soloing_repair_listening_review_package_v2"
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


def _path_exists(path_text: str) -> bool:
    return bool(path_text and Path(path_text).exists())


def _require_no_quality_claim(container: dict[str, Any], *, label: str) -> None:
    claimed = [name for name in QUALITY_CLAIM_KEYS if bool(container.get(name, False))]
    if claimed:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
                f"{label} source-context field required: {key}"
            )
    return {key: container[key] for key in BRIDGE_SOURCE_CONTEXT_KEYS}


def validate_audio_package_report(
    report: dict[str, Any],
    *,
    expected_count: int,
) -> list[dict[str, Any]]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "outside-soloing repair audio package boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "audio package must route to listening review package"
        )
    required_true = [
        "render_attempted",
        "technical_wav_validation",
        "outside_soloing_repair_audio_package_completed",
    ]
    missing = [name for name in required_true if not bool(boundary.get(name, False))]
    if missing:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            f"missing audio package readiness: {missing}"
        )
    if not bool(summary.get("audio_review_required", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "audio review requirement should be recorded"
        )
    if _int(boundary.get("rendered_audio_file_count")) < int(expected_count):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "rendered audio count below expected"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "critical user input should not be required"
        )
    if bool(summary.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(summary.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(summary.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "outside-soloing repair should be targeted before listening package"
        )
    _source_context_fields(summary, label="audio package summary")
    _require_no_quality_claim(boundary, label="audio package boundary")

    rendered = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(rendered) < int(expected_count):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "rendered audio file rows below expected"
        )
    review_items: list[dict[str, Any]] = []
    for item in rendered[: int(expected_count)]:
        wav = _dict(item.get("wav_file"))
        wav_path = str(wav.get("path") or "")
        midi_path = str(item.get("repaired_midi_path") or "")
        if not _path_exists(wav_path) or not _path_exists(midi_path):
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
                "review item MIDI/WAV artifact required"
            )
        review_items.append(
            {
                "candidate_index": _int(item.get("candidate_index")),
                "rank": _int(item.get("rank")),
                "midi_path": midi_path,
                "wav_path": wav_path,
                "duration_seconds": _float(wav.get("duration_seconds")),
                "sample_rate": _int(wav.get("sample_rate")),
                "size_bytes": _int(wav.get("size_bytes")),
                "sha256": str(wav.get("sha256") or ""),
                "changed_note_count": _int(item.get("changed_note_count")),
                "after_bridge_flags": [
                    str(flag) for flag in _list(item.get("after_bridge_flags"))
                ],
                "after_chord_tone_ratio": _float(item.get("after_chord_tone_ratio")),
                "after_max_non_chord_tone_run": _int(
                    item.get("after_max_non_chord_tone_run")
                ),
                "after_cadence_landing_chord_tone": bool(
                    item.get("after_cadence_landing_chord_tone", False)
                ),
                "review_status": "pending",
            }
        )
    return review_items


def build_listening_review_package_report(
    *,
    audio_package_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    expected_count: int,
) -> dict[str, Any]:
    review_items = validate_audio_package_report(
        audio_package_report,
        expected_count=int(expected_count),
    )
    summary = _dict(audio_package_report.get("summary"))
    source_context = _source_context_fields(summary, label="audio package summary")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_summary": {
            "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
            "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
            "sample_rate": _int(summary.get("sample_rate")),
            "duration_min_seconds": _float(summary.get("duration_min_seconds")),
            "duration_max_seconds": _float(summary.get("duration_max_seconds")),
            "changed_note_total": _int(summary.get("changed_note_total")),
            "source_objective_outside_soloing_pitch_role_risk_count": _int(
                summary.get("source_objective_outside_soloing_pitch_role_risk_count")
            ),
            "source_outside_soloing_pitch_role_risk_count_before": _int(
                summary.get("source_outside_soloing_pitch_role_risk_count_before")
            ),
            "source_outside_soloing_pitch_role_risk_count_after": _int(
                summary.get("source_outside_soloing_pitch_role_risk_count_after")
            ),
            "source_outside_soloing_pitch_role_risk_delta": _int(
                summary.get("source_outside_soloing_pitch_role_risk_delta")
            ),
            "source_outside_soloing_repair_targeted": bool(
                summary.get("source_outside_soloing_repair_targeted", True)
            ),
            "source_outside_soloing_residual_risk_preserved": bool(
                summary.get("source_outside_soloing_residual_risk_preserved", False)
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
                summary.get("outside_soloing_repair_targeted", False)
            ),
            "weak_chord_tone_landing_risk_count_after": _int(
                summary.get("weak_chord_tone_landing_risk_count_after")
            ),
            "final_landing_chord_tone_count_after": _int(
                summary.get("final_landing_chord_tone_count_after")
            ),
            "max_non_chord_tone_run_before": _int(
                summary.get("max_non_chord_tone_run_before")
            ),
            "max_non_chord_tone_run_after": _int(
                summary.get("max_non_chord_tone_run_after")
            ),
            "target_supported": bool(summary.get("target_supported", False)),
            "audio_review_required": bool(summary.get("audio_review_required", False)),
            **source_context,
        },
        "review_package": {
            "package_ready": True,
            "review_item_count": int(len(review_items)),
            "review_basis": "human_audio_listening_pending",
            "validated_review_input": False,
            "required_input_fields": [
                "candidate_index",
                "listening_status",
                "preference",
                "issue_notes",
            ],
        },
        "review_items": review_items,
        "readiness": {
            "boundary": BOUNDARY,
            "listening_review_package_ready": True,
            "review_item_count": int(len(review_items)),
            "validated_review_input": False,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
            **source_context,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "outside-soloing repair WAV review items are packaged; preference remains pending until validated listening input",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour phrase/rhythm chord-tone landing outside-soloing repair listening review input guard"
        ),
    }


def validate_listening_review_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_review_item_count: int,
    require_package_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    source = _dict(report.get("source_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "unexpected next boundary"
        )
    if require_package_ready and not bool(readiness.get("listening_review_package_ready", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "listening review package should be ready"
        )
    if _int(readiness.get("review_item_count")) != int(expected_review_item_count):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "review item count mismatch"
        )
    if bool(readiness.get("validated_review_input", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "review input should remain pending"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "critical user input should not be required"
        )
    if bool(source.get("source_outside_soloing_repair_targeted", True)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "source outside-soloing repair target should remain false"
        )
    if not bool(source.get("source_outside_soloing_residual_risk_preserved", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "source outside-soloing residual risk context must be preserved"
        )
    if not bool(source.get("outside_soloing_repair_targeted", False)):
        raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
            "outside-soloing repair should remain targeted in package summary"
        )
    for key in BRIDGE_SOURCE_CONTEXT_KEYS:
        if key not in source:
            raise StageBMidiToSoloChordToneLandingOutsideSoloingRepairListeningReviewPackageError(
                f"listening package source-context field required: {key}"
            )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="listening package readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "listening_review_package_ready": bool(
            readiness.get("listening_review_package_ready", False)
        ),
        "review_item_count": _int(readiness.get("review_item_count")),
        "validated_review_input": bool(readiness.get("validated_review_input", True)),
        "technical_wav_validation": bool(source.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(source.get("rendered_audio_file_count")),
        "sample_rate": _int(source.get("sample_rate")),
        "duration_min_seconds": _float(source.get("duration_min_seconds")),
        "duration_max_seconds": _float(source.get("duration_max_seconds")),
        "changed_note_total": _int(source.get("changed_note_total")),
        "source_objective_outside_soloing_pitch_role_risk_count": _int(
            source.get("source_objective_outside_soloing_pitch_role_risk_count")
        ),
        "source_outside_soloing_pitch_role_risk_count_before": _int(
            source.get("source_outside_soloing_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_targeted": bool(
            source.get("source_outside_soloing_repair_targeted", True)
        ),
        "source_outside_soloing_residual_risk_preserved": bool(
            source.get("source_outside_soloing_residual_risk_preserved", False)
        ),
        "outside_soloing_pitch_role_risk_count_after": _int(
            source.get("outside_soloing_pitch_role_risk_count_after")
        ),
        "outside_soloing_pitch_role_risk_delta": _int(
            source.get("outside_soloing_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_targeted": bool(
            source.get("outside_soloing_repair_targeted", False)
        ),
        "weak_chord_tone_landing_risk_count_after": _int(
            source.get("weak_chord_tone_landing_risk_count_after")
        ),
        "final_landing_chord_tone_count_after": _int(
            source.get("final_landing_chord_tone_count_after")
        ),
        "max_non_chord_tone_run_after": _int(
            source.get("max_non_chord_tone_run_after")
        ),
        "audio_review_required": bool(source.get("audio_review_required", False)),
        **{key: source.get(key) for key in BRIDGE_SOURCE_CONTEXT_KEYS},
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", True)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(
            decision.get("critical_user_input_required", True)
        ),
        "wav_paths": [
            str(_dict(item).get("wav_path") or "") for item in _list(report.get("review_items"))
        ],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    package = report["review_package"]
    source = report["source_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Chord-Tone Landing Outside-Soloing Repair Listening Review Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- package ready: `{_bool_token(package['package_ready'])}`",
        f"- review item count: `{package['review_item_count']}`",
        f"- validated review input: `{_bool_token(package['validated_review_input'])}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- rendered audio file count: `{source['rendered_audio_file_count']}`",
        f"- duration range: `{source['duration_min_seconds']:.3f}s-{source['duration_max_seconds']:.3f}s`",
        f"- changed note total: `{source['changed_note_total']}`",
        f"- source objective outside-soloing pitch-role risk count: `{source['source_objective_outside_soloing_pitch_role_risk_count']}`",
        f"- source outside-soloing pitch-role risk count: `{source['source_outside_soloing_pitch_role_risk_count_before']} -> {source['source_outside_soloing_pitch_role_risk_count_after']}`",
        f"- source outside-soloing pitch-role risk delta: `{source['source_outside_soloing_pitch_role_risk_delta']}`",
        f"- source outside-soloing repair targeted: `{_bool_token(source['source_outside_soloing_repair_targeted'])}`",
        f"- source outside-soloing residual risk preserved: `{_bool_token(source['source_outside_soloing_residual_risk_preserved'])}`",
        f"- outside-soloing pitch-role risk count after: `{source['outside_soloing_pitch_role_risk_count_after']}`",
        f"- outside-soloing pitch-role risk delta: `{source['outside_soloing_pitch_role_risk_delta']}`",
        f"- outside-soloing repair targeted: `{_bool_token(source['outside_soloing_repair_targeted'])}`",
        f"- weak chord-tone landing risk count after: `{source['weak_chord_tone_landing_risk_count_after']}`",
        f"- final landing chord-tone count after: `{source['final_landing_chord_tone_count_after']}`",
        f"- max non-chord-tone run after: `{source['max_non_chord_tone_run_after']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{source['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{source['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {source['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{source['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{source['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {source['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{source['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{source['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {source['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Review Items",
        "",
    ]
    for item in report["review_items"]:
        flags = ",".join(item["after_bridge_flags"]) or "none"
        lines.append(
            f"- candidate `{item['candidate_index']}` rank `{item['rank']}`: "
            f"WAV `{item['wav_path']}`, MIDI `{item['midi_path']}`, "
            f"duration `{item['duration_seconds']:.3f}`, changed `{item['changed_note_count']}`, "
            f"max run `{item['after_max_non_chord_tone_run']}`, flags `{flags}`"
        )
    lines.extend(["", "## Required Input Fields", ""])
    for field in package["required_input_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(
        [
            "",
            "## Claim Boundary",
            "",
            f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
            f"- audio rendered quality claimed: `{_bool_token(readiness['audio_rendered_quality_claimed'])}`",
            f"- broad trained-model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
            "",
            "## Next",
            "",
            f"- `{report['next_recommended_issue']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build outside-soloing repair listening review package"
    )
    parser.add_argument("--audio_package_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_package"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=976)
    parser.add_argument("--expected_review_item_count", type=int, default=6)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_package_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_listening_review_package_report(
        audio_package_report=read_json(Path(args.audio_package_report)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        expected_count=int(args.expected_review_item_count),
    )
    summary = validate_listening_review_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_review_item_count=int(args.expected_review_item_count),
        require_package_ready=bool(args.require_package_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_package.json"
        ),
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_package_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / (
            "stage_b_midi_to_solo_songlike_melody_contour_phrase_rhythm_"
            "chord_tone_landing_outside_soloing_repair_listening_review_package.md"
        ),
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
