"""Build a listening review package for songlike melody contour repaired MIDI-to-solo output."""

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
from scripts.audit_stage_b_midi_to_solo_final_status import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
)
from scripts.render_stage_b_midi_to_solo_songlike_melody_contour_repair_audio import (  # noqa: E402
    BOUNDARY as SOURCE_BOUNDARY,
    EXPECTED_SOURCE_SCHEMA_VERSIONS,
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION,
    SCHEMA_VERSION as SOURCE_AUDIO_SCHEMA_VERSION,
    StageBMidiToSoloSonglikeMelodyContourRepairAudioError,
    validate_audio_render_report,
)


class StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_input_guard"
SCHEMA_VERSION = "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package_v5"

EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS = {
    "songlike_melody_contour_repair_audio_package": SOURCE_AUDIO_SCHEMA_VERSION,
    **EXPECTED_SOURCE_SCHEMA_VERSIONS,
}

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
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _validate_source_schema_versions(
    source_schema_versions: dict[str, Any],
    *,
    label: str,
) -> dict[str, str]:
    normalized = {key: str(value) for key, value in source_schema_versions.items()}
    for key, expected in EXPECTED_AUDIO_SOURCE_SCHEMA_VERSIONS.items():
        if str(normalized.get(key) or "") != expected:
            raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
                f"{label} source schema version mismatch: {key}"
            )
    return normalized


def _source_context_fields(summary: dict[str, Any]) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        objective_key = f"objective_{key}"
        if objective_key not in summary or summary[objective_key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
                f"objective source-context field required: {objective_key}"
            )
        if key not in summary or summary[key] is None:
            raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
                f"source-context field required: {key}"
            )
    missing_preserved = [
        key for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS if not bool(summary.get(key))
    ]
    missing_objective_preserved = [
        f"objective_{key}"
        for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS
        if not bool(summary.get(f"objective_{key}"))
    ]
    if missing_preserved or missing_objective_preserved:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "source-context preserved field must be true: "
            f"{missing_objective_preserved + missing_preserved}"
        )
    return {
        "objective_source_outside_soloing_repair_wav_count": _int(
            summary.get("objective_source_outside_soloing_repair_wav_count")
        ),
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            summary.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_schema_context_preserved": bool(
            summary.get("objective_source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_objective_schema_version": str(
            summary.get("objective_source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            summary.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_delta"
            )
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            summary.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            summary.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            summary.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            summary.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            summary.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_schema_context_preserved": bool(
            summary.get("source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "source_outside_soloing_repair_objective_schema_version": str(
            summary.get("source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            summary.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            summary.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            summary.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            summary.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            summary.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        **{
            f"objective_{key}": summary.get(f"objective_{key}")
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS
        },
        **{key: summary.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
    }


def _validate_source_context(summary: dict[str, Any], *, base: str, label: str) -> None:
    objective_risk = _int(summary.get(f"{base}_source_objective_pitch_role_risk_count"))
    source_before = _int(summary.get(f"{base}_source_pitch_role_risk_count_before"))
    source_after = _int(summary.get(f"{base}_source_pitch_role_risk_count_after"))
    source_delta = _int(summary.get(f"{base}_source_pitch_role_risk_delta"))
    current_after = _int(summary.get(f"{base}_pitch_role_risk_count_after"))
    current_delta = _int(summary.get(f"{base}_pitch_role_risk_delta"))
    if not bool(summary.get(f"{base}_source_context_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} source context preservation required"
        )
    if not bool(summary.get(f"{base}_schema_context_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} schema context preservation required"
        )
    if (
        str(summary.get(f"{base}_objective_schema_version") or "")
        != OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
    ):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} objective schema version mismatch"
        )
    if objective_risk <= 0 or source_before <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} source pitch-role risk context required"
        )
    if objective_risk != source_before:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} objective/source risk mismatch"
        )
    if source_before - source_after != source_delta:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} source pitch-role risk delta mismatch"
        )
    if source_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} positive source pitch-role risk delta required"
        )
    if bool(summary.get(f"{base}_source_targeted", True)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} source-targeted flag should remain false"
        )
    if not bool(summary.get(f"{base}_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} source residual risk preservation required"
        )
    if current_after != 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} current repair residual pitch-role risk should be zero"
        )
    if current_delta <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"{label} positive current repair pitch-role risk delta required"
        )


def validate_audio_package_report(
    report: dict[str, Any],
    *,
    expected_count: int,
) -> dict[str, Any]:
    boundary = _dict(report.get("audio_render_boundary"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    try:
        audio_validation_summary = validate_audio_render_report(
            report,
            expected_boundary=SOURCE_BOUNDARY,
            expected_next_boundary=SOURCE_NEXT_BOUNDARY,
            expected_file_count=int(expected_count),
            expected_sample_rate=_int(summary.get("sample_rate")) or 44100,
            require_audio_package_completed=True,
            require_no_quality_claim=True,
        )
    except StageBMidiToSoloSonglikeMelodyContourRepairAudioError as exc:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            str(exc)
        ) from exc
    source_schema_versions = _validate_source_schema_versions(
        {
            "songlike_melody_contour_repair_audio_package": audio_validation_summary.get(
                "schema_version"
            ),
            **_dict(report.get("source_schema_versions")),
        },
        label="songlike melody contour repair audio package",
    )
    if str(boundary.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "songlike melody contour repair audio package boundary required"
        )
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "audio package must route to listening review package"
        )
    required_true = [
        "render_attempted",
        "technical_wav_validation",
        "songlike_melody_contour_repair_audio_package_completed",
    ]
    missing = [name for name in required_true if not bool(boundary.get(name, False))]
    if missing:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"missing audio package readiness: {missing}"
        )
    if not bool(summary.get("audio_review_required", False)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "audio review requirement should be recorded"
        )
    if not bool(summary.get("source_outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "source outside-soloing repair evidence readiness required"
        )
    if _int(summary.get("objective_source_outside_soloing_repair_wav_count")) < int(
        expected_count
    ):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "objective source outside-soloing WAV count below expected"
        )
    _validate_source_context(
        summary,
        base="objective_source_outside_soloing_repair",
        label="objective source outside-soloing repair",
    )
    _validate_source_context(
        summary,
        base="source_outside_soloing_repair",
        label="source outside-soloing repair",
    )
    if _int(summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "source outside-soloing residual pitch-role risk should be zero"
        )
    if _int(summary.get("source_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "source outside-soloing not-evaluable boundary required"
        )
    if _int(summary.get("repaired_outside_soloing_not_evaluable_count")) <= 0:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "repaired outside-soloing not-evaluable boundary required"
        )
    if _int(boundary.get("rendered_audio_file_count")) < int(expected_count):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "rendered audio count below expected"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(boundary, label="audio package boundary")

    rendered = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if len(rendered) < int(expected_count):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "rendered audio file rows below expected"
        )
    review_items: list[dict[str, Any]] = []
    for item in rendered[: int(expected_count)]:
        wav = _dict(item.get("wav_file"))
        wav_path = str(wav.get("path") or "")
        midi_path = str(item.get("repaired_midi_path") or "")
        if not _path_exists(wav_path) or not _path_exists(midi_path):
            raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
                "review item MIDI/WAV artifact required"
            )
        review_items.append(
            {
                "candidate_index": _int(item.get("candidate_index")),
                "source": str(item.get("source") or ""),
                "rank": _int(item.get("rank")),
                "midi_path": midi_path,
                "wav_path": wav_path,
                "duration_seconds": _float(wav.get("duration_seconds")),
                "sample_rate": _int(wav.get("sample_rate")),
                "size_bytes": _int(wav.get("size_bytes")),
                "sha256": str(wav.get("sha256") or ""),
                "repaired_failure_labels": _list(item.get("repaired_failure_labels")),
                "repaired_dead_air_ratio": _float(item.get("repaired_dead_air_ratio")),
                "repaired_max_interval": _int(item.get("repaired_max_interval")),
                "repaired_unique_pitch_count": _int(item.get("repaired_unique_pitch_count")),
                "changed_pitch_count": _int(item.get("changed_pitch_count")),
                "changed_time_count": _int(item.get("changed_time_count")),
                "review_status": "pending",
            }
        )
    return {
        "review_items": review_items,
        "audio_validation_summary": audio_validation_summary,
        "source_schema_versions": source_schema_versions,
    }


def build_listening_review_package_report(
    *,
    audio_package_report: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    expected_count: int,
) -> dict[str, Any]:
    source = validate_audio_package_report(
        audio_package_report,
        expected_count=int(expected_count),
    )
    review_items = [_dict(item) for item in _list(source.get("review_items"))]
    audio_validation_summary = _dict(source.get("audio_validation_summary"))
    source_schema_versions = {
        key: str(value) for key, value in _dict(source.get("source_schema_versions")).items()
    }
    summary = _dict(audio_package_report.get("summary"))
    source_context = _source_context_fields(summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": SOURCE_BOUNDARY,
        "source_schema_versions": source_schema_versions,
        "source_summary": {
            "source_songlike_melody_contour_repair_audio_package_schema_version": str(
                audio_validation_summary.get("schema_version") or ""
            ),
            "source_songlike_melody_contour_repair_sweep_schema_version": str(
                audio_validation_summary.get(
                    "source_songlike_melody_contour_repair_sweep_schema_version"
                )
                or ""
            ),
            "source_targeted_quality_repair_followup_schema_version": str(
                audio_validation_summary.get(
                    "source_targeted_quality_repair_followup_schema_version"
                )
                or ""
            ),
            "source_targeted_quality_repair_objective_next_schema_version": str(
                audio_validation_summary.get(
                    "source_targeted_quality_repair_objective_next_schema_version"
                )
                or ""
            ),
            "source_targeted_quality_repair_sweep_schema_version": str(
                audio_validation_summary.get("source_targeted_quality_repair_sweep_schema_version")
                or ""
            ),
            "source_targeted_quality_repair_listening_review_input_guard_schema_version": str(
                audio_validation_summary.get(
                    "source_targeted_quality_repair_listening_review_input_guard_schema_version"
                )
                or ""
            ),
            "source_targeted_quality_repair_listening_review_package_schema_version": str(
                audio_validation_summary.get(
                    "source_targeted_quality_repair_listening_review_package_schema_version"
                )
                or ""
            ),
            "source_targeted_quality_repair_audio_package_schema_version": str(
                audio_validation_summary.get(
                    "source_targeted_quality_repair_audio_package_schema_version"
                )
                or ""
            ),
            "source_candidate_failure_labeling_schema_version": str(
                audio_validation_summary.get("source_candidate_failure_labeling_schema_version")
                or ""
            ),
            "source_quality_rubric_schema_version": str(
                audio_validation_summary.get("source_quality_rubric_schema_version") or ""
            ),
            "source_post_mvp_plan_schema_version": str(
                audio_validation_summary.get("source_post_mvp_plan_schema_version") or ""
            ),
            "source_final_status_schema_version": str(
                audio_validation_summary.get("source_final_status_schema_version") or ""
            ),
            "source_delivery_package_schema_version": str(
                audio_validation_summary.get("source_delivery_package_schema_version") or ""
            ),
            "source_listening_gap_schema_version": str(
                audio_validation_summary.get("source_listening_gap_schema_version") or ""
            ),
            "source_quality_gap_schema_version": str(
                audio_validation_summary.get("source_quality_gap_schema_version") or ""
            ),
            "source_current_evidence_schema_version": str(
                audio_validation_summary.get("source_current_evidence_schema_version") or ""
            ),
            "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
            "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
            "sample_rate": _int(summary.get("sample_rate")),
            "duration_min_seconds": _float(summary.get("duration_min_seconds")),
            "duration_max_seconds": _float(summary.get("duration_max_seconds")),
            "source_total_failure_label_count": _int(
                summary.get("source_total_failure_label_count")
            ),
            "repaired_total_failure_label_count": _int(
                summary.get("repaired_total_failure_label_count")
            ),
            "failure_label_delta": _int(summary.get("failure_label_delta")),
            "source_songlike_failure_count": _int(
                summary.get("source_songlike_failure_count")
            ),
            "repaired_songlike_failure_count": _int(
                summary.get("repaired_songlike_failure_count")
            ),
            "songlike_failure_delta": _int(summary.get("songlike_failure_delta")),
            "improved_candidate_count": _int(summary.get("improved_candidate_count")),
            "technical_regression_count": _int(summary.get("technical_regression_count")),
            "source_outside_soloing_repair_evidence_ready": bool(
                summary.get("source_outside_soloing_repair_evidence_ready", False)
            ),
            **source_context,
            "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
                summary.get("source_outside_soloing_repair_pitch_role_risk_count_after")
            ),
            "source_outside_soloing_not_evaluable_count": _int(
                summary.get("source_outside_soloing_not_evaluable_count")
            ),
            "repaired_outside_soloing_not_evaluable_count": _int(
                summary.get("repaired_outside_soloing_not_evaluable_count")
            ),
            "repaired_not_evaluable_counts": _dict(
                summary.get("repaired_not_evaluable_counts")
            ),
            "remaining_failure_counts": _dict(summary.get("remaining_failure_counts")),
            "audio_review_required": bool(summary.get("audio_review_required", False)),
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
            "objective_source_outside_soloing_repair_schema_context_preserved": bool(
                summary.get("objective_source_outside_soloing_repair_schema_context_preserved", False)
            ),
            "objective_source_outside_soloing_repair_objective_schema_version": str(
                summary.get("objective_source_outside_soloing_repair_objective_schema_version") or ""
            ),
            "source_outside_soloing_repair_schema_context_preserved": bool(
                summary.get("source_outside_soloing_repair_schema_context_preserved", False)
            ),
            "source_outside_soloing_repair_objective_schema_version": str(
                summary.get("source_outside_soloing_repair_objective_schema_version") or ""
            ),
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
            "reason": "songlike contour repair WAV review items are packaged; preference remains pending until validated listening input",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": (
            "Stage B MIDI-to-solo songlike melody contour repair listening review input guard source-context refresh"
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
    source_schema_versions = _dict(report.get("source_schema_versions"))
    if str(report.get("schema_version") or "") != SCHEMA_VERSION:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "songlike melody contour repair listening review package schema version mismatch"
        )
    _validate_source_schema_versions(
        source_schema_versions,
        label="songlike melody contour repair listening review package",
    )
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "unexpected next boundary"
        )
    if require_package_ready and not bool(readiness.get("listening_review_package_ready", False)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "listening review package should be ready"
        )
    if _int(readiness.get("review_item_count")) != int(expected_review_item_count):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "review item count mismatch"
        )
    if bool(readiness.get("validated_review_input", True)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "review input should remain pending"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
            "critical user input should not be required"
        )
    for label, schema_context_key, objective_schema_key in (
        (
            "objective",
            "objective_source_outside_soloing_repair_schema_context_preserved",
            "objective_source_outside_soloing_repair_objective_schema_version",
        ),
        (
            "source",
            "source_outside_soloing_repair_schema_context_preserved",
            "source_outside_soloing_repair_objective_schema_version",
        ),
    ):
        if not bool(source.get(schema_context_key, False)):
            raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
                f"{label} outside-soloing schema context preservation required"
            )
        if (
            str(source.get(objective_schema_key) or "")
            != OUTSIDE_SOLOING_REPAIR_OBJECTIVE_SCHEMA_VERSION
        ):
            raise StageBMidiToSoloSonglikeMelodyContourRepairListeningReviewPackageError(
                f"{label} outside-soloing objective schema version mismatch"
            )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="listening package readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "schema_version": str(report.get("schema_version") or ""),
        "source_songlike_melody_contour_repair_audio_package_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_repair_audio_package") or ""
        ),
        "source_songlike_melody_contour_repair_sweep_schema_version": str(
            source_schema_versions.get("songlike_melody_contour_repair_sweep") or ""
        ),
        "source_targeted_quality_repair_followup_schema_version": str(
            source_schema_versions.get("targeted_quality_repair_followup_decision") or ""
        ),
        "source_targeted_quality_repair_objective_next_schema_version": str(
            source_schema_versions.get("targeted_quality_repair_objective_next") or ""
        ),
        "source_targeted_quality_repair_sweep_schema_version": str(
            source_schema_versions.get("targeted_quality_repair_sweep") or ""
        ),
        "source_targeted_quality_repair_listening_review_input_guard_schema_version": str(
            source_schema_versions.get("targeted_quality_repair_listening_review_input_guard")
            or ""
        ),
        "source_targeted_quality_repair_listening_review_package_schema_version": str(
            source_schema_versions.get("targeted_quality_repair_listening_review_package") or ""
        ),
        "source_targeted_quality_repair_audio_package_schema_version": str(
            source_schema_versions.get("targeted_quality_repair_audio_package") or ""
        ),
        "source_candidate_failure_labeling_schema_version": str(
            source_schema_versions.get("candidate_failure_labeling") or ""
        ),
        "source_quality_rubric_schema_version": str(
            source_schema_versions.get("quality_rubric_baseline") or ""
        ),
        "source_post_mvp_plan_schema_version": str(
            source_schema_versions.get("post_mvp_quality_iteration_plan") or ""
        ),
        "source_final_status_schema_version": str(
            source_schema_versions.get("final_status_audit") or ""
        ),
        "source_delivery_package_schema_version": str(
            source_schema_versions.get("delivery_package") or ""
        ),
        "source_listening_gap_schema_version": str(
            source_schema_versions.get("listening_review_quality_gap") or ""
        ),
        "source_quality_gap_schema_version": str(
            source_schema_versions.get("quality_gap_decision") or ""
        ),
        "source_current_evidence_schema_version": str(
            source_schema_versions.get("current_evidence") or ""
        ),
        "listening_review_package_ready": bool(readiness.get("listening_review_package_ready", False)),
        "review_item_count": _int(readiness.get("review_item_count")),
        "validated_review_input": bool(readiness.get("validated_review_input", True)),
        "technical_wav_validation": bool(source.get("technical_wav_validation", False)),
        "rendered_audio_file_count": _int(source.get("rendered_audio_file_count")),
        "sample_rate": _int(source.get("sample_rate")),
        "duration_min_seconds": _float(source.get("duration_min_seconds")),
        "duration_max_seconds": _float(source.get("duration_max_seconds")),
        "failure_label_delta": _int(source.get("failure_label_delta")),
        "songlike_failure_delta": _int(source.get("songlike_failure_delta")),
        "source_outside_soloing_repair_evidence_ready": bool(
            source.get("source_outside_soloing_repair_evidence_ready", False)
        ),
        "objective_source_outside_soloing_repair_wav_count": _int(
            source.get("objective_source_outside_soloing_repair_wav_count")
        ),
        "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_objective_pitch_role_risk_count"
            )
        ),
        "objective_source_outside_soloing_repair_source_context_preserved": bool(
            source.get("objective_source_outside_soloing_repair_source_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_schema_context_preserved": bool(
            source.get("objective_source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "objective_source_outside_soloing_repair_objective_schema_version": str(
            source.get("objective_source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_before"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get(
                "objective_source_outside_soloing_repair_source_pitch_role_risk_count_after"
            )
        ),
        "objective_source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get("objective_source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "objective_source_outside_soloing_repair_source_targeted": bool(
            source.get("objective_source_outside_soloing_repair_source_targeted", True)
        ),
        "objective_source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get(
                "objective_source_outside_soloing_repair_source_residual_risk_preserved",
                False,
            )
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "objective_source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("objective_source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_objective_pitch_role_risk_count": _int(
            source.get("source_outside_soloing_repair_source_objective_pitch_role_risk_count")
        ),
        "source_outside_soloing_repair_source_context_preserved": bool(
            source.get("source_outside_soloing_repair_source_context_preserved", False)
        ),
        "source_outside_soloing_repair_schema_context_preserved": bool(
            source.get("source_outside_soloing_repair_schema_context_preserved", False)
        ),
        "source_outside_soloing_repair_objective_schema_version": str(
            source.get("source_outside_soloing_repair_objective_schema_version") or ""
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_before": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_before")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_count_after")
        ),
        "source_outside_soloing_repair_source_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_source_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_source_targeted": bool(
            source.get("source_outside_soloing_repair_source_targeted", True)
        ),
        "source_outside_soloing_repair_source_residual_risk_preserved": bool(
            source.get("source_outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "source_outside_soloing_repair_pitch_role_risk_delta": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_delta")
        ),
        "source_outside_soloing_repair_pitch_role_risk_count_after": _int(
            source.get("source_outside_soloing_repair_pitch_role_risk_count_after")
        ),
        **{
            f"objective_{key}": source.get(f"objective_{key}")
            for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS
        },
        **{key: source.get(key) for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS},
        "source_outside_soloing_not_evaluable_count": _int(
            source.get("source_outside_soloing_not_evaluable_count")
        ),
        "repaired_outside_soloing_not_evaluable_count": _int(
            source.get("repaired_outside_soloing_not_evaluable_count")
        ),
        "repaired_not_evaluable_counts": _dict(
            source.get("repaired_not_evaluable_counts")
        ),
        "audio_review_required": bool(source.get("audio_review_required", False)),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "wav_paths": [str(_dict(item).get("wav_path") or "") for item in _list(report.get("review_items"))],
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    readiness = report["readiness"]
    decision = report["decision"]
    package = report["review_package"]
    source = report["source_summary"]
    lines = [
        "# Stage B MIDI-to-Solo Songlike Melody Contour Repair Listening Review Package Source Context Refresh",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- schema version: `{report['schema_version']}`",
        f"- source audio package schema version: `{report['source_schema_versions']['songlike_melody_contour_repair_audio_package']}`",
        f"- source sweep schema version: `{report['source_schema_versions']['songlike_melody_contour_repair_sweep']}`",
        f"- source targeted quality repair follow-up schema version: `{report['source_schema_versions']['targeted_quality_repair_followup_decision']}`",
        f"- source targeted quality repair objective next schema version: `{report['source_schema_versions']['targeted_quality_repair_objective_next']}`",
        f"- source targeted quality repair sweep schema version: `{report['source_schema_versions']['targeted_quality_repair_sweep']}`",
        f"- source targeted quality repair listening review input guard schema version: `{report['source_schema_versions']['targeted_quality_repair_listening_review_input_guard']}`",
        f"- source targeted quality repair listening review package schema version: `{report['source_schema_versions']['targeted_quality_repair_listening_review_package']}`",
        f"- source targeted quality repair audio package schema version: `{report['source_schema_versions']['targeted_quality_repair_audio_package']}`",
        f"- source candidate failure labeling schema version: `{report['source_schema_versions']['candidate_failure_labeling']}`",
        f"- source quality rubric schema version: `{report['source_schema_versions']['quality_rubric_baseline']}`",
        f"- source post-MVP plan schema version: `{report['source_schema_versions']['post_mvp_quality_iteration_plan']}`",
        f"- source final status schema version: `{report['source_schema_versions']['final_status_audit']}`",
        f"- source delivery package schema version: `{report['source_schema_versions']['delivery_package']}`",
        f"- source listening gap schema version: `{report['source_schema_versions']['listening_review_quality_gap']}`",
        f"- source quality gap schema version: `{report['source_schema_versions']['quality_gap_decision']}`",
        f"- source current evidence schema version: `{report['source_schema_versions']['current_evidence']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- package ready: `{_bool_token(package['package_ready'])}`",
        f"- review item count: `{package['review_item_count']}`",
        f"- validated review input: `{_bool_token(package['validated_review_input'])}`",
        f"- technical WAV validation: `{_bool_token(source['technical_wav_validation'])}`",
        f"- rendered audio file count: `{source['rendered_audio_file_count']}`",
        f"- duration range: `{source['duration_min_seconds']:.3f}s-{source['duration_max_seconds']:.3f}s`",
        f"- failure label delta: `{source['failure_label_delta']}`",
        f"- songlike failure count: `{source['source_songlike_failure_count']} -> {source['repaired_songlike_failure_count']}`",
        f"- songlike failure delta: `{source['songlike_failure_delta']}`",
        f"- source outside-soloing repair evidence ready: `{_bool_token(source['source_outside_soloing_repair_evidence_ready'])}`",
        f"- objective source outside-soloing repair WAV count: `{source['objective_source_outside_soloing_repair_wav_count']}`",
        f"- objective source outside-soloing source context preserved: `{_bool_token(source['objective_source_outside_soloing_repair_source_context_preserved'])}`",
        f"- objective source outside-soloing schema context preserved: `{_bool_token(source['objective_source_outside_soloing_repair_schema_context_preserved'])}`",
        f"- objective source outside-soloing objective schema version: `{source['objective_source_outside_soloing_repair_objective_schema_version']}`",
        f"- objective source outside-soloing source pitch-role risk before / after / delta: `{source['objective_source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{source['objective_source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{source['objective_source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- objective source outside-soloing source repair targeted: `{_bool_token(source['objective_source_outside_soloing_repair_source_targeted'])}`",
        f"- objective source outside-soloing source residual risk preserved: `{_bool_token(source['objective_source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- objective source outside-soloing current repair pitch-role risk after / delta: `{source['objective_source_outside_soloing_repair_pitch_role_risk_count_after']}` / `{source['objective_source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- objective follow-up objective source outside-soloing source pitch-role risk: `{source['objective_followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['objective_followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- objective follow-up objective source outside-soloing source context preserved: `{_bool_token(source['objective_followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- objective follow-up objective current repair pitch-role risk after/delta: `{source['objective_followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']}` / `{source['objective_followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- objective repair sweep source outside-soloing source pitch-role risk: `{source['objective_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['objective_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- objective follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(source['objective_followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- objective bridge repair sweep source outside-soloing source context preserved: `{_bool_token(source['objective_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- objective repair sweep current repair pitch-role risk after/delta: `{source['objective_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']}` / `{source['objective_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- source outside-soloing source context preserved: `{_bool_token(source['source_outside_soloing_repair_source_context_preserved'])}`",
        f"- source outside-soloing schema context preserved: `{_bool_token(source['source_outside_soloing_repair_schema_context_preserved'])}`",
        f"- source outside-soloing objective schema version: `{source['source_outside_soloing_repair_objective_schema_version']}`",
        f"- source outside-soloing source pitch-role risk before / after / delta: `{source['source_outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{source['source_outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{source['source_outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- source outside-soloing source repair targeted: `{_bool_token(source['source_outside_soloing_repair_source_targeted'])}`",
        f"- source outside-soloing source residual risk preserved: `{_bool_token(source['source_outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- source outside-soloing current repair pitch-role risk after / delta: `{source['source_outside_soloing_repair_pitch_role_risk_count_after']}` / `{source['source_outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{source['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(source['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up objective current repair pitch-role risk after/delta: `{source['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']}` / `{source['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{source['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {source['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(source['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(source['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep current repair pitch-role risk after/delta: `{source['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']}` / `{source['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- source outside-soloing repair pitch-role risk after: `{source['source_outside_soloing_repair_pitch_role_risk_count_after']}`",
        f"- source outside-soloing not evaluable count: `{source['source_outside_soloing_not_evaluable_count']}`",
        f"- repaired outside-soloing not evaluable count: `{source['repaired_outside_soloing_not_evaluable_count']}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        "",
        "## Review Items",
        "",
    ]
    for item in report["review_items"]:
        failure_labels = ",".join(item["repaired_failure_labels"]) or "none"
        lines.append(
            f"- candidate `{item['candidate_index']}` `{item['source']}` rank `{item['rank']}`: "
            f"WAV `{item['wav_path']}`, MIDI `{item['midi_path']}`, duration `{item['duration_seconds']:.3f}`, "
            f"failure labels `{failure_labels}`"
        )
    lines.extend(["", "## Required Input Fields", ""])
    for field in package["required_input_fields"]:
        lines.append(f"- `{field}`")
    lines.extend(["", "## Repaired Not Evaluable Counts", ""])
    for label, count in source["repaired_not_evaluable_counts"].items():
        lines.append(f"- `{label}`: `{count}`")
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
        description="Build songlike melody contour repair listening review package"
    )
    parser.add_argument("--audio_package_report", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=1188)
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
        / "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package.json",
        report,
    )
    write_json(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_songlike_melody_contour_repair_listening_review_package.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
