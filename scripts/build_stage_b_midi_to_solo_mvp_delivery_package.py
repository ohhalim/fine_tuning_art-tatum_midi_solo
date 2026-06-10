"""Build a delivery manifest for the current MIDI-to-solo technical MVP."""

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
from scripts.decide_stage_b_midi_to_solo_listening_review_quality_gap import (  # noqa: E402
    BOUNDARY as LISTENING_GAP_BOUNDARY,
    NEXT_BOUNDARY as LISTENING_GAP_NEXT_BOUNDARY,
)
from scripts.run_stage_b_midi_to_solo_phrase_bank_cli_mvp_package import (  # noqa: E402
    BOUNDARY as CLI_PACKAGE_BOUNDARY,
)


class StageBMidiToSoloMvpDeliveryPackageError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_mvp_delivery_package"
NEXT_BOUNDARY = "stage_b_midi_to_solo_readme_final_evidence_refresh"
SCHEMA_VERSION = "stage_b_midi_to_solo_mvp_delivery_package_v1"
CHANGED_RATIO_AUDIO_BOUNDARY = (
    "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_audio_package"
)

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
    "musical_quality_claimed",
    "phrase_bank_musical_quality_claimed",
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
        raise StageBMidiToSoloMvpDeliveryPackageError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _require_existing_file(path: str, *, label: str) -> str:
    if not path:
        raise StageBMidiToSoloMvpDeliveryPackageError(f"{label} path required")
    if not Path(path).exists():
        raise StageBMidiToSoloMvpDeliveryPackageError(f"{label} missing: {path}")
    return path


def validate_listening_gap(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("quality_gap_summary"))
    if str(report.get("boundary") or "") != LISTENING_GAP_BOUNDARY:
        raise StageBMidiToSoloMvpDeliveryPackageError("listening review quality gap boundary required")
    if str(decision.get("next_boundary") or "") != LISTENING_GAP_NEXT_BOUNDARY:
        raise StageBMidiToSoloMvpDeliveryPackageError("listening gap must route to MVP delivery package")
    if not bool(readiness.get("listening_review_quality_gap_completed", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("listening review quality gap completion required")
    if not bool(readiness.get("technical_mvp_delivery_package_ready", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("technical MVP delivery package readiness required")
    if not bool(summary.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("technical model-core MVP completion required")
    if not bool(summary.get("changed_ratio_repair_objective_completed", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio repair objective completion required")
    if not bool(summary.get("outside_soloing_repair_objective_completed", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair objective completion required"
        )
    if _int(summary.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio repair rendered WAV count below 3")
    if _int(summary.get("max_repaired_interval")) > _int(summary.get("max_interval_threshold")):
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio repair interval threshold exceeded")
    if _float(summary.get("max_repaired_pitch_changed_ratio")) > _float(
        summary.get("target_max_pitch_changed_ratio")
    ):
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio repair ratio threshold exceeded")
    if not bool(summary.get("outside_soloing_repair_objective_path_ready", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair objective path readiness required"
        )
    if not bool(summary.get("outside_soloing_repair_target_supported", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair target support required"
        )
    if _int(summary.get("outside_soloing_repair_rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair rendered WAV count below 6"
        )
    if _int(summary.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair residual pitch-role risk should be zero"
        )
    source_objective_risk = _int(
        summary.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        summary.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(
        summary.get("outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_risk_delta = _int(
        summary.get("outside_soloing_repair_source_pitch_role_risk_delta")
    )
    if source_objective_risk <= 0:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(summary.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(summary.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source residual risk preservation required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpDeliveryPackageError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="listening gap readiness")
    return {
        "technical_model_core_mvp_completed": True,
        "changed_ratio_repair_objective_completed": True,
        "outside_soloing_repair_objective_completed": True,
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
        "max_interval_threshold": _int(summary.get("max_interval_threshold")),
        "max_repaired_pitch_changed_ratio": _float(summary.get("max_repaired_pitch_changed_ratio")),
        "target_max_pitch_changed_ratio": _float(summary.get("target_max_pitch_changed_ratio")),
        "listening_review_quality_gap_open": bool(
            summary.get("listening_review_quality_gap_open", False)
        ),
        "outside_soloing_repair_objective_path_ready": bool(
            summary.get("outside_soloing_repair_objective_path_ready", False)
        ),
        "outside_soloing_repair_target_supported": bool(
            summary.get("outside_soloing_repair_target_supported", False)
        ),
        "outside_soloing_repair_rendered_audio_file_count": _int(
            summary.get("outside_soloing_repair_rendered_audio_file_count")
        ),
        "outside_soloing_repair_changed_note_total": _int(
            summary.get("outside_soloing_repair_changed_note_total")
        ),
        "outside_soloing_repair_source_objective_pitch_role_risk_count": source_objective_risk,
        "outside_soloing_repair_source_pitch_role_risk_count_before": source_risk_before,
        "outside_soloing_repair_source_pitch_role_risk_count_after": source_risk_after,
        "outside_soloing_repair_source_pitch_role_risk_delta": source_risk_delta,
        "outside_soloing_repair_source_targeted": bool(
            summary.get("outside_soloing_repair_source_targeted", True)
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            summary.get("outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            summary.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
    }


def validate_cli_package(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    cli = _dict(report.get("cli"))
    objective = _dict(report.get("objective_summary"))
    package_input = _dict(report.get("input"))
    candidates = [_dict(item) for item in _list(report.get("candidate_manifest"))]
    if str(report.get("boundary") or "") != CLI_PACKAGE_BOUNDARY:
        raise StageBMidiToSoloMvpDeliveryPackageError("phrase-bank CLI MVP package boundary required")
    if not bool(readiness.get("cli_mvp_package_completed", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI MVP package completion required")
    if not bool(objective.get("cli_mvp_package_ready", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI MVP package ready flag required")
    if _int(objective.get("candidate_count")) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI candidate count below 3")
    if _int(objective.get("objective_supported_candidate_count")) != _int(
        objective.get("candidate_count")
    ):
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI objective support count mismatch")
    if not str(cli.get("command") or ""):
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI command required")
    input_midi = _require_existing_file(str(package_input.get("midi_path") or ""), label="CLI input MIDI")
    rows: list[dict[str, Any]] = []
    for item in candidates:
        repaired_midi = _require_existing_file(
            str(item.get("repaired_midi_path") or ""),
            label="CLI repaired MIDI",
        )
        rows.append(
            {
                "rank": _int(item.get("rank")),
                "repaired_midi_path": repaired_midi,
                "note_count": _int(item.get("note_count")),
                "unique_pitch_count": _int(item.get("unique_pitch_count")),
                "dead_air_ratio": _float(item.get("dead_air_ratio")),
                "objective_supported": bool(item.get("objective_supported", False)),
            }
        )
    if len(rows) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI repaired MIDI candidates below 3")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI package critical user input should be false")
    _require_no_quality_claim(readiness, label="CLI package readiness")
    return {
        "input_midi": input_midi,
        "cli_script": str(cli.get("script") or ""),
        "cli_command": str(cli.get("command") or ""),
        "candidate_count": _int(objective.get("candidate_count")),
        "objective_supported_candidate_count": _int(
            objective.get("objective_supported_candidate_count")
        ),
        "min_dead_air_ratio": _float(objective.get("min_dead_air_ratio")),
        "max_dead_air_ratio": _float(objective.get("max_dead_air_ratio")),
        "candidate_manifest": rows,
    }


def validate_changed_ratio_audio_package(report: dict[str, Any]) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    summary = _dict(report.get("summary"))
    source_summary = _dict(report.get("source_summary"))
    audio_files = [_dict(item) for item in _list(report.get("rendered_audio_files"))]
    if str(decision.get("current_boundary") or "") != CHANGED_RATIO_AUDIO_BOUNDARY:
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio audio boundary required")
    if _int(summary.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio audio rendered count below 3")
    if not bool(summary.get("technical_wav_validation", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio audio technical validation required")
    if not bool(source_summary.get("changed_ratio_repair_passed", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio repair pass required")
    if _float(source_summary.get("repaired_max_pitch_changed_ratio")) > _float(
        source_summary.get("max_pitch_changed_ratio")
    ):
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio repair ratio threshold exceeded")
    rows: list[dict[str, Any]] = []
    for item in audio_files:
        wav_file = _dict(item.get("wav_file"))
        wav_path = _require_existing_file(str(wav_file.get("path") or ""), label="changed-ratio WAV")
        repaired_midi = _require_existing_file(
            str(item.get("repaired_midi_path") or ""),
            label="changed-ratio repaired MIDI",
        )
        rows.append(
            {
                "rank": _int(item.get("rank")),
                "repaired_midi_path": repaired_midi,
                "wav_path": wav_path,
                "duration_seconds": _float(wav_file.get("duration_seconds")),
                "sample_rate": _int(wav_file.get("sample_rate")),
                "pitch_changed_ratio": _float(item.get("pitch_changed_ratio")),
                "repaired_max_interval": _int(item.get("repaired_max_interval")),
                "repaired_unique_pitch_count": _int(item.get("repaired_unique_pitch_count")),
            }
        )
    if len(rows) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio WAV rows below 3")
    durations = [_float(item.get("duration_seconds")) for item in rows]
    return {
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "technical_wav_validation": bool(summary.get("technical_wav_validation", False)),
        "duration_min_seconds": min(durations) if durations else 0.0,
        "duration_max_seconds": max(durations) if durations else 0.0,
        "repaired_max_pitch_changed_ratio": _float(
            source_summary.get("repaired_max_pitch_changed_ratio")
        ),
        "target_max_pitch_changed_ratio": _float(source_summary.get("max_pitch_changed_ratio")),
        "repaired_max_interval": _int(source_summary.get("repaired_max_interval")),
        "target_max_interval": _int(source_summary.get("target_max_interval")),
        "audio_manifest": rows,
    }


def build_delivery_package_report(
    *,
    listening_review_quality_gap: dict[str, Any],
    cli_mvp_package: dict[str, Any],
    changed_ratio_audio_package: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    listening = validate_listening_gap(listening_review_quality_gap)
    cli = validate_cli_package(cli_mvp_package)
    audio = validate_changed_ratio_audio_package(changed_ratio_audio_package)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundaries": {
            "listening_review_quality_gap": LISTENING_GAP_BOUNDARY,
            "phrase_bank_cli_mvp_package": CLI_PACKAGE_BOUNDARY,
            "changed_ratio_repair_audio_package": CHANGED_RATIO_AUDIO_BOUNDARY,
        },
        "delivery_package": {
            "technical_mvp_delivery_package_completed": True,
            "runnable_cli_ready": True,
            "input_to_ranked_midi_ready": True,
            "input_to_rendered_wav_evidence_ready": True,
            "changed_ratio_repair_audio_evidence_ready": True,
            "outside_soloing_repair_evidence_ready": True,
            "listening_review_quality_gap_open": bool(
                listening["listening_review_quality_gap_open"]
            ),
            "cli_command": cli["cli_command"],
            "cli_input_midi": cli["input_midi"],
            "cli_candidate_count": cli["candidate_count"],
            "cli_objective_supported_candidate_count": cli[
                "objective_supported_candidate_count"
            ],
            "cli_dead_air_ratio_range": [
                cli["min_dead_air_ratio"],
                cli["max_dead_air_ratio"],
            ],
            "changed_ratio_repair_wav_count": audio["rendered_audio_file_count"],
            "changed_ratio_repair_ratio_target": [
                audio["repaired_max_pitch_changed_ratio"],
                audio["target_max_pitch_changed_ratio"],
            ],
            "changed_ratio_repair_interval_target": [
                audio["repaired_max_interval"],
                audio["target_max_interval"],
            ],
            "changed_ratio_repair_wav_duration_range": [
                audio["duration_min_seconds"],
                audio["duration_max_seconds"],
            ],
            "outside_soloing_repair_wav_count": listening[
                "outside_soloing_repair_rendered_audio_file_count"
            ],
            "outside_soloing_repair_changed_note_total": listening[
                "outside_soloing_repair_changed_note_total"
            ],
            "outside_soloing_repair_source_objective_pitch_role_risk_count": listening[
                "outside_soloing_repair_source_objective_pitch_role_risk_count"
            ],
            "outside_soloing_repair_source_pitch_role_risk_count_before": listening[
                "outside_soloing_repair_source_pitch_role_risk_count_before"
            ],
            "outside_soloing_repair_source_pitch_role_risk_count_after": listening[
                "outside_soloing_repair_source_pitch_role_risk_count_after"
            ],
            "outside_soloing_repair_source_pitch_role_risk_delta": listening[
                "outside_soloing_repair_source_pitch_role_risk_delta"
            ],
            "outside_soloing_repair_source_targeted": listening[
                "outside_soloing_repair_source_targeted"
            ],
            "outside_soloing_repair_source_residual_risk_preserved": listening[
                "outside_soloing_repair_source_residual_risk_preserved"
            ],
            "outside_soloing_repair_pitch_role_risk_count_after": listening[
                "outside_soloing_repair_pitch_role_risk_count_after"
            ],
            "outside_soloing_repair_pitch_role_risk_delta": listening[
                "outside_soloing_repair_pitch_role_risk_delta"
            ],
        },
        "artifact_manifest": {
            "cli_repaired_midi_candidates": cli["candidate_manifest"],
            "changed_ratio_repair_audio_candidates": audio["audio_manifest"],
        },
        "readiness": {
            "boundary": BOUNDARY,
            "mvp_delivery_package_completed": True,
            "technical_mvp_delivery_package_completed": True,
            "runnable_cli_ready": True,
            "local_artifact_paths_recorded": True,
            "raw_artifact_upload_required": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "phrase_bank_musical_quality_claimed": False,
            "audio_rendered_quality_claimed": False,
            "model_checkpoint_generation_quality_claimed": False,
            "model_direct_generation_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "technical MVP delivery package is recorded; refresh README final evidence without quality claims",
        },
        "not_proven": [
            "listening_review_completed",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo README final evidence refresh",
    }


def validate_delivery_package_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    require_delivery_completed: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    package = _dict(report.get("delivery_package"))
    artifact_manifest = _dict(report.get("artifact_manifest"))
    cli_candidates = _list(artifact_manifest.get("cli_repaired_midi_candidates"))
    audio_candidates = _list(artifact_manifest.get("changed_ratio_repair_audio_candidates"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloMvpDeliveryPackageError("unexpected next boundary")
    if require_delivery_completed and not bool(
        readiness.get("mvp_delivery_package_completed", False)
    ):
        raise StageBMidiToSoloMvpDeliveryPackageError("MVP delivery package completion required")
    if not bool(package.get("runnable_cli_ready", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError("runnable CLI readiness required")
    if _int(package.get("cli_candidate_count")) < 3 or len(cli_candidates) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("CLI candidate manifest below 3")
    if _int(package.get("changed_ratio_repair_wav_count")) < 3 or len(audio_candidates) < 3:
        raise StageBMidiToSoloMvpDeliveryPackageError("changed-ratio audio manifest below 3")
    if not bool(package.get("outside_soloing_repair_evidence_ready", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair evidence readiness required"
        )
    if _int(package.get("outside_soloing_repair_wav_count")) < 6:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair WAV count below 6"
        )
    if _int(package.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing repair residual pitch-role risk should be zero"
        )
    source_objective_risk = _int(
        package.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        package.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(
        package.get("outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_risk_delta = _int(
        package.get("outside_soloing_repair_source_pitch_role_risk_delta")
    )
    if source_objective_risk <= 0:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(package.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(package.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloMvpDeliveryPackageError(
            "outside-soloing source residual risk preservation required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloMvpDeliveryPackageError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="MVP delivery package readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "mvp_delivery_package_completed": bool(
            readiness.get("mvp_delivery_package_completed", False)
        ),
        "runnable_cli_ready": bool(package.get("runnable_cli_ready", False)),
        "input_to_ranked_midi_ready": bool(package.get("input_to_ranked_midi_ready", False)),
        "input_to_rendered_wav_evidence_ready": bool(
            package.get("input_to_rendered_wav_evidence_ready", False)
        ),
        "changed_ratio_repair_audio_evidence_ready": bool(
            package.get("changed_ratio_repair_audio_evidence_ready", False)
        ),
        "outside_soloing_repair_evidence_ready": bool(
            package.get("outside_soloing_repair_evidence_ready", False)
        ),
        "cli_candidate_count": _int(package.get("cli_candidate_count")),
        "changed_ratio_repair_wav_count": _int(package.get("changed_ratio_repair_wav_count")),
        "outside_soloing_repair_wav_count": _int(
            package.get("outside_soloing_repair_wav_count")
        ),
        "outside_soloing_repair_changed_note_total": _int(
            package.get("outside_soloing_repair_changed_note_total")
        ),
        "outside_soloing_repair_source_objective_pitch_role_risk_count": source_objective_risk,
        "outside_soloing_repair_source_pitch_role_risk_count_before": source_risk_before,
        "outside_soloing_repair_source_pitch_role_risk_count_after": source_risk_after,
        "outside_soloing_repair_source_pitch_role_risk_delta": source_risk_delta,
        "outside_soloing_repair_source_targeted": bool(
            package.get("outside_soloing_repair_source_targeted", True)
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            package.get("outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            package.get("outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            package.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
        "listening_review_quality_gap_open": bool(
            package.get("listening_review_quality_gap_open", False)
        ),
        "raw_artifact_upload_required": bool(readiness.get("raw_artifact_upload_required", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    package = report["delivery_package"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo MVP Delivery Package",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- technical MVP delivery package completed: `{_bool_token(package['technical_mvp_delivery_package_completed'])}`",
        f"- runnable CLI ready: `{_bool_token(package['runnable_cli_ready'])}`",
        f"- input to ranked MIDI ready: `{_bool_token(package['input_to_ranked_midi_ready'])}`",
        f"- input to rendered WAV evidence ready: `{_bool_token(package['input_to_rendered_wav_evidence_ready'])}`",
        f"- changed-ratio repair audio evidence ready: `{_bool_token(package['changed_ratio_repair_audio_evidence_ready'])}`",
        f"- outside-soloing repair evidence ready: `{_bool_token(package['outside_soloing_repair_evidence_ready'])}`",
        f"- listening review quality gap open: `{_bool_token(package['listening_review_quality_gap_open'])}`",
        "",
        "## Run Command",
        "",
        f"- `{package['cli_command']}`",
        "",
        "## Evidence",
        "",
        f"- CLI candidate count: `{package['cli_candidate_count']}`",
        f"- CLI objective-supported candidate count: `{package['cli_objective_supported_candidate_count']}`",
        f"- CLI dead-air ratio range: `{package['cli_dead_air_ratio_range'][0]:.4f}` - `{package['cli_dead_air_ratio_range'][1]:.4f}`",
        f"- changed-ratio repair WAV count: `{package['changed_ratio_repair_wav_count']}`",
        f"- changed-ratio repair max ratio / target: `{package['changed_ratio_repair_ratio_target'][0]:.4f}` / `{package['changed_ratio_repair_ratio_target'][1]:.4f}`",
        f"- changed-ratio repair max interval / target: `{package['changed_ratio_repair_interval_target'][0]}` / `{package['changed_ratio_repair_interval_target'][1]}`",
        f"- changed-ratio repair WAV duration range: `{package['changed_ratio_repair_wav_duration_range'][0]:.3f}s` - `{package['changed_ratio_repair_wav_duration_range'][1]:.3f}s`",
        f"- outside-soloing repair WAV count: `{package['outside_soloing_repair_wav_count']}`",
        f"- outside-soloing repair changed note total: `{package['outside_soloing_repair_changed_note_total']}`",
        f"- outside-soloing source objective pitch-role risk: `{package['outside_soloing_repair_source_objective_pitch_role_risk_count']}`",
        f"- outside-soloing source pitch-role risk before / after / delta: `{package['outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{package['outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{package['outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- outside-soloing source repair targeted: `{_bool_token(package['outside_soloing_repair_source_targeted'])}`",
        f"- outside-soloing source residual risk preserved: `{_bool_token(package['outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- outside-soloing current repair pitch-role risk after / delta: `{package['outside_soloing_repair_pitch_role_risk_count_after']}` / `{package['outside_soloing_repair_pitch_role_risk_delta']}`",
        "",
        "## Claim Boundary",
        "",
        f"- raw artifact upload required: `{_bool_token(readiness['raw_artifact_upload_required'])}`",
        f"- human/audio preference claimed: `{_bool_token(readiness['human_audio_preference_claimed'])}`",
        f"- MIDI-to-solo musical quality claimed: `{_bool_token(readiness['midi_to_solo_musical_quality_claimed'])}`",
        f"- broad trained model quality claimed: `{_bool_token(readiness['broad_trained_model_quality_claimed'])}`",
        f"- Brad style adaptation claimed: `{_bool_token(readiness['brad_style_adaptation_claimed'])}`",
        "",
        "## Next",
        "",
        f"- `{report['next_recommended_issue']}`",
    ]
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build MIDI-to-solo MVP delivery package manifest")
    parser.add_argument("--listening_review_quality_gap", type=str, required=True)
    parser.add_argument("--cli_mvp_package", type=str, required=True)
    parser.add_argument("--changed_ratio_audio_package", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_mvp_delivery_package",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=738)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_delivery_completed", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_delivery_package_report(
        listening_review_quality_gap=read_json(Path(args.listening_review_quality_gap)),
        cli_mvp_package=read_json(Path(args.cli_mvp_package)),
        changed_ratio_audio_package=read_json(Path(args.changed_ratio_audio_package)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_delivery_package_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_delivery_completed=bool(args.require_delivery_completed),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_delivery_package.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_mvp_delivery_package_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_mvp_delivery_package.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
