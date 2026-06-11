"""Separate the remaining listening-review quality gap after technical MVP evidence."""

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
from scripts.decide_stage_b_midi_to_solo_quality_gap import (  # noqa: E402
    BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_KEYS,
    BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS,
    BOUNDARY as SOURCE_BOUNDARY,
    LISTENING_REVIEW_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    LISTENING_REVIEW_TARGET,
)


class StageBMidiToSoloListeningReviewQualityGapError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_listening_review_quality_gap"
NEXT_BOUNDARY = "stage_b_midi_to_solo_mvp_delivery_package"
SELECTED_TARGET = "mvp_delivery_package"
SCHEMA_VERSION = "stage_b_midi_to_solo_listening_review_quality_gap_v3"

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
        raise StageBMidiToSoloListeningReviewQualityGapError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def _source_context_fields(container: dict[str, Any], *, label: str) -> dict[str, Any]:
    for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS:
        if key not in container:
            raise StageBMidiToSoloListeningReviewQualityGapError(
                f"{label} source-context field required: {key}"
            )
    missing_preserved = [
        key for key in BRIDGE_SOURCE_CONTEXT_PRESERVED_KEYS if not bool(container.get(key))
    ]
    if missing_preserved:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            f"{label} source-context preserved field must be true: {missing_preserved}"
        )
    return {key: container[key] for key in BRIDGE_REQUIRED_SOURCE_CONTEXT_KEYS}


def validate_quality_gap_decision(report: dict[str, Any]) -> dict[str, Any]:
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    quality_gap = _dict(report.get("quality_gap"))
    selected = _dict(report.get("selected_target"))
    summary = _dict(report.get("mvp_completion_summary"))

    if str(report.get("boundary") or "") != SOURCE_BOUNDARY:
        raise StageBMidiToSoloListeningReviewQualityGapError("quality gap decision boundary required")
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "quality gap decision should route to listening review quality gap"
        )
    if not bool(readiness.get("quality_gap_decision_completed", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError("quality gap decision completion required")
    if str(readiness.get("selected_target") or "") != LISTENING_REVIEW_TARGET:
        raise StageBMidiToSoloListeningReviewQualityGapError("listening review target required")
    if str(selected.get("selected_target") or "") != LISTENING_REVIEW_TARGET:
        raise StageBMidiToSoloListeningReviewQualityGapError("selected listening target required")
    if not bool(quality_gap.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError("technical model-core MVP completion required")
    if not bool(quality_gap.get("phrase_bank_cli_technical_path_completed", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError("phrase-bank CLI technical path required")
    if not bool(quality_gap.get("model_conditioned_pitch_contour_objective_completed", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError("pitch-contour objective completion required")
    if not bool(
        quality_gap.get(
            "model_conditioned_pitch_contour_changed_ratio_repair_objective_completed",
            False,
        )
    ):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "changed-ratio repair objective completion required"
        )
    if not bool(quality_gap.get("pitch_contour_changed_ratio_repair_objective_path_ready", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "changed-ratio repair objective path readiness required"
        )
    if not bool(quality_gap.get("pitch_contour_changed_ratio_repair_target_supported", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "changed-ratio repair target support required"
        )
    if not bool(quality_gap.get("outside_soloing_repair_objective_completed", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair objective completion required"
        )
    if not bool(readiness.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair source context readiness required"
        )
    if not bool(quality_gap.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair source context preservation required"
        )
    source_context = _source_context_fields(
        quality_gap,
        label="quality gap decision",
    )
    if not bool(quality_gap.get("outside_soloing_repair_objective_path_ready", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair objective path readiness required"
        )
    if not bool(quality_gap.get("outside_soloing_repair_target_supported", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair target support required"
        )
    if bool(quality_gap.get("model_conditioned_input_path_alignment_required", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "model-conditioned input path alignment should not be selected"
        )
    if bool(quality_gap.get("musical_quality_mvp_completed", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError("musical quality should remain incomplete")
    if bool(quality_gap.get("human_audio_preference_completed", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError("human/audio preference should remain incomplete")
    if bool(quality_gap.get("product_mvp_completed", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError("product MVP should remain incomplete")
    if _int(
        summary.get("model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count")
    ) < 3:
        raise StageBMidiToSoloListeningReviewQualityGapError("changed-ratio repair rendered WAV count below 3")
    if _int(summary.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval")) > _int(
        summary.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold")
    ):
        raise StageBMidiToSoloListeningReviewQualityGapError("changed-ratio repair interval threshold exceeded")
    if _float(
        summary.get("model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio")
    ) > _float(
        summary.get("model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio")
    ):
        raise StageBMidiToSoloListeningReviewQualityGapError("changed-ratio repair ratio threshold exceeded")
    if _int(summary.get("outside_soloing_repair_rendered_audio_file_count")) < 6:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair rendered WAV count below 6"
        )
    if _int(summary.get("outside_soloing_repair_pitch_role_risk_count_after")) != 0:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair residual pitch-role risk should be zero"
        )
    source_objective_risk = _int(
        quality_gap.get("outside_soloing_repair_source_objective_pitch_role_risk_count")
    )
    source_risk_before = _int(
        quality_gap.get("outside_soloing_repair_source_pitch_role_risk_count_before")
    )
    source_risk_after = _int(
        quality_gap.get("outside_soloing_repair_source_pitch_role_risk_count_after")
    )
    source_risk_delta = _int(
        quality_gap.get("outside_soloing_repair_source_pitch_role_risk_delta")
    )
    if source_objective_risk <= 0:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(quality_gap.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(quality_gap.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source residual risk preservation required"
        )
    outside_targets = [
        "outside_soloing_repair_objective_path_supported",
        "outside_soloing_repair_weak_landing_target_supported",
        "outside_soloing_repair_final_landing_target_supported",
        "outside_soloing_repair_non_chord_run_target_supported",
    ]
    missing_outside_targets = [name for name in outside_targets if not bool(summary.get(name, False))]
    if missing_outside_targets:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            f"outside-soloing repair targets missing: {missing_outside_targets}"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError("critical user input should not be required")
    _require_no_quality_claim(readiness, label="quality gap decision readiness")

    return {
        "boundary": SOURCE_BOUNDARY,
        "technical_model_core_mvp_completed": True,
        "phrase_bank_cli_technical_path_completed": True,
        "model_conditioned_pitch_contour_objective_completed": True,
        "changed_ratio_repair_objective_completed": True,
        "outside_soloing_repair_objective_completed": True,
        "outside_soloing_repair_source_context_preserved": bool(
            quality_gap.get("outside_soloing_repair_source_context_preserved", False)
        ),
        "fallback_path_active": bool(quality_gap.get("fallback_path_active", False)),
        "model_conditioned_input_path_alignment_required": bool(
            quality_gap.get("model_conditioned_input_path_alignment_required", False)
        ),
        "rendered_audio_file_count": _int(
            summary.get("model_conditioned_pitch_contour_changed_ratio_repair_rendered_audio_file_count")
        ),
        "max_repaired_interval": _int(
            summary.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval")
        ),
        "max_interval_threshold": _int(
            summary.get("model_conditioned_pitch_contour_changed_ratio_repair_max_interval_threshold")
        ),
        "max_repaired_pitch_changed_ratio": _float(
            summary.get("model_conditioned_pitch_contour_changed_ratio_repair_max_pitch_changed_ratio")
        ),
        "target_max_pitch_changed_ratio": _float(
            summary.get("model_conditioned_pitch_contour_changed_ratio_repair_target_max_pitch_changed_ratio")
        ),
        "outside_soloing_repair_objective_path_ready": bool(
            quality_gap.get("outside_soloing_repair_objective_path_ready", False)
        ),
        "outside_soloing_repair_target_supported": bool(
            quality_gap.get("outside_soloing_repair_target_supported", False)
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
            quality_gap.get("outside_soloing_repair_source_targeted", True)
        ),
        "outside_soloing_repair_source_residual_risk_preserved": bool(
            quality_gap.get("outside_soloing_repair_source_residual_risk_preserved", False)
        ),
        "outside_soloing_repair_pitch_role_risk_count_after": _int(
            summary.get("outside_soloing_repair_pitch_role_risk_count_after")
        ),
        "outside_soloing_repair_pitch_role_risk_delta": _int(
            summary.get("outside_soloing_repair_pitch_role_risk_delta")
        ),
        "outside_soloing_repair_objective_path_supported": bool(
            summary.get("outside_soloing_repair_objective_path_supported", False)
        ),
        "outside_soloing_repair_weak_landing_target_supported": bool(
            summary.get("outside_soloing_repair_weak_landing_target_supported", False)
        ),
        "outside_soloing_repair_final_landing_target_supported": bool(
            summary.get("outside_soloing_repair_final_landing_target_supported", False)
        ),
        "outside_soloing_repair_non_chord_run_target_supported": bool(
            summary.get("outside_soloing_repair_non_chord_run_target_supported", False)
        ),
        **source_context,
        "human_review_required_now": bool(quality_gap.get("human_review_required_now", False)),
        "musical_quality_mvp_completed": False,
        "human_audio_preference_completed": False,
        "product_mvp_completed": False,
    }


def build_listening_review_quality_gap_report(
    *,
    quality_gap_decision: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    source = validate_quality_gap_decision(quality_gap_decision)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["boundary"],
        "quality_gap_summary": {
            **source,
            "listening_review_quality_gap_open": True,
            "technical_mvp_delivery_package_ready": True,
            "human_audio_preference_gap_open": True,
            "musical_quality_gap_open": True,
        },
        "selected_next_target": {
            "selected_target": SELECTED_TARGET,
            "selected_next_boundary": NEXT_BOUNDARY,
            "reason": (
                "technical MIDI-to-solo path, changed-ratio repair, and outside-soloing repair objective evidence are ready; "
                "delivery package can be prepared while listening and musical quality claims remain excluded"
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "listening_review_quality_gap_completed": True,
            "technical_mvp_delivery_package_ready": True,
            "listening_review_quality_gap_open": True,
            "human_review_required_now": False,
            "outside_soloing_repair_source_context_preserved": bool(
                source["outside_soloing_repair_source_context_preserved"]
            ),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
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
            "reason": "listening review gap is open, but technical delivery package preparation does not require a quality claim",
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
        "next_recommended_issue": "Stage B MIDI-to-solo MVP delivery package source-context refresh",
    }


def validate_listening_review_quality_gap_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_delivery_package_ready: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_next_target"))
    summary = _dict(report.get("quality_gap_summary"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloListeningReviewQualityGapError("unexpected next boundary")
    if expected_target and str(selected.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloListeningReviewQualityGapError("unexpected selected target")
    if not bool(readiness.get("listening_review_quality_gap_completed", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError("quality gap boundary completion required")
    if require_delivery_package_ready and not bool(
        readiness.get("technical_mvp_delivery_package_ready", False)
    ):
        raise StageBMidiToSoloListeningReviewQualityGapError("delivery package readiness required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError("critical user input should not be required")
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="listening review quality gap readiness")
    if not bool(readiness.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair source context readiness required"
        )
    if not bool(summary.get("outside_soloing_repair_source_context_preserved", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing repair source context preservation required"
        )
    source_context = _source_context_fields(
        summary,
        label="listening review quality gap",
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
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source objective pitch-role risk count required"
        )
    if source_risk_after > source_risk_before:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source pitch-role risk should not increase"
        )
    if source_risk_delta != source_risk_before - source_risk_after:
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source pitch-role risk delta mismatch"
        )
    if bool(summary.get("outside_soloing_repair_source_targeted", True)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source repair should remain non-targeted"
        )
    if not bool(summary.get("outside_soloing_repair_source_residual_risk_preserved", False)):
        raise StageBMidiToSoloListeningReviewQualityGapError(
            "outside-soloing source residual risk preservation required"
        )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(selected.get("selected_target") or ""),
        "technical_model_core_mvp_completed": bool(
            summary.get("technical_model_core_mvp_completed", False)
        ),
        "changed_ratio_repair_objective_completed": bool(
            summary.get("changed_ratio_repair_objective_completed", False)
        ),
        "outside_soloing_repair_objective_completed": bool(
            summary.get("outside_soloing_repair_objective_completed", False)
        ),
        "outside_soloing_repair_source_context_preserved": bool(
            summary.get("outside_soloing_repair_source_context_preserved", False)
        ),
        "rendered_audio_file_count": _int(summary.get("rendered_audio_file_count")),
        "max_repaired_interval": _int(summary.get("max_repaired_interval")),
        "max_interval_threshold": _int(summary.get("max_interval_threshold")),
        "max_repaired_pitch_changed_ratio": _float(summary.get("max_repaired_pitch_changed_ratio")),
        "target_max_pitch_changed_ratio": _float(summary.get("target_max_pitch_changed_ratio")),
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
        "outside_soloing_repair_objective_path_supported": bool(
            summary.get("outside_soloing_repair_objective_path_supported", False)
        ),
        "outside_soloing_repair_weak_landing_target_supported": bool(
            summary.get("outside_soloing_repair_weak_landing_target_supported", False)
        ),
        "outside_soloing_repair_final_landing_target_supported": bool(
            summary.get("outside_soloing_repair_final_landing_target_supported", False)
        ),
        "outside_soloing_repair_non_chord_run_target_supported": bool(
            summary.get("outside_soloing_repair_non_chord_run_target_supported", False)
        ),
        **source_context,
        "listening_review_quality_gap_open": bool(
            summary.get("listening_review_quality_gap_open", False)
        ),
        "technical_mvp_delivery_package_ready": bool(
            readiness.get("technical_mvp_delivery_package_ready", False)
        ),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["quality_gap_summary"]
    selected = report["selected_next_target"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Listening Review Quality Gap",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- source boundary: `{report['source_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- listening review quality gap open: `{_bool_token(summary['listening_review_quality_gap_open'])}`",
        f"- technical MVP delivery package ready: `{_bool_token(summary['technical_mvp_delivery_package_ready'])}`",
        f"- human review required now: `{_bool_token(readiness['human_review_required_now'])}`",
        "",
        "## Evidence",
        "",
        f"- technical model-core MVP completed: `{_bool_token(summary['technical_model_core_mvp_completed'])}`",
        f"- phrase-bank CLI technical path completed: `{_bool_token(summary['phrase_bank_cli_technical_path_completed'])}`",
        f"- model-conditioned pitch-contour objective completed: `{_bool_token(summary['model_conditioned_pitch_contour_objective_completed'])}`",
        f"- changed-ratio repair objective completed: `{_bool_token(summary['changed_ratio_repair_objective_completed'])}`",
        f"- outside-soloing repair objective completed: `{_bool_token(summary['outside_soloing_repair_objective_completed'])}`",
        f"- outside-soloing repair source context preserved: `{_bool_token(summary['outside_soloing_repair_source_context_preserved'])}`",
        f"- rendered WAV files: `{summary['rendered_audio_file_count']}`",
        f"- changed-ratio repair max interval / threshold: `{summary['max_repaired_interval']}` / `{summary['max_interval_threshold']}`",
        f"- changed-ratio repair max ratio / target: `{summary['max_repaired_pitch_changed_ratio']:.4f}` / `{summary['target_max_pitch_changed_ratio']:.4f}`",
        f"- outside-soloing repair objective path ready: `{_bool_token(summary['outside_soloing_repair_objective_path_ready'])}`",
        f"- outside-soloing repair target supported: `{_bool_token(summary['outside_soloing_repair_target_supported'])}`",
        f"- outside-soloing repair rendered WAV files: `{summary['outside_soloing_repair_rendered_audio_file_count']}`",
        f"- outside-soloing repair changed note total: `{summary['outside_soloing_repair_changed_note_total']}`",
        f"- outside-soloing source objective pitch-role risk: `{summary['outside_soloing_repair_source_objective_pitch_role_risk_count']}`",
        f"- outside-soloing source pitch-role risk before / after / delta: `{summary['outside_soloing_repair_source_pitch_role_risk_count_before']}` / `{summary['outside_soloing_repair_source_pitch_role_risk_count_after']}` / `{summary['outside_soloing_repair_source_pitch_role_risk_delta']}`",
        f"- outside-soloing source repair targeted: `{_bool_token(summary['outside_soloing_repair_source_targeted'])}`",
        f"- outside-soloing source residual risk preserved: `{_bool_token(summary['outside_soloing_repair_source_residual_risk_preserved'])}`",
        f"- outside-soloing current repair pitch-role risk after / delta: `{summary['outside_soloing_repair_pitch_role_risk_count_after']}` / `{summary['outside_soloing_repair_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source pitch-role risk: `{summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_objective_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up objective source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_objective_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_objective_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up objective source outside-soloing source context preserved: `{_bool_token(summary['followup_objective_source_outside_soloing_source_context_preserved'])}`",
        f"- follow-up repair sweep source outside-soloing source pitch-role risk: `{summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['followup_repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- follow-up repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['followup_repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- follow-up repair sweep source outside-soloing source context preserved: `{_bool_token(summary['followup_repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- bridge repair sweep source outside-soloing source pitch-role risk: `{summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_before']} -> {summary['repair_sweep_source_outside_soloing_source_pitch_role_risk_count_after']}`",
        f"- bridge repair sweep source outside-soloing current repair pitch-role risk after/delta: `{summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_count_after']} / {summary['repair_sweep_source_outside_soloing_current_pitch_role_risk_delta']}`",
        f"- bridge repair sweep source outside-soloing source context preserved: `{_bool_token(summary['repair_sweep_source_outside_soloing_source_context_preserved'])}`",
        f"- outside-soloing repair objective path supported: `{_bool_token(summary['outside_soloing_repair_objective_path_supported'])}`",
        f"- outside-soloing repair weak landing target supported: `{_bool_token(summary['outside_soloing_repair_weak_landing_target_supported'])}`",
        f"- outside-soloing repair final landing target supported: `{_bool_token(summary['outside_soloing_repair_final_landing_target_supported'])}`",
        f"- outside-soloing repair non-chord run target supported: `{_bool_token(summary['outside_soloing_repair_non_chord_run_target_supported'])}`",
        f"- musical quality MVP completed: `{_bool_token(summary['musical_quality_mvp_completed'])}`",
        f"- human/audio preference completed: `{_bool_token(summary['human_audio_preference_completed'])}`",
        f"- product MVP completed: `{_bool_token(summary['product_mvp_completed'])}`",
        "",
        "## Claim Boundary",
        "",
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
    parser = argparse.ArgumentParser(description="Decide remaining MIDI-to-solo listening review quality gap")
    parser.add_argument("--quality_gap_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_listening_review_quality_gap",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=736)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_delivery_package_ready", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_listening_review_quality_gap_report(
        quality_gap_decision=read_json(Path(args.quality_gap_decision)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_listening_review_quality_gap_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_delivery_package_ready=bool(args.require_delivery_package_ready),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_listening_review_quality_gap.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_listening_review_quality_gap_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_listening_review_quality_gap.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
