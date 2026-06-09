"""Decide the next boundary for model-conditioned pitch-contour changed-ratio review."""

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
    BOUNDARY as QUALITY_GAP_BOUNDARY,
    PITCH_CONTOUR_CHANGED_RATIO_NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    PITCH_CONTOUR_CHANGED_RATIO_TARGET as SOURCE_TARGET,
)


class StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_repair_probe"
SELECTED_TARGET = "lower_pitch_change_ratio_repair_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision_v1"

QUALITY_CLAIM_KEYS = [
    "human_audio_preference_claimed",
    "midi_to_solo_musical_quality_claimed",
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
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            f"unexpected quality claim in {label}: {claimed}"
        )


def validate_quality_gap_decision(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != QUALITY_GAP_BOUNDARY:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "quality gap decision boundary required"
        )
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    selected = _dict(report.get("selected_target"))
    quality_gap = _dict(report.get("quality_gap"))
    summary = _dict(report.get("mvp_completion_summary"))
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "quality gap decision should route to changed-ratio review"
        )
    if str(readiness.get("selected_target") or "") != SOURCE_TARGET:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "quality gap selected target mismatch"
        )
    if str(selected.get("selected_target") or "") != SOURCE_TARGET:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "selected target payload mismatch"
        )
    if not bool(readiness.get("quality_gap_decision_completed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "quality gap decision completion required"
        )
    if not bool(quality_gap.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "technical model-core MVP completion required"
        )
    if not bool(quality_gap.get("model_conditioned_pitch_contour_objective_completed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "model-conditioned pitch-contour objective completion required"
        )
    if not bool(quality_gap.get("pitch_contour_changed_ratio_review_required", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "pitch-contour changed-ratio review requirement expected"
        )
    if bool(quality_gap.get("model_conditioned_input_path_alignment_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "fallback alignment should not be selected for changed-ratio review"
        )
    if _int(summary.get("model_conditioned_pitch_contour_max_interval")) > _int(
        summary.get("model_conditioned_pitch_contour_max_interval_threshold")
    ):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "pitch-contour interval threshold exceeded"
        )
    if not bool(summary.get("model_conditioned_pitch_contour_target_supported", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "pitch-contour target support required"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "critical user input should not be required"
        )
    _require_no_quality_claim(readiness, label="quality gap readiness")
    return {
        "source_boundary": QUALITY_GAP_BOUNDARY,
        "technical_model_core_mvp_completed": bool(
            quality_gap.get("technical_model_core_mvp_completed", False)
        ),
        "model_conditioned_pitch_contour_objective_completed": bool(
            quality_gap.get("model_conditioned_pitch_contour_objective_completed", False)
        ),
        "fallback_path_active": bool(quality_gap.get("fallback_path_active", False)),
        "model_conditioned_input_path_alignment_required": bool(
            quality_gap.get("model_conditioned_input_path_alignment_required", True)
        ),
        "pitch_contour_changed_ratio_review_required": bool(
            quality_gap.get("pitch_contour_changed_ratio_review_required", False)
        ),
        "max_interval": _int(summary.get("model_conditioned_pitch_contour_max_interval")),
        "max_interval_threshold": _int(
            summary.get("model_conditioned_pitch_contour_max_interval_threshold")
        ),
        "pitch_contour_target_supported": bool(
            summary.get("model_conditioned_pitch_contour_target_supported", False)
        ),
        "audio_review_required": bool(
            summary.get("model_conditioned_pitch_contour_audio_review_required", False)
        ),
        "human_audio_preference_claimed": bool(
            readiness.get("human_audio_preference_claimed", False)
        ),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", False)
        ),
    }


def build_changed_ratio_review_decision_report(
    *,
    quality_gap_decision: dict[str, Any],
    output_dir: Path,
    issue_number: int,
    changed_ratio_review_threshold: float,
) -> dict[str, Any]:
    source = validate_quality_gap_decision(quality_gap_decision)
    repair_required = bool(source["pitch_contour_changed_ratio_review_required"])
    selected_target = SELECTED_TARGET if repair_required else "pitch_contour_listening_review"
    next_boundary = NEXT_BOUNDARY if repair_required else "stage_b_midi_to_solo_listening_review_quality_gap"
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": source["source_boundary"],
        "changed_ratio_review": {
            "technical_model_core_mvp_completed": bool(
                source["technical_model_core_mvp_completed"]
            ),
            "model_conditioned_pitch_contour_objective_completed": bool(
                source["model_conditioned_pitch_contour_objective_completed"]
            ),
            "fallback_path_active": bool(source["fallback_path_active"]),
            "model_conditioned_input_path_alignment_required": bool(
                source["model_conditioned_input_path_alignment_required"]
            ),
            "max_interval": _int(source["max_interval"]),
            "max_interval_threshold": _int(source["max_interval_threshold"]),
            "pitch_contour_target_supported": bool(source["pitch_contour_target_supported"]),
            "changed_ratio_review_threshold": float(changed_ratio_review_threshold),
            "changed_ratio_review_required": bool(repair_required),
            "repair_probe_required": bool(repair_required),
            "audio_review_required": bool(source["audio_review_required"]),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
        },
        "selected_target": {
            "selected_target": selected_target,
            "selected_next_boundary": next_boundary,
            "reason": (
                "pitch-contour interval target passed, but changed-ratio review remains; "
                "route to lower pitch-change repair probe before any quality claim"
                if repair_required
                else "pitch-contour changed-ratio review not required; route to listening quality gap"
            ),
        },
        "readiness": {
            "boundary": BOUNDARY,
            "changed_ratio_review_decision_completed": True,
            "selected_target": selected_target,
            "next_boundary_selected": next_boundary,
            "repair_probe_required": bool(repair_required),
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": next_boundary,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": "changed-ratio review selected the next boundary without quality claim",
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned pitch-contour changed-ratio repair probe",
    }


def validate_changed_ratio_review_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_repair_probe: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    review = _dict(report.get("changed_ratio_review"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "unexpected next boundary"
        )
    if expected_target and str(readiness.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "unexpected selected target"
        )
    if not bool(readiness.get("changed_ratio_review_decision_completed", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "changed-ratio review decision completion required"
        )
    if require_repair_probe and not bool(readiness.get("repair_probe_required", False)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "repair probe requirement expected"
        )
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloPitchContourChangedRatioReviewDecisionError(
            "critical user input should not be required"
        )
    if require_no_quality_claim:
        _require_no_quality_claim(readiness, label="changed-ratio review readiness")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(readiness.get("selected_target") or ""),
        "repair_probe_required": bool(readiness.get("repair_probe_required", False)),
        "technical_model_core_mvp_completed": bool(
            review.get("technical_model_core_mvp_completed", False)
        ),
        "model_conditioned_pitch_contour_objective_completed": bool(
            review.get("model_conditioned_pitch_contour_objective_completed", False)
        ),
        "model_conditioned_input_path_alignment_required": bool(
            review.get("model_conditioned_input_path_alignment_required", True)
        ),
        "max_interval": _int(review.get("max_interval")),
        "max_interval_threshold": _int(review.get("max_interval_threshold")),
        "pitch_contour_target_supported": bool(
            review.get("pitch_contour_target_supported", False)
        ),
        "changed_ratio_review_threshold": _float(review.get("changed_ratio_review_threshold")),
        "changed_ratio_review_required": bool(
            review.get("changed_ratio_review_required", False)
        ),
        "audio_review_required": bool(review.get("audio_review_required", False)),
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
    review = report["changed_ratio_review"]
    selected = report["selected_target"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Pitch-Contour Changed-Ratio Review Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- repair probe required: `{_bool_token(readiness['repair_probe_required'])}`",
        "",
        "## Evidence",
        "",
        f"- technical model-core MVP completed: `{_bool_token(review['technical_model_core_mvp_completed'])}`",
        f"- model-conditioned pitch-contour objective completed: `{_bool_token(review['model_conditioned_pitch_contour_objective_completed'])}`",
        f"- model-conditioned input path alignment required: `{_bool_token(review['model_conditioned_input_path_alignment_required'])}`",
        f"- max interval / threshold: `{review['max_interval']}` / `{review['max_interval_threshold']}`",
        f"- pitch-contour target supported: `{_bool_token(review['pitch_contour_target_supported'])}`",
        f"- changed-ratio review threshold: `{review['changed_ratio_review_threshold']}`",
        f"- changed-ratio review required: `{_bool_token(review['changed_ratio_review_required'])}`",
        f"- audio review required: `{_bool_token(review['audio_review_required'])}`",
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
    parser = argparse.ArgumentParser(description="Decide pitch-contour changed-ratio review boundary")
    parser.add_argument("--quality_gap_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=(
            "outputs/stage_b_midi_to_solo_model_conditioned_pitch_contour_"
            "changed_ratio_review_decision"
        ),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=716)
    parser.add_argument("--changed_ratio_review_threshold", type=float, default=0.5)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_repair_probe", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_changed_ratio_review_decision_report(
        quality_gap_decision=read_json(Path(args.quality_gap_decision)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
        changed_ratio_review_threshold=float(args.changed_ratio_review_threshold),
    )
    summary = validate_changed_ratio_review_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_repair_probe=bool(args.require_repair_probe),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision.json",
        report,
    )
    write_json(
        output_dir
        / (
            "stage_b_midi_to_solo_model_conditioned_pitch_contour_"
            "changed_ratio_review_decision_validation_summary.json"
        ),
        summary,
    )
    markdown = markdown_report(report)
    write_text(
        output_dir
        / "stage_b_midi_to_solo_model_conditioned_pitch_contour_changed_ratio_review_decision.md",
        markdown,
    )
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
