"""Decide the next quality-gap target after the MIDI-to-solo MVP completion audit."""

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
from scripts.audit_stage_b_midi_to_solo_mvp_completion import (  # noqa: E402
    BOUNDARY as MVP_COMPLETION_AUDIT_BOUNDARY,
)


class StageBMidiToSoloQualityGapDecisionError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_quality_gap_decision"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment"
SELECTED_TARGET = "model_conditioned_input_path_quality_alignment"
SCHEMA_VERSION = "stage_b_midi_to_solo_quality_gap_decision_v1"


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


def validate_mvp_completion_audit(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != MVP_COMPLETION_AUDIT_BOUNDARY:
        raise StageBMidiToSoloQualityGapDecisionError("MVP completion audit boundary required")
    readiness = _dict(report.get("readiness"))
    audit = _dict(report.get("completion_audit"))
    evidence = _dict(report.get("current_evidence"))
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != BOUNDARY:
        raise StageBMidiToSoloQualityGapDecisionError("MVP audit should route to quality gap decision")
    if not bool(readiness.get("mvp_completion_audit_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("MVP completion audit completion required")
    if not bool(audit.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("technical model-core MVP completion required")
    if bool(audit.get("musical_quality_mvp_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("musical quality should remain incomplete")
    if bool(audit.get("human_audio_preference_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("human/audio preference should remain incomplete")
    if bool(audit.get("product_mvp_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("product MVP should remain incomplete")
    if _int(evidence.get("exported_candidate_count")) < 3:
        raise StageBMidiToSoloQualityGapDecisionError("exported candidate count below 3")
    if _int(evidence.get("rendered_audio_file_count")) < 3:
        raise StageBMidiToSoloQualityGapDecisionError("rendered WAV count below 3")
    if _int(evidence.get("objective_strict_valid_sample_count")) != _int(evidence.get("objective_sample_count")):
        raise StageBMidiToSoloQualityGapDecisionError("objective strict valid count must match sample count")
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloQualityGapDecisionError(f"unexpected quality claim: {claimed}")
    return {
        "technical_model_core_mvp_completed": True,
        "musical_quality_mvp_completed": False,
        "human_audio_preference_completed": False,
        "product_mvp_completed": False,
        "generation_source": str(evidence.get("generation_source") or ""),
        "exported_candidate_count": _int(evidence.get("exported_candidate_count")),
        "rendered_audio_file_count": _int(evidence.get("rendered_audio_file_count")),
        "objective_sample_count": _int(evidence.get("objective_sample_count")),
        "objective_strict_valid_sample_count": _int(evidence.get("objective_strict_valid_sample_count")),
        "objective_dead_air_failure_count": _int(evidence.get("objective_dead_air_failure_count")),
        "objective_avg_postprocess_removal_ratio": _float(
            evidence.get("objective_avg_postprocess_removal_ratio")
        ),
        "objective_target_avg_postprocess_removal_ratio": _float(
            evidence.get("objective_target_avg_postprocess_removal_ratio")
        ),
    }


def select_quality_gap_target(audit_summary: dict[str, Any]) -> dict[str, Any]:
    generation_source = str(audit_summary.get("generation_source") or "")
    fallback_path_active = generation_source == "context_conditioned_fallback"
    target = SELECTED_TARGET if fallback_path_active else "listening_review_quality_gap"
    next_boundary = NEXT_BOUNDARY if fallback_path_active else "stage_b_midi_to_solo_listening_review_quality_gap"
    reason = (
        "input-to-WAV path still uses context_conditioned_fallback while selected-scale objective repair is separate"
        if fallback_path_active
        else "model-conditioned path is available; listening review gap remains"
    )
    return {
        "selected_target": target,
        "selected_next_boundary": next_boundary,
        "fallback_path_active": fallback_path_active,
        "quality_gap_reason": reason,
        "human_review_required_now": False,
    }


def build_quality_gap_decision_report(
    *,
    mvp_completion_audit: dict[str, Any],
    output_dir: Path,
    issue_number: int,
) -> dict[str, Any]:
    audit_summary = validate_mvp_completion_audit(mvp_completion_audit)
    target = select_quality_gap_target(audit_summary)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "issue_number": int(issue_number),
        "boundary": BOUNDARY,
        "source_boundary": MVP_COMPLETION_AUDIT_BOUNDARY,
        "mvp_completion_summary": audit_summary,
        "quality_gap": {
            "technical_model_core_mvp_completed": True,
            "musical_quality_mvp_completed": False,
            "human_audio_preference_completed": False,
            "product_mvp_completed": False,
            "fallback_path_active": bool(target["fallback_path_active"]),
            "model_conditioned_input_path_alignment_required": bool(target["fallback_path_active"]),
            "human_review_required_now": False,
        },
        "selected_target": target,
        "readiness": {
            "boundary": BOUNDARY,
            "quality_gap_decision_completed": True,
            "selected_target": str(target["selected_target"]),
            "next_boundary_selected": str(target["selected_next_boundary"]),
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": str(target["selected_next_boundary"]),
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": str(target["quality_gap_reason"]),
        },
        "not_proven": [
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned input path quality alignment",
    }


def validate_quality_gap_decision_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_target: str | None,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    quality_gap = _dict(report.get("quality_gap"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloQualityGapDecisionError(f"expected boundary {expected_boundary}, got {boundary}")
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloQualityGapDecisionError("unexpected next boundary")
    if expected_target and str(readiness.get("selected_target") or "") != expected_target:
        raise StageBMidiToSoloQualityGapDecisionError("unexpected selected target")
    if not bool(readiness.get("quality_gap_decision_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("quality gap decision completion required")
    if not bool(quality_gap.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloQualityGapDecisionError("technical model-core MVP completion required")
    if bool(quality_gap.get("musical_quality_mvp_completed", True)):
        raise StageBMidiToSoloQualityGapDecisionError("musical quality should remain incomplete")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloQualityGapDecisionError("critical user input should not be required")
    if require_no_quality_claim:
        blocked = [
            "human_audio_preference_claimed",
            "midi_to_solo_musical_quality_claimed",
            "broad_trained_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_claimed",
        ]
        claimed = [name for name in blocked if bool(readiness.get(name, True))]
        if claimed:
            raise StageBMidiToSoloQualityGapDecisionError(f"unexpected quality claim: {claimed}")
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_target": str(readiness.get("selected_target") or ""),
        "fallback_path_active": bool(quality_gap.get("fallback_path_active", False)),
        "model_conditioned_input_path_alignment_required": bool(
            quality_gap.get("model_conditioned_input_path_alignment_required", False)
        ),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "technical_model_core_mvp_completed": bool(
            quality_gap.get("technical_model_core_mvp_completed", False)
        ),
        "musical_quality_mvp_completed": bool(quality_gap.get("musical_quality_mvp_completed", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["mvp_completion_summary"]
    quality_gap = report["quality_gap"]
    selected = report["selected_target"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Quality Gap Decision",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected target: `{selected['selected_target']}`",
        f"- fallback path active: `{_bool_token(quality_gap['fallback_path_active'])}`",
        f"- human review required now: `{_bool_token(quality_gap['human_review_required_now'])}`",
        "",
        "## Evidence",
        "",
        f"- technical model-core MVP completed: `{_bool_token(summary['technical_model_core_mvp_completed'])}`",
        f"- musical quality MVP completed: `{_bool_token(summary['musical_quality_mvp_completed'])}`",
        f"- generation source: `{summary['generation_source']}`",
        f"- exported candidates: `{summary['exported_candidate_count']}`",
        f"- rendered WAV files: `{summary['rendered_audio_file_count']}`",
        f"- objective strict/sample: `{summary['objective_strict_valid_sample_count']}` / `{summary['objective_sample_count']}`",
        f"- objective dead-air failure: `{summary['objective_dead_air_failure_count']}`",
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
    parser = argparse.ArgumentParser(description="Decide next MIDI-to-solo quality-gap target")
    parser.add_argument("--mvp_completion_audit", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_quality_gap_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=618)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_target", type=str, default="")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_quality_gap_decision_report(
        mvp_completion_audit=read_json(Path(args.mvp_completion_audit)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_quality_gap_decision_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_target=str(args.expected_target or ""),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_quality_gap_decision.json", report)
    write_json(output_dir / "stage_b_midi_to_solo_quality_gap_decision_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_quality_gap_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
