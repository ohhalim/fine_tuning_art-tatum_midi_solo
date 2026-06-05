"""Decide the model-conditioned input-path alignment boundary."""

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
    NEXT_BOUNDARY as SOURCE_NEXT_BOUNDARY,
    SELECTED_TARGET,
)


class StageBMidiToSoloModelConditionedInputPathAlignmentError(ValueError):
    pass


BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment"
NEXT_BOUNDARY = "stage_b_midi_to_solo_model_conditioned_input_path_probe"
SELECTED_PROBE_TARGET = "replace_fallback_with_model_conditioned_input_path_probe"
SCHEMA_VERSION = "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment_v1"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _bool_token(value: Any) -> str:
    return "true" if bool(value) else "false"


def validate_quality_gap_decision(report: dict[str, Any]) -> dict[str, Any]:
    if str(report.get("boundary") or "") != QUALITY_GAP_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "quality gap decision boundary required"
        )
    readiness = _dict(report.get("readiness"))
    quality_gap = _dict(report.get("quality_gap"))
    selected = _dict(report.get("selected_target"))
    decision = _dict(report.get("decision"))
    if str(decision.get("next_boundary") or "") != SOURCE_NEXT_BOUNDARY:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "quality gap decision must route to model-conditioned input path alignment"
        )
    if str(readiness.get("selected_target") or "") != SELECTED_TARGET:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "model-conditioned input path alignment target required"
        )
    if not bool(readiness.get("quality_gap_decision_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "quality gap decision completion required"
        )
    if not bool(quality_gap.get("technical_model_core_mvp_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "technical model-core MVP completion required"
        )
    if not bool(quality_gap.get("fallback_path_active", False)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "fallback path should be active for this alignment boundary"
        )
    if not bool(quality_gap.get("model_conditioned_input_path_alignment_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "model-conditioned input path alignment should be required"
        )
    if bool(quality_gap.get("musical_quality_mvp_completed", True)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            "musical quality MVP should remain incomplete"
        )
    blocked_claims = [
        "human_audio_preference_claimed",
        "midi_to_solo_musical_quality_claimed",
        "broad_trained_model_quality_claimed",
        "brad_style_adaptation_claimed",
        "production_ready_claimed",
    ]
    claimed = [name for name in blocked_claims if bool(readiness.get(name, False))]
    if claimed:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            f"unexpected quality claim: {claimed}"
        )
    return {
        "source_boundary": QUALITY_GAP_BOUNDARY,
        "technical_model_core_mvp_completed": True,
        "fallback_path_active": True,
        "model_conditioned_input_path_alignment_required": True,
        "selected_quality_gap_target": str(readiness.get("selected_target") or ""),
        "human_review_required_now": bool(readiness.get("human_review_required_now", False)),
    }


def build_alignment_report(
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
        "source_boundary": QUALITY_GAP_BOUNDARY,
        "alignment_source": source,
        "alignment_requirements": {
            "replace_context_conditioned_fallback_in_input_to_wav_path": True,
            "reuse_selected_scale_objective_repair_guardrails": True,
            "preserve_ranked_midi_export_min_count": 3,
            "preserve_rendered_wav_min_count": 3,
            "preserve_objective_strict_sample_support": True,
            "preserve_no_quality_claim": True,
        },
        "alignment_decision": {
            "model_conditioned_input_path_aligned": False,
            "fallback_replacement_probe_required": True,
            "selected_probe_target": SELECTED_PROBE_TARGET,
            "selected_next_boundary": NEXT_BOUNDARY,
            "human_review_required_now": False,
        },
        "readiness": {
            "boundary": BOUNDARY,
            "model_conditioned_input_path_quality_alignment_decision_completed": True,
            "model_conditioned_input_path_aligned": False,
            "fallback_replacement_probe_required": True,
            "selected_probe_target": SELECTED_PROBE_TARGET,
            "human_review_required_now": False,
            "human_audio_preference_claimed": False,
            "midi_to_solo_musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_claimed": False,
        },
        "decision": {
            "current_boundary": BOUNDARY,
            "next_boundary": NEXT_BOUNDARY,
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "reason": (
                "input-to-WAV path still depends on context-conditioned fallback; next probe should "
                "replace that path with a model-conditioned input-path candidate while preserving objective guardrails"
            ),
        },
        "not_proven": [
            "model_conditioned_input_path_quality",
            "human_audio_preference",
            "midi_to_solo_musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B MIDI-to-solo model-conditioned input path probe",
    }


def validate_alignment_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_next_boundary: str | None,
    expected_probe_target: str | None,
    require_probe_required: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    boundary = str(report.get("boundary") or "")
    readiness = _dict(report.get("readiness"))
    decision = _dict(report.get("decision"))
    alignment_decision = _dict(report.get("alignment_decision"))
    if expected_boundary and boundary != expected_boundary:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
            f"expected boundary {expected_boundary}, got {boundary}"
        )
    if expected_next_boundary and str(decision.get("next_boundary") or "") != expected_next_boundary:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError("unexpected next boundary")
    if expected_probe_target and str(readiness.get("selected_probe_target") or "") != expected_probe_target:
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError("unexpected probe target")
    if not bool(readiness.get("model_conditioned_input_path_quality_alignment_decision_completed", False)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError("alignment decision completion required")
    if bool(readiness.get("model_conditioned_input_path_aligned", True)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError("input path should not be marked aligned yet")
    if require_probe_required and not bool(readiness.get("fallback_replacement_probe_required", False)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError("fallback replacement probe required")
    if bool(decision.get("critical_user_input_required", True)):
        raise StageBMidiToSoloModelConditionedInputPathAlignmentError("critical user input should not be required")
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
            raise StageBMidiToSoloModelConditionedInputPathAlignmentError(
                f"unexpected quality claim: {claimed}"
            )
    return {
        "boundary": boundary,
        "next_boundary": str(decision.get("next_boundary") or ""),
        "selected_probe_target": str(readiness.get("selected_probe_target") or ""),
        "model_conditioned_input_path_aligned": bool(
            readiness.get("model_conditioned_input_path_aligned", True)
        ),
        "fallback_replacement_probe_required": bool(
            readiness.get("fallback_replacement_probe_required", False)
        ),
        "human_review_required_now": bool(readiness.get("human_review_required_now", True)),
        "human_audio_preference_claimed": bool(readiness.get("human_audio_preference_claimed", True)),
        "midi_to_solo_musical_quality_claimed": bool(
            readiness.get("midi_to_solo_musical_quality_claimed", True)
        ),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
        "fallback_replacement_probe_required_from_decision": bool(
            alignment_decision.get("fallback_replacement_probe_required", False)
        ),
    }


def markdown_report(report: dict[str, Any]) -> str:
    source = report["alignment_source"]
    requirements = report["alignment_requirements"]
    readiness = report["readiness"]
    decision = report["decision"]
    lines = [
        "# Stage B MIDI-to-Solo Model-Conditioned Input Path Quality Alignment",
        "",
        "## Summary",
        "",
        f"- boundary: `{report['boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- selected probe target: `{readiness['selected_probe_target']}`",
        f"- model-conditioned input path aligned: `{_bool_token(readiness['model_conditioned_input_path_aligned'])}`",
        f"- fallback replacement probe required: `{_bool_token(readiness['fallback_replacement_probe_required'])}`",
        f"- human review required now: `{_bool_token(readiness['human_review_required_now'])}`",
        "",
        "## Source",
        "",
        f"- source boundary: `{source['source_boundary']}`",
        f"- technical model-core MVP completed: `{_bool_token(source['technical_model_core_mvp_completed'])}`",
        f"- fallback path active: `{_bool_token(source['fallback_path_active'])}`",
        f"- model-conditioned input path alignment required: `{_bool_token(source['model_conditioned_input_path_alignment_required'])}`",
        "",
        "## Alignment Requirements",
        "",
    ]
    for key, value in requirements.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
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
    )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide model-conditioned input-path alignment boundary")
    parser.add_argument("--quality_gap_decision", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--issue_number", type=int, default=620)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--expected_probe_target", type=str, default="")
    parser.add_argument("--require_probe_required", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_alignment_report(
        quality_gap_decision=read_json(Path(args.quality_gap_decision)),
        output_dir=output_dir,
        issue_number=int(args.issue_number),
    )
    summary = validate_alignment_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_next_boundary=str(args.expected_next_boundary or ""),
        expected_probe_target=str(args.expected_probe_target or ""),
        require_probe_required=bool(args.require_probe_required),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment.json", report)
    write_json(
        output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment_validation_summary.json",
        summary,
    )
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_midi_to_solo_model_conditioned_input_path_quality_alignment.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
