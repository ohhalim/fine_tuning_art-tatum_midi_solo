"""Decide phrase-continuation repair boundary after generic tiny checkpoint listening rejection."""

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
from scripts.run_stage_b_generic_tiny_checkpoint_generation_probe import (  # noqa: E402
    _bool_token,
    _dict,
)


class StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError(ValueError):
    pass


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def validate_user_listening_review(user_listening_review: dict[str, Any]) -> dict[str, Any]:
    review = _dict(user_listening_review.get("user_listening_review"))
    claim = _dict(user_listening_review.get("claim_boundary"))
    decision = _dict(user_listening_review.get("decision"))
    if str(claim.get("boundary") or "") != "generic_tiny_checkpoint_repair_audio_review_reject_all":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("reject-all audio review boundary required")
    if str(review.get("overall_decision") or "") != "reject_all":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("overall reject_all decision required")
    if str(review.get("candidate_decision") or "") != "reject":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("candidate reject decision required")
    if str(review.get("primary_failure") or "") != "plunk_and_stop":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("plunk_and_stop primary failure required")
    if bool(claim.get("human_audio_keep_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("human/audio keep must not be claimed")
    if bool(claim.get("musical_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("musical quality must not be claimed")
    if bool(claim.get("broad_trained_model_quality_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("broad trained-model quality must not be claimed")
    if bool(claim.get("brad_style_adaptation_claimed", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("Brad style adaptation must not be claimed")
    candidate_reviews = _list(review.get("candidate_reviews"))
    if len(candidate_reviews) < 5:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("five candidate reviews required")
    for item in candidate_reviews:
        if not isinstance(item, dict):
            continue
        if str(item.get("decision") or "") != "reject":
            raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("all candidate reviews must be reject")
        if str(item.get("primary_failure") or "") != "plunk_and_stop":
            raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError(
                "all candidate reviews must carry plunk_and_stop failure"
            )
    if str(decision.get("next_boundary") or "") != "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_decision":
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("unexpected next boundary in user review")
    return {
        "overall_decision": str(review.get("overall_decision") or ""),
        "candidate_decision": str(review.get("candidate_decision") or ""),
        "primary_failure": str(review.get("primary_failure") or ""),
        "timing": str(review.get("timing") or ""),
        "phrase": str(review.get("phrase") or ""),
        "vocabulary": str(review.get("vocabulary") or ""),
        "assessment": str(review.get("assessment") or ""),
        "reviewed_audio_file_count": len(_list(user_listening_review.get("reviewed_audio_files"))),
        "candidate_review_count": len(candidate_reviews),
    }


def build_phrase_continuation_decision(
    user_listening_review: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    review_summary = validate_user_listening_review(user_listening_review)
    repair_targets = [
        "increase_min_note_events_per_review_window",
        "require_phrase_continuation_after_initial_cell",
        "limit_terminal_dead_air_after_last_note",
        "penalize_single_cell_or_two_hit_outputs",
        "require_cadence_or_contour_resolution_before_end",
        "prefer_motif_extension_over_isolated_hits",
    ]
    return {
        "schema_version": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_user_listening_review_schema": str(user_listening_review.get("schema_version") or ""),
        "input_boundary": "generic_tiny_checkpoint_repair_audio_review_reject_all",
        "user_review_summary": review_summary,
        "decision": {
            "next_boundary": "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_repair_sweep",
            "reason": "current repaired candidates pass technical MIDI/WAV checks but user listening rejects all as short plunk-and-stop fragments",
            "auto_progress_allowed": True,
            "critical_user_input_required": False,
            "blocked_reason": "",
        },
        "repair_targets": repair_targets,
        "selection_constraints": {
            "keep_stage_b_grammar_gate": True,
            "keep_required_midi_decode": True,
            "require_audio_render_after_repair": True,
            "require_user_listening_review_after_repair": True,
            "do_not_claim_broad_model_quality": True,
        },
        "claim_boundary": {
            "boundary": "generic_tiny_checkpoint_repair_phrase_continuation_decision",
            "human_audio_keep_claimed": False,
            "musical_quality_claimed": False,
            "broad_trained_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
        },
        "not_proven": [
            "human_audio_keep",
            "musical_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
        ],
        "next_recommended_issue": "Stage B generic tiny checkpoint repair phrase continuation repair sweep",
    }


def validate_phrase_continuation_decision(
    report: dict[str, Any],
    *,
    expected_next_boundary: str | None,
    require_auto_progress_allowed: bool,
    require_no_critical_user_input: bool,
    require_no_quality_claim: bool,
) -> dict[str, Any]:
    decision = _dict(report.get("decision"))
    claim = _dict(report.get("claim_boundary"))
    next_boundary = str(decision.get("next_boundary") or "")
    if expected_next_boundary and next_boundary != expected_next_boundary:
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError(
            f"expected next boundary {expected_next_boundary}, got {next_boundary}"
        )
    if require_auto_progress_allowed and not bool(decision.get("auto_progress_allowed", False)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("auto progress must be allowed")
    if require_no_critical_user_input and bool(decision.get("critical_user_input_required", True)):
        raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("critical user input must not be required")
    if require_no_quality_claim:
        claimed = [
            bool(claim.get("human_audio_keep_claimed", True)),
            bool(claim.get("musical_quality_claimed", True)),
            bool(claim.get("broad_trained_model_quality_claimed", True)),
            bool(claim.get("brad_style_adaptation_claimed", True)),
        ]
        if any(claimed):
            raise StageBGenericTinyCheckpointRepairPhraseContinuationDecisionError("quality or keep claims must not be set")
    return {
        "input_boundary": str(report.get("input_boundary") or ""),
        "next_boundary": next_boundary,
        "auto_progress_allowed": bool(decision.get("auto_progress_allowed", False)),
        "critical_user_input_required": bool(decision.get("critical_user_input_required", True)),
        "repair_target_count": len(_list(report.get("repair_targets"))),
        "human_audio_keep_claimed": bool(claim.get("human_audio_keep_claimed", True)),
        "musical_quality_claimed": bool(claim.get("musical_quality_claimed", True)),
        "broad_trained_model_quality_claimed": bool(claim.get("broad_trained_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    decision = report["decision"]
    review = report["user_review_summary"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Generic Tiny Checkpoint Repair Phrase Continuation Decision",
        "",
        "## Summary",
        "",
        f"- input boundary: `{report['input_boundary']}`",
        f"- next boundary: `{decision['next_boundary']}`",
        f"- auto progress allowed: `{_bool_token(decision['auto_progress_allowed'])}`",
        f"- critical user input required: `{_bool_token(decision['critical_user_input_required'])}`",
        f"- human/audio keep claimed: `{_bool_token(claim['human_audio_keep_claimed'])}`",
        f"- musical quality claimed: `{_bool_token(claim['musical_quality_claimed'])}`",
        "",
        "## User Review",
        "",
        f"- overall decision: `{review['overall_decision']}`",
        f"- candidate decision: `{review['candidate_decision']}`",
        f"- primary failure: `{review['primary_failure']}`",
        f"- timing: `{review['timing']}`",
        f"- phrase: `{review['phrase']}`",
        f"- vocabulary: `{review['vocabulary']}`",
        f"- assessment: {review['assessment']}",
        "",
        "## Repair Targets",
        "",
    ]
    for item in report.get("repair_targets", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Decide generic tiny checkpoint phrase continuation repair boundary")
    parser.add_argument(
        "--user_listening_review",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_user_listening_review/"
        "harness_stage_b_generic_tiny_checkpoint_repair_user_listening_review/"
        "stage_b_generic_tiny_checkpoint_repair_user_listening_review.json",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--doc_path", type=str, default="")
    parser.add_argument("--expected_next_boundary", type=str, default="")
    parser.add_argument("--require_auto_progress_allowed", action="store_true")
    parser.add_argument("--require_no_critical_user_input", action="store_true")
    parser.add_argument("--require_no_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_phrase_continuation_decision(
        read_json(Path(args.user_listening_review)),
        output_dir=output_dir,
    )
    summary = validate_phrase_continuation_decision(
        report,
        expected_next_boundary=str(args.expected_next_boundary or ""),
        require_auto_progress_allowed=bool(args.require_auto_progress_allowed),
        require_no_critical_user_input=bool(args.require_no_critical_user_input),
        require_no_quality_claim=bool(args.require_no_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision.json", report)
    write_json(output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision_validation_summary.json", summary)
    markdown = markdown_report(report)
    write_text(output_dir / "stage_b_generic_tiny_checkpoint_repair_phrase_continuation_decision.md", markdown)
    if args.doc_path:
        write_text(Path(args.doc_path), markdown)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
