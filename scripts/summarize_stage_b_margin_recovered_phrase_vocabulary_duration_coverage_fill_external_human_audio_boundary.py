"""Summarize external human/audio review boundary for duration coverage fill evidence."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillExternalHumanAudioBoundaryError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_external_boundary_report(
    midi_evidence_consolidation: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    source_claim = (
        midi_evidence_consolidation.get("claim_boundary")
        if isinstance(midi_evidence_consolidation.get("claim_boundary"), dict)
        else {}
    )
    midi_summary = (
        midi_evidence_consolidation.get("midi_evidence_summary")
        if isinstance(midi_evidence_consolidation.get("midi_evidence_summary"), dict)
        else {}
    )
    source_boundary = str(source_claim.get("boundary") or "")
    preference = str(midi_summary.get("preference") or "")
    if source_boundary != "midi_evidence_preference_support":
        raise DurationCoverageFillExternalHumanAudioBoundaryError(f"unexpected source boundary: {source_boundary}")
    if preference != "duration_coverage_fill_keep":
        raise DurationCoverageFillExternalHumanAudioBoundaryError(f"unexpected MIDI evidence preference: {preference}")
    if not bool(source_claim.get("midi_evidence_preference_claimed", False)):
        raise DurationCoverageFillExternalHumanAudioBoundaryError("MIDI evidence preference must be claimed first")
    if bool(source_claim.get("human_audio_preference_claimed", True)):
        raise DurationCoverageFillExternalHumanAudioBoundaryError("human/audio preference must not be claimed")
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_midi_evidence_consolidation_schema": str(midi_evidence_consolidation.get("schema_version") or ""),
        "candidate_id": str(midi_evidence_consolidation.get("candidate_id") or ""),
        "decision_path": [
            "duration_coverage_fill_repair",
            "focused_context",
            "focused_listening_fill",
            "keep_consolidation",
            "audio_review_package",
            "midi_evidence_review",
            "midi_evidence_consolidation",
            "external_human_audio_boundary",
        ],
        "source_boundary": {
            "boundary": source_boundary,
            "preference": preference,
            "midi_evidence_preference_claimed": bool(source_claim.get("midi_evidence_preference_claimed", False)),
            "human_audio_preference_claimed": False,
            "audio_render_used": bool(source_claim.get("audio_render_used", False)),
        },
        "midi_evidence_summary": {
            "preference": preference,
            "source_score": float(midi_summary.get("source_score", 0.0) or 0.0),
            "fill_score": float(midi_summary.get("fill_score", 0.0) or 0.0),
            "score_delta_fill_minus_source": float(midi_summary.get("score_delta_fill_minus_source", 0.0) or 0.0),
            "dead_air_delta_fill_minus_source": float(midi_summary.get("dead_air_delta_fill_minus_source", 0.0) or 0.0),
            "focused_note_count_delta_fill_minus_source": int(
                midi_summary.get("focused_note_count_delta_fill_minus_source", 0) or 0
            ),
            "focused_unique_pitch_count_delta_fill_minus_source": int(
                midi_summary.get("focused_unique_pitch_count_delta_fill_minus_source", 0) or 0
            ),
            "max_simultaneous_notes_delta_fill_minus_source": int(
                midi_summary.get("max_simultaneous_notes_delta_fill_minus_source", 0) or 0
            ),
        },
        "external_review_boundary": {
            "boundary": "external_human_audio_review_required_for_human_preference_claim",
            "status": "pending_external_review_input",
            "reviewer_input_present": False,
            "midi_evidence_preference_claimed": True,
            "human_audio_preference_claimed": False,
            "audio_render_used": False,
            "requires_external_human_or_audio_review": True,
        },
        "required_review_input": {
            "reviewer": {"required": True},
            "audio_render_used": {"required": True, "allowed_values": [True]},
            "preference": {
                "required": True,
                "allowed_values": [
                    "source_constrained_partial",
                    "duration_coverage_fill_keep",
                    "no_preference",
                    "needs_followup",
                ],
            },
            "timing": {"required": True, "allowed_values": ["weak", "acceptable", "strong"]},
            "phrase": {"required": True, "allowed_values": ["weak", "acceptable", "strong"]},
            "vocabulary": {"required": True, "allowed_values": ["thin", "acceptable", "strong"]},
            "notes": {"required": True},
        },
        "claim_policy": [
            "MIDI evidence preference can support review prioritization only",
            "human/audio preference requires validated external review input",
            "audio rendered quality requires an audio render and reviewer notes",
            "broad trained-model quality remains out of scope",
        ],
        "not_proven": [
            "human_audio_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render package",
        "external_review_input_followup": "Stage B margin-recovered phrase/vocabulary duration coverage fill external review input fill",
    }


def validate_external_boundary(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_boundary: str | None,
    require_no_human_audio_preference: bool,
    require_pending_external_review: bool,
) -> dict[str, Any]:
    candidate_id = str(report.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillExternalHumanAudioBoundaryError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    boundary = (
        report.get("external_review_boundary")
        if isinstance(report.get("external_review_boundary"), dict)
        else {}
    )
    boundary_name = str(boundary.get("boundary") or "")
    if expected_boundary and boundary_name != expected_boundary:
        raise DurationCoverageFillExternalHumanAudioBoundaryError(
            f"expected boundary {expected_boundary}, got {boundary_name}"
        )
    if require_no_human_audio_preference and bool(boundary.get("human_audio_preference_claimed", True)):
        raise DurationCoverageFillExternalHumanAudioBoundaryError("human/audio preference must not be claimed")
    if require_pending_external_review and str(boundary.get("status") or "") != "pending_external_review_input":
        raise DurationCoverageFillExternalHumanAudioBoundaryError("external review input must remain pending")
    summary = report.get("midi_evidence_summary") if isinstance(report.get("midi_evidence_summary"), dict) else {}
    return {
        "candidate_id": candidate_id,
        "source_boundary": str(report.get("source_boundary", {}).get("boundary") or ""),
        "external_boundary": boundary_name,
        "external_review_status": str(boundary.get("status") or ""),
        "midi_evidence_preference": str(summary.get("preference") or ""),
        "score_delta_fill_minus_source": float(summary.get("score_delta_fill_minus_source", 0.0) or 0.0),
        "human_audio_preference_claimed": bool(boundary.get("human_audio_preference_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["midi_evidence_summary"]
    boundary = report["external_review_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill External Human/Audio Boundary",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- source boundary: `{report['source_boundary']['boundary']}`",
        f"- external boundary: `{boundary['boundary']}`",
        f"- external review status: `{boundary['status']}`",
        f"- MIDI evidence preference: `{summary['preference']}`",
        f"- score delta fill-source: `{summary['score_delta_fill_minus_source']:.3f}`",
        f"- human/audio preference claimed: `{boundary['human_audio_preference_claimed']}`",
        "",
        "## Required Review Input",
        "",
    ]
    for field, config in report.get("required_review_input", {}).items():
        allowed = config.get("allowed_values") if isinstance(config, dict) else None
        suffix = f", allowed `{allowed}`" if allowed else ""
        lines.append(f"- `{field}`: required `{bool(config.get('required', False))}`{suffix}")
    lines.extend(["", "## Claim Policy", ""])
    for item in report.get("claim_policy", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize external human/audio review boundary")
    parser.add_argument("--midi_evidence_consolidation", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_human_audio_preference", action="store_true")
    parser.add_argument("--require_pending_external_review", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_external_boundary_report(read_json(Path(args.midi_evidence_consolidation)), output_dir=output_dir)
    summary = validate_external_boundary(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_boundary=str(args.expected_boundary or ""),
        require_no_human_audio_preference=bool(args.require_no_human_audio_preference),
        require_pending_external_review=bool(args.require_pending_external_review),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_external_human_audio_boundary.json"
    markdown_path = output_dir / "duration_coverage_fill_external_human_audio_boundary.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_external_human_audio_boundary_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
