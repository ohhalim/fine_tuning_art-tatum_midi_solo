"""Consolidate duration/coverage fill MIDI evidence review boundaries."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillMidiEvidenceConsolidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_consolidation_report(
    midi_evidence_review: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    review = (
        midi_evidence_review.get("midi_evidence_review")
        if isinstance(midi_evidence_review.get("midi_evidence_review"), dict)
        else {}
    )
    claim = (
        midi_evidence_review.get("claim_boundary")
        if isinstance(midi_evidence_review.get("claim_boundary"), dict)
        else {}
    )
    score = midi_evidence_review.get("score") if isinstance(midi_evidence_review.get("score"), dict) else {}
    metric_delta = (
        midi_evidence_review.get("metric_delta")
        if isinstance(midi_evidence_review.get("metric_delta"), dict)
        else {}
    )
    preference = str(review.get("preference") or "")
    if preference != "duration_coverage_fill_keep":
        raise DurationCoverageFillMidiEvidenceConsolidationError(f"unexpected MIDI evidence preference: {preference}")
    if bool(claim.get("human_audio_preference_claimed", True)):
        raise DurationCoverageFillMidiEvidenceConsolidationError("human/audio preference must not be claimed")
    if bool(claim.get("audio_render_used", True)):
        raise DurationCoverageFillMidiEvidenceConsolidationError("audio render must not be claimed")
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_midi_evidence_schema": str(midi_evidence_review.get("schema_version") or ""),
        "candidate_id": str(midi_evidence_review.get("candidate_id") or ""),
        "decision_path": [
            "duration_coverage_fill_repair",
            "focused_context",
            "focused_listening_fill",
            "keep_consolidation",
            "audio_review_package",
            "midi_evidence_review",
            "midi_evidence_consolidation",
        ],
        "midi_evidence_summary": {
            "preference": preference,
            "review_basis": str(midi_evidence_review.get("review_basis") or ""),
            "source_score": float(score.get("source_constrained_partial", 0.0) or 0.0),
            "fill_score": float(score.get("duration_coverage_fill_keep", 0.0) or 0.0),
            "score_delta_fill_minus_source": float(score.get("score_delta_fill_minus_source", 0.0) or 0.0),
            "dead_air_delta_fill_minus_source": float(
                metric_delta.get("dead_air_delta_fill_minus_source", 0.0) or 0.0
            ),
            "focused_note_count_delta_fill_minus_source": int(
                metric_delta.get("focused_note_count_delta_fill_minus_source", 0) or 0
            ),
            "focused_unique_pitch_count_delta_fill_minus_source": int(
                metric_delta.get("focused_unique_pitch_count_delta_fill_minus_source", 0) or 0
            ),
            "max_simultaneous_notes_delta_fill_minus_source": int(
                metric_delta.get("max_simultaneous_notes_delta_fill_minus_source", 0) or 0
            ),
        },
        "claim_boundary": {
            "boundary": "midi_evidence_preference_support",
            "midi_evidence_preference_claimed": bool(claim.get("midi_evidence_preference_claimed", False)),
            "human_audio_preference_claimed": False,
            "audio_render_used": False,
            "not_human_audio_review": True,
        },
        "proven": [
            "midi_metric_preference_for_duration_coverage_fill_keep",
            "dead_air_reduced_against_source_partial",
            "focused_note_count_increased_against_source_partial",
            "focused_unique_pitch_count_increased_against_source_partial",
            "max_simultaneous_notes_reduced_against_source_partial",
        ],
        "not_proven": [
            "human_audio_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill external human/audio review boundary",
    }


def validate_consolidation(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_boundary: str | None,
    require_no_human_audio_preference: bool,
) -> dict[str, Any]:
    candidate_id = str(report.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillMidiEvidenceConsolidationError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    claim = report.get("claim_boundary") if isinstance(report.get("claim_boundary"), dict) else {}
    boundary = str(claim.get("boundary") or "")
    if expected_boundary and boundary != expected_boundary:
        raise DurationCoverageFillMidiEvidenceConsolidationError(f"expected boundary {expected_boundary}, got {boundary}")
    if require_no_human_audio_preference and bool(claim.get("human_audio_preference_claimed", True)):
        raise DurationCoverageFillMidiEvidenceConsolidationError("human/audio preference must not be claimed")
    summary = report.get("midi_evidence_summary") if isinstance(report.get("midi_evidence_summary"), dict) else {}
    return {
        "candidate_id": candidate_id,
        "boundary": boundary,
        "preference": str(summary.get("preference") or ""),
        "score_delta_fill_minus_source": float(summary.get("score_delta_fill_minus_source", 0.0) or 0.0),
        "dead_air_delta_fill_minus_source": float(summary.get("dead_air_delta_fill_minus_source", 0.0) or 0.0),
        "human_audio_preference_claimed": bool(claim.get("human_audio_preference_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    summary = report["midi_evidence_summary"]
    claim = report["claim_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill MIDI Evidence Consolidation",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- boundary: `{claim['boundary']}`",
        f"- preference: `{summary['preference']}`",
        f"- score delta fill-source: `{summary['score_delta_fill_minus_source']:.3f}`",
        f"- dead-air delta fill-source: `{summary['dead_air_delta_fill_minus_source']:.4f}`",
        f"- human/audio preference claimed: `{claim['human_audio_preference_claimed']}`",
        "",
        "## Proven",
        "",
    ]
    for item in report.get("proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate duration fill MIDI evidence review")
    parser.add_argument("--midi_evidence_review", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_no_human_audio_preference", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_consolidation_report(read_json(Path(args.midi_evidence_review)), output_dir=output_dir)
    summary = validate_consolidation(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_boundary=str(args.expected_boundary or ""),
        require_no_human_audio_preference=bool(args.require_no_human_audio_preference),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_midi_evidence_consolidation.json"
    markdown_path = output_dir / "duration_coverage_fill_midi_evidence_consolidation.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_midi_evidence_consolidation_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
