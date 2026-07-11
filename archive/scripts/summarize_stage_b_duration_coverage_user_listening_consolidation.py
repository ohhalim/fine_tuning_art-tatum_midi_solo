"""Consolidate MIDI evidence, WAV render validation, and user listening review."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageBDurationCoverageUserListeningConsolidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def require_candidate_match(*reports: dict[str, Any]) -> str:
    candidate_ids = [str(report.get("candidate_id") or "") for report in reports]
    unique_ids = {candidate_id for candidate_id in candidate_ids if candidate_id}
    if len(unique_ids) != 1:
        raise StageBDurationCoverageUserListeningConsolidationError(
            f"candidate mismatch: {candidate_ids}"
        )
    return next(iter(unique_ids))


def build_consolidation_report(
    midi_evidence_consolidation: dict[str, Any],
    audio_render_attempt: dict[str, Any],
    user_listening_review_fill: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    candidate_id = require_candidate_match(
        midi_evidence_consolidation,
        audio_render_attempt,
        user_listening_review_fill,
    )
    midi_summary = (
        midi_evidence_consolidation.get("midi_evidence_summary")
        if isinstance(midi_evidence_consolidation.get("midi_evidence_summary"), dict)
        else {}
    )
    midi_claim = (
        midi_evidence_consolidation.get("claim_boundary")
        if isinstance(midi_evidence_consolidation.get("claim_boundary"), dict)
        else {}
    )
    audio_boundary = (
        audio_render_attempt.get("audio_render_boundary")
        if isinstance(audio_render_attempt.get("audio_render_boundary"), dict)
        else {}
    )
    user_review = (
        user_listening_review_fill.get("user_listening_review")
        if isinstance(user_listening_review_fill.get("user_listening_review"), dict)
        else {}
    )
    user_claim = (
        user_listening_review_fill.get("claim_boundary")
        if isinstance(user_listening_review_fill.get("claim_boundary"), dict)
        else {}
    )
    midi_preference = str(midi_summary.get("preference") or "")
    user_preference = str(user_review.get("preference") or "")
    if midi_preference != "duration_coverage_fill_keep":
        raise StageBDurationCoverageUserListeningConsolidationError(
            f"unexpected MIDI evidence preference: {midi_preference}"
        )
    if user_preference != "duration_coverage_fill_keep":
        raise StageBDurationCoverageUserListeningConsolidationError(
            f"unexpected user listening preference: {user_preference}"
        )
    if not bool(midi_claim.get("midi_evidence_preference_claimed", False)):
        raise StageBDurationCoverageUserListeningConsolidationError("MIDI evidence preference is not claimed")
    if not bool(audio_boundary.get("technical_wav_validation", False)):
        raise StageBDurationCoverageUserListeningConsolidationError("technical WAV validation is required")
    if not bool(user_claim.get("human_audio_preference_claimed", False)):
        raise StageBDurationCoverageUserListeningConsolidationError("user listening preference is not claimed")
    rendered_files = audio_render_attempt.get("rendered_audio_files")
    if not isinstance(rendered_files, list) or len(rendered_files) != 2:
        raise StageBDurationCoverageUserListeningConsolidationError("expected two rendered WAV files")
    return {
        "schema_version": "stage_b_duration_coverage_fill_user_listening_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "candidate_id": candidate_id,
        "source_schemas": {
            "midi_evidence_consolidation": str(midi_evidence_consolidation.get("schema_version") or ""),
            "audio_render_attempt": str(audio_render_attempt.get("schema_version") or ""),
            "user_listening_review_fill": str(user_listening_review_fill.get("schema_version") or ""),
        },
        "evidence_alignment": {
            "midi_evidence_preference": midi_preference,
            "user_listening_preference": user_preference,
            "same_preferred_candidate": midi_preference == user_preference,
            "technical_wav_validation": True,
            "rendered_audio_file_count": int(audio_boundary.get("rendered_audio_file_count", 0) or 0),
            "single_user_review": bool(user_claim.get("single_user_review", False)),
        },
        "midi_evidence_summary": {
            "review_basis": str(midi_summary.get("review_basis") or ""),
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
        "audio_render_summary": {
            "render_attempted": bool(audio_boundary.get("render_attempted", False)),
            "technical_wav_validation": bool(audio_boundary.get("technical_wav_validation", False)),
            "rendered_audio_file_count": int(audio_boundary.get("rendered_audio_file_count", 0) or 0),
            "sample_rates": sorted(
                {
                    int(item.get("wav_file", {}).get("sample_rate", 0) or 0)
                    for item in rendered_files
                    if isinstance(item, dict)
                }
            ),
            "durations_seconds": {
                str(item.get("role") or ""): float(item.get("wav_file", {}).get("duration_seconds", 0.0) or 0.0)
                for item in rendered_files
                if isinstance(item, dict)
            },
        },
        "user_listening_summary": {
            "review_basis": str(user_review.get("review_basis") or ""),
            "preference": user_preference,
            "timing": str(user_review.get("timing") or ""),
            "phrase": str(user_review.get("phrase") or ""),
            "vocabulary": str(user_review.get("vocabulary") or ""),
            "source_assessment": str(user_review.get("source_assessment") or ""),
            "fill_assessment": str(user_review.get("fill_assessment") or ""),
            "notes": str(user_review.get("notes") or ""),
        },
        "consolidated_claim_boundary": {
            "boundary": "midi_evidence_and_single_user_listening_support_duration_coverage_fill_keep",
            "preferred_candidate": "duration_coverage_fill_keep",
            "midi_evidence_preference_claimed": True,
            "technical_wav_validation_claimed": True,
            "single_user_human_audio_preference_claimed": True,
            "multi_reviewer_preference_claimed": False,
            "audio_rendered_quality_claimed": False,
            "broad_model_quality_claimed": False,
            "brad_style_adaptation_claimed": False,
            "production_ready_improviser_claimed": False,
        },
        "proven": [
            "midi_metric_preference_for_duration_coverage_fill_keep",
            "technical_wav_render_validation_completed",
            "single_user_listening_preference_for_duration_coverage_fill_keep",
        ],
        "not_proven": [
            "multi_reviewer_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill next repair or repeatability decision",
    }


def validate_consolidation_report(
    report: dict[str, Any],
    *,
    expected_boundary: str | None,
    expected_preferred_candidate: str | None,
    require_no_broad_quality_claim: bool,
) -> dict[str, Any]:
    boundary = (
        report.get("consolidated_claim_boundary")
        if isinstance(report.get("consolidated_claim_boundary"), dict)
        else {}
    )
    boundary_name = str(boundary.get("boundary") or "")
    preferred = str(boundary.get("preferred_candidate") or "")
    if expected_boundary and boundary_name != expected_boundary:
        raise StageBDurationCoverageUserListeningConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary_name}"
        )
    if expected_preferred_candidate and preferred != expected_preferred_candidate:
        raise StageBDurationCoverageUserListeningConsolidationError(
            f"expected preferred candidate {expected_preferred_candidate}, got {preferred}"
        )
    if require_no_broad_quality_claim:
        blocked = [
            "multi_reviewer_preference_claimed",
            "audio_rendered_quality_claimed",
            "broad_model_quality_claimed",
            "brad_style_adaptation_claimed",
            "production_ready_improviser_claimed",
        ]
        claimed = [name for name in blocked if bool(boundary.get(name, True))]
        if claimed:
            raise StageBDurationCoverageUserListeningConsolidationError(
                f"unexpected broad claim: {claimed}"
            )
    alignment = report.get("evidence_alignment") if isinstance(report.get("evidence_alignment"), dict) else {}
    if not bool(alignment.get("same_preferred_candidate", False)):
        raise StageBDurationCoverageUserListeningConsolidationError("evidence preference mismatch")
    return {
        "candidate_id": str(report.get("candidate_id") or ""),
        "boundary": boundary_name,
        "preferred_candidate": preferred,
        "same_preferred_candidate": bool(alignment.get("same_preferred_candidate", False)),
        "rendered_audio_file_count": int(alignment.get("rendered_audio_file_count", 0) or 0),
        "single_user_review": bool(alignment.get("single_user_review", False)),
        "broad_model_quality_claimed": bool(boundary.get("broad_model_quality_claimed", True)),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["consolidated_claim_boundary"]
    midi = report["midi_evidence_summary"]
    user = report["user_listening_summary"]
    audio = report["audio_render_summary"]
    lines = [
        "# Stage B Duration Coverage Fill User Listening Review Consolidation",
        "",
        f"- candidate: `{report['candidate_id']}`",
        f"- boundary: `{boundary['boundary']}`",
        f"- preferred candidate: `{boundary['preferred_candidate']}`",
        f"- MIDI score delta fill-source: `{midi['score_delta_fill_minus_source']:.3f}`",
        f"- rendered audio file count: `{audio['rendered_audio_file_count']}`",
        f"- user listening preference: `{user['preference']}`",
        f"- broad model quality claimed: `{boundary['broad_model_quality_claimed']}`",
        "",
        "## User Assessment",
        "",
        f"- source: {user['source_assessment']}",
        f"- fill: {user['fill_assessment']}",
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
    parser = argparse.ArgumentParser(description="Consolidate Stage B duration coverage user listening review")
    parser.add_argument("--midi_evidence_consolidation", type=str, required=True)
    parser.add_argument("--audio_render_attempt", type=str, required=True)
    parser.add_argument("--user_listening_review_fill", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--expected_preferred_candidate", type=str, default="")
    parser.add_argument("--require_no_broad_quality_claim", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_consolidation_report(
        read_json(Path(args.midi_evidence_consolidation)),
        read_json(Path(args.audio_render_attempt)),
        read_json(Path(args.user_listening_review_fill)),
        output_dir=output_dir,
    )
    summary = validate_consolidation_report(
        report,
        expected_boundary=str(args.expected_boundary or ""),
        expected_preferred_candidate=str(args.expected_preferred_candidate or ""),
        require_no_broad_quality_claim=bool(args.require_no_broad_quality_claim),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "stage_b_duration_coverage_fill_user_listening_review_consolidation.json"
    markdown_path = output_dir / "stage_b_duration_coverage_fill_user_listening_review_consolidation.md"
    write_json(report_path, report)
    write_json(output_dir / "stage_b_duration_coverage_fill_user_listening_review_consolidation_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
