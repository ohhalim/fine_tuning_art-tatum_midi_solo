"""Build a human/audio review boundary for the duration/coverage fill keep candidate."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi


class DurationCoverageFillHumanAudioBoundaryError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def non_drum_notes(path: Path) -> list[pretty_midi.Note]:
    if not path.exists():
        raise DurationCoverageFillHumanAudioBoundaryError(f"MIDI file not found: {path}")
    midi = pretty_midi.PrettyMIDI(str(path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)
    notes.sort(key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    return notes


def note_signature(path: Path) -> list[dict[str, Any]]:
    signature = []
    for note in non_drum_notes(path):
        signature.append(
            {
                "pitch": int(note.pitch),
                "start_sec": round(float(note.start), 6),
                "duration_sec": round(float(note.end) - float(note.start), 6),
                "velocity": int(note.velocity),
            }
        )
    return signature


def metric_summary(candidate: dict[str, Any], *, source: dict[str, Any] | None = None) -> dict[str, Any]:
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    focused = candidate.get("focused_solo_metrics") if isinstance(candidate.get("focused_solo_metrics"), dict) else {}
    if source:
        source_metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
        source_focused = source.get("focused_solo_metrics") if isinstance(source.get("focused_solo_metrics"), dict) else {}
        metrics = source_metrics
        focused = source_focused
    return {
        "note_count": int(metrics.get("note_count", 0) or focused.get("focused_note_count", 0) or 0),
        "focused_note_count": int(focused.get("focused_note_count", 0) or metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "max_simultaneous_notes": int(metrics.get("max_simultaneous_notes", 0) or 0),
        "focused_max_simultaneous_notes": int(focused.get("focused_max_simultaneous_notes", 0) or 0),
        "adjacent_pitch_repeats": int(focused.get("focused_adjacent_pitch_repeats", 0) or 0),
        "duplicated_3_note_pitch_class_chunks": int(
            focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0
        ),
        "max_interval": int(focused.get("focused_max_interval", 0) or 0),
    }


def keep_candidate(keep_consolidation: dict[str, Any]) -> dict[str, Any]:
    candidate = keep_consolidation.get("candidate")
    if not isinstance(candidate, dict):
        raise DurationCoverageFillHumanAudioBoundaryError("keep consolidation must contain candidate")
    if str(candidate.get("decision") or "") != "keep":
        raise DurationCoverageFillHumanAudioBoundaryError("keep consolidation candidate decision must be keep")
    return candidate


def compact_review_item(
    *,
    role: str,
    candidate_id: str,
    midi_path: Path,
    metrics: dict[str, Any],
    context_midi_path: str = "",
    prior_decision: str = "",
) -> dict[str, Any]:
    signature = note_signature(midi_path)
    return {
        "role": role,
        "candidate_id": candidate_id,
        "midi_path": str(midi_path),
        "context_midi_path": context_midi_path,
        "prior_decision": prior_decision,
        "metric_summary": metrics,
        "note_signature": signature,
        "note_signature_count": len(signature),
        "human_audio_review": {
            "status": "pending",
            "preference": "pending",
            "timing": "pending",
            "phrase": "pending",
            "vocabulary": "pending",
            "audio_render_used": "",
            "reviewer": "",
            "notes": "",
        },
    }


def build_human_audio_boundary(
    keep_consolidation: dict[str, Any],
    duration_fill_summary: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    keep = keep_candidate(keep_consolidation)
    duration_repair = (
        keep_consolidation.get("duration_coverage_repair")
        if isinstance(keep_consolidation.get("duration_coverage_repair"), dict)
        else {}
    )
    source = duration_fill_summary.get("source_candidate")
    selected = duration_fill_summary.get("selected_candidate")
    if not isinstance(source, dict) or not isinstance(selected, dict):
        raise DurationCoverageFillHumanAudioBoundaryError("duration fill summary must contain source/selected")
    if str(keep.get("candidate_id") or "") != str(selected.get("candidate_id") or ""):
        raise DurationCoverageFillHumanAudioBoundaryError("keep candidate does not match selected duration fill candidate")

    source_path = Path(str(source.get("midi_path") or ""))
    selected_path = Path(str(selected.get("midi_path") or ""))
    source_item = compact_review_item(
        role="source_constrained_partial",
        candidate_id=str(source.get("candidate_id") or ""),
        midi_path=source_path,
        metrics=metric_summary({}, source=source),
        prior_decision="needs_duration_coverage_fill",
    )
    selected_item = compact_review_item(
        role="duration_coverage_fill_keep",
        candidate_id=str(selected.get("candidate_id") or ""),
        midi_path=selected_path,
        context_midi_path=str(keep.get("context_midi_path") or ""),
        metrics=metric_summary(selected),
        prior_decision=str(keep.get("decision") or ""),
    )
    note_sequence_match = source_item["note_signature"] == selected_item["note_signature"]
    metric_match = source_item["metric_summary"] == selected_item["metric_summary"]
    boundary = (
        "pending_human_audio_review_same_midi_content"
        if note_sequence_match
        else "pending_human_audio_review_source_vs_fill_distinct_midi_content"
    )
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_keep_consolidation_schema": str(keep_consolidation.get("schema_version") or ""),
        "source_duration_fill_schema": str(duration_fill_summary.get("schema_version") or ""),
        "review_items": [source_item, selected_item],
        "objective_comparison": {
            "note_sequence_match": note_sequence_match,
            "metric_summary_match": metric_match,
            "source_note_signature_count": int(source_item["note_signature_count"]),
            "selected_note_signature_count": int(selected_item["note_signature_count"]),
            "fill_addition_count": int(duration_repair.get("fill_addition_count", 0) or 0),
            "dead_air_delta_from_baseline": float(duration_repair.get("dead_air_delta_from_baseline", 0.0) or 0.0),
        },
        "human_audio_boundary": {
            "boundary": boundary,
            "status": "pending",
            "preference_claimed": False,
            "audio_render_used": False,
            "not_human_audio_review": True,
            "requires_audio_render_or_human_review": True,
        },
        "not_proven": [
            "human_audio_preference",
            "audio_rendered_quality",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "production_ready_improviser",
        ],
        "next_recommended_issue": "Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio review fill",
    }


def validate_human_audio_boundary(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    require_pending: bool,
    require_no_preference: bool,
    expect_distinct_midi_content: bool,
) -> dict[str, Any]:
    items = report.get("review_items")
    if not isinstance(items, list) or len(items) != 2:
        raise DurationCoverageFillHumanAudioBoundaryError("report must contain exactly two review items")
    selected = [item for item in items if item.get("role") == "duration_coverage_fill_keep"]
    if len(selected) != 1:
        raise DurationCoverageFillHumanAudioBoundaryError("report must contain one duration fill keep item")
    candidate_id = str(selected[0].get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillHumanAudioBoundaryError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    if require_pending:
        statuses = [str(item.get("human_audio_review", {}).get("status") or "") for item in items]
        if any(status != "pending" for status in statuses):
            raise DurationCoverageFillHumanAudioBoundaryError(f"expected pending statuses, got {statuses}")
    boundary = report.get("human_audio_boundary") if isinstance(report.get("human_audio_boundary"), dict) else {}
    if require_no_preference and bool(boundary.get("preference_claimed", True)):
        raise DurationCoverageFillHumanAudioBoundaryError("human/audio preference must not be claimed")
    note_sequence_match = bool(report.get("objective_comparison", {}).get("note_sequence_match", False))
    if expect_distinct_midi_content and note_sequence_match:
        raise DurationCoverageFillHumanAudioBoundaryError("expected distinct MIDI content")
    return {
        "review_item_count": len(items),
        "candidate_id": candidate_id,
        "human_status": str(boundary.get("status") or ""),
        "boundary": str(boundary.get("boundary") or ""),
        "preference_claimed": bool(boundary.get("preference_claimed", True)),
        "note_sequence_match": note_sequence_match,
        "metric_summary_match": bool(report.get("objective_comparison", {}).get("metric_summary_match", False)),
        "fill_addition_count": int(report.get("objective_comparison", {}).get("fill_addition_count", 0) or 0),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    boundary = report["human_audio_boundary"]
    objective = report["objective_comparison"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Human/Audio Boundary",
        "",
        f"- review items: `{len(report.get('review_items') or [])}`",
        f"- human/audio status: `{boundary['status']}`",
        f"- boundary: `{boundary['boundary']}`",
        f"- note sequence match: `{objective['note_sequence_match']}`",
        f"- fill additions: `{objective['fill_addition_count']}`",
        f"- dead-air delta: `{objective['dead_air_delta_from_baseline']:.4f}`",
        "",
        "This file prepares a human/audio review boundary. It does not claim a listening preference.",
        "",
        "| role | candidate | prior decision | notes | focused notes | unique | focused unique | dead-air | max active | human status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report.get("review_items", []):
        metrics = item["metric_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["role"]),
                    str(item["candidate_id"]),
                    str(item["prior_decision"]),
                    str(metrics["note_count"]),
                    str(metrics["focused_note_count"]),
                    str(metrics["unique_pitch_count"]),
                    str(metrics["focused_unique_pitch_count"]),
                    f"{float(metrics['dead_air_ratio']):.4f}",
                    str(metrics["max_simultaneous_notes"]),
                    str(item["human_audio_review"]["status"]),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Not Proven",
            "",
        ]
    )
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build duration/coverage fill human/audio review boundary")
    parser.add_argument("--keep_consolidation", type=str, required=True)
    parser.add_argument("--duration_fill_summary", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--require_pending", action="store_true")
    parser.add_argument("--require_no_preference", action="store_true")
    parser.add_argument("--expect_distinct_midi_content", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_human_audio_boundary(
        read_json(Path(args.keep_consolidation)),
        read_json(Path(args.duration_fill_summary)),
        output_dir=output_dir,
    )
    summary = validate_human_audio_boundary(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        require_pending=bool(args.require_pending),
        require_no_preference=bool(args.require_no_preference),
        expect_distinct_midi_content=bool(args.expect_distinct_midi_content),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_human_audio_boundary.json"
    markdown_path = output_dir / "duration_coverage_fill_human_audio_boundary.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_human_audio_boundary_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
