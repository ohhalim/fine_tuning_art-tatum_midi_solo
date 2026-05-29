"""Consolidate the duration/coverage fill keep candidate boundary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class DurationCoverageFillKeepConsolidationError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def single_keep_candidate(filled_notes: dict[str, Any]) -> dict[str, Any]:
    candidates = filled_notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise DurationCoverageFillKeepConsolidationError("filled notes must contain candidates")
    keep_candidates = [
        candidate
        for candidate in candidates
        if str(candidate.get("listening", {}).get("decision") or "") == "keep"
    ]
    if len(keep_candidates) != 1:
        raise DurationCoverageFillKeepConsolidationError(
            f"expected exactly one keep candidate, got {len(keep_candidates)}"
        )
    return keep_candidates[0]


def compact_keep_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    metadata = candidate.get("review_metadata") if isinstance(candidate.get("review_metadata"), dict) else {}
    listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
    metrics = candidate.get("focused_context_metrics") if isinstance(candidate.get("focused_context_metrics"), dict) else {}
    evidence = candidate.get("listening_fill_evidence") if isinstance(candidate.get("listening_fill_evidence"), dict) else {}
    files = candidate.get("review_files") if isinstance(candidate.get("review_files"), dict) else {}
    return {
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "mode": str(metadata.get("mode") or ""),
        "decision": str(listening.get("decision") or ""),
        "timing": str(listening.get("timing") or ""),
        "chord_fit": str(listening.get("chord_fit") or ""),
        "phrase_continuation": str(listening.get("phrase_continuation") or ""),
        "landing": str(listening.get("landing") or ""),
        "jazz_vocabulary": str(listening.get("jazz_vocabulary") or ""),
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "range": str(metrics.get("range") or ""),
        "phrase_span_beats": float(metrics.get("phrase_span_beats", 0.0) or 0.0),
        "max_active_notes": int(metrics.get("max_simultaneous_notes", 0) or 0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "onset_coverage_ratio": float(metrics.get("onset_coverage_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(metrics.get("sustained_coverage_ratio", 0.0) or 0.0),
        "adjacent_pitch_repeats": int(metrics.get("adjacent_pitch_repeats", 0) or 0),
        "duplicated_3_note_pitch_class_chunks": int(
            metrics.get("duplicated_3_note_pitch_class_chunks", 0) or 0
        ),
        "max_interval": int(metrics.get("max_interval", 0) or 0),
        "final_note": str(metrics.get("final_note") or ""),
        "final_chord": str(metrics.get("final_chord") or ""),
        "final_note_role": str(metrics.get("final_note_role") or ""),
        "review_risks": list(candidate.get("review_risks") or []),
        "not_human_audio_review": bool(evidence.get("not_human_audio_review", False)),
        "midi_path": str(files.get("midi_path") or ""),
        "context_midi_path": str(files.get("context_midi_path") or ""),
        "source_midi_path": str(files.get("source_midi_path") or ""),
    }


def compact_duration_repair(duration_fill_summary: dict[str, Any]) -> dict[str, Any]:
    repair = duration_fill_summary.get("repair_summary")
    selected = duration_fill_summary.get("selected_candidate")
    source = duration_fill_summary.get("source_candidate")
    if not isinstance(repair, dict) or not isinstance(selected, dict) or not isinstance(source, dict):
        raise DurationCoverageFillKeepConsolidationError("duration fill summary is missing repair/source data")
    fill_repair = selected.get("fill_repair") if isinstance(selected.get("fill_repair"), dict) else {}
    gate = selected.get("duration_coverage_gate") if isinstance(selected.get("duration_coverage_gate"), dict) else {}
    selected_metrics = selected.get("metrics") if isinstance(selected.get("metrics"), dict) else {}
    source_metrics = source.get("metrics") if isinstance(source.get("metrics"), dict) else {}
    selected_focused = (
        selected.get("focused_solo_metrics") if isinstance(selected.get("focused_solo_metrics"), dict) else {}
    )
    source_focused = source.get("focused_solo_metrics") if isinstance(source.get("focused_solo_metrics"), dict) else {}
    return {
        "source_candidate_id": str(source.get("candidate_id") or ""),
        "selected_candidate_id": str(repair.get("selected_candidate_id") or selected.get("candidate_id") or ""),
        "variant_count": int(duration_fill_summary.get("variant_count", 0) or 0),
        "qualified_variant_count": int(duration_fill_summary.get("qualified_variant_count", 0) or 0),
        "qualified": bool(repair.get("qualified", False)),
        "remaining_flags": list(repair.get("remaining_flags") or gate.get("flags") or []),
        "fill_addition_count": int(repair.get("selected_fill_addition_count", 0) or 0),
        "max_additions": int(fill_repair.get("max_additions", 0) or 0),
        "baseline_dead_air_ratio": float(repair.get("baseline_dead_air_ratio", 0.0) or 0.0),
        "selected_dead_air_ratio": float(repair.get("selected_dead_air_ratio", 0.0) or 0.0),
        "dead_air_delta_from_baseline": float(repair.get("dead_air_delta_from_baseline", 0.0) or 0.0),
        "source_note_count": int(source_focused.get("focused_note_count", 0) or source_metrics.get("note_count", 0) or 0),
        "source_unique_pitch_count": int(
            source_focused.get("focused_unique_pitch_count", 0) or source_metrics.get("unique_pitch_count", 0) or 0
        ),
        "selected_note_count": int(
            selected_focused.get("focused_note_count", 0) or selected_metrics.get("note_count", 0) or 0
        ),
        "selected_unique_pitch_count": int(
            selected_focused.get("focused_unique_pitch_count", 0)
            or selected_metrics.get("unique_pitch_count", 0)
            or 0
        ),
        "selected_adjacent_pitch_repeats": int(
            repair.get("selected_adjacent_pitch_repeats", 0)
            or selected_focused.get("focused_adjacent_pitch_repeats", 0)
            or 0
        ),
        "selected_duplicated_3_note_pitch_class_chunks": int(
            repair.get("selected_duplicated_3_note_pitch_class_chunks", 0)
            or selected_focused.get("focused_duplicated_3_note_pitch_class_chunks", 0)
            or 0
        ),
        "selected_max_interval": int(
            repair.get("selected_max_interval", 0) or selected_focused.get("focused_max_interval", 0) or 0
        ),
        "duration_coverage_fill_improved": bool(repair.get("duration_coverage_fill_improved", False)),
        "claim_boundary": str(repair.get("claim_boundary") or ""),
    }


def build_keep_consolidation_report(
    filled_notes: dict[str, Any],
    duration_fill_summary: dict[str, Any],
    *,
    output_dir: Path,
) -> dict[str, Any]:
    keep = compact_keep_candidate(single_keep_candidate(filled_notes))
    duration = compact_duration_repair(duration_fill_summary)
    if keep["candidate_id"] != duration["selected_candidate_id"]:
        raise DurationCoverageFillKeepConsolidationError(
            f"filled keep candidate does not match duration fill selected candidate: {keep['candidate_id']}"
        )

    boundary = (
        "single_postprocess_candidate_keep_support"
        if keep["decision"] == "keep"
        and duration["qualified"]
        and duration["duration_coverage_fill_improved"]
        and keep["not_human_audio_review"]
        else "insufficient_keep_support"
    )
    return {
        "schema_version": "stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_filled_notes_schema": str(filled_notes.get("schema_version") or ""),
        "source_duration_fill_schema": str(duration_fill_summary.get("schema_version") or ""),
        "decision_path": [
            "duration_coverage_fill_repair",
            "focused_context_package",
            "focused_listening_notes",
            "midi_context_evidence_fill",
            "keep_consolidation",
        ],
        "candidate": keep,
        "duration_coverage_repair": duration,
        "evidence_boundary": {
            "boundary": boundary,
            "claim": "single_postprocess_candidate_midi_context_keep",
            "postprocess_claim_boundary": duration["claim_boundary"],
            "selected_candidate_is_keep": keep["decision"] == "keep",
            "duration_fill_qualified": duration["qualified"],
            "duration_fill_improved_dead_air": duration["duration_coverage_fill_improved"],
            "not_human_audio_review": keep["not_human_audio_review"],
        },
        "proven": [
            "midi_context_evidence_keep",
            "dead_air_reduced_from_constrained_partial_candidate",
            "adjacent_repeat_blocker_repaired",
            "wide_interval_blocker_repaired",
            "solo_context_review_artifact_available",
            "final_landing_chord_tone",
        ],
        "not_proven": [
            "human_audio_preference",
            "broad_trained_model_quality",
            "brad_style_adaptation",
            "broad_repeatability",
            "production_ready_improviser",
        ],
        "next_recommended_issue": (
            "Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio comparison boundary"
        ),
    }


def validate_keep_consolidation(
    report: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    expected_boundary: str | None,
    require_not_human_audio_review: bool,
    require_postprocess_claim_boundary: str | None,
) -> dict[str, Any]:
    candidate = report.get("candidate") if isinstance(report.get("candidate"), dict) else {}
    boundary = report.get("evidence_boundary") if isinstance(report.get("evidence_boundary"), dict) else {}
    candidate_id = str(candidate.get("candidate_id") or "")
    if expected_candidate_id and candidate_id != expected_candidate_id:
        raise DurationCoverageFillKeepConsolidationError(
            f"expected candidate {expected_candidate_id}, got {candidate_id}"
        )
    if candidate.get("decision") != "keep":
        raise DurationCoverageFillKeepConsolidationError("candidate decision must be keep")
    if expected_boundary and str(boundary.get("boundary") or "") != expected_boundary:
        raise DurationCoverageFillKeepConsolidationError(
            f"expected boundary {expected_boundary}, got {boundary.get('boundary')}"
        )
    if require_not_human_audio_review and not bool(boundary.get("not_human_audio_review", False)):
        raise DurationCoverageFillKeepConsolidationError("expected not_human_audio_review boundary")
    if require_postprocess_claim_boundary and str(boundary.get("postprocess_claim_boundary") or "") != (
        require_postprocess_claim_boundary
    ):
        raise DurationCoverageFillKeepConsolidationError(
            "expected postprocess claim boundary "
            f"{require_postprocess_claim_boundary}, got {boundary.get('postprocess_claim_boundary')}"
        )
    return {
        "candidate_id": candidate_id,
        "decision": str(candidate.get("decision") or ""),
        "boundary": str(boundary.get("boundary") or ""),
        "postprocess_claim_boundary": str(boundary.get("postprocess_claim_boundary") or ""),
        "not_human_audio_review": bool(boundary.get("not_human_audio_review", False)),
        "note_count": int(candidate.get("note_count", 0) or 0),
        "unique_pitch_count": int(candidate.get("unique_pitch_count", 0) or 0),
        "dead_air_ratio": float(candidate.get("dead_air_ratio", 0.0) or 0.0),
        "onset_coverage_ratio": float(candidate.get("onset_coverage_ratio", 0.0) or 0.0),
        "sustained_coverage_ratio": float(candidate.get("sustained_coverage_ratio", 0.0) or 0.0),
        "adjacent_pitch_repeats": int(candidate.get("adjacent_pitch_repeats", 0) or 0),
        "max_interval": int(candidate.get("max_interval", 0) or 0),
        "fill_addition_count": int(report.get("duration_coverage_repair", {}).get("fill_addition_count", 0) or 0),
        "next_recommended_issue": str(report.get("next_recommended_issue") or ""),
    }


def markdown_report(report: dict[str, Any]) -> str:
    candidate = report["candidate"]
    repair = report["duration_coverage_repair"]
    boundary = report["evidence_boundary"]
    lines = [
        "# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Keep Consolidation",
        "",
        f"- candidate: `{candidate['candidate_id']}`",
        f"- source candidate: `{repair['source_candidate_id']}`",
        f"- decision: `{candidate['decision']}`",
        f"- boundary: `{boundary['boundary']}`",
        f"- postprocess claim boundary: `{boundary['postprocess_claim_boundary']}`",
        f"- fill additions: `{repair['fill_addition_count']}`",
        f"- dead-air: `{repair['baseline_dead_air_ratio']:.4f}` -> `{repair['selected_dead_air_ratio']:.4f}`",
        f"- grid coverage: onset `{candidate['onset_coverage_ratio']:.4f}`, sustained `{candidate['sustained_coverage_ratio']:.4f}`",
        "",
        "This is single postprocess candidate MIDI/context evidence, not human audio proof or broad model quality.",
        "",
        "| notes | unique | range | phrase span | max active | dead-air | adj repeat | dup3 | max interval | final landing | risks |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    risks = ",".join(candidate["review_risks"]) if candidate["review_risks"] else "none"
    final_landing = f"{candidate['final_note']} over {candidate['final_chord']} ({candidate['final_note_role']})"
    lines.append(
        "| "
        + " | ".join(
            [
                str(candidate["note_count"]),
                str(candidate["unique_pitch_count"]),
                str(candidate["range"]),
                f"{float(candidate['phrase_span_beats']):.3f}",
                str(candidate["max_active_notes"]),
                f"{float(candidate['dead_air_ratio']):.4f}",
                str(candidate["adjacent_pitch_repeats"]),
                str(candidate["duplicated_3_note_pitch_class_chunks"]),
                str(candidate["max_interval"]),
                final_landing,
                risks,
            ]
        )
        + " |"
    )
    lines.extend(["", "## Proven", ""])
    for item in report.get("proven", []):
        lines.append(f"- `{item}`")
    lines.extend(["", "## Not Proven", ""])
    for item in report.get("not_proven", []):
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Consolidate duration/coverage fill keep candidate evidence")
    parser.add_argument("--filled_notes", type=str, required=True)
    parser.add_argument("--duration_fill_summary", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation",
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--expected_boundary", type=str, default="")
    parser.add_argument("--require_not_human_audio_review", action="store_true")
    parser.add_argument("--require_postprocess_claim_boundary", type=str, default="")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_keep_consolidation_report(
        read_json(Path(args.filled_notes)),
        read_json(Path(args.duration_fill_summary)),
        output_dir=output_dir,
    )
    summary = validate_keep_consolidation(
        report,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        expected_boundary=str(args.expected_boundary or ""),
        require_not_human_audio_review=bool(args.require_not_human_audio_review),
        require_postprocess_claim_boundary=str(args.require_postprocess_claim_boundary or ""),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "duration_coverage_fill_keep_consolidation.json"
    markdown_path = output_dir / "duration_coverage_fill_keep_consolidation.md"
    write_json(report_path, report)
    write_json(output_dir / "duration_coverage_fill_keep_consolidation_validation_summary.json", summary)
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({**summary, "report_path": str(report_path), "markdown_path": str(markdown_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
