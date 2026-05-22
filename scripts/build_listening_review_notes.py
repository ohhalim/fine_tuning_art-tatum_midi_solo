"""Build and validate structured listening review notes for Stage B candidates."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


SCHEMA_VERSION = "stage_b_listening_review_notes_v1"

PHRASE_QUALITY_VALUES = {"pending", "phrase", "fragment", "exercise", "invalid"}
TIMING_VALUES = {"pending", "acceptable", "too_stiff", "too_loose", "off_grid"}
CHORD_FIT_VALUES = {"pending", "fits", "too_safe", "too_outside", "unclear"}
DECISION_VALUES = {"pending", "keep", "reject", "needs_followup"}
OBJECTIVE_BUCKET_VALUES = {"clean", "warning", "problem"}
ISSUE_VALUES = {
    "too_safe",
    "too_scalar",
    "too_mechanical",
    "weak_phrase",
    "bad_timing",
    "bad_chord_fit",
    "too_repetitive",
    "too_sparse",
    "other",
}


class ReviewNotesError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def build_candidate_note(sample: dict[str, Any]) -> dict[str, Any]:
    ratios = sample.get("role_ratios", {})
    return {
        "candidate_id": str(sample["sample_id"]),
        "source_metrics": {
            "note_count": int(sample.get("note_count", 0) or 0),
            "unique_pitch_count": int(sample.get("unique_pitch_count", 0) or 0),
            "chord_tone_ratio": float(ratios.get("chord_tone_ratio", 0.0) or 0.0),
            "tension_ratio": float(ratios.get("tension_ratio", 0.0) or 0.0),
            "approach_ratio": float(ratios.get("approach_ratio", 0.0) or 0.0),
            "outside_ratio": float(ratios.get("outside_ratio", 0.0) or 0.0),
        },
        "listening": {
            "status": "pending",
            "phrase_quality": "pending",
            "timing": "pending",
            "chord_fit": "pending",
            "issues": [],
            "decision": "pending",
            "notes": "",
        },
    }


def _candidate_id_from_review_manifest(sample: dict[str, Any]) -> str:
    mode = str(sample.get("mode") or "candidate").strip() or "candidate"
    rank = int(sample.get("review_rank", 0) or 0)
    sample_index = int(sample.get("sample_index", 0) or 0)
    if rank and sample_index:
        return f"{mode}_rank_{rank}_sample_{sample_index}"
    return str(sample.get("sample_id") or sample.get("review_midi_path") or mode)


def build_candidate_note_from_review_manifest(sample: dict[str, Any]) -> dict[str, Any]:
    ratios = sample.get("role_ratios", {})
    source_metrics = {
        "note_count": int(sample.get("note_count", 0) or 0),
        "unique_pitch_count": int(sample.get("unique_pitch_count", 0) or 0),
        "chord_tone_ratio": float(sample.get("chord_tone_ratio", ratios.get("chord_tone_ratio", 0.0)) or 0.0),
        "tension_ratio": float(sample.get("tension_ratio", ratios.get("tension_ratio", 0.0)) or 0.0),
        "approach_ratio": float(sample.get("approach_ratio", ratios.get("approach_ratio", 0.0)) or 0.0),
        "outside_ratio": float(sample.get("outside_ratio", ratios.get("outside_ratio", 0.0)) or 0.0),
        "root_tone_ratio": float(sample.get("root_tone_ratio", 0.0) or 0.0),
        "dead_air_ratio": float(sample.get("dead_air_ratio", 0.0) or 0.0),
        "syncopated_onset_ratio": float(sample.get("syncopated_onset_ratio", 0.0) or 0.0),
        "unique_bar_position_pattern_ratio": float(sample.get("unique_bar_position_pattern_ratio", 0.0) or 0.0),
        "duration_diversity_ratio": float(sample.get("duration_diversity_ratio", 0.0) or 0.0),
        "most_common_duration_ratio": float(sample.get("most_common_duration_ratio", 0.0) or 0.0),
        "ioi_diversity_ratio": float(sample.get("ioi_diversity_ratio", 0.0) or 0.0),
        "most_common_ioi_ratio": float(sample.get("most_common_ioi_ratio", 0.0) or 0.0),
    }
    return {
        "candidate_id": _candidate_id_from_review_manifest(sample),
        "review_metadata": {
            "mode": str(sample.get("mode", "")),
            "review_rank": int(sample.get("review_rank", 0) or 0),
            "sample_index": int(sample.get("sample_index", 0) or 0),
            "sample_seed": sample.get("sample_seed"),
            "valid": bool(sample.get("valid", False)),
            "strict_valid": bool(sample.get("strict_valid", False)),
        },
        "review_files": {
            "midi_path": str(sample.get("review_midi_path") or sample.get("midi_path") or ""),
            "source_midi_path": str(sample.get("midi_path") or ""),
            "context_midi_path": str(sample.get("context_midi_path") or ""),
        },
        "source_metrics": source_metrics,
        "listening": {
            "status": "pending",
            "phrase_quality": "pending",
            "timing": "pending",
            "chord_fit": "pending",
            "issues": [],
            "decision": "pending",
            "notes": "",
        },
    }


def objective_review_by_candidate(objective_midi_review_report: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not objective_midi_review_report:
        return {}
    candidates = objective_midi_review_report.get("candidates")
    if not isinstance(candidates, list):
        raise ReviewNotesError("objective MIDI review report candidates must be a list")
    indexed: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ReviewNotesError("objective MIDI review candidate must be an object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ReviewNotesError("objective MIDI review candidate_id is required")
        indexed[candidate_id] = candidate
    return indexed


def attach_objective_review(candidate_note: dict[str, Any], objective_candidate: dict[str, Any] | None) -> None:
    if not objective_candidate:
        return
    metrics = objective_candidate.get("metrics", {})
    candidate_note["objective_review"] = {
        "objective_flags": list(objective_candidate.get("objective_flags", [])),
        "objective_penalty": int(objective_candidate.get("objective_penalty", 0) or 0),
        "objective_priority_score": int(objective_candidate.get("objective_priority_score", 0) or 0),
        "objective_reviewable": bool(objective_candidate.get("objective_reviewable", False)),
        "objective_bucket": str(objective_candidate.get("objective_bucket", "problem")),
        "metrics": {
            "max_active_notes": int(metrics.get("max_active_notes", 0) or 0),
            "polyphonic_tick_ratio": float(metrics.get("polyphonic_tick_ratio", 0.0) or 0.0),
            "off_sixteenth_grid_count": int(metrics.get("off_sixteenth_grid_count", 0) or 0),
            "stepwise_interval_ratio": float(metrics.get("stepwise_interval_ratio", 0.0) or 0.0),
            "chromatic_interval_ratio": float(metrics.get("chromatic_interval_ratio", 0.0) or 0.0),
            "chord_tone_ratio": float(metrics.get("chord_tone_ratio", 0.0) or 0.0),
            "tension_ratio": float(metrics.get("tension_ratio", 0.0) or 0.0),
            "outside_ratio": float(metrics.get("outside_ratio", 0.0) or 0.0),
            "most_common_duration_ratio": float(metrics.get("most_common_duration_ratio", 0.0) or 0.0),
        },
    }


def build_review_notes_template(
    generated_chord_eval_report: dict[str, Any],
    source_review_markdown: str | None = None,
) -> dict[str, Any]:
    samples = generated_chord_eval_report.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ReviewNotesError("generated chord eval report must contain non-empty samples")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_generated_chord_eval_report": str(generated_chord_eval_report.get("source_report_path", "")),
        "source_review_markdown": source_review_markdown,
        "reviewer": "",
        "review_context": {
            "listen_with_context_midi": True,
            "compare_solo_and_context": True,
            "do_not_infer_real_reference_chords": True,
        },
        "candidates": [build_candidate_note(sample) for sample in samples],
    }


def build_review_notes_from_review_manifest(
    review_manifest: dict[str, Any],
    source_review_markdown: str | None = None,
    objective_midi_review_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    candidates = review_manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ReviewNotesError("review manifest must contain non-empty candidates")
    objective_by_candidate = objective_review_by_candidate(objective_midi_review_report)
    candidate_notes = [build_candidate_note_from_review_manifest(sample) for sample in candidates]
    for note in candidate_notes:
        attach_objective_review(note, objective_by_candidate.get(note["candidate_id"]))
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_review_manifest": str(review_manifest.get("source_manifest_path", "")),
        "source_review_markdown": source_review_markdown,
        "source_objective_midi_review_report": str(
            objective_midi_review_report.get("source_report_path", "") if objective_midi_review_report else ""
        ),
        "reviewer": "",
        "review_context": {
            "listen_with_context_midi": True,
            "compare_solo_and_context": True,
            "do_not_infer_real_reference_chords": True,
        },
        "chord_progression": review_manifest.get("chord_progression", []),
        "candidates": candidate_notes,
    }


def _require_enum(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise ReviewNotesError(f"{path} must be one of {sorted(allowed)}")


def validate_review_notes(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ReviewNotesError(f"schema_version must be {SCHEMA_VERSION!r}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ReviewNotesError("candidates must be a non-empty list")
    seen: set[str] = set()
    completed = 0
    decisions: dict[str, int] = {decision: 0 for decision in DECISION_VALUES}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise ReviewNotesError(f"candidate {index} must be an object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ReviewNotesError(f"candidate {index} candidate_id is required")
        if candidate_id in seen:
            raise ReviewNotesError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)

        listening = candidate.get("listening")
        if not isinstance(listening, dict):
            raise ReviewNotesError(f"{candidate_id}: listening object is required")
        _require_enum(listening.get("status"), {"pending", "reviewed"}, f"{candidate_id}.listening.status")
        _require_enum(listening.get("phrase_quality"), PHRASE_QUALITY_VALUES, f"{candidate_id}.listening.phrase_quality")
        _require_enum(listening.get("timing"), TIMING_VALUES, f"{candidate_id}.listening.timing")
        _require_enum(listening.get("chord_fit"), CHORD_FIT_VALUES, f"{candidate_id}.listening.chord_fit")
        _require_enum(listening.get("decision"), DECISION_VALUES, f"{candidate_id}.listening.decision")
        issues = listening.get("issues", [])
        if not isinstance(issues, list):
            raise ReviewNotesError(f"{candidate_id}.listening.issues must be a list")
        for issue in issues:
            _require_enum(issue, ISSUE_VALUES, f"{candidate_id}.listening.issues[]")
        objective_review = candidate.get("objective_review")
        if objective_review is not None:
            if not isinstance(objective_review, dict):
                raise ReviewNotesError(f"{candidate_id}.objective_review must be an object")
            _require_enum(
                objective_review.get("objective_bucket"),
                OBJECTIVE_BUCKET_VALUES,
                f"{candidate_id}.objective_review.objective_bucket",
            )
            if not isinstance(objective_review.get("objective_flags", []), list):
                raise ReviewNotesError(f"{candidate_id}.objective_review.objective_flags must be a list")
            if not isinstance(objective_review.get("objective_reviewable"), bool):
                raise ReviewNotesError(f"{candidate_id}.objective_review.objective_reviewable must be a boolean")
        if listening.get("status") == "reviewed":
            completed += 1
        decisions[str(listening.get("decision"))] += 1
    return {
        "candidate_count": int(len(candidates)),
        "reviewed_count": int(completed),
        "pending_count": int(len(candidates) - completed),
        "decision_counts": decisions,
    }


def markdown_summary(summary: dict[str, Any], output_path: Path) -> str:
    lines = [
        "# Stage B Listening Review Notes",
        "",
        f"- output: `{output_path}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- reviewed: `{summary['reviewed_count']}`",
        f"- pending: `{summary['pending_count']}`",
        "",
        "Decision counts:",
        "",
    ]
    for decision, count in sorted(summary["decision_counts"].items()):
        lines.append(f"- {decision}: `{count}`")
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or validate Stage B listening review notes")
    parser.add_argument("--generated_chord_eval_report", type=str, default=None)
    parser.add_argument("--review_manifest", type=str, default=None)
    parser.add_argument("--source_review_markdown", type=str, default=None)
    parser.add_argument("--objective_midi_review_report", type=str, default=None)
    parser.add_argument("--review_notes", type=str, default=None)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_listening_review_notes"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    if args.review_notes:
        notes_path = Path(args.review_notes)
        notes = read_json(notes_path)
    else:
        if args.review_manifest:
            review_manifest_path = Path(args.review_manifest)
            review_manifest = read_json(review_manifest_path)
            review_manifest["source_manifest_path"] = str(review_manifest_path)
            objective_report = None
            if args.objective_midi_review_report:
                objective_report_path = Path(args.objective_midi_review_report)
                objective_report = read_json(objective_report_path)
                objective_report["source_report_path"] = str(objective_report_path)
            notes = build_review_notes_from_review_manifest(
                review_manifest,
                source_review_markdown=args.source_review_markdown,
                objective_midi_review_report=objective_report,
            )
        else:
            if not args.generated_chord_eval_report:
                raise ReviewNotesError(
                    "--generated_chord_eval_report or --review_manifest is required when --review_notes is not provided"
                )
            generated_report = read_json(Path(args.generated_chord_eval_report))
            notes = build_review_notes_template(generated_report, source_review_markdown=args.source_review_markdown)
        notes_path = run_dir / "review_notes_template.json"
        write_json(notes_path, notes)
    summary = validate_review_notes(notes)
    write_json(run_dir / "review_notes_summary.json", summary)
    (run_dir / "review_notes_summary.md").write_text(markdown_summary(summary, notes_path), encoding="utf-8")
    print(json.dumps({**summary, "review_notes_path": str(notes_path)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
