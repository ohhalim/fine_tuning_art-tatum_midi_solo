"""Build focused listening review notes from a focused Stage B package."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "stage_b_focused_listening_review_notes_v1"

TIMING_VALUES = {"pending", "strong", "acceptable", "stiff", "rushed", "dragging"}
CHORD_FIT_VALUES = {"pending", "strong", "acceptable", "too_safe", "too_outside", "unclear"}
PHRASE_CONTINUATION_VALUES = {"pending", "strong", "acceptable", "weak", "broken"}
LANDING_VALUES = {"pending", "strong", "acceptable", "weak", "unresolved"}
JAZZ_VOCABULARY_VALUES = {"pending", "strong", "acceptable", "thin", "exercise_like"}
DECISION_VALUES = {"pending", "keep", "needs_followup", "reject"}


class FocusedListeningReviewNotesError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compact_source_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "note_count": int(metrics.get("note_count", 0) or 0),
        "unique_pitch_count": int(metrics.get("unique_pitch_count", 0) or 0),
        "chord_tone_ratio": float(metrics.get("chord_tone_ratio", 0.0) or 0.0),
        "tension_ratio": float(metrics.get("tension_ratio", 0.0) or 0.0),
        "outside_ratio": float(metrics.get("outside_ratio", 0.0) or 0.0),
        "root_tone_ratio": float(metrics.get("root_tone_ratio", 0.0) or 0.0),
        "dead_air_ratio": float(metrics.get("dead_air_ratio", 0.0) or 0.0),
        "syncopated_onset_ratio": float(metrics.get("syncopated_onset_ratio", 0.0) or 0.0),
        "unique_bar_position_pattern_ratio": float(metrics.get("unique_bar_position_pattern_ratio", 0.0) or 0.0),
        "duration_diversity_ratio": float(metrics.get("duration_diversity_ratio", 0.0) or 0.0),
        "most_common_duration_ratio": float(metrics.get("most_common_duration_ratio", 0.0) or 0.0),
        "ioi_diversity_ratio": float(metrics.get("ioi_diversity_ratio", 0.0) or 0.0),
        "most_common_ioi_ratio": float(metrics.get("most_common_ioi_ratio", 0.0) or 0.0),
    }


def build_candidate_note(candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise FocusedListeningReviewNotesError("focused package candidate_id is required")
    files = _as_mapping(candidate.get("review_files"))
    source_metrics = _compact_source_metrics(_as_mapping(candidate.get("source_metrics")))
    proxy_review = _as_mapping(candidate.get("listening"))
    objective_review = _as_mapping(candidate.get("objective_review"))
    return {
        "candidate_id": candidate_id,
        "review_metadata": dict(_as_mapping(candidate.get("review_metadata"))),
        "review_files": {
            "midi_path": str(files.get("midi_path") or ""),
            "context_midi_path": str(files.get("context_midi_path") or ""),
            "source_midi_path": str(files.get("source_midi_path") or ""),
        },
        "source_metrics": source_metrics,
        "proxy_review": {
            "status": str(proxy_review.get("status") or ""),
            "phrase_quality": str(proxy_review.get("phrase_quality") or ""),
            "timing": str(proxy_review.get("timing") or ""),
            "chord_fit": str(proxy_review.get("chord_fit") or ""),
            "issues": list(proxy_review.get("issues") or []),
            "decision": str(proxy_review.get("decision") or ""),
            "notes": str(proxy_review.get("notes") or ""),
            "objective_flags": list(objective_review.get("objective_flags") or []),
            "objective_bucket": str(objective_review.get("objective_bucket") or ""),
        },
        "objective_first_16_notes": list(candidate.get("objective_first_16_notes") or []),
        "listening": {
            "status": "pending",
            "timing": "pending",
            "chord_fit": "pending",
            "phrase_continuation": "pending",
            "landing": "pending",
            "jazz_vocabulary": "pending",
            "decision": "pending",
            "notes": "",
        },
    }


def build_focused_listening_review_notes(focused_package: dict[str, Any]) -> dict[str, Any]:
    candidates = focused_package.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FocusedListeningReviewNotesError("focused package must contain non-empty candidates")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_focused_review_package": str(focused_package.get("output_dir") or ""),
        "reviewer": "",
        "review_context": {
            "listen_with_context_midi": True,
            "compare_solo_and_context": True,
            "real_listening_fields_are_separate_from_proxy_review": True,
            "focus": ["timing", "chord_fit", "phrase_continuation", "landing", "jazz_vocabulary"],
        },
        "candidates": [build_candidate_note(candidate) for candidate in candidates],
    }


def _require_enum(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise FocusedListeningReviewNotesError(f"{path} must be one of {sorted(allowed)}")


def validate_focused_listening_review_notes(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise FocusedListeningReviewNotesError(f"schema_version must be {SCHEMA_VERSION!r}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise FocusedListeningReviewNotesError("candidates must be a non-empty list")
    seen: set[str] = set()
    reviewed_count = 0
    decision_counts = {key: 0 for key in DECISION_VALUES}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise FocusedListeningReviewNotesError(f"candidate {index} must be an object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise FocusedListeningReviewNotesError(f"candidate {index} candidate_id is required")
        if candidate_id in seen:
            raise FocusedListeningReviewNotesError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        files = _as_mapping(candidate.get("review_files"))
        if not files.get("midi_path") or not files.get("context_midi_path"):
            raise FocusedListeningReviewNotesError(f"{candidate_id}.review_files must include midi_path and context_midi_path")
        proxy_review = candidate.get("proxy_review")
        if not isinstance(proxy_review, dict):
            raise FocusedListeningReviewNotesError(f"{candidate_id}.proxy_review must be an object")
        listening = candidate.get("listening")
        if not isinstance(listening, dict):
            raise FocusedListeningReviewNotesError(f"{candidate_id}.listening must be an object")
        _require_enum(listening.get("status"), {"pending", "reviewed"}, f"{candidate_id}.listening.status")
        _require_enum(listening.get("timing"), TIMING_VALUES, f"{candidate_id}.listening.timing")
        _require_enum(listening.get("chord_fit"), CHORD_FIT_VALUES, f"{candidate_id}.listening.chord_fit")
        _require_enum(
            listening.get("phrase_continuation"),
            PHRASE_CONTINUATION_VALUES,
            f"{candidate_id}.listening.phrase_continuation",
        )
        _require_enum(listening.get("landing"), LANDING_VALUES, f"{candidate_id}.listening.landing")
        _require_enum(
            listening.get("jazz_vocabulary"),
            JAZZ_VOCABULARY_VALUES,
            f"{candidate_id}.listening.jazz_vocabulary",
        )
        _require_enum(listening.get("decision"), DECISION_VALUES, f"{candidate_id}.listening.decision")
        if listening.get("status") == "reviewed":
            reviewed_count += 1
        decision_counts[str(listening.get("decision"))] += 1
    return {
        "candidate_count": len(candidates),
        "reviewed_count": reviewed_count,
        "pending_count": len(candidates) - reviewed_count,
        "decision_counts": decision_counts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage B focused listening review notes")
    parser.add_argument("--focused_package", type=str, required=True)
    parser.add_argument("--review_notes", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_focused_listening_review_notes"))
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
        focused_package_path = Path(args.focused_package)
        focused_package = read_json(focused_package_path)
        focused_package["output_dir"] = str(focused_package_path.parent)
        notes = build_focused_listening_review_notes(focused_package)
        notes_path = run_dir / "focused_listening_review_notes_template.json"
        write_json(notes_path, notes)
    summary = validate_focused_listening_review_notes(notes)
    write_json(run_dir / "focused_listening_review_notes_summary.json", summary)
    print(json.dumps({**summary, "review_notes_path": str(notes_path)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
