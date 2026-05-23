"""Build and validate Stage B clean context listening review notes."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "stage_b_clean_listening_review_notes_v1"

TIMING_VALUES = {"pending", "strong", "acceptable", "stiff", "rushed", "dragging"}
CHORD_FIT_VALUES = {"pending", "strong", "acceptable", "too_safe", "too_outside", "unclear"}
PHRASE_CONTINUATION_VALUES = {"pending", "strong", "acceptable", "weak", "broken"}
LANDING_VALUES = {"pending", "strong", "acceptable", "weak", "unresolved"}
JAZZ_VOCABULARY_VALUES = {"pending", "strong", "acceptable", "thin", "exercise_like"}
DECISION_VALUES = {"pending", "keep", "needs_followup", "reject"}


class CleanListeningReviewNotesError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _first_present(*values: Any, default: Any = "") -> Any:
    for value in values:
        if value not in (None, ""):
            return value
    return default


def build_candidate_note(candidate: dict[str, Any], diagnostics_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_id = str(candidate.get("candidate_id") or "").strip()
    if not candidate_id:
        raise CleanListeningReviewNotesError("clean package candidate_id is required")
    diagnostics = diagnostics_by_id.get(candidate_id, {})
    package_metrics = _as_mapping(candidate.get("metrics"))
    if not package_metrics:
        package_metrics = candidate
    phrase_metrics = _as_mapping(diagnostics.get("solo_metrics"))
    if not phrase_metrics:
        phrase_metrics = _as_mapping(diagnostics.get("phrase_metrics"))
    return {
        "candidate_id": candidate_id,
        "review_files": {
            "midi_path": str(_first_present(candidate.get("review_midi_path"), candidate.get("midi_path"))),
            "context_midi_path": str(candidate.get("context_midi_path") or ""),
            "chord_guide_path": str(candidate.get("chord_guide_path") or ""),
            "bass_root_guide_path": str(candidate.get("bass_root_guide_path") or ""),
        },
        "source_metrics": {
            "note_count": int(_first_present(package_metrics.get("note_count"), phrase_metrics.get("note_count"), default=0) or 0),
            "unique_pitch_count": int(
                _first_present(package_metrics.get("unique_pitch_count"), phrase_metrics.get("unique_pitch_count"), default=0)
                or 0
            ),
            "chord_tone_ratio": float(package_metrics.get("chord_tone_ratio", 0.0) or 0.0),
            "tension_ratio": float(package_metrics.get("tension_ratio", 0.0) or 0.0),
            "unresolved_large_leap_ratio": float(package_metrics.get("unresolved_large_leap_ratio", 0.0) or 0.0),
            "bar_coverage_ratio": float(phrase_metrics.get("bar_coverage_ratio", 0.0) or 0.0),
            "off_grid_ratio": float(
                _first_present(phrase_metrics.get("off_sixteenth_grid_ratio"), phrase_metrics.get("off_grid_ratio"), default=0.0)
                or 0.0
            ),
            "max_duration_beats": float(phrase_metrics.get("max_duration_beats", 0.0) or 0.0),
            "max_simultaneous_notes": int(phrase_metrics.get("max_simultaneous_notes", 0) or 0),
        },
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


def _require_enum(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise CleanListeningReviewNotesError(f"{path} must be one of {sorted(allowed)}")


def build_clean_listening_review_notes(clean_package: dict[str, Any], clean_context_diagnostics: dict[str, Any]) -> dict[str, Any]:
    candidates = clean_package.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CleanListeningReviewNotesError("clean package must contain non-empty candidates")
    diagnostic_candidates = clean_context_diagnostics.get("candidates", [])
    diagnostics_by_id = {
        str(item.get("candidate_id") or ""): item
        for item in diagnostic_candidates
        if isinstance(item, dict) and str(item.get("candidate_id") or "").strip()
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_clean_review_package": str(clean_package.get("output_dir") or ""),
        "source_clean_context_diagnostics": str(clean_context_diagnostics.get("output_dir") or ""),
        "reviewer": "",
        "review_context": {
            "listen_with_context_midi": True,
            "compare_solo_and_context": True,
            "focus": ["timing", "chord_fit", "phrase_continuation", "landing", "jazz_vocabulary"],
        },
        "candidates": [build_candidate_note(candidate, diagnostics_by_id) for candidate in candidates],
    }


def validate_clean_listening_review_notes(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CleanListeningReviewNotesError(f"schema_version must be {SCHEMA_VERSION!r}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CleanListeningReviewNotesError("candidates must be a non-empty list")
    seen: set[str] = set()
    reviewed_count = 0
    decision_counts = {key: 0 for key in DECISION_VALUES}
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise CleanListeningReviewNotesError(f"candidate {index} must be an object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise CleanListeningReviewNotesError(f"candidate {index} candidate_id is required")
        if candidate_id in seen:
            raise CleanListeningReviewNotesError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        listening = candidate.get("listening")
        if not isinstance(listening, dict):
            raise CleanListeningReviewNotesError(f"{candidate_id}.listening must be an object")
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
    parser = argparse.ArgumentParser(description="Build Stage B clean listening review notes")
    parser.add_argument("--clean_package", type=str, required=True)
    parser.add_argument("--clean_context_diagnostics", type=str, required=True)
    parser.add_argument("--review_notes", type=str, default=None)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_clean_listening_review_notes"))
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
        clean_package_path = Path(args.clean_package)
        diagnostics_path = Path(args.clean_context_diagnostics)
        clean_package = read_json(clean_package_path)
        clean_package["output_dir"] = str(clean_package_path.parent)
        diagnostics = read_json(diagnostics_path)
        diagnostics["output_dir"] = str(diagnostics_path.parent)
        notes = build_clean_listening_review_notes(clean_package, diagnostics)
        notes_path = run_dir / "clean_listening_review_notes_template.json"
        write_json(notes_path, notes)
    summary = validate_clean_listening_review_notes(notes)
    write_json(run_dir / "clean_listening_review_notes_summary.json", summary)
    print(json.dumps({**summary, "review_notes_path": str(notes_path)}, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
