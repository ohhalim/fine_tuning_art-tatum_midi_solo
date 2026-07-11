"""Build listening review notes for Stage B margin-recovered candidates."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "stage_b_margin_recovered_listening_notes_v1"

TIMING_VALUES = {"pending", "strong", "acceptable", "stiff", "rushed", "dragging"}
PHRASE_VALUES = {"pending", "strong", "acceptable", "weak", "broken", "exercise_like"}
VOCAB_VALUES = {"pending", "strong", "acceptable", "thin", "too_repetitive", "too_safe"}
DECISION_VALUES = {"pending", "keep", "needs_followup", "reject"}


class MarginRecoveredListeningNotesError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")


def _candidate_id(candidate: dict[str, Any]) -> str:
    return "margin_recovered_rank_{rank}_seed_{seed}_sample_{sample}".format(
        rank=int(candidate.get("review_rank", 0) or 0),
        seed=int(candidate.get("seed", 0) or 0),
        sample=int(candidate.get("sample_index", 0) or 0),
    )


def _float(candidate: dict[str, Any], key: str) -> float:
    value = candidate.get(key, 0.0)
    return float(value if value is not None else 0.0)


def _int(candidate: dict[str, Any], key: str) -> int:
    value = candidate.get(key, 0)
    return int(value if value is not None else 0)


def build_candidate_note(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": _candidate_id(candidate),
        "review_metadata": {
            "review_rank": _int(candidate, "review_rank"),
            "is_selected_best": bool(candidate.get("is_selected_best", False)),
            "seed": _int(candidate, "seed"),
            "sample_index": _int(candidate, "sample_index"),
            "seed_strict_valid_sample_count": _int(candidate, "seed_strict_valid_sample_count"),
            "seed_sample_count": _int(candidate, "seed_sample_count"),
            "seed_dead_air_outlier_count": _int(candidate, "seed_dead_air_outlier_count"),
        },
        "review_files": {
            "midi_path": str(candidate.get("midi_path") or ""),
        },
        "source_metrics": {
            "dead_air_ratio": _float(candidate, "dead_air_ratio"),
            "note_count": _int(candidate, "note_count"),
            "unique_pitch_count": _int(candidate, "unique_pitch_count"),
            "phrase_coverage_ratio": _float(candidate, "phrase_coverage_ratio"),
            "onset_coverage_ratio": _float(candidate, "onset_coverage_ratio"),
            "sustained_coverage_ratio": _float(candidate, "sustained_coverage_ratio"),
            "postprocess_removal_ratio": _float(candidate, "postprocess_removal_ratio"),
        },
        "seed_failure_reasons": candidate.get("seed_failure_reasons", {}),
        "seed_strict_failure_reasons": candidate.get("seed_strict_failure_reasons", {}),
        "listening": {
            "status": "pending",
            "timing": "pending",
            "phrase": "pending",
            "jazz_vocabulary": "pending",
            "decision": "pending",
            "notes": "",
        },
    }


def build_listening_notes(review_export: dict[str, Any]) -> dict[str, Any]:
    candidates = review_export.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise MarginRecoveredListeningNotesError("review export must contain non-empty candidates")
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_review_export": str(review_export.get("source_summary_path", "")),
        "source_run_id": str(review_export.get("source_run_id", "")),
        "reviewer": "",
        "review_context": {
            "listen_to_generated_midi": True,
            "compare_rank_order_with_listening_preference": True,
            "generated_midi_files_are_not_committed": True,
            "fields_are_pending_until_explicit_review": True,
        },
        "source_summary": review_export.get("summary", {}),
        "candidates": [build_candidate_note(candidate) for candidate in candidates],
    }


def _require_enum(value: Any, allowed: set[str], path: str) -> None:
    if value not in allowed:
        raise MarginRecoveredListeningNotesError(f"{path} must be one of {sorted(allowed)}")


def validate_listening_notes(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise MarginRecoveredListeningNotesError(f"schema_version must be {SCHEMA_VERSION!r}")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise MarginRecoveredListeningNotesError("candidates must be a non-empty list")
    seen: set[str] = set()
    decision_counts = {value: 0 for value in DECISION_VALUES}
    reviewed_count = 0
    selected_count = 0
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise MarginRecoveredListeningNotesError(f"candidate {index} must be an object")
        candidate_id = str(candidate.get("candidate_id") or "").strip()
        if not candidate_id:
            raise MarginRecoveredListeningNotesError(f"candidate {index} candidate_id is required")
        if candidate_id in seen:
            raise MarginRecoveredListeningNotesError(f"duplicate candidate_id: {candidate_id}")
        seen.add(candidate_id)
        files = candidate.get("review_files")
        if not isinstance(files, dict) or not str(files.get("midi_path") or "").strip():
            raise MarginRecoveredListeningNotesError(f"{candidate_id}.review_files.midi_path is required")
        metadata = candidate.get("review_metadata")
        if not isinstance(metadata, dict):
            raise MarginRecoveredListeningNotesError(f"{candidate_id}.review_metadata is required")
        if metadata.get("is_selected_best"):
            selected_count += 1
        listening = candidate.get("listening")
        if not isinstance(listening, dict):
            raise MarginRecoveredListeningNotesError(f"{candidate_id}.listening is required")
        _require_enum(listening.get("status"), {"pending", "reviewed"}, f"{candidate_id}.listening.status")
        _require_enum(listening.get("timing"), TIMING_VALUES, f"{candidate_id}.listening.timing")
        _require_enum(listening.get("phrase"), PHRASE_VALUES, f"{candidate_id}.listening.phrase")
        _require_enum(
            listening.get("jazz_vocabulary"),
            VOCAB_VALUES,
            f"{candidate_id}.listening.jazz_vocabulary",
        )
        _require_enum(listening.get("decision"), DECISION_VALUES, f"{candidate_id}.listening.decision")
        if listening.get("status") == "reviewed":
            reviewed_count += 1
        decision_counts[str(listening.get("decision"))] += 1
    if selected_count != 1:
        raise MarginRecoveredListeningNotesError("exactly one selected best candidate is required")
    return {
        "candidate_count": int(len(candidates)),
        "selected_best_count": int(selected_count),
        "reviewed_count": int(reviewed_count),
        "pending_count": int(len(candidates) - reviewed_count),
        "decision_counts": decision_counts,
    }


def markdown_report(notes: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# Stage B Margin-Recovered Listening Review Notes",
        "",
        f"- source run: `{notes['source_run_id']}`",
        f"- candidate count: `{summary['candidate_count']}`",
        f"- pending count: `{summary['pending_count']}`",
        f"- selected best count: `{summary['selected_best_count']}`",
        "",
        "| candidate | selected | seed | sample | rank | dead-air | notes | pitches | phrase | onset | sustained | decision |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for candidate in notes["candidates"]:
        metadata = candidate["review_metadata"]
        metrics = candidate["source_metrics"]
        listening = candidate["listening"]
        lines.append(
            "| `{candidate_id}` | {selected} | {seed} | {sample} | {rank} | {dead_air:.3f} | "
            "{notes_count} | {pitches} | {phrase:.3f} | {onset:.3f} | {sustained:.3f} | {decision} |".format(
                candidate_id=candidate["candidate_id"],
                selected=metadata["is_selected_best"],
                seed=metadata["seed"],
                sample=metadata["sample_index"],
                rank=metadata["review_rank"],
                dead_air=metrics["dead_air_ratio"],
                notes_count=metrics["note_count"],
                pitches=metrics["unique_pitch_count"],
                phrase=metrics["phrase_coverage_ratio"],
                onset=metrics["onset_coverage_ratio"],
                sustained=metrics["sustained_coverage_ratio"],
                decision=listening["decision"],
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage B margin-recovered listening review notes")
    parser.add_argument("--review_export", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_listening_notes"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--expected_candidate_count", type=int, default=None)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    review_export = read_json(Path(args.review_export))
    notes = build_listening_notes(review_export)
    summary = validate_listening_notes(notes)
    write_json(run_dir / "listening_review_notes_template.json", notes)
    write_json(run_dir / "listening_review_notes_summary.json", summary)
    (run_dir / "listening_review_notes_template.md").write_text(markdown_report(notes, summary), encoding="utf-8")
    print(json.dumps({**summary, "review_notes_path": str(run_dir / "listening_review_notes_template.json")}, indent=2))

    if args.expected_candidate_count is not None and summary["candidate_count"] != int(args.expected_candidate_count):
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
