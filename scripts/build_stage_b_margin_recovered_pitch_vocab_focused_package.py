"""Build a focused package from a margin-recovered pitch vocabulary sweep result."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_margin_recovered_focused_package import (  # noqa: E402
    build_margin_recovered_focused_package,
    markdown_report,
    read_json,
    validate_package,
)
from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402


class MarginRecoveredPitchVocabFocusedPackageError(ValueError):
    pass


DEFAULT_DECISION = "pitch_vocab_qualified"


def selected_candidate(sweep_summary: dict[str, Any]) -> dict[str, Any]:
    candidate = sweep_summary.get("selected_candidate")
    if not isinstance(candidate, dict):
        raise MarginRecoveredPitchVocabFocusedPackageError("sweep summary must contain selected_candidate")
    return candidate


def review_notes_from_sweep(
    sweep_summary: dict[str, Any],
    *,
    decision: str = DEFAULT_DECISION,
) -> dict[str, Any]:
    candidate = selected_candidate(sweep_summary)
    candidate_id = str(candidate.get("candidate_id") or "")
    midi_path = str(candidate.get("midi_path") or "")
    if not candidate_id:
        raise MarginRecoveredPitchVocabFocusedPackageError("selected candidate_id is required")
    if not midi_path:
        raise MarginRecoveredPitchVocabFocusedPackageError("selected candidate midi_path is required")
    metrics = dict(candidate.get("metrics") or {})
    temporal = dict(candidate.get("temporal_coverage") or {})
    focused = dict(candidate.get("focused_solo_metrics") or {})
    gate = candidate.get("pitch_vocab_gate") if isinstance(candidate.get("pitch_vocab_gate"), dict) else {}
    summary = sweep_summary.get("sweep_summary") if isinstance(sweep_summary.get("sweep_summary"), dict) else {}
    return {
        "schema_version": "stage_b_margin_recovered_pitch_vocab_focused_review_notes_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_pitch_vocab_sweep": str(sweep_summary.get("output_dir") or ""),
        "candidates": [
            {
                "candidate_id": candidate_id,
                "review_metadata": {
                    "mode": "margin_recovered_pitch_vocab_sweep",
                    "review_rank": 1,
                    "source_run_id": str(candidate.get("source_run_id") or ""),
                    "sample_index": int(candidate.get("sample_index", 0) or 0),
                    "sample_seed": int(candidate.get("sample_seed", 0) or 0),
                },
                "review_files": {
                    "midi_path": midi_path,
                },
                "source_metrics": {
                    **metrics,
                    **temporal,
                    "focused_note_count": int(focused.get("focused_note_count", 0) or 0),
                    "focused_unique_pitch_count": int(focused.get("focused_unique_pitch_count", 0) or 0),
                    "focused_adjacent_pitch_repeats": int(focused.get("focused_adjacent_pitch_repeats", 0) or 0),
                    "focused_duplicated_3_note_pitch_class_chunks": int(
                        focused.get("focused_duplicated_3_note_pitch_class_chunks", 0) or 0
                    ),
                    "previous_dead_air_delta": float(summary.get("dead_air_delta_from_previous", 0.0) or 0.0),
                    "previous_unique_pitch_delta": int(
                        summary.get("focused_unique_pitch_delta_from_previous", 0) or 0
                    ),
                },
                "listening": {
                    "decision": str(decision),
                    "phrase_quality": "pending_context",
                    "timing": "pending_context",
                    "chord_fit": "pending_context",
                    "jazz_vocabulary": "pending_context",
                    "notes": "Selected by pitch vocabulary sweep; not a human listening decision.",
                },
                "objective_review": {
                    "objective_flags": list(gate.get("flags") or []),
                    "qualified": bool(gate.get("qualified", False)),
                },
            }
        ],
    }


def build_pitch_vocab_focused_package(
    sweep_summary: dict[str, Any],
    *,
    output_dir: Path,
    decision: str = DEFAULT_DECISION,
) -> dict[str, Any]:
    review_notes = review_notes_from_sweep(sweep_summary, decision=decision)
    package = build_margin_recovered_focused_package(
        review_notes,
        output_dir=output_dir,
        decision=decision,
    )
    package["source_pitch_vocab_sweep"] = str(sweep_summary.get("output_dir") or "")
    package["pitch_vocab_sweep_summary"] = dict(sweep_summary.get("sweep_summary") or {})
    return package


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build focused package from pitch vocabulary sweep")
    parser.add_argument("--sweep_summary", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_pitch_vocab_focused_package"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--decision", type=str, default=DEFAULT_DECISION)
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    sweep_summary = read_json(Path(args.sweep_summary))
    package = build_pitch_vocab_focused_package(
        sweep_summary,
        output_dir=output_dir,
        decision=str(args.decision),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "focused_review_package.json", package)
    (output_dir / "focused_review_package.md").write_text(markdown_report(package), encoding="utf-8")
    summary = validate_package(
        package,
        expected_candidate_id=str(args.expected_candidate_id or ""),
        min_candidates=int(args.min_candidates),
    )
    summary.update(
        {
            "package_path": str(output_dir / "focused_review_package.json"),
            "markdown_path": str(output_dir / "focused_review_package.md"),
        }
    )
    write_json(output_dir / "focused_review_package_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
