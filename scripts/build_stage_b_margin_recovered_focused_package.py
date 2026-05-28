"""Build a focused package for the Stage B margin-recovered proxy keep candidate."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_focused_review_package import (  # noqa: E402
    build_focused_review_package,
    markdown_report,
)
from scripts.build_stage_b_margin_recovered_listening_notes import write_json  # noqa: E402
from scripts.run_stage_b_data_motif_generation_compare import write_context_midi  # noqa: E402
from scripts.run_stage_b_generation_probe import postprocess_stage_b_midi  # noqa: E402


class MarginRecoveredFocusedPackageError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_report_path(candidate: dict[str, Any]) -> Path:
    files = candidate.get("review_files") if isinstance(candidate.get("review_files"), dict) else {}
    midi_path_text = str(files.get("midi_path") or "").strip()
    if not midi_path_text:
        raise MarginRecoveredFocusedPackageError(f"{candidate.get('candidate_id')}.review_files.midi_path is required")
    midi_path = Path(midi_path_text)
    return midi_path.parent.parent / "report.json"


def request_context(report: dict[str, Any]) -> tuple[list[str], float, int]:
    request = report.get("request") if isinstance(report.get("request"), dict) else {}
    chords = request.get("chord_progression")
    if not isinstance(chords, list) or not chords:
        raise MarginRecoveredFocusedPackageError("generation report request.chord_progression is required")
    bpm = float(request.get("bpm") or 124.0)
    bars = int(request.get("bars") or 2)
    return [str(chord) for chord in chords], bpm, bars


def note_summary(midi_path: Path, *, limit: int = 16) -> list[dict[str, Any]]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    notes.sort(key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    rows: list[dict[str, Any]] = []
    for note in notes[:limit]:
        rows.append(
            {
                "pitch": int(note.pitch),
                "start_sec": round(float(note.start), 6),
                "end_sec": round(float(note.end), 6),
                "duration_sec": round(float(note.end) - float(note.start), 6),
                "velocity": int(note.velocity),
            }
        )
    return rows


def non_drum_notes(midi_path: Path) -> list[pretty_midi.Note]:
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def max_simultaneous_notes(notes: list[pretty_midi.Note]) -> int:
    if not notes:
        return 0
    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((float(note.start), 1))
        events.append((float(note.end), -1))
    active = 0
    peak = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        peak = max(peak, active)
    return int(peak)


def midi_metric_summary(midi_path: Path) -> dict[str, Any]:
    notes = non_drum_notes(midi_path)
    return {
        "note_count": int(len(notes)),
        "unique_pitch_count": int(len({int(note.pitch) for note in notes})),
        "max_simultaneous_notes": max_simultaneous_notes(notes),
    }


def write_solo_line_midi(source_midi_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi = pretty_midi.PrettyMIDI(str(source_midi_path))
    summary = postprocess_stage_b_midi(midi, simultaneous_limit=1)
    midi.write(str(output_path))
    summary["solo_line_midi_path"] = str(output_path)
    return summary


def selected_keep_candidates(review_notes: dict[str, Any], *, decision: str) -> list[dict[str, Any]]:
    candidates = review_notes.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise MarginRecoveredFocusedPackageError("review notes must contain non-empty candidates")
    selected = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise MarginRecoveredFocusedPackageError("candidate must be an object")
        listening = candidate.get("listening") if isinstance(candidate.get("listening"), dict) else {}
        if str(listening.get("decision") or "") == decision:
            selected.append(candidate)
    return selected


def enrich_review_notes_with_context(
    review_notes: dict[str, Any],
    *,
    output_dir: Path,
    decision: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    enriched = json.loads(json.dumps(review_notes))
    selected_ids = {
        str(candidate.get("candidate_id") or "")
        for candidate in selected_keep_candidates(enriched, decision=decision)
    }
    objective_report = {"candidates": []}
    solo_source_dir = output_dir / "source_solo_midi"
    context_source_dir = output_dir / "source_context_midi"
    for candidate in enriched["candidates"]:
        candidate_id = str(candidate.get("candidate_id") or "")
        if candidate_id not in selected_ids:
            continue
        files = candidate.setdefault("review_files", {})
        listening = candidate.setdefault("listening", {})
        if "phrase_quality" not in listening:
            listening["phrase_quality"] = str(listening.get("phrase") or "")
        if "chord_fit" not in listening:
            listening["chord_fit"] = "not_scored"
        source_midi_path = Path(str(files.get("midi_path") or ""))
        solo_midi_path = solo_source_dir / f"{candidate_id}_solo_line.mid"
        solo_postprocess = write_solo_line_midi(source_midi_path, solo_midi_path)
        original_metrics = dict(candidate.get("source_metrics") or {})
        candidate["source_metrics_before_focused_package"] = original_metrics
        focused_metrics = midi_metric_summary(solo_midi_path)
        candidate["source_metrics"] = {
            **original_metrics,
            "original_note_count": int(original_metrics.get("note_count", 0) or 0),
            "original_unique_pitch_count": int(original_metrics.get("unique_pitch_count", 0) or 0),
            **focused_metrics,
            "focused_postprocess_removed_note_count": int(solo_postprocess.get("removed_note_count", 0) or 0),
            "focused_postprocess_removal_ratio": (
                float(solo_postprocess.get("removed_note_count", 0) or 0)
                / max(1.0, float(solo_postprocess.get("before_note_count", 0) or 0))
            ),
        }
        report = read_json(candidate_report_path(candidate))
        chords, bpm, bars = request_context(report)
        context_path = context_source_dir / f"{candidate_id}_with_context.mid"
        write_context_midi(solo_midi_path, context_path, chords, bpm=bpm, bars=bars)
        files["source_midi_path"] = str(source_midi_path)
        files["midi_path"] = str(solo_midi_path)
        files["context_midi_path"] = str(context_path)
        candidate["focused_package_transform"] = {
            "solo_line_postprocess": solo_postprocess,
            "context_chords": chords,
            "context_bpm": bpm,
            "context_bars": bars,
        }
        objective_report["candidates"].append(
            {
                "candidate_id": candidate_id,
                "first_16_notes": note_summary(solo_midi_path),
            }
        )
    return enriched, objective_report


def build_margin_recovered_focused_package(
    review_notes: dict[str, Any],
    *,
    output_dir: Path,
    decision: str = "keep",
) -> dict[str, Any]:
    enriched_notes, objective_report = enrich_review_notes_with_context(
        review_notes,
        output_dir=output_dir,
        decision=decision,
    )
    package = build_focused_review_package(
        enriched_notes,
        output_dir=output_dir,
        decision=decision,
        copy_files=True,
        objective_report=objective_report,
    )
    enriched_by_id = {
        str(candidate.get("candidate_id") or ""): candidate
        for candidate in enriched_notes.get("candidates", [])
        if isinstance(candidate, dict)
    }
    for candidate in package.get("candidates", []):
        source_candidate = enriched_by_id.get(str(candidate.get("candidate_id") or ""), {})
        if "source_metrics_before_focused_package" in source_candidate:
            candidate["source_metrics_before_focused_package"] = source_candidate[
                "source_metrics_before_focused_package"
            ]
        if "focused_package_transform" in source_candidate:
            candidate["focused_package_transform"] = source_candidate["focused_package_transform"]
    package["source_context_generation"] = {
        "source": "generation report request chord progression",
        "not_human_listening": True,
    }
    return package


def validate_package(
    package: dict[str, Any],
    *,
    expected_candidate_id: str | None,
    min_candidates: int,
) -> dict[str, Any]:
    candidates = package.get("candidates")
    if not isinstance(candidates, list):
        raise MarginRecoveredFocusedPackageError("package candidates must be a list")
    if len(candidates) < int(min_candidates):
        raise MarginRecoveredFocusedPackageError(f"candidate_count {len(candidates)} < {min_candidates}")
    if expected_candidate_id:
        actual = [str(candidate.get("candidate_id") or "") for candidate in candidates]
        if actual != [expected_candidate_id]:
            raise MarginRecoveredFocusedPackageError(f"expected candidate {expected_candidate_id}, got {actual}")
    copied_files = 0
    for candidate in candidates:
        files = candidate.get("review_files") if isinstance(candidate.get("review_files"), dict) else {}
        for key in ("midi_path", "context_midi_path"):
            path = Path(str(files.get(key) or ""))
            if not path.exists():
                raise MarginRecoveredFocusedPackageError(f"{candidate.get('candidate_id')}.{key} does not exist: {path}")
            copied_files += 1
        metrics = candidate.get("source_metrics") if isinstance(candidate.get("source_metrics"), dict) else {}
        if int(metrics.get("max_simultaneous_notes", 0) or 0) > 1:
            raise MarginRecoveredFocusedPackageError(
                f"{candidate.get('candidate_id')} max_simultaneous_notes must be <= 1"
            )
    return {
        "candidate_count": int(len(candidates)),
        "candidate_ids": [str(candidate.get("candidate_id") or "") for candidate in candidates],
        "copied_midi_files": int(copied_files),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Stage B margin-recovered proxy keep focused package")
    parser.add_argument("--review_notes", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_margin_recovered_focused_package"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--decision", type=str, default="keep")
    parser.add_argument("--expected_candidate_id", type=str, default="")
    parser.add_argument("--min_candidates", type=int, default=1)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    review_notes_path = Path(args.review_notes)
    review_notes = read_json(review_notes_path)
    review_notes["review_notes_path"] = str(review_notes_path)
    package = build_margin_recovered_focused_package(
        review_notes,
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
