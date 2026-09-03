"""Build note-level review tables for bebop-language best-of packages."""

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

from scripts.build_stage_b_midi_to_solo_bebop_language_package import (  # noqa: E402
    BebopLanguagePackageError,
    chord_for_time,
    chord_pitch_classes,
    parse_chord,
    pc_name,
    scale_intervals,
    solo_notes,
    write_json,
    write_text,
)


SCHEMA_VERSION = "stage_b_midi_to_solo_bebop_language_note_review_v1"


def pitch_name(pitch: int) -> str:
    return f"{pc_name(pitch)}{pitch // 12 - 1}"


def scale_pitch_classes(chord: str) -> set[int]:
    root, _ = parse_chord(chord)
    return {(root + interval) % 12 for interval in scale_intervals(chord)}


def pitch_role(pitch: int, chord: str) -> str:
    pitch_class = int(pitch) % 12
    if pitch_class in chord_pitch_classes(chord):
        return "chord_tone"
    if pitch_class in scale_pitch_classes(chord):
        return "scale_tension"
    return "outside"


def listen_first_ranks(package: dict[str, Any]) -> set[int]:
    ranks: set[int] = set()
    for item in package.get("listen_first", {}).get("files", []):
        ranks.add(int(item["rank"]))
    return ranks


def selected_rows(package: dict[str, Any], *, all_candidates: bool) -> list[dict[str, Any]]:
    selected = list(package.get("selected_candidates") or [])
    if all_candidates:
        return selected
    ranks = listen_first_ranks(package)
    return [item for item in selected if int(item["rank"]) in ranks]


def note_rows(candidate: dict[str, Any], *, bars: int, bpm: float, max_notes: int) -> list[dict[str, Any]]:
    midi_path = Path(str(candidate["midi_path"]))
    if not midi_path.exists():
        raise BebopLanguagePackageError(f"missing candidate MIDI: {midi_path}")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = solo_notes(pm)
    seconds_per_beat = 60.0 / float(bpm)
    rows: list[dict[str, Any]] = []
    previous_pitch: int | None = None
    for index, note in enumerate(notes[:max_notes], start=1):
        start = float(note.start)
        beat_position = start / seconds_per_beat
        bar_index = int(beat_position // 4) + 1
        beat_in_bar = (beat_position % 4) + 1
        chord = chord_for_time(candidate["chords"], start, bars=bars, bpm=bpm)
        interval = None if previous_pitch is None else int(note.pitch) - int(previous_pitch)
        rows.append(
            {
                "index": int(index),
                "bar": int(bar_index),
                "beat": round(float(beat_in_bar), 3),
                "start_seconds": round(start, 4),
                "duration_seconds": round(float(note.end) - start, 4),
                "pitch": int(note.pitch),
                "pitch_name": pitch_name(int(note.pitch)),
                "interval_from_previous": interval,
                "active_chord": chord,
                "role": pitch_role(int(note.pitch), chord),
            }
        )
        previous_pitch = int(note.pitch)
    return rows


def build_review(package: dict[str, Any], *, source_package: Path, all_candidates: bool, max_notes: int) -> dict[str, Any]:
    generation = package.get("generation", {})
    bars = int(generation.get("bars") or 8)
    bpm = float(generation.get("bpm") or 124.0)
    candidates = []
    for item in selected_rows(package, all_candidates=all_candidates):
        metrics = dict(item.get("objective_metrics") or {})
        candidates.append(
            {
                "case_label": str(item["case_label"]),
                "rank": int(item["rank"]),
                "variant_index": int(item["variant_index"]),
                "source_run_id": str(item.get("source_run_id") or ""),
                "chords": list(item["chords"]),
                "midi_path": str(item["midi_path"]),
                "context_midi_path": str(item.get("context_midi_path") or ""),
                "context_wav": str(item.get("context_audio", {}).get("wav_file", {}).get("path") or ""),
                "metrics": {
                    "score": float(item.get("score") or 0.0),
                    "gate_penalty": float(item.get("gate_penalty") or 0.0),
                    "offbeat_non_chord_ratio": float(metrics.get("offbeat_non_chord_ratio") or 0.0),
                    "offbeat_non_chord_resolution_ratio": float(
                        metrics.get("offbeat_non_chord_resolution_ratio") or 0.0
                    ),
                    "offbeat_unresolved_non_chord_ratio": float(
                        metrics.get("offbeat_unresolved_non_chord_ratio") or 0.0
                    ),
                    "dominant_altered_offbeat_ratio": float(metrics.get("dominant_altered_offbeat_ratio") or 0.0),
                    "interval_trigram_repeat_ratio": float(metrics.get("interval_trigram_repeat_ratio") or 0.0),
                    "max_bar_pitch_class_jaccard": float(metrics.get("max_bar_pitch_class_jaccard") or 0.0),
                    "unique_pitch_count": int(metrics.get("unique_pitch_count") or 0),
                },
                "first_notes": note_rows(item, bars=bars, bpm=bpm, max_notes=max_notes),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "source_package": str(source_package),
        "candidate_scope": "all_selected" if all_candidates else "listen_first",
        "candidate_count": len(candidates),
        "max_notes_per_candidate": int(max_notes),
        "quality_claimed": False,
        "model_direct_claimed": False,
        "candidates": candidates,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage B MIDI-to-Solo Bebop Language Note Review",
        "",
        f"- source package: `{report['source_package']}`",
        f"- candidate scope: `{report['candidate_scope']}`",
        f"- candidate count: `{report['candidate_count']}`",
        f"- quality claimed: `{str(bool(report['quality_claimed'])).lower()}`",
        "",
        "## Candidate Metrics",
        "",
        "| case | rank | score | gate | offbeat | resolved | unresolved | altered | interval repeat | bar sim | unique | context WAV |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in report["candidates"]:
        metrics = item["metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    item["case_label"],
                    str(item["rank"]),
                    f"{metrics['score']:.4f}",
                    f"{metrics['gate_penalty']:.4f}",
                    f"{metrics['offbeat_non_chord_ratio']:.4f}",
                    f"{metrics['offbeat_non_chord_resolution_ratio']:.4f}",
                    f"{metrics['offbeat_unresolved_non_chord_ratio']:.4f}",
                    f"{metrics['dominant_altered_offbeat_ratio']:.4f}",
                    f"{metrics['interval_trigram_repeat_ratio']:.4f}",
                    f"{metrics['max_bar_pitch_class_jaccard']:.4f}",
                    str(metrics["unique_pitch_count"]),
                    item["context_wav"],
                ]
            )
            + " |"
        )
    lines.extend(["", "## First Notes", ""])
    for item in report["candidates"]:
        lines.extend(
            [
                f"### {item['case_label']} rank {item['rank']}",
                "",
                "| # | bar | beat | pitch | interval | chord | role | duration |",
                "|---:|---:|---:|---|---:|---|---|---:|",
            ]
        )
        for note in item["first_notes"]:
            interval = "" if note["interval_from_previous"] is None else str(note["interval_from_previous"])
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(note["index"]),
                        str(note["bar"]),
                        f"{note['beat']:.3f}",
                        str(note["pitch_name"]),
                        interval,
                        str(note["active_chord"]),
                        str(note["role"]),
                        f"{note['duration_seconds']:.4f}",
                    ]
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build bebop-language note-level review tables")
    parser.add_argument("--package", required=True)
    parser.add_argument(
        "--output_root",
        default=str(ROOT_DIR / "outputs" / "stage_b_midi_to_solo_bebop_language_note_review"),
    )
    parser.add_argument("--run_id", default="")
    parser.add_argument("--max_notes", type=int, default=32)
    parser.add_argument("--all_candidates", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_package = Path(args.package)
    package = json.loads(source_package.read_text(encoding="utf-8"))
    run_id = str(args.run_id or datetime.now(timezone.utc).strftime("manual_%Y_%m_%d_bebop_note_review_%H%M%S"))
    output_dir = Path(args.output_root) / run_id
    report = build_review(
        package,
        source_package=source_package,
        all_candidates=bool(args.all_candidates),
        max_notes=int(args.max_notes),
    )
    write_json(output_dir / "bebop_language_note_review.json", report)
    write_text(output_dir / "bebop_language_note_review.md", markdown_report(report))
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "candidate_count": report["candidate_count"],
                "candidate_scope": report["candidate_scope"],
                "quality_claimed": report["quality_claimed"],
                "report_path": str(output_dir / "bebop_language_note_review.json"),
                "markdown_path": str(output_dir / "bebop_language_note_review.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
