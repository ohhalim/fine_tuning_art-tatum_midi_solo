"""Build phrase diagnostics for objective-clean Stage B context review candidates."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_listening_review_notes import write_json  # noqa: E402


SCHEMA_VERSION = "stage_b_clean_context_diagnostics_v1"


class CleanContextDiagnosticsError(ValueError):
    pass


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def non_drum_notes(midi: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in midi.instruments:
        if instrument.is_drum:
            continue
        notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (note.start, note.pitch, note.end))


def infer_bpm(midi: pretty_midi.PrettyMIDI, fallback_bpm: float = 120.0) -> float:
    _, tempi = midi.get_tempo_changes()
    if len(tempi) == 0:
        return float(fallback_bpm)
    return float(tempi[0])


def seconds_to_beats(seconds: float, bpm: float) -> float:
    return float(seconds * bpm / 60.0)


def nearest_grid_distance(beats: float, grid_beats: float = 0.25) -> float:
    remainder = beats % grid_beats
    return float(min(remainder, grid_beats - remainder))


def ratio(part: int | float, total: int | float) -> float:
    if total <= 0:
        return 0.0
    return float(part / total)


def pitch_name(pitch: int) -> str:
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[pitch % 12]}{pitch // 12 - 1}"


def run_lengths(values: list[int]) -> list[int]:
    if not values:
        return []
    lengths: list[int] = []
    current = values[0]
    count = 1
    for value in values[1:]:
        if value == current:
            count += 1
        else:
            lengths.append(count)
            current = value
            count = 1
    lengths.append(count)
    return lengths


def analyze_solo_path(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise CleanContextDiagnosticsError(f"solo MIDI does not exist: {path}")
    midi = pretty_midi.PrettyMIDI(str(path))
    bpm = infer_bpm(midi)
    notes = non_drum_notes(midi)
    if not notes:
        return {
            "bpm": bpm,
            "note_count": 0,
            "unique_pitch_count": 0,
            "bar_count": 0,
            "bar_coverage_ratio": 0.0,
            "diagnostic_flags": ["empty_solo_midi"],
        }

    starts_beats = [seconds_to_beats(note.start, bpm) for note in notes]
    ends_beats = [seconds_to_beats(note.end, bpm) for note in notes]
    durations_beats = [max(0.0, end - start) for start, end in zip(starts_beats, ends_beats)]
    pitches = [int(note.pitch) for note in notes]
    bar_indexes = [int(start // 4.0) for start in starts_beats]
    bar_count = max(1, int(math.ceil(max(ends_beats) / 4.0)))
    bar_note_counts = Counter(bar_indexes)
    off_grid_count = sum(1 for start in starts_beats if nearest_grid_distance(start) > 0.035)
    pitch_counts = Counter(pitches)
    most_common_pitch_count = pitch_counts.most_common(1)[0][1]
    intervals = [pitches[index + 1] - pitches[index] for index in range(len(pitches) - 1)]
    repeated_interval_count = sum(1 for interval in intervals if interval == 0)
    same_pitch_runs = run_lengths(pitches)
    first_notes = [
        {
            "start_beats": round(starts_beats[index], 3),
            "duration_beats": round(durations_beats[index], 3),
            "pitch": pitches[index],
            "pitch_name": pitch_name(pitches[index]),
        }
        for index in range(min(16, len(notes)))
    ]
    metrics = {
        "bpm": bpm,
        "note_count": int(len(notes)),
        "unique_pitch_count": int(len(set(pitches))),
        "pitch_min": int(min(pitches)),
        "pitch_max": int(max(pitches)),
        "pitch_range_semitones": int(max(pitches) - min(pitches)),
        "bar_count": int(bar_count),
        "covered_bar_count": int(len(bar_note_counts)),
        "bar_coverage_ratio": ratio(len(bar_note_counts), bar_count),
        "bar_note_counts": {str(key + 1): int(value) for key, value in sorted(bar_note_counts.items())},
        "off_sixteenth_grid_count": int(off_grid_count),
        "off_sixteenth_grid_ratio": ratio(off_grid_count, len(notes)),
        "max_duration_beats": float(max(durations_beats)),
        "max_duration_bar_ratio": ratio(max(durations_beats), 4.0),
        "long_note_ratio": ratio(sum(1 for duration in durations_beats if duration >= 1.0), len(notes)),
        "most_common_pitch_ratio": ratio(most_common_pitch_count, len(notes)),
        "repeated_pitch_interval_ratio": ratio(repeated_interval_count, len(intervals)),
        "longest_same_pitch_run": int(max(same_pitch_runs)),
        "first_16_notes": first_notes,
    }
    flags = diagnostic_flags(metrics)
    metrics["diagnostic_flags"] = flags
    metrics["decision_hint"] = decision_hint(flags)
    return metrics


def context_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"context_exists": False, "instrument_count": 0, "instruments": []}
    midi = pretty_midi.PrettyMIDI(str(path))
    instruments = [
        {
            "name": instrument.name,
            "program": int(instrument.program),
            "is_drum": bool(instrument.is_drum),
            "note_count": int(len(instrument.notes)),
        }
        for instrument in midi.instruments
    ]
    names = [item["name"].lower() for item in instruments]
    return {
        "context_exists": True,
        "instrument_count": int(len(instruments)),
        "instruments": instruments,
        "has_chord_guide": any("chord" in name for name in names),
        "has_bass_guide": any("bass" in name for name in names),
        "has_solo_track": any("solo" in name for name in names),
    }


def diagnostic_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if int(metrics.get("note_count", 0) or 0) < 32:
        flags.append("too_sparse_for_phrase_review")
    if int(metrics.get("unique_pitch_count", 0) or 0) < 12:
        flags.append("low_pitch_variety")
    if float(metrics.get("bar_coverage_ratio", 0.0) or 0.0) < 0.75:
        flags.append("low_bar_coverage")
    if float(metrics.get("off_sixteenth_grid_ratio", 0.0) or 0.0) > 0.10:
        flags.append("timing_needs_review")
    if float(metrics.get("max_duration_bar_ratio", 0.0) or 0.0) > 0.50:
        flags.append("long_sustain_risk")
    if float(metrics.get("most_common_pitch_ratio", 0.0) or 0.0) > 0.20:
        flags.append("dominant_pitch_reuse")
    if int(metrics.get("longest_same_pitch_run", 0) or 0) >= 3:
        flags.append("same_pitch_run")
    return flags


def decision_hint(flags: list[str]) -> str:
    if not flags:
        return "listen_with_context"
    if "timing_needs_review" in flags:
        return "review_timing_before_generation_changes"
    if "dominant_pitch_reuse" in flags or "low_pitch_variety" in flags:
        return "review_phrase_vocabulary_and_pitch_reuse"
    if "low_bar_coverage" in flags or "too_sparse_for_phrase_review" in flags:
        return "increase_phrase_coverage_before_quality_claim"
    return "listen_with_context_and_record_issue_flags"


def analyze_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    solo_path = Path(str(candidate.get("review_midi_path") or ""))
    context_path = Path(str(candidate.get("context_midi_path") or ""))
    solo_metrics = analyze_solo_path(solo_path)
    context = context_summary(context_path)
    package_metrics = dict(candidate.get("metrics") or {})
    return {
        "candidate_id": str(candidate.get("candidate_id") or solo_path.stem),
        "mode": str(candidate.get("mode") or ""),
        "review_rank": int(candidate.get("review_rank", 0) or 0),
        "sample_index": int(candidate.get("sample_index", 0) or 0),
        "review_midi_path": str(solo_path),
        "context_midi_path": str(context_path),
        "package_metrics": package_metrics,
        "solo_metrics": solo_metrics,
        "context_summary": context,
        "review_checklist": {
            "timing": "pending",
            "chord_fit": "pending",
            "phrase_continuation": "pending",
            "landing": "pending",
            "jazz_vocabulary": "pending",
            "notes": "",
        },
    }


def build_clean_context_diagnostics(clean_package: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    candidates = clean_package.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise CleanContextDiagnosticsError("clean package must contain non-empty candidates")
    analyzed = [analyze_candidate(candidate) for candidate in candidates]
    flag_counts = Counter(flag for item in analyzed for flag in item["solo_metrics"].get("diagnostic_flags", []))
    decision_counts = Counter(item["solo_metrics"].get("decision_hint", "") for item in analyzed)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_clean_package": str(clean_package.get("output_dir") or ""),
        "candidate_count": int(len(analyzed)),
        "diagnostic_flag_counts": dict(sorted(flag_counts.items())),
        "decision_hint_counts": dict(sorted(decision_counts.items())),
        "candidates": analyzed,
    }


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Stage B Clean Context Diagnostics",
        "",
        f"- candidate count: `{report['candidate_count']}`",
        f"- diagnostic flags: `{report['diagnostic_flag_counts']}`",
        f"- decision hints: `{report['decision_hint_counts']}`",
        "",
        "| candidate | notes | unique | bars | grid off | max dur | pitch reuse | flags | hint | context |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for candidate in report["candidates"]:
        solo = candidate["solo_metrics"]
        context = candidate["context_summary"]
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate["candidate_id"],
                    str(solo.get("note_count", 0)),
                    str(solo.get("unique_pitch_count", 0)),
                    f"{solo.get('covered_bar_count', 0)}/{solo.get('bar_count', 0)}",
                    f"{float(solo.get('off_sixteenth_grid_ratio', 0.0)):.3f}",
                    f"{float(solo.get('max_duration_beats', 0.0)):.3f}",
                    f"{float(solo.get('most_common_pitch_ratio', 0.0)):.3f}",
                    ", ".join(solo.get("diagnostic_flags", [])) or "none",
                    str(solo.get("decision_hint", "")),
                    "yes" if context.get("context_exists") else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Review Checklist",
            "",
            "For each candidate, listen to the context MIDI and fill:",
            "",
            "- timing: `good`, `too_loose`, `too_straight`, `unclear`",
            "- chord_fit: `fits`, `too_safe`, `too_outside`, `unclear`",
            "- phrase_continuation: `phrase_like`, `fragmented`, `exercise_like`, `unclear`",
            "- landing: `clear`, `weak`, `missing`, `unclear`",
            "- jazz_vocabulary: `present`, `weak`, `absent`, `unclear`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build clean context phrase diagnostics")
    parser.add_argument("--clean_package", type=str, required=True)
    parser.add_argument("--output_root", type=str, default=str(ROOT_DIR / "outputs" / "stage_b_clean_context_diagnostics"))
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--min_candidates", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_clean_context_diagnostics(read_json(Path(args.clean_package)), output_dir=output_dir)
    write_json(output_dir / "clean_context_diagnostics.json", report)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "clean_context_diagnostics.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "diagnostic_path": str(output_dir / "clean_context_diagnostics.json"),
                "markdown_path": str(output_dir / "clean_context_diagnostics.md"),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0 if int(report["candidate_count"]) >= int(args.min_candidates) else 3


if __name__ == "__main__":
    raise SystemExit(main())
