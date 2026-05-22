"""Review generated MIDI candidates with objective note-level diagnostics."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mido

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_listening_review_notes import write_json


SCHEMA_VERSION = "stage_b_objective_midi_note_review_v1"
NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
OBJECTIVE_FLAG_PENALTIES = {
    "overlap_polyphonic": 40,
    "off_sixteenth_grid": 35,
    "duration_pattern_collapse": 20,
    "chromatic_walk": 18,
    "too_stepwise_or_scalar": 16,
    "unresolved_large_leaps": 16,
    "repeated_pitch": 14,
    "too_safe_chord_tones": 12,
}
OBJECTIVE_SEVERE_FLAGS = {"overlap_polyphonic", "off_sixteenth_grid"}
PITCH_CLASSES = {
    "C": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
}


class ObjectiveMidiReviewError(ValueError):
    pass


def pitch_name(pitch: int) -> str:
    return f"{NOTE_NAMES[pitch % 12]}{pitch // 12 - 1}"


def candidate_id(candidate: dict[str, Any]) -> str:
    mode = str(candidate.get("mode") or "candidate")
    rank = int(candidate.get("review_rank", 0) or 0)
    sample_index = int(candidate.get("sample_index", 0) or 0)
    if rank and sample_index:
        return f"{mode}_rank_{rank}_sample_{sample_index}"
    return str(candidate.get("sample_id") or Path(str(candidate.get("review_midi_path", "candidate"))).stem)


def chord_root(chord: str) -> int:
    token = chord[:2] if len(chord) > 1 and chord[1] in "#b" else chord[:1]
    if token not in PITCH_CLASSES:
        raise ObjectiveMidiReviewError(f"unsupported chord root: {chord}")
    return PITCH_CLASSES[token]


def chord_pitch_classes(chord: str) -> set[int]:
    root = chord_root(chord)
    lowered = chord.lower()
    if "m7b5" in lowered or "m7♭5" in lowered or "ø" in chord or "half-diminished" in lowered:
        intervals = (0, 3, 6, 10)
    elif "maj7" in chord:
        intervals = (0, 4, 7, 11)
    elif "m7" in chord:
        intervals = (0, 3, 7, 10)
    elif "7" in chord:
        intervals = (0, 4, 7, 10)
    else:
        intervals = (0, 4, 7)
    return {(root + interval) % 12 for interval in intervals}


def broad_tension_pitch_classes(chord: str) -> set[int]:
    root = chord_root(chord)
    return {(root + interval) % 12 for interval in (1, 2, 3, 5, 6, 8, 9)}


def read_midi_notes(path: Path) -> tuple[int, list[dict[str, int]]]:
    midi = mido.MidiFile(path)
    notes: list[dict[str, int]] = []
    for track in midi.tracks:
        tick = 0
        active: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for message in track:
            tick += message.time
            if message.type == "note_on" and message.velocity > 0:
                active.setdefault((message.channel, message.note), []).append((tick, message.velocity))
            elif message.type in {"note_off", "note_on"}:
                key = (getattr(message, "channel", 0), getattr(message, "note", -1))
                starts = active.get(key)
                if starts:
                    start, velocity = starts.pop(0)
                    if tick > start:
                        notes.append({"start": start, "end": tick, "pitch": key[1], "velocity": velocity})
    notes.sort(key=lambda note: (note["start"], note["pitch"], note["end"]))
    return midi.ticks_per_beat, notes


def active_polyphony_summary(notes: list[dict[str, int]]) -> dict[str, float | int]:
    events: list[tuple[int, int]] = []
    for note in notes:
        events.append((note["start"], 1))
        events.append((note["end"], -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    max_active = 0
    poly_ticks = 0
    last_tick: int | None = None
    for tick, delta in events:
        if last_tick is not None and tick > last_tick and active > 1:
            poly_ticks += tick - last_tick
        active += delta
        max_active = max(max_active, active)
        last_tick = tick
    total_ticks = max((note["end"] for note in notes), default=1)
    return {
        "max_active_notes": int(max_active),
        "polyphonic_tick_ratio": float(poly_ticks / total_ticks) if total_ticks else 0.0,
    }


def ratio(part: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return float(part / total)


def interval_summary(notes: list[dict[str, int]]) -> dict[str, float | int]:
    intervals = [notes[index + 1]["pitch"] - notes[index]["pitch"] for index in range(len(notes) - 1)]
    abs_intervals = [abs(interval) for interval in intervals]
    directions = [1 if interval > 0 else -1 if interval < 0 else 0 for interval in intervals]
    direction_changes = sum(
        1
        for left, right in zip(directions, directions[1:])
        if left != 0 and right != 0 and left != right
    )
    large_leap_indexes = [index for index, interval in enumerate(abs_intervals) if interval >= 7]
    resolved_large_leaps = 0
    for index in large_leap_indexes:
        if index + 1 >= len(intervals):
            continue
        current_direction = directions[index]
        next_direction = directions[index + 1]
        next_size = abs_intervals[index + 1]
        if current_direction != 0 and next_direction == -current_direction and 1 <= next_size <= 5:
            resolved_large_leaps += 1
    large_leap_count = len(large_leap_indexes)
    unresolved_large_leaps = large_leap_count - resolved_large_leaps
    total = len(abs_intervals)
    return {
        "stepwise_interval_ratio": ratio(sum(1 for interval in abs_intervals if interval in {1, 2}), total),
        "chromatic_interval_ratio": ratio(sum(1 for interval in abs_intervals if interval == 1), total),
        "repeated_pitch_interval_ratio": ratio(sum(1 for interval in abs_intervals if interval == 0), total),
        "large_leap_interval_ratio": ratio(sum(1 for interval in abs_intervals if interval >= 7), total),
        "large_leap_count": int(large_leap_count),
        "resolved_large_leap_count": int(resolved_large_leaps),
        "unresolved_large_leap_count": int(unresolved_large_leaps),
        "unresolved_large_leap_ratio": ratio(unresolved_large_leaps, large_leap_count),
        "direction_change_ratio": ratio(direction_changes, max(0, len(directions) - 1)),
    }


def duration_summary(notes: list[dict[str, int]], ticks_per_beat: int) -> dict[str, Any]:
    durations = [note["end"] - note["start"] for note in notes]
    counts = Counter(durations)
    most_common = counts.most_common(5)
    most_common_count = most_common[0][1] if most_common else 0
    return {
        "unique_duration_count": int(len(counts)),
        "most_common_duration_ratio": ratio(most_common_count, len(durations)),
        "most_common_durations_beats": [
            {"duration_beats": duration / ticks_per_beat, "count": count}
            for duration, count in most_common
        ],
    }


def grid_summary(notes: list[dict[str, int]], ticks_per_beat: int) -> dict[str, Any]:
    sixteenth_ticks = max(1, ticks_per_beat // 4)
    off_grid = 0
    positions = set()
    for note in notes:
        distance = min(note["start"] % sixteenth_ticks, sixteenth_ticks - (note["start"] % sixteenth_ticks))
        if distance > 1:
            off_grid += 1
        positions.add(round((note["start"] % (ticks_per_beat * 4)) / ticks_per_beat, 3))
    return {
        "sixteenth_grid_ticks": int(sixteenth_ticks),
        "off_sixteenth_grid_count": int(off_grid),
        "off_sixteenth_grid_ratio": ratio(off_grid, len(notes)),
        "unique_bar_position_count": int(len(positions)),
        "bar_positions_beats": sorted(positions),
    }


def chord_role_summary(
    notes: list[dict[str, int]],
    ticks_per_beat: int,
    chord_progression: list[str],
) -> dict[str, float | int]:
    if not chord_progression:
        return {
            "chord_tone_ratio": 0.0,
            "tension_ratio": 0.0,
            "outside_ratio": 0.0,
            "root_tone_ratio": 0.0,
        }
    chord_tones = 0
    tensions = 0
    outside = 0
    roots = 0
    bar_ticks = ticks_per_beat * 4
    for note in notes:
        chord = chord_progression[(note["start"] // bar_ticks) % len(chord_progression)]
        pitch_class = note["pitch"] % 12
        root = chord_root(chord)
        if pitch_class == root:
            roots += 1
        if pitch_class in chord_pitch_classes(chord):
            chord_tones += 1
        elif pitch_class in broad_tension_pitch_classes(chord):
            tensions += 1
        else:
            outside += 1
    total = len(notes)
    return {
        "chord_tone_ratio": ratio(chord_tones, total),
        "tension_ratio": ratio(tensions, total),
        "outside_ratio": ratio(outside, total),
        "root_tone_ratio": ratio(roots, total),
    }


def objective_flags(metrics: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if metrics["max_active_notes"] > 1 or metrics["polyphonic_tick_ratio"] > 0.05:
        flags.append("overlap_polyphonic")
    if metrics["off_sixteenth_grid_count"] > 0:
        flags.append("off_sixteenth_grid")
    if metrics["stepwise_interval_ratio"] >= 0.70:
        flags.append("too_stepwise_or_scalar")
    if metrics["chromatic_interval_ratio"] >= 0.35:
        flags.append("chromatic_walk")
    if metrics["large_leap_count"] >= 3 and metrics["unresolved_large_leap_ratio"] >= 0.45:
        flags.append("unresolved_large_leaps")
    if metrics["chord_tone_ratio"] >= 0.80 and metrics["tension_ratio"] <= 0.10:
        flags.append("too_safe_chord_tones")
    if metrics["most_common_duration_ratio"] >= 0.75:
        flags.append("duration_pattern_collapse")
    if metrics["repeated_pitch_interval_ratio"] >= 0.20:
        flags.append("repeated_pitch")
    return flags


def objective_penalty(flags: list[str]) -> int:
    return int(sum(OBJECTIVE_FLAG_PENALTIES.get(flag, 10) for flag in flags))


def objective_reviewable(metrics: dict[str, Any], flags: list[str]) -> bool:
    if any(flag in OBJECTIVE_SEVERE_FLAGS for flag in flags):
        return False
    if int(metrics.get("note_count", 0) or 0) < 8:
        return False
    if int(metrics.get("unique_pitch_count", 0) or 0) < 4:
        return False
    return True


def objective_bucket(metrics: dict[str, Any], flags: list[str]) -> str:
    if not objective_reviewable(metrics, flags):
        return "problem"
    if flags:
        return "warning"
    return "clean"


def objective_priority_score(metrics: dict[str, Any], flags: list[str]) -> int:
    score = 100 - objective_penalty(flags)
    if int(metrics.get("note_count", 0) or 0) >= 24:
        score += 3
    if int(metrics.get("unique_pitch_count", 0) or 0) >= 8:
        score += 3
    return int(max(0, min(100, score)))


def analyze_candidate(candidate: dict[str, Any], chord_progression: list[str]) -> dict[str, Any]:
    path = Path(str(candidate.get("review_midi_path") or candidate.get("midi_path") or ""))
    if not path.exists():
        raise ObjectiveMidiReviewError(f"candidate MIDI does not exist: {path}")
    ticks_per_beat, notes = read_midi_notes(path)
    pitch_values = [note["pitch"] for note in notes]
    metrics: dict[str, Any] = {
        "note_count": int(len(notes)),
        "unique_pitch_count": int(len(set(pitch_values))),
        "pitch_min": int(min(pitch_values)) if pitch_values else None,
        "pitch_max": int(max(pitch_values)) if pitch_values else None,
        "ticks_per_beat": int(ticks_per_beat),
    }
    metrics.update(active_polyphony_summary(notes))
    metrics.update(interval_summary(notes))
    metrics.update(duration_summary(notes, ticks_per_beat))
    metrics.update(grid_summary(notes, ticks_per_beat))
    metrics.update(chord_role_summary(notes, ticks_per_beat, chord_progression))
    first_notes = [
        {
            "start_beats": note["start"] / ticks_per_beat,
            "duration_beats": (note["end"] - note["start"]) / ticks_per_beat,
            "pitch": int(note["pitch"]),
            "pitch_name": pitch_name(note["pitch"]),
        }
        for note in notes[:16]
    ]
    flags = objective_flags(metrics)
    penalty = objective_penalty(flags)
    reviewable = objective_reviewable(metrics, flags)
    bucket = objective_bucket(metrics, flags)
    priority_score = objective_priority_score(metrics, flags)
    return {
        "candidate_id": candidate_id(candidate),
        "mode": str(candidate.get("mode", "")),
        "review_rank": int(candidate.get("review_rank", 0) or 0),
        "sample_index": int(candidate.get("sample_index", 0) or 0),
        "review_midi_path": str(path),
        "context_midi_path": str(candidate.get("context_midi_path") or ""),
        "metrics": metrics,
        "objective_flags": flags,
        "objective_penalty": penalty,
        "objective_priority_score": priority_score,
        "objective_reviewable": reviewable,
        "objective_bucket": bucket,
        "first_16_notes": first_notes,
    }


def analyze_review_manifest(review_manifest: dict[str, Any]) -> dict[str, Any]:
    candidates = review_manifest.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ObjectiveMidiReviewError("review_manifest must contain non-empty candidates")
    chord_progression = list(review_manifest.get("chord_progression") or [])
    analyzed = [analyze_candidate(candidate, chord_progression) for candidate in candidates]
    flag_counts = Counter(flag for candidate in analyzed for flag in candidate["objective_flags"])
    bucket_counts = Counter(candidate["objective_bucket"] for candidate in analyzed)
    mode_counts: dict[str, Counter[str]] = {}
    for candidate in analyzed:
        counter = mode_counts.setdefault(candidate["mode"], Counter())
        for flag in candidate["objective_flags"]:
            counter[flag] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "candidate_count": int(len(analyzed)),
        "chord_progression": chord_progression,
        "flag_counts": dict(sorted(flag_counts.items())),
        "objective_bucket_counts": dict(sorted(bucket_counts.items())),
        "objective_reviewable_count": int(sum(1 for candidate in analyzed if candidate["objective_reviewable"])),
        "mode_flag_counts": {mode: dict(sorted(counts.items())) for mode, counts in sorted(mode_counts.items())},
        "candidates": analyzed,
    }


def markdown_report(report: dict[str, Any], output_path: Path) -> str:
    lines = [
        "# Stage B Objective MIDI Note Review",
        "",
        f"- output: `{output_path}`",
        f"- candidate count: `{report['candidate_count']}`",
        f"- objective reviewable: `{report['objective_reviewable_count']}`",
        f"- chord progression: `{' '.join(report['chord_progression'])}`",
        "",
        "## Flag Counts",
        "",
        "| flag | count |",
        "|---|---:|",
    ]
    if report["flag_counts"]:
        for flag, count in report["flag_counts"].items():
            lines.append(f"| {flag} | {count} |")
    else:
        lines.append("| none | 0 |")

    lines.extend(["", "## Objective Buckets", "", "| bucket | count |", "|---|---:|"])
    for bucket, count in report["objective_bucket_counts"].items():
        lines.append(f"| {bucket} | {count} |")

    lines.extend(
        [
            "",
            "## Candidates",
            "",
            "| candidate | bucket | reviewable | priority | penalty | notes | unique | max active | poly ratio | stepwise | chromatic | large leap | unresolved leap | chord tone | tension | outside | flags |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for candidate in report["candidates"]:
        metrics = candidate["metrics"]
        flags = ", ".join(candidate["objective_flags"]) or "ok_objective"
        lines.append(
            "| "
            + " | ".join(
                [
                    candidate["candidate_id"],
                    candidate["objective_bucket"],
                    str(candidate["objective_reviewable"]).lower(),
                    str(candidate["objective_priority_score"]),
                    str(candidate["objective_penalty"]),
                    str(metrics["note_count"]),
                    str(metrics["unique_pitch_count"]),
                    str(metrics["max_active_notes"]),
                    f"{metrics['polyphonic_tick_ratio']:.3f}",
                    f"{metrics['stepwise_interval_ratio']:.3f}",
                    f"{metrics['chromatic_interval_ratio']:.3f}",
                    f"{metrics['large_leap_interval_ratio']:.3f}",
                    f"{metrics['unresolved_large_leap_ratio']:.3f}",
                    f"{metrics['chord_tone_ratio']:.3f}",
                    f"{metrics['tension_ratio']:.3f}",
                    f"{metrics['outside_ratio']:.3f}",
                    flags,
                ]
            )
            + " |"
        )
    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run objective note-level review on Stage B MIDI candidates")
    parser.add_argument("--review_manifest", type=str, required=True)
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(ROOT_DIR / "outputs" / "stage_b_objective_midi_review"),
    )
    parser.add_argument("--run_id", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_root) / run_id
    review_manifest = json.loads(Path(args.review_manifest).read_text(encoding="utf-8"))
    report = analyze_review_manifest(review_manifest)
    report_path = run_dir / "objective_midi_note_review.json"
    write_json(report_path, report)
    markdown_path = run_dir / "objective_midi_note_review.md"
    markdown_path.write_text(markdown_report(report, report_path), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate_count": report["candidate_count"],
                "flag_counts": report["flag_counts"],
                "report_path": str(report_path),
                "markdown_path": str(markdown_path),
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
