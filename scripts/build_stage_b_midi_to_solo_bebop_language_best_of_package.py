"""Build a best-of bebop-language MIDI/WAV package from prior packages."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.build_stage_b_midi_to_solo_bebop_language_package import (  # noqa: E402
    DEFAULT_OUTPUT_ROOT,
    BebopLanguagePackageError,
    RenderConfig,
    add_context,
    build_listen_first_package,
    candidate_gate_penalty,
    candidate_score,
    chord_for_time,
    chord_pitch_classes,
    markdown_report,
    objective_metrics,
    parse_chord,
    pitches_for_pcs,
    read_json,
    render_wav,
    scale_intervals,
    sha256_file,
    solo_notes,
    validate_report,
    write_json,
    write_text,
)
from scripts.render_stage_b_midi_to_solo_candidate_audio import resolve_soundfont  # noqa: E402


DEFAULT_PACKAGE_GLOBS = ("manual_2026_06_13_bebop_language_*/bebop_language_package.json",)


def parse_globs(raw: str) -> list[str]:
    return [item.strip() for item in str(raw or "").split(",") if item.strip()] or list(DEFAULT_PACKAGE_GLOBS)


def package_paths(source_root: Path, patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(source_root.glob(pattern))
    return sorted({path.resolve() for path in paths if path.exists()})


def candidate_rows(
    *,
    paths: list[Path],
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for package_path in paths:
        report = read_json(package_path)
        generation = report.get("generation", {})
        bars = int(generation.get("bars") or 8)
        bpm = float(generation.get("bpm") or 124.0)
        for item in report.get("selected_candidates", []):
            midi_path = Path(str(item.get("midi_path") or item.get("raw_midi_path") or ""))
            if not midi_path.exists():
                continue
            midi_sha = sha256_file(midi_path)
            if midi_sha in seen_hashes:
                continue
            seen_hashes.add(midi_sha)
            chords = list(item.get("chords") or [])
            if not chords:
                continue
            pm = pretty_midi.PrettyMIDI(str(midi_path))
            metrics = objective_metrics(pm, chords, bars=bars, bpm=bpm)
            score = candidate_score(
                metrics,
                target_chord_tone_ratio=target_chord_tone_ratio,
                target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            )
            gate_penalty = candidate_gate_penalty(metrics)
            rows.append(
                {
                    "source_run_id": package_path.parent.name,
                    "source_package": str(package_path),
                    "source_midi_path": str(midi_path),
                    "source_midi_sha256": midi_sha,
                    "case_label": str(item.get("case_label") or "unknown_case"),
                    "chords": chords,
                    "variant_index": int(item.get("variant_index") or 0),
                    "seed": int(item.get("seed") or 0),
                    "generation_meta": dict(item.get("generation_meta") or {}),
                    "objective_metrics": metrics,
                    "score": float(score),
                    "gate_penalty": float(gate_penalty),
                }
            )
    return sorted(
        rows,
        key=lambda row: (
            float(row["score"]),
            float(row["gate_penalty"]),
            str(row["case_label"]),
            str(row["source_run_id"]),
            int(row["variant_index"]),
        ),
    )


def filter_candidate_rows(
    rows: list[dict[str, Any]],
    *,
    max_gate_penalty: float | None,
    max_offbeat_non_chord_ratio: float | None,
    max_unresolved_offbeat_non_chord_ratio: float | None,
    max_dominant_altered_offbeat_ratio: float | None,
    max_adjacent_repeat_ratio: float | None,
    max_bar_pitch_class_jaccard: float | None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for row in rows:
        metrics = row["objective_metrics"]
        if max_gate_penalty is not None and float(row["gate_penalty"]) > float(max_gate_penalty):
            continue
        if (
            max_offbeat_non_chord_ratio is not None
            and float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio)
        ):
            continue
        if (
            max_unresolved_offbeat_non_chord_ratio is not None
            and float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(max_unresolved_offbeat_non_chord_ratio)
        ):
            continue
        if (
            max_dominant_altered_offbeat_ratio is not None
            and float(metrics["dominant_altered_offbeat_ratio"]) > float(max_dominant_altered_offbeat_ratio)
        ):
            continue
        if (
            max_adjacent_repeat_ratio is not None
            and float(metrics["adjacent_repeat_ratio"]) > float(max_adjacent_repeat_ratio)
        ):
            continue
        if (
            max_bar_pitch_class_jaccard is not None
            and float(metrics["max_bar_pitch_class_jaccard"]) > float(max_bar_pitch_class_jaccard)
        ):
            continue
        filtered.append(row)
    return filtered


def select_candidates(rows: list[dict[str, Any]], *, selected_count: int, max_per_case: int) -> list[dict[str, Any]]:
    if not rows:
        raise BebopLanguagePackageError("no candidate rows for best-of package")
    selected: list[dict[str, Any]] = []
    selected_hashes: set[str] = set()
    case_counts: dict[str, int] = defaultdict(int)
    cases = sorted({str(row["case_label"]) for row in rows})
    for case_label in cases:
        case_best = next(row for row in rows if str(row["case_label"]) == case_label)
        selected.append(case_best)
        selected_hashes.add(str(case_best["source_midi_sha256"]))
        case_counts[case_label] += 1
    for row in rows:
        if len(selected) >= int(selected_count):
            break
        midi_sha = str(row["source_midi_sha256"])
        case_label = str(row["case_label"])
        if midi_sha in selected_hashes:
            continue
        if case_counts[case_label] >= int(max_per_case):
            continue
        selected.append(row)
        selected_hashes.add(midi_sha)
        case_counts[case_label] += 1
    if len(selected) < int(selected_count):
        raise BebopLanguagePackageError("not enough selected candidates after max-per-case filter")
    return selected[: int(selected_count)]


def aggregate_metrics(*, generated_count: int, rendered: list[dict[str, Any]], listen_first: dict[str, Any]) -> dict[str, Any]:
    metrics = [item["objective_metrics"] for item in rendered]
    rhythm_metrics = [item.get("rhythm_articulation_metrics") or {} for item in rendered]

    def avg(key: str) -> float:
        return mean([float(item[key]) for item in metrics]) if metrics else 0.0

    def rhythm_avg(key: str) -> float:
        values = [float(item[key]) for item in rhythm_metrics if key in item]
        return mean(values) if values else 0.0

    return {
        "generated_candidate_count": int(generated_count),
        "selected_candidate_count": len(rendered),
        "listen_first_case_count": int(listen_first["case_count"]),
        "avg_score": mean([float(item["score"]) for item in rendered]) if rendered else 0.0,
        "avg_gate_penalty": mean([float(item.get("gate_penalty") or 0.0) for item in rendered]) if rendered else 0.0,
        "max_gate_penalty": max((float(item.get("gate_penalty") or 0.0) for item in rendered), default=0.0),
        "avg_unique_pitch_count": avg("unique_pitch_count"),
        "avg_step_motion_ratio": avg("step_motion_ratio"),
        "avg_third_fourth_motion_ratio": avg("third_fourth_motion_ratio"),
        "avg_large_leap_ratio": avg("large_leap_ratio"),
        "avg_adjacent_repeat_ratio": avg("adjacent_repeat_ratio"),
        "avg_chord_tone_ratio": avg("chord_tone_ratio"),
        "avg_tension_ratio": avg("tension_ratio"),
        "avg_strong_beat_chord_tone_ratio": avg("strong_beat_chord_tone_ratio"),
        "avg_offbeat_non_chord_ratio": avg("offbeat_non_chord_ratio"),
        "avg_offbeat_non_chord_resolution_ratio": avg("offbeat_non_chord_resolution_ratio"),
        "avg_offbeat_unresolved_non_chord_ratio": avg("offbeat_unresolved_non_chord_ratio"),
        "avg_chromatic_step_ratio": avg("chromatic_step_ratio"),
        "avg_enclosure_proxy_ratio": avg("enclosure_proxy_ratio"),
        "avg_dominant_altered_offbeat_ratio": avg("dominant_altered_offbeat_ratio"),
        "avg_two_note_cycle_ratio": avg("two_note_cycle_ratio"),
        "avg_interval_trigram_repeat_ratio": avg("interval_trigram_repeat_ratio"),
        "avg_bar_half_repeat_ratio": avg("bar_half_repeat_ratio"),
        "avg_max_bar_pitch_class_jaccard": avg("max_bar_pitch_class_jaccard"),
        "avg_bar_pitch_shape_repeat_ratio": avg("bar_pitch_shape_repeat_ratio"),
        "avg_duration_template_repeat_ratio": rhythm_avg("duration_template_repeat_ratio"),
        "avg_most_common_duration_ratio": rhythm_avg("most_common_duration_ratio"),
        "avg_unique_duration_bucket_count": rhythm_avg("unique_duration_bucket_count"),
        "avg_unique_velocity_count": rhythm_avg("unique_velocity_count"),
        "max_offbeat_start_shift_seconds": max(
            (float(item.get("max_offbeat_start_shift_seconds") or 0.0) for item in rhythm_metrics),
            default=0.0,
        ),
        "min_bar_unique_pitch_count_min": min((int(item["min_bar_unique_pitch_count"]) for item in metrics), default=0),
        "all_final_landings_chord_tone": all(bool(item["final_landing_is_chord_tone"]) for item in metrics),
    }


def listen_first_consonance_score(item: dict[str, Any]) -> float:
    metrics = item["objective_metrics"]
    return (
        float(item.get("gate_penalty") or 0.0) * 3.0
        + float(metrics["offbeat_unresolved_non_chord_ratio"]) * 6.0
        + abs(float(metrics["offbeat_non_chord_ratio"]) - 0.40625) * 1.4
        + max(0.0, float(metrics["dominant_altered_offbeat_ratio"]) - 0.1875) * 2.0
        + max(0.0, 0.10 - float(metrics["dominant_altered_offbeat_ratio"])) * 0.5
        + max(0.0, float(metrics["max_bar_pitch_class_jaccard"]) - 0.72) * 1.2
        + max(0.0, 14.0 - float(metrics["unique_pitch_count"])) * 0.04
        + float(metrics["two_note_cycle_ratio"]) * 1.0
        + max(0.0, float(metrics["interval_trigram_repeat_ratio"]) - 0.02) * 1.4
        + float(item["score"]) * 0.12
    )


def bebop_language_selection_score(item: dict[str, Any]) -> float:
    metrics = item["objective_metrics"]
    return (
        float(item.get("gate_penalty") or 0.0) * 5.0
        + float(metrics["offbeat_unresolved_non_chord_ratio"]) * 10.0
        + max(0.0, 0.99 - float(metrics["offbeat_non_chord_resolution_ratio"])) * 6.0
        + max(0.0, float(metrics["large_leap_ratio"]) - 0.0625) * 3.0
        + max(0.0, int(metrics["max_abs_interval"]) - 9) * 0.08
        + max(0.0, 0.3125 - float(metrics["enclosure_proxy_ratio"])) * 2.2
        + max(0.0, float(metrics["max_bar_pitch_class_jaccard"]) - 0.625) * 1.4
        + float(metrics["adjacent_repeat_ratio"]) * 2.4
        + float(metrics["two_note_cycle_ratio"]) * 3.0
        + max(0.0, float(metrics["interval_trigram_repeat_ratio"]) - 0.0125) * 2.0
        + float(metrics["bar_half_repeat_ratio"]) * 2.5
        + abs(float(metrics["offbeat_non_chord_ratio"]) - 0.390625) * 1.0
        + abs(float(metrics["chord_tone_ratio"]) - 0.8046875) * 0.8
        + max(0.0, 0.18 - float(metrics["chromatic_step_ratio"])) * 0.6
        + max(0.0, 0.125 - float(metrics["dominant_altered_offbeat_ratio"])) * 0.35
        + max(0, 14 - int(metrics["unique_pitch_count"])) * 0.03
        + float(item["score"]) * 0.10
    )


def bebop_stepwise_chromatic_selection_score(item: dict[str, Any]) -> float:
    metrics = item["objective_metrics"]
    return (
        bebop_language_selection_score(item)
        + max(0.0, 0.40 - float(metrics["step_motion_ratio"])) * 2.5
        + max(0.0, 0.22 - float(metrics["chromatic_step_ratio"])) * 3.0
        + max(0.0, float(metrics["third_fourth_motion_ratio"]) - 0.56) * 1.0
        + max(0.0, float(metrics["large_leap_ratio"]) - 0.09) * 2.0
        + max(0.0, float(metrics["max_bar_pitch_class_jaccard"]) - 0.70) * 1.0
        + max(0.0, 0.28 - float(metrics["enclosure_proxy_ratio"])) * 0.7
    )


def selection_sort_key(row: dict[str, Any], *, selection_profile: str) -> tuple[float, float, str, str, int]:
    if selection_profile == "score":
        primary_score = float(row["score"])
    elif selection_profile == "bebop_language":
        primary_score = bebop_language_selection_score(row)
    elif selection_profile == "bebop_stepwise_chromatic":
        primary_score = bebop_stepwise_chromatic_selection_score(row)
    else:
        raise BebopLanguagePackageError(f"unknown selection profile: {selection_profile}")
    return (
        primary_score,
        float(row["gate_penalty"]),
        str(row["case_label"]),
        str(row["source_run_id"]),
        int(row["variant_index"]),
    )


def order_rendered_for_listen_first(rendered: list[dict[str, Any]], *, listen_first_mode: str) -> list[dict[str, Any]]:
    if listen_first_mode == "rank":
        return rendered
    if listen_first_mode != "consonance":
        raise BebopLanguagePackageError(f"unknown listen-first mode: {listen_first_mode}")

    best_by_case: dict[str, dict[str, Any]] = {}
    for item in rendered:
        case_label = str(item["case_label"])
        current = best_by_case.get(case_label)
        if current is None or listen_first_consonance_score(item) < listen_first_consonance_score(current):
            best_by_case[case_label] = item

    priority_hashes = {str(item["midi_sha256"]) for item in best_by_case.values()}
    priority_rows = sorted(best_by_case.values(), key=lambda item: str(item["case_label"]))
    remaining = [item for item in rendered if str(item["midi_sha256"]) not in priority_hashes]
    return priority_rows + remaining


def repair_bar_similarity(
    pm: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_iterations: int = 4,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(pm)
    current_metrics = objective_metrics(current, chords, bars=bars, bpm=bpm)
    current_score = candidate_score(
        current_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    current_gate = candidate_gate_penalty(current_metrics)
    original_max_bar_similarity = float(current_metrics["max_bar_pitch_class_jaccard"])
    repair_steps: list[dict[str, Any]] = []
    seconds_per_beat = 60.0 / float(bpm)

    for _ in range(int(max_iterations)):
        best_candidate: tuple[tuple[float, float, float], pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]] | None = None
        notes = solo_notes(current)
        for note_index, note in enumerate(notes):
            beat_position = float(note.start) / seconds_per_beat
            in_bar_beat = beat_position % 4
            if abs(in_bar_beat - round(in_bar_beat)) < 0.05:
                continue
            chord = chord_for_time(chords, float(note.start), bars=bars, bpm=bpm)
            scale_pcs = set()
            chord_pcs = chord_pitch_classes(chord)
            chord_root, _intervals = parse_chord(chord)
            root_pitch_class = chord_root % 12
            for interval in scale_intervals(chord):
                scale_pcs.add((root_pitch_class + interval) % 12)
            candidate_pitches = pitches_for_pcs(
                scale_pcs | chord_pcs,
                low=max(56, int(note.pitch) - 7),
                high=min(86, int(note.pitch) + 7),
            )
            candidate_pitches = [pitch for pitch in candidate_pitches if int(pitch) % 12 != int(note.pitch) % 12]
            if not candidate_pitches:
                continue
            for replacement in sorted(candidate_pitches, key=lambda pitch: abs(int(pitch) - int(note.pitch)))[:6]:
                trial = copy.deepcopy(current)
                trial_notes = solo_notes(trial)
                trial_notes[note_index].pitch = int(replacement)
                metrics = objective_metrics(trial, chords, bars=bars, bpm=bpm)
                gate = candidate_gate_penalty(metrics)
                if gate > current_gate:
                    continue
                if float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(current_metrics["offbeat_unresolved_non_chord_ratio"]):
                    continue
                if float(metrics["offbeat_non_chord_ratio"]) > max(0.4375, float(current_metrics["offbeat_non_chord_ratio"])):
                    continue
                if float(metrics["dominant_altered_offbeat_ratio"]) > 0.25:
                    continue
                score = candidate_score(
                    metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                improvement_key = (
                    float(metrics["max_bar_pitch_class_jaccard"]),
                    float(metrics.get("interval_trigram_repeat_ratio") or 0.0),
                    float(score),
                )
                current_key = (
                    float(current_metrics["max_bar_pitch_class_jaccard"]),
                    float(current_metrics.get("interval_trigram_repeat_ratio") or 0.0),
                    float(current_score),
                )
                if improvement_key >= current_key:
                    continue
                step = {
                    "note_index": int(note_index + 1),
                    "old_pitch": int(note.pitch),
                    "new_pitch": int(replacement),
                    "old_max_bar_pitch_class_jaccard": float(current_metrics["max_bar_pitch_class_jaccard"]),
                    "new_max_bar_pitch_class_jaccard": float(metrics["max_bar_pitch_class_jaccard"]),
                    "old_score": float(current_score),
                    "new_score": float(score),
                }
                if best_candidate is None or improvement_key < best_candidate[0]:
                    best_candidate = (improvement_key, trial, metrics, step)
        if best_candidate is None:
            break
        _key, current, current_metrics, step = best_candidate
        current_score = float(step["new_score"])
        current_gate = candidate_gate_penalty(current_metrics)
        repair_steps.append(step)

    repair_report = {
        "attempted": True,
        "changed": bool(repair_steps),
        "step_count": len(repair_steps),
        "original_max_bar_pitch_class_jaccard": original_max_bar_similarity,
        "final_max_bar_pitch_class_jaccard": float(current_metrics["max_bar_pitch_class_jaccard"]),
        "steps": repair_steps,
    }
    return current, current_metrics, repair_report


def is_strong_beat_note(note: pretty_midi.Note, *, bpm: float) -> bool:
    seconds_per_beat = 60.0 / float(bpm)
    beat_position = float(note.start) / seconds_per_beat
    in_bar_beat = beat_position % 4
    return abs(in_bar_beat - round(in_bar_beat)) < 0.05


def notes_bracket_target(notes: list[pretty_midi.Note], target_index: int) -> bool:
    if target_index < 2:
        return False
    left = int(notes[target_index - 2].pitch)
    right = int(notes[target_index - 1].pitch)
    target = int(notes[target_index].pitch)
    return (left - target) * (right - target) < 0 and abs(left - target) <= 4 and abs(right - target) <= 4


def enclosure_replacement_pitches(chord: str, *, left_pitch: int, target_pitch: int) -> list[int]:
    left_distance = int(left_pitch) - int(target_pitch)
    if left_distance == 0 or abs(left_distance) > 4:
        return []
    replacement_side = -1 if left_distance > 0 else 1
    chord_root, _intervals = parse_chord(chord)
    root_pitch_class = chord_root % 12
    scale_pcs = {(root_pitch_class + interval) % 12 for interval in scale_intervals(chord)}
    allowed_pcs = scale_pcs | chord_pitch_classes(chord)
    candidates: list[int] = []
    for offset in (replacement_side, replacement_side * 2, replacement_side * 3, replacement_side * 4):
        pitch = int(target_pitch) + int(offset)
        if 56 <= pitch <= 86 and pitch % 12 in allowed_pcs:
            candidates.append(pitch)
    return candidates


def repair_enclosure_density(
    pm: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_offbeat_non_chord_ratio: float,
    max_iterations: int = 8,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(pm)
    current_metrics = objective_metrics(current, chords, bars=bars, bpm=bpm)
    current_score = candidate_score(
        current_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    current_gate = candidate_gate_penalty(current_metrics)
    original_enclosure = float(current_metrics["enclosure_proxy_ratio"])
    repair_steps: list[dict[str, Any]] = []

    for _ in range(int(max_iterations)):
        best_candidate: tuple[tuple[float, float, float, float, float], pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]] | None = None
        notes = solo_notes(current)
        for note_index in range(2, len(notes)):
            target_note = notes[note_index]
            if not is_strong_beat_note(target_note, bpm=bpm):
                continue
            chord = chord_for_time(chords, float(target_note.start), bars=bars, bpm=bpm)
            if int(target_note.pitch) % 12 not in chord_pitch_classes(chord):
                continue
            if notes_bracket_target(notes, note_index):
                continue
            replacement_pitches = enclosure_replacement_pitches(
                chord,
                left_pitch=int(notes[note_index - 2].pitch),
                target_pitch=int(target_note.pitch),
            )
            for replacement in replacement_pitches:
                trial = copy.deepcopy(current)
                trial_notes = solo_notes(trial)
                old_pitch = int(trial_notes[note_index - 1].pitch)
                if old_pitch == int(replacement):
                    continue
                trial_notes[note_index - 1].pitch = int(replacement)
                metrics = objective_metrics(trial, chords, bars=bars, bpm=bpm)
                gate = candidate_gate_penalty(metrics)
                if gate > current_gate:
                    continue
                if float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(current_metrics["offbeat_unresolved_non_chord_ratio"]):
                    continue
                if float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio):
                    continue
                if float(metrics["dominant_altered_offbeat_ratio"]) > 0.25:
                    continue
                if float(metrics["two_note_cycle_ratio"]) > 0.0:
                    continue
                if float(metrics["interval_trigram_repeat_ratio"]) > max(
                    0.02,
                    float(current_metrics["interval_trigram_repeat_ratio"]),
                ):
                    continue
                if float(metrics["max_bar_pitch_class_jaccard"]) > max(
                    0.72,
                    float(current_metrics["max_bar_pitch_class_jaccard"]) + 0.05,
                ):
                    continue
                score = candidate_score(
                    metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                if score > current_score + 0.08:
                    continue
                improvement_key = (
                    -float(metrics["enclosure_proxy_ratio"]),
                    float(metrics["offbeat_non_chord_ratio"]),
                    float(metrics["offbeat_unresolved_non_chord_ratio"]),
                    float(metrics["interval_trigram_repeat_ratio"]),
                    float(metrics["max_bar_pitch_class_jaccard"]),
                    float(score),
                )
                current_key = (
                    -float(current_metrics["enclosure_proxy_ratio"]),
                    float(current_metrics["offbeat_non_chord_ratio"]),
                    float(current_metrics["offbeat_unresolved_non_chord_ratio"]),
                    float(current_metrics["interval_trigram_repeat_ratio"]),
                    float(current_metrics["max_bar_pitch_class_jaccard"]),
                    float(current_score),
                )
                if improvement_key >= current_key:
                    continue
                step = {
                    "note_index": int(note_index),
                    "target_note_index": int(note_index + 1),
                    "old_pitch": old_pitch,
                    "new_pitch": int(replacement),
                    "target_pitch": int(target_note.pitch),
                    "old_enclosure_proxy_ratio": float(current_metrics["enclosure_proxy_ratio"]),
                    "new_enclosure_proxy_ratio": float(metrics["enclosure_proxy_ratio"]),
                    "old_offbeat_non_chord_ratio": float(current_metrics["offbeat_non_chord_ratio"]),
                    "new_offbeat_non_chord_ratio": float(metrics["offbeat_non_chord_ratio"]),
                    "old_score": float(current_score),
                    "new_score": float(score),
                }
                if best_candidate is None or improvement_key < best_candidate[0]:
                    best_candidate = (improvement_key, trial, metrics, step)
        if best_candidate is None:
            break
        _key, current, current_metrics, step = best_candidate
        current_score = float(step["new_score"])
        current_gate = candidate_gate_penalty(current_metrics)
        repair_steps.append(step)

    repair_report = {
        "attempted": True,
        "changed": bool(repair_steps),
        "step_count": len(repair_steps),
        "original_enclosure_proxy_ratio": original_enclosure,
        "final_enclosure_proxy_ratio": float(current_metrics["enclosure_proxy_ratio"]),
        "max_offbeat_non_chord_ratio": float(max_offbeat_non_chord_ratio),
        "steps": repair_steps,
    }
    return current, current_metrics, repair_report


def offbeat_resolution_trials(
    notes: list[pretty_midi.Note],
    note_index: int,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
) -> list[tuple[int, int]]:
    note = notes[note_index]
    next_note = notes[note_index + 1] if note_index + 1 < len(notes) else None
    if next_note is None:
        return []
    chord = chord_for_time(chords, float(note.start), bars=bars, bpm=bpm)
    next_chord = chord_for_time(chords, float(next_note.start), bars=bars, bpm=bpm)
    if int(note.pitch) % 12 in chord_pitch_classes(chord):
        return []
    if int(next_note.pitch) % 12 in chord_pitch_classes(next_chord) and abs(int(next_note.pitch) - int(note.pitch)) <= 2:
        return []

    trials: list[tuple[int, int]] = []
    chord_root, _intervals = parse_chord(chord)
    root_pitch_class = chord_root % 12
    scale_pcs = {(root_pitch_class + interval) % 12 for interval in scale_intervals(chord)}
    allowed_pcs = scale_pcs | chord_pitch_classes(chord)
    for pitch in range(max(56, int(next_note.pitch) - 2), min(86, int(next_note.pitch) + 2) + 1):
        if pitch != int(note.pitch) and pitch % 12 in allowed_pcs:
            trials.append((note_index, int(pitch)))

    next_chord_pcs = chord_pitch_classes(next_chord)
    for pitch in range(max(56, int(note.pitch) - 2), min(86, int(note.pitch) + 2) + 1):
        if pitch != int(next_note.pitch) and pitch % 12 in next_chord_pcs:
            trials.append((note_index + 1, int(pitch)))
    return sorted(set(trials), key=lambda item: (item[0], abs(item[1] - int(notes[item[0]].pitch)), item[1]))


def repair_unresolved_offbeats(
    pm: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_offbeat_non_chord_ratio: float,
    max_iterations: int = 4,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(pm)
    current_metrics = objective_metrics(current, chords, bars=bars, bpm=bpm)
    current_score = candidate_score(
        current_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    current_gate = candidate_gate_penalty(current_metrics)
    original_unresolved = float(current_metrics["offbeat_unresolved_non_chord_ratio"])
    repair_steps: list[dict[str, Any]] = []

    for _ in range(int(max_iterations)):
        best_candidate: tuple[tuple[float, float, float, float, float], pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]] | None = None
        notes = solo_notes(current)
        for note_index, note in enumerate(notes[:-1]):
            if is_strong_beat_note(note, bpm=bpm):
                continue
            for target_note_index, replacement in offbeat_resolution_trials(
                notes,
                note_index,
                chords,
                bars=bars,
                bpm=bpm,
            ):
                trial = copy.deepcopy(current)
                trial_notes = solo_notes(trial)
                old_pitch = int(trial_notes[target_note_index].pitch)
                trial_notes[target_note_index].pitch = int(replacement)
                metrics = objective_metrics(trial, chords, bars=bars, bpm=bpm)
                gate = candidate_gate_penalty(metrics)
                if gate > current_gate:
                    continue
                if float(metrics["offbeat_unresolved_non_chord_ratio"]) >= float(
                    current_metrics["offbeat_unresolved_non_chord_ratio"]
                ):
                    continue
                if float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio):
                    continue
                if float(metrics["dominant_altered_offbeat_ratio"]) > 0.25:
                    continue
                if float(metrics["two_note_cycle_ratio"]) > 0.0:
                    continue
                if float(metrics["interval_trigram_repeat_ratio"]) > max(
                    0.02,
                    float(current_metrics["interval_trigram_repeat_ratio"]),
                ):
                    continue
                if float(metrics["max_bar_pitch_class_jaccard"]) > max(
                    0.72,
                    float(current_metrics["max_bar_pitch_class_jaccard"]) + 0.05,
                ):
                    continue
                if int(metrics["max_abs_interval"]) > max(12, int(current_metrics["max_abs_interval"])):
                    continue
                score = candidate_score(
                    metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                improvement_key = (
                    float(metrics["offbeat_unresolved_non_chord_ratio"]),
                    float(gate),
                    float(score),
                    float(metrics["max_bar_pitch_class_jaccard"]),
                    float(metrics["interval_trigram_repeat_ratio"]),
                )
                step = {
                    "unresolved_note_index": int(note_index + 1),
                    "changed_note_index": int(target_note_index + 1),
                    "old_pitch": old_pitch,
                    "new_pitch": int(replacement),
                    "old_offbeat_unresolved_non_chord_ratio": float(
                        current_metrics["offbeat_unresolved_non_chord_ratio"]
                    ),
                    "new_offbeat_unresolved_non_chord_ratio": float(
                        metrics["offbeat_unresolved_non_chord_ratio"]
                    ),
                    "old_score": float(current_score),
                    "new_score": float(score),
                }
                if best_candidate is None or improvement_key < best_candidate[0]:
                    best_candidate = (improvement_key, trial, metrics, step)
        if best_candidate is None:
            break
        _key, current, current_metrics, step = best_candidate
        current_score = float(step["new_score"])
        current_gate = candidate_gate_penalty(current_metrics)
        repair_steps.append(step)

    repair_report = {
        "attempted": True,
        "changed": bool(repair_steps),
        "step_count": len(repair_steps),
        "original_offbeat_unresolved_non_chord_ratio": original_unresolved,
        "final_offbeat_unresolved_non_chord_ratio": float(current_metrics["offbeat_unresolved_non_chord_ratio"]),
        "max_offbeat_non_chord_ratio": float(max_offbeat_non_chord_ratio),
        "steps": repair_steps,
    }
    return current, current_metrics, repair_report


def adjacent_repeat_replacement_pitches(
    notes: list[pretty_midi.Note],
    note_index: int,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
) -> list[int]:
    if note_index <= 0:
        return []
    previous = notes[note_index - 1]
    note = notes[note_index]
    if int(previous.pitch) != int(note.pitch):
        return []
    next_note = notes[note_index + 1] if note_index + 1 < len(notes) else None
    chord = chord_for_time(chords, float(note.start), bars=bars, bpm=bpm)
    chord_pcs = chord_pitch_classes(chord)
    if is_strong_beat_note(note, bpm=bpm):
        allowed_pcs = chord_pcs
    else:
        chord_root, _intervals = parse_chord(chord)
        root_pitch_class = chord_root % 12
        allowed_pcs = {(root_pitch_class + interval) % 12 for interval in scale_intervals(chord)}
        allowed_pcs |= chord_pcs
    reference = int(next_note.pitch) if next_note is not None else int(previous.pitch) + 3
    candidates = pitches_for_pcs(
        allowed_pcs,
        low=max(56, int(previous.pitch) - 8),
        high=min(86, int(previous.pitch) + 8),
    )
    return sorted(
        [
            pitch
            for pitch in candidates
            if int(pitch) != int(note.pitch)
            and (next_note is None or int(pitch) != int(next_note.pitch))
            and abs(int(pitch) - int(previous.pitch)) <= 8
            and (next_note is None or abs(int(next_note.pitch) - int(pitch)) <= 7)
        ],
        key=lambda pitch: (abs(int(pitch) - reference), abs(int(pitch) - int(note.pitch)), int(pitch)),
    )


def repair_adjacent_repeats(
    pm: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_offbeat_non_chord_ratio: float,
    max_iterations: int = 4,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(pm)
    current_metrics = objective_metrics(current, chords, bars=bars, bpm=bpm)
    current_score = candidate_score(
        current_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    current_gate = candidate_gate_penalty(current_metrics)
    original_adjacent_repeat = float(current_metrics["adjacent_repeat_ratio"])
    repair_steps: list[dict[str, Any]] = []

    for _ in range(int(max_iterations)):
        best_candidate: tuple[tuple[float, float, int, float, float], pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]] | None = None
        notes = solo_notes(current)
        for note_index in range(1, len(notes)):
            for replacement in adjacent_repeat_replacement_pitches(notes, note_index, chords, bars=bars, bpm=bpm):
                trial = copy.deepcopy(current)
                trial_notes = solo_notes(trial)
                old_pitch = int(trial_notes[note_index].pitch)
                trial_notes[note_index].pitch = int(replacement)
                metrics = objective_metrics(trial, chords, bars=bars, bpm=bpm)
                gate = candidate_gate_penalty(metrics)
                if gate > current_gate:
                    continue
                if float(metrics["adjacent_repeat_ratio"]) >= float(current_metrics["adjacent_repeat_ratio"]):
                    continue
                if float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(
                    current_metrics["offbeat_unresolved_non_chord_ratio"]
                ):
                    continue
                if float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio):
                    continue
                if float(metrics["dominant_altered_offbeat_ratio"]) > 0.25:
                    continue
                if float(metrics["two_note_cycle_ratio"]) > float(current_metrics["two_note_cycle_ratio"]):
                    continue
                if float(metrics["interval_trigram_repeat_ratio"]) > max(
                    0.02,
                    float(current_metrics["interval_trigram_repeat_ratio"]),
                ):
                    continue
                if float(metrics["max_bar_pitch_class_jaccard"]) > max(
                    0.72,
                    float(current_metrics["max_bar_pitch_class_jaccard"]) + 0.05,
                ):
                    continue
                if float(metrics["large_leap_ratio"]) > max(0.065, float(current_metrics["large_leap_ratio"])):
                    continue
                score = candidate_score(
                    metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                if score > current_score + 0.10:
                    continue
                improvement_key = (
                    float(metrics["adjacent_repeat_ratio"]),
                    float(metrics["large_leap_ratio"]),
                    int(metrics["max_abs_interval"]),
                    -float(metrics["enclosure_proxy_ratio"]),
                    float(score),
                )
                current_key = (
                    float(current_metrics["adjacent_repeat_ratio"]),
                    float(current_metrics["large_leap_ratio"]),
                    int(current_metrics["max_abs_interval"]),
                    -float(current_metrics["enclosure_proxy_ratio"]),
                    float(current_score),
                )
                if improvement_key >= current_key:
                    continue
                step = {
                    "changed_note_index": int(note_index + 1),
                    "old_pitch": old_pitch,
                    "new_pitch": int(replacement),
                    "old_adjacent_repeat_ratio": float(current_metrics["adjacent_repeat_ratio"]),
                    "new_adjacent_repeat_ratio": float(metrics["adjacent_repeat_ratio"]),
                    "old_score": float(current_score),
                    "new_score": float(score),
                }
                if best_candidate is None or improvement_key < best_candidate[0]:
                    best_candidate = (improvement_key, trial, metrics, step)
        if best_candidate is None:
            break
        _key, current, current_metrics, step = best_candidate
        current_score = float(step["new_score"])
        current_gate = candidate_gate_penalty(current_metrics)
        repair_steps.append(step)

    repair_report = {
        "attempted": True,
        "changed": bool(repair_steps),
        "step_count": len(repair_steps),
        "original_adjacent_repeat_ratio": original_adjacent_repeat,
        "final_adjacent_repeat_ratio": float(current_metrics["adjacent_repeat_ratio"]),
        "steps": repair_steps,
    }
    return current, current_metrics, repair_report


def large_leap_replacement_pitches(
    notes: list[pretty_midi.Note],
    note_index: int,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
) -> list[int]:
    if note_index <= 0:
        return []
    previous = notes[note_index - 1]
    note = notes[note_index]
    if abs(int(note.pitch) - int(previous.pitch)) < 6:
        return []
    chord = chord_for_time(chords, float(note.start), bars=bars, bpm=bpm)
    if is_strong_beat_note(note, bpm=bpm):
        allowed_pcs = chord_pitch_classes(chord)
    else:
        chord_root, _intervals = parse_chord(chord)
        root_pitch_class = chord_root % 12
        allowed_pcs = {(root_pitch_class + interval) % 12 for interval in scale_intervals(chord)}
        allowed_pcs |= chord_pitch_classes(chord)
    candidates = pitches_for_pcs(
        allowed_pcs,
        low=max(56, int(previous.pitch) - 5),
        high=min(86, int(previous.pitch) + 5),
    )
    return sorted(
        [
            pitch
            for pitch in candidates
            if int(pitch) != int(note.pitch)
            and abs(int(pitch) - int(previous.pitch)) < abs(int(note.pitch) - int(previous.pitch))
        ],
        key=lambda pitch: (abs(int(pitch) - int(previous.pitch)), abs(int(pitch) - int(note.pitch)), int(pitch)),
    )


def repair_large_leaps(
    pm: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_offbeat_non_chord_ratio: float,
    min_enclosure_proxy_ratio: float,
    max_iterations: int = 4,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(pm)
    current_metrics = objective_metrics(current, chords, bars=bars, bpm=bpm)
    current_score = candidate_score(
        current_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    current_gate = candidate_gate_penalty(current_metrics)
    original_large_leap = float(current_metrics["large_leap_ratio"])
    repair_steps: list[dict[str, Any]] = []

    for _ in range(int(max_iterations)):
        best_candidate: tuple[tuple[float, int, float, float, float], pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]] | None = None
        notes = solo_notes(current)
        for note_index in range(1, len(notes)):
            for replacement in large_leap_replacement_pitches(notes, note_index, chords, bars=bars, bpm=bpm):
                trial = copy.deepcopy(current)
                trial_notes = solo_notes(trial)
                old_pitch = int(trial_notes[note_index].pitch)
                trial_notes[note_index].pitch = int(replacement)
                metrics = objective_metrics(trial, chords, bars=bars, bpm=bpm)
                gate = candidate_gate_penalty(metrics)
                if gate > current_gate:
                    continue
                if float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(
                    current_metrics["offbeat_unresolved_non_chord_ratio"]
                ):
                    continue
                if float(metrics["offbeat_non_chord_resolution_ratio"]) < float(
                    current_metrics["offbeat_non_chord_resolution_ratio"]
                ):
                    continue
                if float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio):
                    continue
                if float(metrics["dominant_altered_offbeat_ratio"]) > 0.25:
                    continue
                if float(metrics["two_note_cycle_ratio"]) > 0.0:
                    continue
                if float(metrics["interval_trigram_repeat_ratio"]) > max(
                    0.02,
                    float(current_metrics["interval_trigram_repeat_ratio"]),
                ):
                    continue
                if float(metrics["max_bar_pitch_class_jaccard"]) > 0.72:
                    continue
                if float(metrics["enclosure_proxy_ratio"]) < float(min_enclosure_proxy_ratio):
                    continue
                score = candidate_score(
                    metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                if score > current_score + 0.10:
                    continue
                improvement_key = (
                    float(metrics["large_leap_ratio"]),
                    int(metrics["max_abs_interval"]),
                    -float(metrics["enclosure_proxy_ratio"]),
                    float(metrics["max_bar_pitch_class_jaccard"]),
                    float(score),
                )
                current_key = (
                    float(current_metrics["large_leap_ratio"]),
                    int(current_metrics["max_abs_interval"]),
                    -float(current_metrics["enclosure_proxy_ratio"]),
                    float(current_metrics["max_bar_pitch_class_jaccard"]),
                    float(current_score),
                )
                if improvement_key >= current_key:
                    continue
                step = {
                    "changed_note_index": int(note_index + 1),
                    "old_pitch": old_pitch,
                    "new_pitch": int(replacement),
                    "old_large_leap_ratio": float(current_metrics["large_leap_ratio"]),
                    "new_large_leap_ratio": float(metrics["large_leap_ratio"]),
                    "old_max_abs_interval": int(current_metrics["max_abs_interval"]),
                    "new_max_abs_interval": int(metrics["max_abs_interval"]),
                    "old_enclosure_proxy_ratio": float(current_metrics["enclosure_proxy_ratio"]),
                    "new_enclosure_proxy_ratio": float(metrics["enclosure_proxy_ratio"]),
                    "old_score": float(current_score),
                    "new_score": float(score),
                }
                if best_candidate is None or improvement_key < best_candidate[0]:
                    best_candidate = (improvement_key, trial, metrics, step)
        if best_candidate is None:
            break
        _key, current, current_metrics, step = best_candidate
        current_score = float(step["new_score"])
        current_gate = candidate_gate_penalty(current_metrics)
        repair_steps.append(step)

    repair_report = {
        "attempted": True,
        "changed": bool(repair_steps),
        "step_count": len(repair_steps),
        "original_large_leap_ratio": original_large_leap,
        "final_large_leap_ratio": float(current_metrics["large_leap_ratio"]),
        "min_enclosure_proxy_ratio": float(min_enclosure_proxy_ratio),
        "steps": repair_steps,
    }
    return current, current_metrics, repair_report


def motion_balance_score(
    metrics: dict[str, Any],
    *,
    target_min_step_motion_ratio: float,
    target_min_chromatic_step_ratio: float,
    target_max_large_leap_ratio: float,
) -> float:
    return (
        max(0.0, float(target_min_step_motion_ratio) - float(metrics["step_motion_ratio"])) * 5.0
        + max(0.0, float(target_min_chromatic_step_ratio) - float(metrics["chromatic_step_ratio"])) * 5.0
        + max(0.0, float(metrics["large_leap_ratio"]) - float(target_max_large_leap_ratio)) * 4.0
        + max(0.0, 0.30 - float(metrics["enclosure_proxy_ratio"])) * 1.2
        + float(metrics["adjacent_repeat_ratio"]) * 3.0
        + float(metrics["offbeat_unresolved_non_chord_ratio"]) * 8.0
        + max(0.0, float(metrics["max_bar_pitch_class_jaccard"]) - 0.70) * 2.0
    )


def motion_balance_replacement_pitches(
    notes: list[pretty_midi.Note],
    note_index: int,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
) -> list[int]:
    note = notes[note_index]
    previous = notes[note_index - 1] if note_index > 0 else None
    next_note = notes[note_index + 1] if note_index + 1 < len(notes) else None
    chord = chord_for_time(chords, float(note.start), bars=bars, bpm=bpm)
    if is_strong_beat_note(note, bpm=bpm):
        allowed_pcs = chord_pitch_classes(chord)
    else:
        chord_root, _intervals = parse_chord(chord)
        root_pitch_class = chord_root % 12
        allowed_pcs = {(root_pitch_class + interval) % 12 for interval in scale_intervals(chord)}
        allowed_pcs |= chord_pitch_classes(chord)

    references = {int(note.pitch)}
    if previous is not None:
        for offset in (-4, -3, -2, -1, 1, 2, 3, 4):
            references.add(int(previous.pitch) + offset)
    if next_note is not None:
        for offset in (-4, -3, -2, -1, 1, 2, 3, 4):
            references.add(int(next_note.pitch) + offset)

    candidates: set[int] = set()
    for reference in references:
        for pitch in pitches_for_pcs(
            allowed_pcs,
            low=max(56, int(reference) - 2),
            high=min(86, int(reference) + 2),
        ):
            if int(pitch) != int(note.pitch):
                candidates.add(int(pitch))

    def local_key(pitch: int) -> tuple[int, int, int, int]:
        left = abs(int(pitch) - int(previous.pitch)) if previous is not None else 0
        right = abs(int(next_note.pitch) - int(pitch)) if next_note is not None else 0
        large_count = int(left >= 6) + int(right >= 6)
        chromatic_count = int(left == 1) + int(right == 1)
        step_count = int(left <= 2) + int(right <= 2)
        return (large_count, -chromatic_count, -step_count, abs(int(pitch) - int(note.pitch)))

    return sorted(candidates, key=local_key)[:10]


def repair_motion_balance(
    pm: pretty_midi.PrettyMIDI,
    chords: list[str],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_offbeat_non_chord_ratio: float,
    max_bar_pitch_class_jaccard: float,
    target_min_step_motion_ratio: float,
    target_min_chromatic_step_ratio: float,
    target_max_large_leap_ratio: float,
    max_iterations: int = 8,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]]:
    current = copy.deepcopy(pm)
    current_metrics = objective_metrics(current, chords, bars=bars, bpm=bpm)
    current_score = candidate_score(
        current_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    current_gate = candidate_gate_penalty(current_metrics)
    current_motion_score = motion_balance_score(
        current_metrics,
        target_min_step_motion_ratio=target_min_step_motion_ratio,
        target_min_chromatic_step_ratio=target_min_chromatic_step_ratio,
        target_max_large_leap_ratio=target_max_large_leap_ratio,
    )
    original_metrics = {
        "step_motion_ratio": float(current_metrics["step_motion_ratio"]),
        "chromatic_step_ratio": float(current_metrics["chromatic_step_ratio"]),
        "large_leap_ratio": float(current_metrics["large_leap_ratio"]),
        "enclosure_proxy_ratio": float(current_metrics["enclosure_proxy_ratio"]),
        "max_bar_pitch_class_jaccard": float(current_metrics["max_bar_pitch_class_jaccard"]),
    }
    repair_steps: list[dict[str, Any]] = []

    for _ in range(int(max_iterations)):
        best_candidate: tuple[tuple[float, float, float, float, float], pretty_midi.PrettyMIDI, dict[str, Any], dict[str, Any]] | None = None
        notes = solo_notes(current)
        for note_index in range(len(notes)):
            for replacement in motion_balance_replacement_pitches(notes, note_index, chords, bars=bars, bpm=bpm):
                trial = copy.deepcopy(current)
                trial_notes = solo_notes(trial)
                old_pitch = int(trial_notes[note_index].pitch)
                trial_notes[note_index].pitch = int(replacement)
                metrics = objective_metrics(trial, chords, bars=bars, bpm=bpm)
                gate = candidate_gate_penalty(metrics)
                if gate > current_gate:
                    continue
                if float(metrics["offbeat_unresolved_non_chord_ratio"]) > float(
                    current_metrics["offbeat_unresolved_non_chord_ratio"]
                ):
                    continue
                if float(metrics["offbeat_non_chord_resolution_ratio"]) < float(
                    current_metrics["offbeat_non_chord_resolution_ratio"]
                ):
                    continue
                if float(metrics["offbeat_non_chord_ratio"]) > float(max_offbeat_non_chord_ratio):
                    continue
                if float(metrics["dominant_altered_offbeat_ratio"]) > 0.25:
                    continue
                if float(metrics["adjacent_repeat_ratio"]) > float(current_metrics["adjacent_repeat_ratio"]):
                    continue
                if float(metrics["two_note_cycle_ratio"]) > float(current_metrics["two_note_cycle_ratio"]):
                    continue
                if float(metrics["interval_trigram_repeat_ratio"]) > max(
                    0.02,
                    float(current_metrics["interval_trigram_repeat_ratio"]),
                ):
                    continue
                if float(metrics["max_bar_pitch_class_jaccard"]) > float(max_bar_pitch_class_jaccard):
                    continue
                motion_score = motion_balance_score(
                    metrics,
                    target_min_step_motion_ratio=target_min_step_motion_ratio,
                    target_min_chromatic_step_ratio=target_min_chromatic_step_ratio,
                    target_max_large_leap_ratio=target_max_large_leap_ratio,
                )
                if motion_score >= current_motion_score:
                    continue
                score = candidate_score(
                    metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                if score > current_score + 0.12:
                    continue
                improvement_key = (
                    float(motion_score),
                    float(metrics["large_leap_ratio"]),
                    -float(metrics["chromatic_step_ratio"]),
                    -float(metrics["step_motion_ratio"]),
                    float(score),
                )
                step = {
                    "changed_note_index": int(note_index + 1),
                    "old_pitch": old_pitch,
                    "new_pitch": int(replacement),
                    "old_motion_balance_score": float(current_motion_score),
                    "new_motion_balance_score": float(motion_score),
                    "old_step_motion_ratio": float(current_metrics["step_motion_ratio"]),
                    "new_step_motion_ratio": float(metrics["step_motion_ratio"]),
                    "old_chromatic_step_ratio": float(current_metrics["chromatic_step_ratio"]),
                    "new_chromatic_step_ratio": float(metrics["chromatic_step_ratio"]),
                    "old_large_leap_ratio": float(current_metrics["large_leap_ratio"]),
                    "new_large_leap_ratio": float(metrics["large_leap_ratio"]),
                    "old_score": float(current_score),
                    "new_score": float(score),
                }
                if best_candidate is None or improvement_key < best_candidate[0]:
                    best_candidate = (improvement_key, trial, metrics, step)
        if best_candidate is None:
            break
        _key, current, current_metrics, step = best_candidate
        current_score = float(step["new_score"])
        current_gate = candidate_gate_penalty(current_metrics)
        current_motion_score = float(step["new_motion_balance_score"])
        repair_steps.append(step)

    repair_report = {
        "attempted": True,
        "changed": bool(repair_steps),
        "step_count": len(repair_steps),
        "original": original_metrics,
        "final": {
            "step_motion_ratio": float(current_metrics["step_motion_ratio"]),
            "chromatic_step_ratio": float(current_metrics["chromatic_step_ratio"]),
            "large_leap_ratio": float(current_metrics["large_leap_ratio"]),
            "enclosure_proxy_ratio": float(current_metrics["enclosure_proxy_ratio"]),
            "max_bar_pitch_class_jaccard": float(current_metrics["max_bar_pitch_class_jaccard"]),
        },
        "target_min_step_motion_ratio": float(target_min_step_motion_ratio),
        "target_min_chromatic_step_ratio": float(target_min_chromatic_step_ratio),
        "target_max_large_leap_ratio": float(target_max_large_leap_ratio),
        "max_bar_pitch_class_jaccard": float(max_bar_pitch_class_jaccard),
        "steps": repair_steps,
    }
    return current, current_metrics, repair_report


def rhythm_articulation_metrics(pm: pretty_midi.PrettyMIDI, *, bars: int, bpm: float) -> dict[str, Any]:
    notes = solo_notes(pm)
    if not notes:
        return {
            "note_count": 0,
            "duration_template_repeat_ratio": 0.0,
            "most_common_duration_ratio": 0.0,
            "unique_duration_bucket_count": 0,
            "unique_velocity_count": 0,
            "velocity_range": 0,
            "max_offbeat_start_shift_seconds": 0.0,
        }
    seconds_per_beat = 60.0 / float(bpm)
    durations = [max(0.0, float(note.end) - float(note.start)) / seconds_per_beat for note in notes]
    duration_buckets = [round(duration, 2) for duration in durations]
    duration_counts = Counter(duration_buckets)
    most_common_duration_ratio = max(duration_counts.values(), default=0) / max(1, len(duration_buckets))
    bar_templates = [
        tuple(duration_buckets[index : index + 8])
        for index in range(0, min(len(duration_buckets), int(bars) * 8), 8)
        if len(duration_buckets[index : index + 8]) == 8
    ]
    template_counts = Counter(bar_templates)
    duration_template_repeat_ratio = max(template_counts.values(), default=0) / max(1, len(bar_templates))
    velocities = [int(note.velocity) for note in notes]
    ideal_starts = []
    for bar_index in range(int(bars)):
        bar_start = bar_index * 4 * seconds_per_beat
        for beat_index in range(4):
            ideal_starts.append(bar_start + beat_index * seconds_per_beat)
            ideal_starts.append(bar_start + (beat_index + 2 / 3) * seconds_per_beat)
    offbeat_shifts = []
    for index, note in enumerate(notes[: len(ideal_starts)]):
        if index % 2 == 1:
            offbeat_shifts.append(abs(float(note.start) - float(ideal_starts[index])))
    return {
        "note_count": len(notes),
        "duration_template_repeat_ratio": float(duration_template_repeat_ratio),
        "most_common_duration_ratio": float(most_common_duration_ratio),
        "unique_duration_bucket_count": len(set(duration_buckets)),
        "unique_velocity_count": len(set(velocities)),
        "velocity_range": max(velocities) - min(velocities) if velocities else 0,
        "max_offbeat_start_shift_seconds": max(offbeat_shifts or [0.0]),
    }


def apply_rhythm_articulation_variation(
    pm: pretty_midi.PrettyMIDI,
    *,
    bars: int,
    bpm: float,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    current = copy.deepcopy(pm)
    notes = solo_notes(current)
    before_metrics = rhythm_articulation_metrics(current, bars=bars, bpm=bpm)
    if not notes:
        return current, {
            "attempted": True,
            "changed": False,
            "reason": "no solo notes",
            "before": before_metrics,
            "after": before_metrics,
        }

    seconds_per_beat = 60.0 / float(bpm)
    total_duration = int(bars) * 4 * seconds_per_beat
    duration_patterns = (
        (0.78, 0.54, 0.90, 0.62, 0.74, 0.58, 0.86, 0.66),
        (0.70, 0.68, 0.84, 0.52, 0.82, 0.60, 0.76, 0.72),
        (0.88, 0.50, 0.76, 0.70, 0.92, 0.56, 0.80, 0.64),
    )
    velocity_offsets = (
        (10, -14, 4, -10, 8, -16, 2, -8),
        (6, -8, 12, -16, 2, -10, 10, -12),
        (12, -18, 0, -6, 10, -12, 4, -14),
    )
    offbeat_shift_beats = (
        (0.0, -0.030, 0.0, 0.018, 0.0, -0.014, 0.0, 0.024),
        (0.0, 0.020, 0.0, -0.026, 0.0, 0.014, 0.0, -0.018),
        (0.0, -0.018, 0.0, 0.026, 0.0, -0.024, 0.0, 0.012),
    )

    original_starts = [float(note.start) for note in notes]
    shifted_starts: list[float] = []
    for index, note in enumerate(notes):
        slot = index % 8
        bar_index = min(index // 8, int(bars) - 1)
        pattern_index = bar_index % len(duration_patterns)
        shift_seconds = offbeat_shift_beats[pattern_index][slot] * seconds_per_beat
        if slot % 2 == 0:
            shift_seconds = 0.0
        new_start = max(0.0, min(total_duration - 0.02, original_starts[index] + shift_seconds))
        if shifted_starts:
            new_start = max(new_start, shifted_starts[-1] + 0.02)
        shifted_starts.append(float(new_start))
        note.velocity = int(max(52, min(118, int(note.velocity) + velocity_offsets[pattern_index][slot])))

    for index, note in enumerate(notes):
        slot = index % 8
        bar_index = min(index // 8, int(bars) - 1)
        pattern_index = bar_index % len(duration_patterns)
        next_start = shifted_starts[index + 1] if index + 1 < len(shifted_starts) else total_duration
        available = max(0.05, next_start - shifted_starts[index])
        duration = max(0.055, available * duration_patterns[pattern_index][slot])
        note.start = float(shifted_starts[index])
        note.end = float(min(total_duration, shifted_starts[index] + duration))
        if index + 1 < len(notes):
            note.end = min(float(note.end), float(next_start - 0.006))
        if index == len(notes) - 1:
            note.end = float(total_duration)

    for instrument in current.instruments:
        instrument.notes.sort(key=lambda note: (float(note.start), int(note.pitch), float(note.end)))

    after_metrics = rhythm_articulation_metrics(current, bars=bars, bpm=bpm)
    return current, {
        "attempted": True,
        "changed": after_metrics != before_metrics,
        "before": before_metrics,
        "after": after_metrics,
    }


def apply_candidate_repairs(
    pm: pretty_midi.PrettyMIDI,
    item: dict[str, Any],
    *,
    bars: int,
    bpm: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    repair_bar_similarity_enabled: bool,
    repair_bar_similarity_iterations: int,
    repair_enclosure_density_enabled: bool,
    repair_enclosure_density_iterations: int,
    max_enclosure_repair_offbeat_non_chord_ratio: float,
    repair_unresolved_offbeat_enabled: bool,
    repair_unresolved_offbeat_iterations: int,
    repair_adjacent_repeats_enabled: bool,
    repair_adjacent_repeats_iterations: int,
    repair_large_leaps_enabled: bool,
    repair_large_leaps_iterations: int,
    min_large_leap_repair_enclosure_proxy_ratio: float,
    repair_motion_balance_enabled: bool,
    repair_motion_balance_iterations: int,
    target_min_step_motion_ratio: float,
    target_min_chromatic_step_ratio: float,
    target_max_large_leap_ratio: float,
    max_motion_balance_bar_pitch_class_jaccard: float,
) -> tuple[
    pretty_midi.PrettyMIDI,
    dict[str, Any],
    float,
    float,
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    bar_similarity_repair = {"attempted": False, "changed": False, "step_count": 0}
    enclosure_density_repair = {"attempted": False, "changed": False, "step_count": 0}
    unresolved_offbeat_repair = {"attempted": False, "changed": False, "step_count": 0}
    adjacent_repeat_repair = {"attempted": False, "changed": False, "step_count": 0}
    large_leap_repair = {"attempted": False, "changed": False, "step_count": 0}
    motion_balance_repair = {"attempted": False, "changed": False, "step_count": 0}
    item_metrics = objective_metrics(pm, item["chords"], bars=bars, bpm=bpm)
    item_score = candidate_score(
        item_metrics,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    item_gate_penalty = candidate_gate_penalty(item_metrics)
    if repair_bar_similarity_enabled:
        pm, item_metrics, bar_similarity_repair = repair_bar_similarity(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            max_iterations=repair_bar_similarity_iterations,
        )
        item_score = candidate_score(
            item_metrics,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
        )
        item_gate_penalty = candidate_gate_penalty(item_metrics)
    if repair_enclosure_density_enabled:
        pm, item_metrics, enclosure_density_repair = repair_enclosure_density(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            max_iterations=repair_enclosure_density_iterations,
            max_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
        )
        item_score = candidate_score(
            item_metrics,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
        )
        item_gate_penalty = candidate_gate_penalty(item_metrics)
    if repair_unresolved_offbeat_enabled:
        pm, item_metrics, unresolved_offbeat_repair = repair_unresolved_offbeats(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            max_iterations=repair_unresolved_offbeat_iterations,
            max_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
        )
        item_score = candidate_score(
            item_metrics,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
        )
        item_gate_penalty = candidate_gate_penalty(item_metrics)
    if repair_large_leaps_enabled:
        pm, item_metrics, large_leap_repair = repair_large_leaps(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            max_iterations=repair_large_leaps_iterations,
            max_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
            min_enclosure_proxy_ratio=min_large_leap_repair_enclosure_proxy_ratio,
        )
        item_score = candidate_score(
            item_metrics,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
        )
        item_gate_penalty = candidate_gate_penalty(item_metrics)
    if repair_adjacent_repeats_enabled:
        pm, item_metrics, adjacent_repeat_repair = repair_adjacent_repeats(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            max_iterations=repair_adjacent_repeats_iterations,
            max_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
        )
        item_score = candidate_score(
            item_metrics,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
        )
        item_gate_penalty = candidate_gate_penalty(item_metrics)
    if repair_motion_balance_enabled:
        pm, item_metrics, motion_balance_repair = repair_motion_balance(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            max_iterations=repair_motion_balance_iterations,
            max_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
            max_bar_pitch_class_jaccard=max_motion_balance_bar_pitch_class_jaccard,
            target_min_step_motion_ratio=target_min_step_motion_ratio,
            target_min_chromatic_step_ratio=target_min_chromatic_step_ratio,
            target_max_large_leap_ratio=target_max_large_leap_ratio,
        )
        item_score = candidate_score(
            item_metrics,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
        )
        item_gate_penalty = candidate_gate_penalty(item_metrics)
    return (
        pm,
        item_metrics,
        item_score,
        item_gate_penalty,
        bar_similarity_repair,
        enclosure_density_repair,
        unresolved_offbeat_repair,
        adjacent_repeat_repair,
        large_leap_repair,
        motion_balance_repair,
    )


def build_best_of_package(
    *,
    output_dir: Path,
    source_root: Path,
    package_globs: list[str],
    render_config: RenderConfig,
    bars: int,
    bpm: float,
    selected_count: int,
    max_per_case: int,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
    max_gate_penalty: float | None = None,
    max_offbeat_non_chord_ratio: float | None = None,
    max_unresolved_offbeat_non_chord_ratio: float | None = None,
    max_dominant_altered_offbeat_ratio: float | None = None,
    max_adjacent_repeat_ratio: float | None = None,
    max_bar_pitch_class_jaccard: float | None = None,
    listen_first_mode: str = "rank",
    repair_bar_similarity_enabled: bool = False,
    repair_bar_similarity_iterations: int = 4,
    repair_enclosure_density_enabled: bool = False,
    repair_enclosure_density_iterations: int = 8,
    max_enclosure_repair_offbeat_non_chord_ratio: float = 0.421875,
    context_bass_velocity_boost: int = 0,
    context_comp_velocity_boost: int = 0,
    select_after_repair: bool = False,
    repair_unresolved_offbeat_enabled: bool = False,
    repair_unresolved_offbeat_iterations: int = 4,
    repair_adjacent_repeats_enabled: bool = False,
    repair_adjacent_repeats_iterations: int = 4,
    repair_large_leaps_enabled: bool = False,
    repair_large_leaps_iterations: int = 4,
    min_large_leap_repair_enclosure_proxy_ratio: float = 0.28125,
    repair_motion_balance_enabled: bool = False,
    repair_motion_balance_iterations: int = 8,
    target_min_step_motion_ratio: float = 0.40,
    target_min_chromatic_step_ratio: float = 0.22,
    target_max_large_leap_ratio: float = 0.055,
    max_motion_balance_bar_pitch_class_jaccard: float = 0.70,
    repair_rhythm_articulation_enabled: bool = False,
    selection_profile: str = "score",
) -> dict[str, Any]:
    paths = package_paths(source_root, package_globs)
    rows = candidate_rows(
        paths=paths,
        target_chord_tone_ratio=target_chord_tone_ratio,
        target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
    )
    selection_rows = filter_candidate_rows(
        rows,
        max_gate_penalty=max_gate_penalty,
        max_offbeat_non_chord_ratio=max_offbeat_non_chord_ratio,
        max_unresolved_offbeat_non_chord_ratio=max_unresolved_offbeat_non_chord_ratio,
        max_dominant_altered_offbeat_ratio=max_dominant_altered_offbeat_ratio,
        max_adjacent_repeat_ratio=None if select_after_repair else max_adjacent_repeat_ratio,
        max_bar_pitch_class_jaccard=None if select_after_repair else max_bar_pitch_class_jaccard,
    )
    if not selection_rows:
        raise BebopLanguagePackageError("no candidate rows after best-of selection filters")
    if select_after_repair and (
        repair_bar_similarity_enabled or repair_enclosure_density_enabled or repair_unresolved_offbeat_enabled
        or repair_adjacent_repeats_enabled
        or repair_large_leaps_enabled
        or repair_motion_balance_enabled
    ):
        repaired_selection_rows: list[dict[str, Any]] = []
        for item in selection_rows:
            source_midi = Path(str(item["source_midi_path"]))
            pm = pretty_midi.PrettyMIDI(str(source_midi))
            (
                _pm,
                item_metrics,
                item_score,
                item_gate_penalty,
                _bar_repair,
                _enclosure_repair,
                _unresolved_repair,
                _adjacent_repair,
                _large_leap_repair,
                _motion_balance_repair,
            ) = apply_candidate_repairs(
                pm,
                item,
                bars=bars,
                bpm=bpm,
                target_chord_tone_ratio=target_chord_tone_ratio,
                target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                repair_bar_similarity_enabled=repair_bar_similarity_enabled,
                repair_bar_similarity_iterations=repair_bar_similarity_iterations,
                repair_enclosure_density_enabled=repair_enclosure_density_enabled,
                repair_enclosure_density_iterations=repair_enclosure_density_iterations,
                max_enclosure_repair_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
                repair_unresolved_offbeat_enabled=repair_unresolved_offbeat_enabled,
                repair_unresolved_offbeat_iterations=repair_unresolved_offbeat_iterations,
                repair_adjacent_repeats_enabled=repair_adjacent_repeats_enabled,
                repair_adjacent_repeats_iterations=repair_adjacent_repeats_iterations,
                repair_large_leaps_enabled=repair_large_leaps_enabled,
                repair_large_leaps_iterations=repair_large_leaps_iterations,
                min_large_leap_repair_enclosure_proxy_ratio=min_large_leap_repair_enclosure_proxy_ratio,
                repair_motion_balance_enabled=repair_motion_balance_enabled,
                repair_motion_balance_iterations=repair_motion_balance_iterations,
                target_min_step_motion_ratio=target_min_step_motion_ratio,
                target_min_chromatic_step_ratio=target_min_chromatic_step_ratio,
                target_max_large_leap_ratio=target_max_large_leap_ratio,
                max_motion_balance_bar_pitch_class_jaccard=max_motion_balance_bar_pitch_class_jaccard,
            )
            repaired_selection_rows.append(
                {
                    **item,
                    "objective_metrics": item_metrics,
                    "score": float(item_score),
                    "gate_penalty": float(item_gate_penalty),
                }
            )
        selection_rows = sorted(
            filter_candidate_rows(
                repaired_selection_rows,
                max_gate_penalty=max_gate_penalty,
                max_offbeat_non_chord_ratio=max_offbeat_non_chord_ratio,
                max_unresolved_offbeat_non_chord_ratio=max_unresolved_offbeat_non_chord_ratio,
                max_dominant_altered_offbeat_ratio=max_dominant_altered_offbeat_ratio,
                max_adjacent_repeat_ratio=max_adjacent_repeat_ratio,
                max_bar_pitch_class_jaccard=max_bar_pitch_class_jaccard,
            ),
            key=lambda row: selection_sort_key(row, selection_profile=selection_profile),
        )
        if not selection_rows:
            raise BebopLanguagePackageError("no candidate rows after repaired best-of selection filters")
    elif selection_profile != "score":
        selection_rows = sorted(selection_rows, key=lambda row: selection_sort_key(row, selection_profile=selection_profile))
    selected = select_candidates(selection_rows, selected_count=selected_count, max_per_case=max_per_case)
    solo_dir = output_dir / "midi"
    mix_midi_dir = output_dir / "midi_with_context"
    solo_audio_dir = output_dir / "audio"
    mix_audio_dir = output_dir / "audio_with_context"
    rendered: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        source_midi = Path(str(item["source_midi_path"]))
        pm = pretty_midi.PrettyMIDI(str(source_midi))
        (
            pm,
            item_metrics,
            item_score,
            item_gate_penalty,
            bar_similarity_repair,
            enclosure_density_repair,
            unresolved_offbeat_repair,
            adjacent_repeat_repair,
            large_leap_repair,
            motion_balance_repair,
        ) = apply_candidate_repairs(
            pm,
            item,
            bars=bars,
            bpm=bpm,
            target_chord_tone_ratio=target_chord_tone_ratio,
            target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            repair_bar_similarity_enabled=repair_bar_similarity_enabled,
            repair_bar_similarity_iterations=repair_bar_similarity_iterations,
            repair_enclosure_density_enabled=repair_enclosure_density_enabled,
            repair_enclosure_density_iterations=repair_enclosure_density_iterations,
            max_enclosure_repair_offbeat_non_chord_ratio=max_enclosure_repair_offbeat_non_chord_ratio,
            repair_unresolved_offbeat_enabled=repair_unresolved_offbeat_enabled,
            repair_unresolved_offbeat_iterations=repair_unresolved_offbeat_iterations,
            repair_adjacent_repeats_enabled=repair_adjacent_repeats_enabled,
            repair_adjacent_repeats_iterations=repair_adjacent_repeats_iterations,
            repair_large_leaps_enabled=repair_large_leaps_enabled,
            repair_large_leaps_iterations=repair_large_leaps_iterations,
            min_large_leap_repair_enclosure_proxy_ratio=min_large_leap_repair_enclosure_proxy_ratio,
            repair_motion_balance_enabled=repair_motion_balance_enabled,
            repair_motion_balance_iterations=repair_motion_balance_iterations,
            target_min_step_motion_ratio=target_min_step_motion_ratio,
            target_min_chromatic_step_ratio=target_min_chromatic_step_ratio,
            target_max_large_leap_ratio=target_max_large_leap_ratio,
            max_motion_balance_bar_pitch_class_jaccard=max_motion_balance_bar_pitch_class_jaccard,
        )
        rhythm_articulation_repair = {"attempted": False, "changed": False, "accepted": False}
        if repair_rhythm_articulation_enabled:
            candidate_pm, rhythm_articulation_repair = apply_rhythm_articulation_variation(
                pm,
                bars=bars,
                bpm=bpm,
            )
            candidate_metrics = objective_metrics(candidate_pm, item["chords"], bars=bars, bpm=bpm)
            candidate_gate = candidate_gate_penalty(candidate_metrics)
            guard_passed = (
                candidate_gate <= item_gate_penalty
                and float(candidate_metrics["offbeat_unresolved_non_chord_ratio"])
                <= float(item_metrics["offbeat_unresolved_non_chord_ratio"])
                and float(candidate_metrics["offbeat_non_chord_resolution_ratio"])
                >= float(item_metrics["offbeat_non_chord_resolution_ratio"])
                and float(candidate_metrics["adjacent_repeat_ratio"]) <= float(item_metrics["adjacent_repeat_ratio"])
                and float(candidate_metrics["max_bar_pitch_class_jaccard"])
                <= max(0.72, float(item_metrics["max_bar_pitch_class_jaccard"]) + 0.01)
            )
            rhythm_articulation_repair["accepted"] = bool(guard_passed)
            rhythm_articulation_repair["guard"] = {
                "before_gate_penalty": float(item_gate_penalty),
                "after_gate_penalty": float(candidate_gate),
                "before_offbeat_unresolved_non_chord_ratio": float(
                    item_metrics["offbeat_unresolved_non_chord_ratio"]
                ),
                "after_offbeat_unresolved_non_chord_ratio": float(
                    candidate_metrics["offbeat_unresolved_non_chord_ratio"]
                ),
                "before_offbeat_non_chord_resolution_ratio": float(
                    item_metrics["offbeat_non_chord_resolution_ratio"]
                ),
                "after_offbeat_non_chord_resolution_ratio": float(
                    candidate_metrics["offbeat_non_chord_resolution_ratio"]
                ),
                "before_adjacent_repeat_ratio": float(item_metrics["adjacent_repeat_ratio"]),
                "after_adjacent_repeat_ratio": float(candidate_metrics["adjacent_repeat_ratio"]),
                "before_max_bar_pitch_class_jaccard": float(item_metrics["max_bar_pitch_class_jaccard"]),
                "after_max_bar_pitch_class_jaccard": float(candidate_metrics["max_bar_pitch_class_jaccard"]),
            }
            if guard_passed:
                pm = candidate_pm
                item_metrics = candidate_metrics
                item_score = candidate_score(
                    item_metrics,
                    target_chord_tone_ratio=target_chord_tone_ratio,
                    target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
                )
                item_gate_penalty = candidate_gate
        rhythm_metrics = rhythm_articulation_metrics(pm, bars=bars, bpm=bpm)
        context_pm = add_context(
            pm,
            item["chords"],
            bars=bars,
            bpm=bpm,
            bass_velocity_boost=context_bass_velocity_boost,
            comp_velocity_boost=context_comp_velocity_boost,
        )
        safe_case = str(item["case_label"]).replace("/", "_").replace(" ", "_")
        stem = f"candidate_{rank:02d}_{safe_case}_variant_{int(item['variant_index']):02d}_best_of"
        solo_midi_path = solo_dir / f"{stem}.mid"
        mix_midi_path = mix_midi_dir / f"{stem}_with_context.mid"
        solo_midi_path.parent.mkdir(parents=True, exist_ok=True)
        mix_midi_path.parent.mkdir(parents=True, exist_ok=True)
        if (
            (repair_bar_similarity_enabled and bool(bar_similarity_repair.get("changed")))
            or (repair_enclosure_density_enabled and bool(enclosure_density_repair.get("changed")))
            or (repair_unresolved_offbeat_enabled and bool(unresolved_offbeat_repair.get("changed")))
            or (repair_adjacent_repeats_enabled and bool(adjacent_repeat_repair.get("changed")))
            or (repair_large_leaps_enabled and bool(large_leap_repair.get("changed")))
            or (repair_motion_balance_enabled and bool(motion_balance_repair.get("changed")))
            or (repair_rhythm_articulation_enabled and bool(rhythm_articulation_repair.get("accepted")))
        ):
            pm.write(str(solo_midi_path))
        else:
            shutil.copy2(source_midi, solo_midi_path)
        context_pm.write(str(mix_midi_path))
        rendered.append(
            {
                **item,
                "objective_metrics": item_metrics,
                "score": float(item_score),
                "gate_penalty": float(item_gate_penalty),
                "bar_similarity_repair": bar_similarity_repair,
                "enclosure_density_repair": enclosure_density_repair,
                "unresolved_offbeat_repair": unresolved_offbeat_repair,
                "adjacent_repeat_repair": adjacent_repeat_repair,
                "large_leap_repair": large_leap_repair,
                "motion_balance_repair": motion_balance_repair,
                "rhythm_articulation_repair": rhythm_articulation_repair,
                "rhythm_articulation_metrics": rhythm_metrics,
                "rank": int(rank),
                "midi_path": str(solo_midi_path),
                "midi_sha256": sha256_file(solo_midi_path),
                "context_midi_path": str(mix_midi_path),
                "context_midi_sha256": sha256_file(mix_midi_path),
                "solo_audio": render_wav(render_config, solo_midi_path, solo_audio_dir / f"{stem}.wav"),
                "context_audio": render_wav(render_config, mix_midi_path, mix_audio_dir / f"{stem}_with_context.wav"),
            }
        )
    listen_first_rendered = order_rendered_for_listen_first(rendered, listen_first_mode=listen_first_mode)
    listen_first = build_listen_first_package(output_dir, listen_first_rendered)
    report = {
        "schema_version": "stage_b_midi_to_solo_bebop_language_best_of_package_v1",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_root": str(source_root),
        "source_package_globs": package_globs,
        "source_package_count": len(paths),
        "boundary": "stage_b_midi_to_solo_bebop_language_best_of_package",
        "generation": {
            "bars": int(bars),
            "bpm": float(bpm),
            "selected_count": int(selected_count),
            "max_per_case": int(max_per_case),
            "candidate_pool_count": len(rows),
            "target_chord_tone_ratio": float(target_chord_tone_ratio),
            "target_offbeat_non_chord_ratio": float(target_offbeat_non_chord_ratio),
            "selection_pool_count": len(selection_rows),
            "max_gate_penalty": max_gate_penalty,
            "max_offbeat_non_chord_ratio": max_offbeat_non_chord_ratio,
            "max_unresolved_offbeat_non_chord_ratio": max_unresolved_offbeat_non_chord_ratio,
            "max_dominant_altered_offbeat_ratio": max_dominant_altered_offbeat_ratio,
            "max_adjacent_repeat_ratio": max_adjacent_repeat_ratio,
            "max_bar_pitch_class_jaccard": max_bar_pitch_class_jaccard,
            "listen_first_mode": listen_first_mode,
            "repair_bar_similarity": bool(repair_bar_similarity_enabled),
            "repair_bar_similarity_iterations": int(repair_bar_similarity_iterations),
            "repair_enclosure_density": bool(repair_enclosure_density_enabled),
            "repair_enclosure_density_iterations": int(repair_enclosure_density_iterations),
            "repair_unresolved_offbeat": bool(repair_unresolved_offbeat_enabled),
            "repair_unresolved_offbeat_iterations": int(repair_unresolved_offbeat_iterations),
            "repair_adjacent_repeats": bool(repair_adjacent_repeats_enabled),
            "repair_adjacent_repeats_iterations": int(repair_adjacent_repeats_iterations),
            "repair_large_leaps": bool(repair_large_leaps_enabled),
            "repair_large_leaps_iterations": int(repair_large_leaps_iterations),
            "repair_motion_balance": bool(repair_motion_balance_enabled),
            "repair_motion_balance_iterations": int(repair_motion_balance_iterations),
            "target_min_step_motion_ratio": float(target_min_step_motion_ratio),
            "target_min_chromatic_step_ratio": float(target_min_chromatic_step_ratio),
            "target_max_large_leap_ratio": float(target_max_large_leap_ratio),
            "max_motion_balance_bar_pitch_class_jaccard": float(max_motion_balance_bar_pitch_class_jaccard),
            "repair_rhythm_articulation": bool(repair_rhythm_articulation_enabled),
            "min_large_leap_repair_enclosure_proxy_ratio": float(min_large_leap_repair_enclosure_proxy_ratio),
            "max_enclosure_repair_offbeat_non_chord_ratio": float(max_enclosure_repair_offbeat_non_chord_ratio),
            "context_bass_velocity_boost": int(context_bass_velocity_boost),
            "context_comp_velocity_boost": int(context_comp_velocity_boost),
            "select_after_repair": bool(select_after_repair),
            "selection_profile": selection_profile,
        },
        "renderer": render_config.renderer,
        "soundfont": render_config.soundfont,
        "aggregate": aggregate_metrics(generated_count=len(rows), rendered=rendered, listen_first=listen_first),
        "listen_first": listen_first,
        "selected_candidates": rendered,
        "quality_claimed": False,
        "model_direct_claimed": False,
        "not_proven": [
            "human_audio_preference",
            "musical_quality",
            "model_direct_quality",
        ],
        "decision": {
            "current_boundary": "stage_b_midi_to_solo_bebop_language_best_of_package",
            "next_boundary": "listening_review_input",
            "critical_user_input_required": False,
        },
    }
    write_json(output_dir / "bebop_language_best_of_package.json", report)
    write_json(output_dir / "bebop_language_best_of_package_summary.json", validate_report(report, render_config.sample_rate))
    write_text(output_dir / "bebop_language_best_of_package.md", markdown_report(report))
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a best-of bebop-language MIDI/WAV review package")
    parser.add_argument("--source_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--package_globs", default=",".join(DEFAULT_PACKAGE_GLOBS))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT / "best_of"))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--renderer", default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=float, default=124.0)
    parser.add_argument("--selected_count", type=int, default=16)
    parser.add_argument("--max_per_case", type=int, default=4)
    parser.add_argument("--target_chord_tone_ratio", type=float, default=0.78)
    parser.add_argument("--target_offbeat_non_chord_ratio", type=float, default=0.38)
    parser.add_argument("--max_gate_penalty", type=float, default=None)
    parser.add_argument("--max_offbeat_non_chord_ratio", type=float, default=None)
    parser.add_argument("--max_unresolved_offbeat_non_chord_ratio", type=float, default=None)
    parser.add_argument("--max_dominant_altered_offbeat_ratio", type=float, default=None)
    parser.add_argument("--max_adjacent_repeat_ratio", type=float, default=None)
    parser.add_argument("--max_bar_pitch_class_jaccard", type=float, default=None)
    parser.add_argument("--listen_first_mode", choices=["rank", "consonance"], default="rank")
    parser.add_argument("--repair_bar_similarity", action="store_true")
    parser.add_argument("--repair_bar_similarity_iterations", type=int, default=4)
    parser.add_argument("--repair_enclosure_density", action="store_true")
    parser.add_argument("--repair_enclosure_density_iterations", type=int, default=8)
    parser.add_argument("--repair_unresolved_offbeat", action="store_true")
    parser.add_argument("--repair_unresolved_offbeat_iterations", type=int, default=4)
    parser.add_argument("--repair_adjacent_repeats", action="store_true")
    parser.add_argument("--repair_adjacent_repeats_iterations", type=int, default=4)
    parser.add_argument("--repair_large_leaps", action="store_true")
    parser.add_argument("--repair_large_leaps_iterations", type=int, default=4)
    parser.add_argument("--repair_motion_balance", action="store_true")
    parser.add_argument("--repair_motion_balance_iterations", type=int, default=8)
    parser.add_argument("--target_min_step_motion_ratio", type=float, default=0.40)
    parser.add_argument("--target_min_chromatic_step_ratio", type=float, default=0.22)
    parser.add_argument("--target_max_large_leap_ratio", type=float, default=0.055)
    parser.add_argument("--max_motion_balance_bar_pitch_class_jaccard", type=float, default=0.70)
    parser.add_argument("--repair_rhythm_articulation", action="store_true")
    parser.add_argument("--min_large_leap_repair_enclosure_proxy_ratio", type=float, default=0.28125)
    parser.add_argument("--max_enclosure_repair_offbeat_non_chord_ratio", type=float, default=0.421875)
    parser.add_argument("--context_bass_velocity_boost", type=int, default=0)
    parser.add_argument("--context_comp_velocity_boost", type=int, default=0)
    parser.add_argument("--select_after_repair", action="store_true")
    parser.add_argument(
        "--selection_profile",
        choices=["score", "bebop_language", "bebop_stepwise_chromatic"],
        default="score",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    renderer = str(args.renderer or "")
    if not renderer:
        raise BebopLanguagePackageError("fluidsynth renderer not found")
    soundfont = resolve_soundfont(str(args.soundfont or ""))
    if not soundfont:
        raise BebopLanguagePackageError("soundfont not found")
    run_id = str(args.run_id or datetime.now(timezone.utc).strftime("manual_%Y_%m_%d_bebop_language_best_of_%H%M%S"))
    output_dir = Path(args.output_root) / run_id
    report = build_best_of_package(
        output_dir=output_dir,
        source_root=Path(args.source_root),
        package_globs=parse_globs(str(args.package_globs)),
        render_config=RenderConfig(renderer=renderer, soundfont=soundfont, sample_rate=int(args.sample_rate)),
        bars=int(args.bars),
        bpm=float(args.bpm),
        selected_count=int(args.selected_count),
        max_per_case=int(args.max_per_case),
        target_chord_tone_ratio=float(args.target_chord_tone_ratio),
        target_offbeat_non_chord_ratio=float(args.target_offbeat_non_chord_ratio),
        max_gate_penalty=args.max_gate_penalty,
        max_offbeat_non_chord_ratio=args.max_offbeat_non_chord_ratio,
        max_unresolved_offbeat_non_chord_ratio=args.max_unresolved_offbeat_non_chord_ratio,
        max_dominant_altered_offbeat_ratio=args.max_dominant_altered_offbeat_ratio,
        max_adjacent_repeat_ratio=args.max_adjacent_repeat_ratio,
        max_bar_pitch_class_jaccard=args.max_bar_pitch_class_jaccard,
        listen_first_mode=str(args.listen_first_mode),
        repair_bar_similarity_enabled=bool(args.repair_bar_similarity),
        repair_bar_similarity_iterations=int(args.repair_bar_similarity_iterations),
        repair_enclosure_density_enabled=bool(args.repair_enclosure_density),
        repair_enclosure_density_iterations=int(args.repair_enclosure_density_iterations),
        repair_unresolved_offbeat_enabled=bool(args.repair_unresolved_offbeat),
        repair_unresolved_offbeat_iterations=int(args.repair_unresolved_offbeat_iterations),
        repair_adjacent_repeats_enabled=bool(args.repair_adjacent_repeats),
        repair_adjacent_repeats_iterations=int(args.repair_adjacent_repeats_iterations),
        repair_large_leaps_enabled=bool(args.repair_large_leaps),
        repair_large_leaps_iterations=int(args.repair_large_leaps_iterations),
        min_large_leap_repair_enclosure_proxy_ratio=float(args.min_large_leap_repair_enclosure_proxy_ratio),
        repair_motion_balance_enabled=bool(args.repair_motion_balance),
        repair_motion_balance_iterations=int(args.repair_motion_balance_iterations),
        target_min_step_motion_ratio=float(args.target_min_step_motion_ratio),
        target_min_chromatic_step_ratio=float(args.target_min_chromatic_step_ratio),
        target_max_large_leap_ratio=float(args.target_max_large_leap_ratio),
        max_motion_balance_bar_pitch_class_jaccard=float(args.max_motion_balance_bar_pitch_class_jaccard),
        repair_rhythm_articulation_enabled=bool(args.repair_rhythm_articulation),
        max_enclosure_repair_offbeat_non_chord_ratio=float(args.max_enclosure_repair_offbeat_non_chord_ratio),
        context_bass_velocity_boost=int(args.context_bass_velocity_boost),
        context_comp_velocity_boost=int(args.context_comp_velocity_boost),
        select_after_repair=bool(args.select_after_repair),
        selection_profile=str(args.selection_profile),
    )
    print(json.dumps(validate_report(report, int(args.sample_rate)), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
