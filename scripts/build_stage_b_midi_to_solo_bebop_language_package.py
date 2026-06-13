"""Build a high-yield bebop language MIDI/WAV review package.

This script is a local quality-iteration package builder. It does not claim
direct model quality. It generates chord-guided bebop-style solo candidates
with strong-beat chord tones, offbeat approach notes, motif variation, and a
rendered chord/bass context for listening review.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pretty_midi

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.app.fallback import parse_chord  # noqa: E402
from scripts.render_stage_b_midi_to_solo_candidate_audio import resolve_soundfont  # noqa: E402


SCHEMA_VERSION = "stage_b_midi_to_solo_bebop_language_package_v1"
DEFAULT_SOURCE_PACKAGE = (
    ROOT_DIR
    / "outputs/music_transformer_finetune_mvp/solo_yield_rhythm_syncopation_balance_repair/"
    / "issue_1384_rhythm_syncopation_balance_repair_package/rhythm_syncopation_balance_repair_package.json"
)
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs/stage_b_midi_to_solo_bebop_language_package"

PC_NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
LOW_PITCH = 56
HIGH_PITCH = 86
DEFAULT_NON_CHORD_PROBABILITY = 0.38
DEFAULT_TARGET_CHORD_TONE_RATIO = 0.72
DEFAULT_TARGET_OFFBEAT_NON_CHORD_RATIO = 0.46
DATA_CONTOUR_INTERVAL_CELLS: tuple[tuple[int, ...], ...] = (
    (2, -4, 1, 3, 3, 4, -2),
    (4, -3, -4, -3, 1, -2, 4),
    (4, -3, 4, 4, -3, -4, 3),
    (3, 3, -6, -1, 4, -4, 7),
    (-3, 4, 1, 4, -2, 3, -1),
)


class BebopLanguagePackageError(ValueError):
    pass


@dataclass(frozen=True)
class RenderConfig:
    renderer: str
    soundfont: str
    sample_rate: int


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise BebopLanguagePackageError(f"json not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def wav_meta(path: Path) -> dict[str, Any]:
    with wave.open(str(path), "rb") as handle:
        channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        sample_rate = handle.getframerate()
        frame_count = handle.getnframes()
    return {
        "path": str(path),
        "exists": path.exists(),
        "size_bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
        "channels": int(channels),
        "sample_width_bytes": int(sample_width),
        "sample_rate": int(sample_rate),
        "frame_count": int(frame_count),
        "duration_seconds": float(frame_count / sample_rate) if sample_rate else 0.0,
    }


def pc_name(pitch_or_pc: int) -> str:
    return PC_NAMES[int(pitch_or_pc) % 12]


def chord_kind(chord: str) -> str:
    lowered = chord.lower()
    if "maj7" in lowered or "ma7" in lowered:
        return "maj7"
    if "m7" in lowered or "min" in lowered:
        return "m7"
    if "dim" in lowered or "m7b5" in lowered:
        return "m7b5"
    return "7"


def chord_pitch_classes(chord: str) -> set[int]:
    root, intervals = parse_chord(chord)
    return {(root + interval) % 12 for interval in intervals}


def guide_intervals(chord: str) -> tuple[int, int]:
    kind = chord_kind(chord)
    if kind == "m7":
        return 3, 10
    if kind == "maj7":
        return 4, 11
    if kind == "m7b5":
        return 3, 10
    return 4, 10


def scale_intervals(chord: str) -> list[int]:
    kind = chord_kind(chord)
    if kind == "m7":
        return [0, 2, 3, 5, 7, 9, 10]
    if kind == "maj7":
        return [0, 2, 4, 5, 7, 9, 11]
    if kind == "m7b5":
        return [0, 2, 3, 5, 6, 8, 10]
    return [0, 1, 2, 4, 5, 7, 9, 10, 11]


def chord_extension_intervals(chord: str) -> list[int]:
    kind = chord_kind(chord)
    if kind == "m7":
        return [9, 14, 17]
    if kind == "maj7":
        return [9, 14, 21]
    if kind == "m7b5":
        return [13, 17]
    return [13, 14, 15, 20, 21]


def pitches_for_pcs(pcs: set[int], low: int = LOW_PITCH, high: int = HIGH_PITCH) -> list[int]:
    return [pitch for pitch in range(low, high + 1) if pitch % 12 in pcs]


def pitches_for_intervals(chord: str, intervals: Sequence[int], low: int = LOW_PITCH, high: int = HIGH_PITCH) -> list[int]:
    root, _ = parse_chord(chord)
    pcs = {(root + interval) % 12 for interval in intervals}
    return pitches_for_pcs(pcs, low, high)


def nearest(values: Sequence[int], reference: int) -> int:
    if not values:
        return int(reference)
    return int(min(values, key=lambda value: (abs(int(value) - int(reference)), int(value))))


def nearest_chord_tone(chord: str, reference: int, *, avoid: int | None = None) -> int:
    choices = pitches_for_pcs(chord_pitch_classes(chord))
    if avoid is not None:
        filtered = [pitch for pitch in choices if int(pitch) != int(avoid)]
        if filtered:
            choices = filtered
    return nearest(choices, reference)


def nearest_guide_tone(chord: str, reference: int, *, selector: int = 0) -> int:
    intervals = guide_intervals(chord)
    return nearest(pitches_for_intervals(chord, [intervals[selector % len(intervals)]]), reference)


def nearest_scale_tone(chord: str, reference: int, *, avoid_chord: bool) -> int:
    root, _ = parse_chord(chord)
    pcs = {(root + interval) % 12 for interval in scale_intervals(chord)}
    choices = pitches_for_pcs(pcs)
    if avoid_chord:
        chord_pcs = chord_pitch_classes(chord)
        non_chord = [pitch for pitch in choices if pitch % 12 not in chord_pcs]
        if non_chord:
            choices = non_chord
    return nearest(choices, reference)


def chromatic_approach(target: int, previous: int, chord: str, rng: random.Random) -> int:
    candidates: list[int] = []
    if previous <= target:
        candidates.extend([target - 1, target + 1, target - 2])
    else:
        candidates.extend([target + 1, target - 1, target + 2])
    candidates.extend(
        [
            nearest_scale_tone(chord, target - 2, avoid_chord=True),
            nearest_scale_tone(chord, target + 2, avoid_chord=True),
        ]
    )
    valid = [pitch for pitch in candidates if LOW_PITCH <= pitch <= HIGH_PITCH and pitch != previous]
    if not valid:
        valid = [nearest_scale_tone(chord, target, avoid_chord=True)]
    non_chord = [pitch for pitch in valid if pitch % 12 not in chord_pitch_classes(chord)]
    pool = non_chord or valid
    return int(min(pool, key=lambda pitch: (abs(pitch - target), rng.random())))


def enclosure_note(target: int, previous: int, chord: str, rng: random.Random) -> int:
    candidates = [target + 1, target - 1, target + 2, target - 2]
    valid = [pitch for pitch in candidates if LOW_PITCH <= pitch <= HIGH_PITCH and pitch != previous]
    non_chord = [pitch for pitch in valid if pitch % 12 not in chord_pitch_classes(chord)]
    pool = non_chord or valid
    if not pool:
        return chromatic_approach(target, previous, chord, rng)
    return int(pool[rng.randrange(len(pool))])


def altered_approach_note(target: int, previous: int, chord: str, rng: random.Random) -> int:
    root, _ = parse_chord(chord)
    altered_pcs = {(root + interval) % 12 for interval in (1, 3, 6, 8)}
    choices = pitches_for_pcs(altered_pcs)
    valid = [pitch for pitch in choices if LOW_PITCH <= pitch <= HIGH_PITCH and pitch != previous]
    close = [pitch for pitch in valid if abs(pitch - target) <= 4]
    pool = close or valid
    if not pool:
        return chromatic_approach(target, previous, chord, rng)
    return int(min(pool, key=lambda pitch: (abs(pitch - target), abs(pitch - previous), rng.random())))


def offbeat_note(
    chord: str,
    target: int,
    previous: int,
    rng: random.Random,
    *,
    next_chord: str | None = None,
    non_chord_probability: float = 0.48,
) -> int:
    """Choose an offbeat note without making every offbeat chromatic.

    Bebop lines need approach notes, but the current listening feedback shows
    that too many non-chord offbeats read as outside-soloing. This keeps
    chromatic approach motion while allowing chord/scale support notes.
    """

    active_chord = next_chord or chord
    roll = rng.random()
    if roll < non_chord_probability:
        if chord_kind(active_chord) == "7" and rng.random() < 0.42:
            return altered_approach_note(target, previous, active_chord, rng)
        return chromatic_approach(target, previous, active_chord, rng)
    if roll < non_chord_probability + 0.30:
        return nearest_chord_tone(chord, previous + (1 if target >= previous else -1) * rng.choice([2, 3, 4]))
    return nearest_scale_tone(chord, previous + (1 if target >= previous else -1) * rng.choice([1, 2, 3]), avoid_chord=False)


def interval_direction(left: int, right: int) -> int:
    if right > left:
        return 1
    if right < left:
        return -1
    return 0


def avoid_large_leap(pitch: int, previous: int, chord: str) -> int:
    if abs(pitch - previous) <= 10:
        return int(pitch)
    direction = 1 if pitch > previous else -1
    return nearest_chord_tone(chord, previous + direction * 5, avoid=previous)


def build_targets(chords: list[str], bars: int, seed: int) -> list[int]:
    rng = random.Random(seed)
    targets: list[int] = []
    reference = 64 + (seed % 8)
    for bar_index in range(bars + 1):
        chord = chords[bar_index % len(chords)]
        selector = (bar_index + seed) % 2
        guide = nearest_guide_tone(chord, reference, selector=selector)
        root_target = nearest_chord_tone(chord, reference)
        if targets and abs(guide - targets[-1]) > 8:
            guide = nearest([guide, root_target, nearest_guide_tone(chord, targets[-1], selector=selector)], targets[-1])
        target = guide
        if targets:
            target = avoid_large_leap(target, targets[-1], chord)
        targets.append(int(target))
        reference = target + rng.choice([-5, -4, -3, 3, 4, 5])
    return targets


def arpeggio_cell(chord: str, start: int, direction: int) -> list[int]:
    root, intervals = parse_chord(chord)
    usable = list(intervals)
    if 14 not in usable:
        usable.append(14)
    pitches = pitches_for_intervals(chord, usable)
    ordered = sorted(pitches, key=lambda pitch: pitch if direction > 0 else -pitch)
    near_index = min(range(len(ordered)), key=lambda index: abs(ordered[index] - start)) if ordered else 0
    cell: list[int] = []
    index = near_index
    step = 1 if direction > 0 else -1
    while len(cell) < 4 and ordered:
        cell.append(int(ordered[index % len(ordered)]))
        index += step
    return cell


def scalar_fragment(chord: str, start: int, direction: int) -> list[int]:
    root, _ = parse_chord(chord)
    pcs = {(root + interval) % 12 for interval in scale_intervals(chord)}
    scale = pitches_for_pcs(pcs)
    scale = sorted(scale, reverse=direction < 0)
    start_index = min(range(len(scale)), key=lambda index: abs(scale[index] - start)) if scale else 0
    return [int(scale[(start_index + offset) % len(scale)]) for offset in range(4)] if scale else [start] * 4


def contour_cell_line(
    chord: str,
    next_chord: str,
    start_target: int,
    next_target: int,
    *,
    seed: int,
    bar_index: int,
    rng: random.Random,
    non_chord_probability: float,
) -> list[int]:
    cell = DATA_CONTOUR_INTERVAL_CELLS[(seed + bar_index) % len(DATA_CONTOUR_INTERVAL_CELLS)]
    desired = [nearest_chord_tone(chord, start_target)]
    for interval in cell:
        desired.append(max(LOW_PITCH, min(HIGH_PITCH, int(desired[-1] + interval))))

    line: list[int] = []
    for index, reference in enumerate(desired[:8]):
        active_chord = chord if index < 7 else next_chord
        if index % 2 == 0:
            pitch = nearest_chord_tone(active_chord, int(reference), avoid=line[-1] if line else None)
        else:
            target_chord = next_chord if index == 7 else chord
            target_reference = desired[index + 1] if index + 1 < len(desired) else next_target
            target = nearest_chord_tone(target_chord, int(target_reference))
            if rng.random() < non_chord_probability:
                pitch = chromatic_approach(target, line[-1], active_chord, rng)
            else:
                pitch = nearest_scale_tone(active_chord, int(reference), avoid_chord=False)
        if line:
            pitch = avoid_large_leap(int(pitch), line[-1], active_chord)
        line.append(max(LOW_PITCH, min(HIGH_PITCH, int(pitch))))

    line[6] = nearest_chord_tone(chord, int(line[6]), avoid=line[5])
    line[7] = offbeat_note(
        next_chord,
        next_target,
        line[6],
        rng,
        next_chord=next_chord,
        non_chord_probability=max(0.20, non_chord_probability - 0.04),
    )
    return line


def build_bar_line(
    chord: str,
    next_chord: str,
    start_target: int,
    next_target: int,
    *,
    bar_index: int,
    seed: int,
    non_chord_probability: float,
) -> list[int]:
    rng = random.Random(seed + bar_index * 7919)
    direction = 1 if (bar_index + seed) % 2 == 0 else -1
    pattern = (seed + bar_index) % 7

    beat1 = nearest_chord_tone(chord, start_target)
    beat2 = nearest_chord_tone(chord, beat1 + direction * rng.choice([3, 4, 5, 7]), avoid=beat1)
    beat2 = avoid_large_leap(beat2, beat1, chord)
    beat3 = nearest_chord_tone(chord, beat2 - direction * rng.choice([3, 4, 5]), avoid=beat2)
    beat3 = avoid_large_leap(beat3, beat2, chord)
    beat4 = nearest_chord_tone(chord, next_target + (-2 if next_target > beat3 else 2), avoid=beat3)
    beat4 = avoid_large_leap(beat4, beat3, chord)

    if pattern == 0:
        line = [
            beat1,
            offbeat_note(chord, beat2, beat1, rng, non_chord_probability=non_chord_probability),
            beat2,
            offbeat_note(chord, beat3, beat2, rng, non_chord_probability=non_chord_probability),
            beat3,
            offbeat_note(chord, beat4, beat3, rng, non_chord_probability=non_chord_probability),
            beat4,
            offbeat_note(
                chord,
                next_target,
                beat4,
                rng,
                next_chord=next_chord,
                non_chord_probability=non_chord_probability,
            ),
        ]
    elif pattern == 1:
        cell = arpeggio_cell(chord, beat1, direction)
        beat1 = nearest_chord_tone(chord, cell[0])
        beat2 = nearest_chord_tone(chord, cell[2])
        beat3 = nearest_chord_tone(chord, next_target - direction * 4)
        beat4 = nearest_chord_tone(chord, next_target - direction * 2)
        line = [
            beat1,
            nearest_scale_tone(chord, cell[1], avoid_chord=False),
            beat2,
            offbeat_note(chord, beat3, beat2, rng, non_chord_probability=non_chord_probability),
            beat3,
            offbeat_note(chord, beat4, beat3, rng, non_chord_probability=non_chord_probability),
            beat4,
            offbeat_note(
                chord,
                next_target,
                beat4,
                rng,
                next_chord=next_chord,
                non_chord_probability=non_chord_probability,
            ),
        ]
    elif pattern == 2:
        scalar = scalar_fragment(chord, beat1, direction)
        line = [
            nearest_chord_tone(chord, scalar[0]),
            offbeat_note(
                chord,
                scalar[1],
                scalar[0],
                rng,
                non_chord_probability=max(0.20, non_chord_probability - 0.12),
            ),
            nearest_chord_tone(chord, scalar[2]),
            nearest_scale_tone(chord, scalar[3], avoid_chord=False),
            beat3,
            offbeat_note(chord, beat4, beat3, rng, non_chord_probability=non_chord_probability),
            beat4,
            offbeat_note(
                chord,
                next_target,
                beat4,
                rng,
                next_chord=next_chord,
                non_chord_probability=non_chord_probability,
            ),
        ]
    elif pattern == 3:
        extension = nearest(pitches_for_intervals(chord, chord_extension_intervals(chord)), beat2 + direction * 2)
        line = [
            beat1,
            offbeat_note(chord, beat2, beat1, rng, non_chord_probability=non_chord_probability),
            beat2,
            nearest_scale_tone(chord, extension, avoid_chord=False),
            beat3,
            offbeat_note(chord, beat4, beat3, rng, non_chord_probability=non_chord_probability),
            beat4,
            offbeat_note(
                chord,
                next_target,
                beat4,
                rng,
                next_chord=next_chord,
                non_chord_probability=non_chord_probability,
            ),
        ]
    elif pattern == 4:
        line = [
            beat1,
            offbeat_note(chord, beat2, beat1, rng, non_chord_probability=non_chord_probability),
            beat2,
            offbeat_note(chord, beat3, beat2, rng, non_chord_probability=non_chord_probability),
            beat3,
            nearest_scale_tone(chord, beat3 - direction * 2, avoid_chord=True),
            beat4,
            offbeat_note(
                chord,
                next_target,
                beat4,
                rng,
                next_chord=next_chord,
                non_chord_probability=non_chord_probability,
            ),
        ]
    else:
        line = contour_cell_line(
            chord,
            next_chord,
            beat1,
            next_target,
            seed=seed,
            bar_index=bar_index,
            rng=rng,
            non_chord_probability=non_chord_probability,
        )

    repaired: list[int] = []
    for index, pitch in enumerate(line):
        active_chord = chord if index < 7 else next_chord
        if repaired:
            pitch = avoid_large_leap(int(pitch), repaired[-1], active_chord)
            if int(pitch) == repaired[-1]:
                direction_hint = direction if direction != 0 else 1
                if index % 2 == 0:
                    pitch = nearest_chord_tone(active_chord, repaired[-1] + direction_hint * 3, avoid=repaired[-1])
                else:
                    target = line[index + 1] if index + 1 < len(line) else next_target
                    pitch = offbeat_note(
                        active_chord,
                        int(target),
                        repaired[-1],
                        rng,
                        non_chord_probability=max(0.20, non_chord_probability - 0.08),
                    )
        if len(repaired) >= 3:
            would_repeat_two_note_cycle = (
                repaired[-3] == repaired[-1]
                and repaired[-2] == int(pitch)
                and repaired[-3] != repaired[-2]
            )
            if would_repeat_two_note_cycle:
                direction_hint = direction if direction != 0 else 1
                if index % 2 == 0:
                    pitch = nearest_chord_tone(active_chord, int(pitch) + direction_hint * 3, avoid=repaired[-1])
                    if int(pitch) == repaired[-2]:
                        pitch = nearest_chord_tone(active_chord, int(pitch) + direction_hint * 4, avoid=repaired[-1])
                else:
                    target = line[index + 1] if index + 1 < len(line) else next_target
                    pitch = offbeat_note(
                        active_chord,
                        int(target),
                        repaired[-1],
                        rng,
                        non_chord_probability=max(0.20, non_chord_probability - 0.06),
                    )
                    if int(pitch) == repaired[-2]:
                        pitch = nearest_scale_tone(
                            active_chord,
                            int(pitch) + direction_hint * 2,
                            avoid_chord=True,
                        )
        repaired.append(max(LOW_PITCH, min(HIGH_PITCH, int(pitch))))
    if len(repaired) == 8 and repaired[:4] == repaired[4:]:
        repaired[5] = offbeat_note(
            chord,
            repaired[6],
            repaired[4],
            rng,
            non_chord_probability=max(0.20, non_chord_probability - 0.06),
        )
        if repaired[5] == repaired[1]:
            repaired[5] = nearest_scale_tone(chord, repaired[5] + direction * 2, avoid_chord=True)
        repaired[7] = offbeat_note(
            next_chord,
            next_target,
            repaired[6],
            rng,
            next_chord=next_chord,
            non_chord_probability=max(0.20, non_chord_probability - 0.06),
        )
        if repaired[7] == repaired[3]:
            repaired[7] = nearest_scale_tone(next_chord, repaired[7] - direction * 2, avoid_chord=True)
    return repaired


def build_swing_starts(bars: int, bpm: float) -> list[float]:
    seconds_per_beat = 60.0 / float(bpm)
    starts: list[float] = []
    for bar_index in range(bars):
        bar_start = bar_index * 4 * seconds_per_beat
        for beat_index in range(4):
            starts.append(bar_start + beat_index * seconds_per_beat)
            starts.append(bar_start + (beat_index + 2 / 3) * seconds_per_beat)
    return starts


def build_bebop_candidate(
    chords: list[str],
    *,
    seed: int,
    bars: int,
    bpm: float,
    non_chord_probability: float,
) -> tuple[pretty_midi.PrettyMIDI, dict[str, Any]]:
    targets = build_targets(chords, bars, seed)
    notes_by_slot: list[int] = []
    for bar_index in range(bars):
        chord = chords[bar_index % len(chords)]
        next_chord = chords[(bar_index + 1) % len(chords)]
        notes_by_slot.extend(
            build_bar_line(
                chord,
                next_chord,
                targets[bar_index],
                targets[bar_index + 1],
                bar_index=bar_index,
                seed=seed,
                non_chord_probability=non_chord_probability,
            )
        )

    starts = build_swing_starts(bars, bpm)
    seconds_per_beat = 60.0 / float(bpm)
    total_duration = bars * 4 * seconds_per_beat
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    solo = pretty_midi.Instrument(program=0, is_drum=False, name="bebop_language_solo")

    for index, (pitch, start) in enumerate(zip(notes_by_slot, starts)):
        slot = index % 8
        next_start = starts[index + 1] if index + 1 < len(starts) else total_duration
        duration = max(0.07, (next_start - start) * (0.86 if slot % 2 == 0 else 0.78))
        velocity = 96 if slot in (0, 2, 4, 6) else 76
        if slot == 0:
            velocity = 104
        solo.notes.append(
            pretty_midi.Note(
                velocity=int(velocity),
                pitch=max(21, min(108, int(pitch))),
                start=float(start),
                end=min(float(total_duration), float(start + duration)),
            )
        )

    # Replace the last audible note with a clear chord-tone cadence.
    if solo.notes:
        final_note = solo.notes[-1]
        final_chord = chords[(bars - 1) % len(chords)]
        final_note.pitch = nearest_chord_tone(final_chord, int(final_note.pitch))
        final_note.velocity = max(int(final_note.velocity), 98)
        final_note.end = float(total_duration)

    solo.notes.sort(key=lambda note: (float(note.start), int(note.pitch), float(note.end)))
    pm.instruments.append(solo)
    return pm, {
        "seed": int(seed),
        "bars": int(bars),
        "bpm": float(bpm),
        "targets": targets,
        "slot_pitch_count": len(notes_by_slot),
        "non_chord_probability": float(non_chord_probability),
    }


def chord_for_time(chords: list[str], start: float, *, bars: int, bpm: float) -> str:
    total_duration = bars * 4 * (60.0 / float(bpm))
    ratio = max(0.0, min(0.999999, (float(start) + 0.0001) / max(1e-6, total_duration)))
    bar_index = int(ratio * bars)
    return chords[bar_index % len(chords)]


def direction_change_ratio(pitches: list[int]) -> float:
    if len(pitches) < 3:
        return 0.0
    signs: list[int] = []
    for left, right in zip(pitches, pitches[1:]):
        signs.append(interval_direction(left, right))
    signs = [sign for sign in signs if sign != 0]
    if len(signs) < 2:
        return 0.0
    return sum(1 for left, right in zip(signs, signs[1:]) if left != right) / max(1, len(signs) - 1)


def adjacent_repeat_ratio(pitches: list[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    return sum(1 for left, right in zip(pitches, pitches[1:]) if left == right) / (len(pitches) - 1)


def chromatic_step_ratio(pitches: list[int]) -> float:
    if len(pitches) < 2:
        return 0.0
    intervals = [abs(right - left) for left, right in zip(pitches, pitches[1:])]
    return sum(1 for interval in intervals if interval == 1) / max(1, len(intervals))


def two_note_cycle_ratio(pitches: list[int]) -> float:
    if len(pitches) < 4:
        return 0.0
    cycle_count = 0
    window_count = 0
    for index in range(len(pitches) - 3):
        left_a, left_b, right_a, right_b = pitches[index : index + 4]
        window_count += 1
        if left_a == right_a and left_b == right_b and left_a != left_b:
            cycle_count += 1
    return cycle_count / max(1, window_count)


def bar_unique_pitch_counts(pitches: list[int], *, slots_per_bar: int = 8) -> list[int]:
    return [
        len(set(pitches[index : index + slots_per_bar]))
        for index in range(0, len(pitches), slots_per_bar)
        if pitches[index : index + slots_per_bar]
    ]


def bar_half_repeat_ratio(pitches: list[int], *, slots_per_bar: int = 8) -> float:
    bars = [
        pitches[index : index + slots_per_bar]
        for index in range(0, len(pitches), slots_per_bar)
        if len(pitches[index : index + slots_per_bar]) == slots_per_bar
    ]
    if not bars:
        return 0.0
    half = slots_per_bar // 2
    repeat_count = sum(1 for bar in bars if bar[:half] == bar[half:])
    return repeat_count / len(bars)


def bar_pitch_class_similarity_metrics(pitches: list[int], *, slots_per_bar: int = 8) -> dict[str, float]:
    bars = [
        pitches[index : index + slots_per_bar]
        for index in range(0, len(pitches), slots_per_bar)
        if len(pitches[index : index + slots_per_bar]) == slots_per_bar
    ]
    if len(bars) < 2:
        return {
            "max_bar_pitch_class_jaccard": 0.0,
            "avg_bar_pitch_class_jaccard": 0.0,
            "bar_pitch_shape_repeat_ratio": 0.0,
        }
    similarities: list[float] = []
    repeated_shapes = 0
    comparisons = 0
    for left_index, left in enumerate(bars):
        left_pc = {pitch % 12 for pitch in left}
        left_shape = tuple((pitch - left[0]) for pitch in left)
        for right in bars[left_index + 1 :]:
            right_pc = {pitch % 12 for pitch in right}
            union = left_pc | right_pc
            similarities.append(len(left_pc & right_pc) / max(1, len(union)))
            right_shape = tuple((pitch - right[0]) for pitch in right)
            comparisons += 1
            repeated_shapes += int(left_shape == right_shape)
    return {
        "max_bar_pitch_class_jaccard": max(similarities or [0.0]),
        "avg_bar_pitch_class_jaccard": mean(similarities) if similarities else 0.0,
        "bar_pitch_shape_repeat_ratio": repeated_shapes / max(1, comparisons),
    }


def solo_notes(pm: pretty_midi.PrettyMIDI) -> list[pretty_midi.Note]:
    notes: list[pretty_midi.Note] = []
    for instrument in pm.instruments:
        if instrument.is_drum:
            continue
        if instrument.name == "bebop_language_solo":
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda note: (float(note.start), int(note.pitch), float(note.end)))


def objective_metrics(pm: pretty_midi.PrettyMIDI, chords: list[str], *, bars: int, bpm: float) -> dict[str, Any]:
    notes = solo_notes(pm)
    if not notes:
        return {
            "note_count": 0,
            "chord_tone_ratio": 0.0,
            "strong_beat_chord_tone_ratio": 0.0,
            "offbeat_non_chord_ratio": 0.0,
            "final_landing_is_chord_tone": False,
        }

    seconds_per_beat = 60.0 / float(bpm)
    chord_tone_count = 0
    strong_total = 0
    strong_chord_count = 0
    offbeat_total = 0
    offbeat_non_chord_count = 0
    offbeat_non_chord_resolved_count = 0
    offbeat_non_chord_unresolved_count = 0
    enclosure_proxy_count = 0
    strong_chord_tone_count_for_enclosure = 0
    dominant_offbeat_total = 0
    dominant_altered_offbeat_count = 0
    non_chord_examples: list[dict[str, Any]] = []
    pitches = [int(note.pitch) for note in notes]
    durations = [max(0.0, float(note.end) - float(note.start)) for note in notes]
    gaps = [max(0.0, float(notes[index].start) - float(notes[index - 1].start)) for index in range(1, len(notes))]
    bar_uniques = bar_unique_pitch_counts(pitches)
    bar_similarity = bar_pitch_class_similarity_metrics(pitches)

    for note_index, note in enumerate(notes):
        chord = chord_for_time(chords, float(note.start), bars=bars, bpm=bpm)
        is_chord_tone = int(note.pitch) % 12 in chord_pitch_classes(chord)
        chord_tone_count += int(is_chord_tone)
        beat_position = float(note.start) / seconds_per_beat
        in_bar_beat = beat_position % 4
        is_strong = abs(in_bar_beat - round(in_bar_beat)) < 0.05
        if is_strong:
            strong_total += 1
            strong_chord_count += int(is_chord_tone)
            if is_chord_tone:
                strong_chord_tone_count_for_enclosure += 1
                if note_index >= 2:
                    left = int(notes[note_index - 2].pitch)
                    right = int(notes[note_index - 1].pitch)
                    target = int(note.pitch)
                    brackets_target = (left - target) * (right - target) < 0
                    close_enough = abs(left - target) <= 4 and abs(right - target) <= 4
                    if brackets_target and close_enough:
                        enclosure_proxy_count += 1
        else:
            offbeat_total += 1
            offbeat_non_chord_count += int(not is_chord_tone)
            if chord_kind(chord) == "7":
                dominant_offbeat_total += 1
                root, _ = parse_chord(chord)
                altered_pcs = {(root + interval) % 12 for interval in (1, 3, 6, 8)}
                dominant_altered_offbeat_count += int(int(note.pitch) % 12 in altered_pcs)
            if not is_chord_tone:
                next_note = notes[note_index + 1] if note_index + 1 < len(notes) else None
                if next_note is not None:
                    next_chord = chord_for_time(chords, float(next_note.start), bars=bars, bpm=bpm)
                    next_is_chord_tone = int(next_note.pitch) % 12 in chord_pitch_classes(next_chord)
                    resolves_by_step = abs(int(next_note.pitch) - int(note.pitch)) <= 2
                    if next_is_chord_tone and resolves_by_step:
                        offbeat_non_chord_resolved_count += 1
                    else:
                        offbeat_non_chord_unresolved_count += 1
                else:
                    offbeat_non_chord_unresolved_count += 1
        if not is_chord_tone and len(non_chord_examples) < 12:
            non_chord_examples.append(
                {
                    "start_seconds": round(float(note.start), 4),
                    "pitch": int(note.pitch),
                    "pitch_class": pc_name(int(note.pitch)),
                    "active_chord": chord,
                }
            )

    final_note = notes[-1]
    final_chord = chord_for_time(chords, float(final_note.start), bars=bars, bpm=bpm)
    final_ok = int(final_note.pitch) % 12 in chord_pitch_classes(final_chord)
    intervals = [abs(right - left) for left, right in zip(pitches, pitches[1:])]
    step_motion_ratio = sum(1 for interval in intervals if interval <= 2) / max(1, len(intervals))
    third_fourth_motion_ratio = sum(1 for interval in intervals if 3 <= interval <= 5) / max(1, len(intervals))
    large_leap_ratio = sum(1 for interval in intervals if interval >= 6) / max(1, len(intervals))
    return {
        "note_count": len(notes),
        "duration_seconds": max(float(note.end) for note in notes) - min(float(note.start) for note in notes),
        "unique_pitch_count": len(set(pitches)),
        "unique_pitch_class_count": len({pitch % 12 for pitch in pitches}),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "pitch_span": max(pitches) - min(pitches),
        "avg_note_duration_seconds": mean(durations),
        "max_gap_seconds": max(gaps or [0.0]),
        "avg_gap_seconds": mean(gaps) if gaps else 0.0,
        "max_abs_interval": max(intervals or [0]),
        "avg_abs_interval": mean(intervals) if intervals else 0.0,
        "step_motion_ratio": step_motion_ratio,
        "third_fourth_motion_ratio": third_fourth_motion_ratio,
        "large_leap_ratio": large_leap_ratio,
        "direction_change_ratio": direction_change_ratio(pitches),
        "adjacent_repeat_ratio": adjacent_repeat_ratio(pitches),
        "chromatic_step_ratio": chromatic_step_ratio(pitches),
        "two_note_cycle_ratio": two_note_cycle_ratio(pitches),
        "bar_half_repeat_ratio": bar_half_repeat_ratio(pitches),
        "max_bar_pitch_class_jaccard": bar_similarity["max_bar_pitch_class_jaccard"],
        "avg_bar_pitch_class_jaccard": bar_similarity["avg_bar_pitch_class_jaccard"],
        "bar_pitch_shape_repeat_ratio": bar_similarity["bar_pitch_shape_repeat_ratio"],
        "min_bar_unique_pitch_count": min(bar_uniques or [0]),
        "avg_bar_unique_pitch_count": mean(bar_uniques) if bar_uniques else 0.0,
        "chord_tone_ratio": chord_tone_count / max(1, len(notes)),
        "tension_ratio": 1.0 - (chord_tone_count / max(1, len(notes))),
        "strong_beat_chord_tone_ratio": strong_chord_count / max(1, strong_total),
        "offbeat_non_chord_ratio": offbeat_non_chord_count / max(1, offbeat_total),
        "offbeat_non_chord_resolution_ratio": offbeat_non_chord_resolved_count / max(1, offbeat_non_chord_count),
        "offbeat_unresolved_non_chord_ratio": offbeat_non_chord_unresolved_count / max(1, offbeat_total),
        "enclosure_proxy_ratio": enclosure_proxy_count / max(1, strong_chord_tone_count_for_enclosure),
        "enclosure_proxy_count": int(enclosure_proxy_count),
        "dominant_altered_offbeat_ratio": dominant_altered_offbeat_count / max(1, dominant_offbeat_total),
        "dominant_altered_offbeat_count": int(dominant_altered_offbeat_count),
        "final_landing_chord": final_chord,
        "final_landing_pitch": int(final_note.pitch),
        "final_landing_is_chord_tone": final_ok,
        "non_chord_examples": non_chord_examples,
    }


def candidate_score(
    metrics: dict[str, Any],
    *,
    target_chord_tone_ratio: float = DEFAULT_TARGET_CHORD_TONE_RATIO,
    target_offbeat_non_chord_ratio: float = DEFAULT_TARGET_OFFBEAT_NON_CHORD_RATIO,
) -> float:
    chord_tone = float(metrics.get("chord_tone_ratio") or 0.0)
    strong = float(metrics.get("strong_beat_chord_tone_ratio") or 0.0)
    offbeat_non = float(metrics.get("offbeat_non_chord_ratio") or 0.0)
    direction = float(metrics.get("direction_change_ratio") or 0.0)
    repeat = float(metrics.get("adjacent_repeat_ratio") or 0.0)
    resolution = float(metrics.get("offbeat_non_chord_resolution_ratio") or 0.0)
    unresolved = float(metrics.get("offbeat_unresolved_non_chord_ratio") or 0.0)
    chromatic = float(metrics.get("chromatic_step_ratio") or 0.0)
    enclosure = float(metrics.get("enclosure_proxy_ratio") or 0.0)
    dominant_altered = float(metrics.get("dominant_altered_offbeat_ratio") or 0.0)
    cycle = float(metrics.get("two_note_cycle_ratio") or 0.0)
    half_repeat = float(metrics.get("bar_half_repeat_ratio") or 0.0)
    max_bar_similarity = float(metrics.get("max_bar_pitch_class_jaccard") or 0.0)
    shape_repeat = float(metrics.get("bar_pitch_shape_repeat_ratio") or 0.0)
    min_bar_unique = int(metrics.get("min_bar_unique_pitch_count") or 0)
    max_interval = int(metrics.get("max_abs_interval") or 0)
    unique_pitch = int(metrics.get("unique_pitch_count") or 0)
    unique_pc = int(metrics.get("unique_pitch_class_count") or 0)
    max_gap = float(metrics.get("max_gap_seconds") or 0.0)
    step_motion = float(metrics.get("step_motion_ratio") or 0.0)
    third_fourth_motion = float(metrics.get("third_fourth_motion_ratio") or 0.0)
    large_leap = float(metrics.get("large_leap_ratio") or 0.0)
    final_ok = bool(metrics.get("final_landing_is_chord_tone", False))
    return (
        abs(chord_tone - float(target_chord_tone_ratio)) * 2.2
        + abs(offbeat_non - float(target_offbeat_non_chord_ratio)) * 1.5
        + max(0.0, 1.0 - strong) * 4.0
        + max(0.0, 0.36 - direction) * 1.2
        + repeat * 1.6
        + max(0.0, 0.86 - resolution) * 2.4
        + unresolved * 4.5
        + max(0.0, 0.18 - chromatic) * 0.9
        + max(0.0, 0.04 - enclosure) * 0.6
        + max(0.0, 0.14 - dominant_altered) * 0.7
        + cycle * 1.5
        + half_repeat * 1.2
        + max(0.0, max_bar_similarity - 0.72) * 1.4
        + shape_repeat * 1.4
        + max(0, 4 - min_bar_unique) * 0.3
        + max(0, max_interval - 12) * 0.1
        + max(0, 14 - unique_pitch) * 0.04
        + max(0, 7 - unique_pc) * 0.2
        + max(0.0, max_gap - 0.4) * 0.8
        + max(0.0, step_motion - 0.48) * 0.8
        + max(0.0, 0.48 - third_fourth_motion) * 0.8
        + max(0.0, large_leap - 0.10) * 0.8
        + (0.0 if final_ok else 3.0)
    )


def candidate_gate_penalty(metrics: dict[str, Any]) -> float:
    """Penalty for objective defects that should sort before fine-grained score."""

    penalty = 0.0
    if not bool(metrics.get("final_landing_is_chord_tone", False)):
        penalty += 10.0
    penalty += max(0.0, 0.99 - float(metrics.get("strong_beat_chord_tone_ratio") or 0.0)) * 8.0
    penalty += max(0.0, 0.84 - float(metrics.get("offbeat_non_chord_resolution_ratio") or 0.0)) * 4.0
    penalty += max(0.0, float(metrics.get("offbeat_unresolved_non_chord_ratio") or 0.0) - 0.07) * 6.0
    penalty += max(0.0, float(metrics.get("two_note_cycle_ratio") or 0.0) - 0.02) * 5.0
    penalty += float(metrics.get("bar_half_repeat_ratio") or 0.0) * 6.0
    penalty += max(0.0, float(metrics.get("max_bar_pitch_class_jaccard") or 0.0) - 0.86) * 2.0
    penalty += float(metrics.get("bar_pitch_shape_repeat_ratio") or 0.0) * 4.0
    penalty += max(0, 4 - int(metrics.get("min_bar_unique_pitch_count") or 0)) * 2.0
    return float(penalty)


def generated_sort_key(item: dict[str, Any]) -> tuple[float, float, str, int]:
    return (
        float(item["score"]),
        float(item.get("gate_penalty") or 0.0),
        str(item["case_label"]),
        int(item["variant_index"]),
    )


def add_context(pm: pretty_midi.PrettyMIDI, chords: list[str], *, bars: int, bpm: float) -> pretty_midi.PrettyMIDI:
    out = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    for instrument in pm.instruments:
        copied = pretty_midi.Instrument(program=instrument.program, is_drum=instrument.is_drum, name=instrument.name)
        for note in instrument.notes:
            copied.notes.append(
                pretty_midi.Note(
                    velocity=int(note.velocity),
                    pitch=int(note.pitch),
                    start=float(note.start),
                    end=float(note.end),
                )
            )
        out.instruments.append(copied)

    bass = pretty_midi.Instrument(program=32, is_drum=False, name="review_root_fifth_bass")
    comp = pretty_midi.Instrument(program=0, is_drum=False, name="review_guide_tone_comp")
    seconds_per_beat = 60.0 / float(bpm)
    for bar_index in range(bars):
        chord = chords[bar_index % len(chords)]
        start = bar_index * 4 * seconds_per_beat
        end = (bar_index + 1) * 4 * seconds_per_beat
        root, intervals = parse_chord(chord)
        bass_root = 36 + root
        while bass_root > 47:
            bass_root -= 12
        bass.notes.append(pretty_midi.Note(velocity=76, pitch=bass_root, start=start, end=start + seconds_per_beat * 1.85))
        bass.notes.append(pretty_midi.Note(velocity=66, pitch=bass_root + 7, start=start + seconds_per_beat * 2, end=end))

        guides = list(guide_intervals(chord))
        if 14 not in intervals:
            guides.append(14)
        guide_notes = [nearest(pitches_for_intervals(chord, [interval], 50, 74), 62) for interval in guides]
        for comp_start in (start + seconds_per_beat, start + 3 * seconds_per_beat):
            comp_end = min(end, comp_start + seconds_per_beat * 0.58)
            for pitch in guide_notes:
                comp.notes.append(pretty_midi.Note(velocity=62, pitch=int(pitch), start=comp_start, end=comp_end))
    out.instruments.extend([bass, comp])
    return out


def render_wav(render_config: RenderConfig, midi_path: Path, wav_path: Path) -> dict[str, Any]:
    command = [
        render_config.renderer,
        "-ni",
        "-F",
        str(wav_path),
        "-r",
        str(render_config.sample_rate),
        render_config.soundfont,
        str(midi_path),
    ]
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    if completed.returncode != 0:
        raise BebopLanguagePackageError(
            f"render failed for {midi_path}: {(completed.stderr or completed.stdout)[-1000:]}"
        )
    return {
        "wav_file": wav_meta(wav_path),
        "command": command,
        "stdout_tail": (completed.stdout or "")[-800:],
        "stderr_tail": (completed.stderr or "")[-800:],
    }


def build_listen_first_package(output_dir: Path, rendered: list[dict[str, Any]]) -> dict[str, Any]:
    listen_dir = output_dir / "listen_first_by_progression"
    listen_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in rendered:
        case_label = str(item["case_label"])
        if case_label in seen:
            continue
        seen.add(case_label)
        index = len(rows) + 1
        context_source = Path(str(item["context_audio"]["wav_file"]["path"]))
        solo_source = Path(str(item["solo_audio"]["wav_file"]["path"]))
        context_target = listen_dir / f"{index:02d}_{case_label}_rank_{int(item['rank']):02d}_with_context.wav"
        solo_target = listen_dir / f"{index:02d}_{case_label}_rank_{int(item['rank']):02d}_solo_only.wav"
        shutil.copy2(context_source, context_target)
        shutil.copy2(solo_source, solo_target)
        rows.append(
            {
                "case_label": case_label,
                "rank": int(item["rank"]),
                "variant_index": int(item["variant_index"]),
                "chords": list(item["chords"]),
                "context_wav": str(context_target),
                "solo_wav": str(solo_target),
                "objective_metrics": dict(item["objective_metrics"]),
            }
        )
    report = {
        "path": str(listen_dir),
        "case_count": len(rows),
        "files": rows,
    }
    write_json(listen_dir / "listen_first_by_progression.json", report)
    return report


def source_progressions(source_package: dict[str, Any]) -> list[dict[str, Any]]:
    seen: set[tuple[str, ...]] = set()
    rows: list[dict[str, Any]] = []
    for item in source_package.get("selected_candidates", []):
        chords = tuple(part.strip() for part in str(item.get("chords") or "").split(",") if part.strip())
        if not chords or chords in seen:
            continue
        seen.add(chords)
        rows.append(
            {
                "case_label": str(item.get("case_label") or f"case_{len(rows) + 1}"),
                "chords": list(chords),
            }
        )
    if not rows:
        raise BebopLanguagePackageError("no source chord progressions")
    return rows


def build_package(
    *,
    source_package_path: Path,
    output_dir: Path,
    render_config: RenderConfig,
    bars: int,
    bpm: float,
    variants_per_progression: int,
    selected_count: int,
    seed_base: int,
    non_chord_probability: float,
    target_chord_tone_ratio: float,
    target_offbeat_non_chord_ratio: float,
) -> dict[str, Any]:
    source_package = read_json(source_package_path)
    progressions = source_progressions(source_package)
    raw_dir = output_dir / "midi_raw"
    solo_dir = output_dir / "midi"
    mix_midi_dir = output_dir / "midi_with_context"
    solo_audio_dir = output_dir / "audio"
    mix_audio_dir = output_dir / "audio_with_context"

    generated: list[dict[str, Any]] = []
    for progression_index, progression in enumerate(progressions):
        for variant_index in range(variants_per_progression):
            seed = seed_base + progression_index * 1000 + variant_index * 37
            pm, generation_meta = build_bebop_candidate(
                progression["chords"],
                seed=seed,
                bars=bars,
                bpm=bpm,
                non_chord_probability=non_chord_probability,
            )
            metrics = objective_metrics(pm, progression["chords"], bars=bars, bpm=bpm)
            score = candidate_score(
                metrics,
                target_chord_tone_ratio=target_chord_tone_ratio,
                target_offbeat_non_chord_ratio=target_offbeat_non_chord_ratio,
            )
            gate_penalty = candidate_gate_penalty(metrics)
            raw_path = raw_dir / f"{progression['case_label']}_variant_{variant_index + 1:02d}_seed_{seed}.mid"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            pm.write(str(raw_path))
            generated.append(
                {
                    "case_label": progression["case_label"],
                    "chords": progression["chords"],
                    "variant_index": int(variant_index + 1),
                    "seed": int(seed),
                    "raw_midi_path": str(raw_path),
                    "raw_midi_sha256": sha256_file(raw_path),
                    "generation_meta": generation_meta,
                    "objective_metrics": metrics,
                    "score": float(score),
                    "gate_penalty": float(gate_penalty),
                }
            )

    generated.sort(key=generated_sort_key)
    selected: list[dict[str, Any]] = []
    selected_paths: set[str] = set()
    for progression in progressions:
        best_for_progression = next(
            (item for item in generated if str(item["case_label"]) == str(progression["case_label"])),
            None,
        )
        if best_for_progression is not None:
            selected.append(best_for_progression)
            selected_paths.add(str(best_for_progression["raw_midi_path"]))
    for item in generated:
        if len(selected) >= selected_count:
            break
        raw_path = str(item["raw_midi_path"])
        if raw_path in selected_paths:
            continue
        selected.append(item)
        selected_paths.add(raw_path)
    rendered: list[dict[str, Any]] = []
    for rank, item in enumerate(selected, start=1):
        raw_path = Path(str(item["raw_midi_path"]))
        pm = pretty_midi.PrettyMIDI(str(raw_path))
        context_pm = add_context(pm, item["chords"], bars=bars, bpm=bpm)
        safe_case = str(item["case_label"]).replace("/", "_").replace(" ", "_")
        stem = f"candidate_{rank:02d}_{safe_case}_variant_{int(item['variant_index']):02d}_bebop_language"
        solo_midi_path = solo_dir / f"{stem}.mid"
        mix_midi_path = mix_midi_dir / f"{stem}_with_context.mid"
        solo_midi_path.parent.mkdir(parents=True, exist_ok=True)
        mix_midi_path.parent.mkdir(parents=True, exist_ok=True)
        pm.write(str(solo_midi_path))
        context_pm.write(str(mix_midi_path))
        solo_render = render_wav(render_config, solo_midi_path, solo_audio_dir / f"{stem}.wav")
        mix_render = render_wav(render_config, mix_midi_path, mix_audio_dir / f"{stem}_with_context.wav")
        rendered.append(
            {
                **item,
                "rank": int(rank),
                "midi_path": str(solo_midi_path),
                "midi_sha256": sha256_file(solo_midi_path),
                "context_midi_path": str(mix_midi_path),
                "context_midi_sha256": sha256_file(mix_midi_path),
                "solo_audio": solo_render,
                "context_audio": mix_render,
            }
        )

    listen_first = build_listen_first_package(output_dir, rendered)
    aggregate_metrics = {
        "generated_candidate_count": len(generated),
        "selected_candidate_count": len(rendered),
        "listen_first_case_count": int(listen_first["case_count"]),
        "avg_score": mean([float(item["score"]) for item in rendered]) if rendered else 0.0,
        "avg_gate_penalty": mean([float(item.get("gate_penalty") or 0.0) for item in rendered]) if rendered else 0.0,
        "max_gate_penalty": max((float(item.get("gate_penalty") or 0.0) for item in rendered), default=0.0),
        "avg_unique_pitch_count": mean([float(item["objective_metrics"]["unique_pitch_count"]) for item in rendered])
        if rendered
        else 0.0,
        "avg_step_motion_ratio": mean([float(item["objective_metrics"]["step_motion_ratio"]) for item in rendered])
        if rendered
        else 0.0,
        "avg_third_fourth_motion_ratio": mean(
            [float(item["objective_metrics"]["third_fourth_motion_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_large_leap_ratio": mean([float(item["objective_metrics"]["large_leap_ratio"]) for item in rendered])
        if rendered
        else 0.0,
        "avg_chord_tone_ratio": mean([float(item["objective_metrics"]["chord_tone_ratio"]) for item in rendered])
        if rendered
        else 0.0,
        "avg_tension_ratio": mean([float(item["objective_metrics"]["tension_ratio"]) for item in rendered])
        if rendered
        else 0.0,
        "avg_strong_beat_chord_tone_ratio": mean(
            [float(item["objective_metrics"]["strong_beat_chord_tone_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_offbeat_non_chord_ratio": mean(
            [float(item["objective_metrics"]["offbeat_non_chord_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_offbeat_non_chord_resolution_ratio": mean(
            [float(item["objective_metrics"]["offbeat_non_chord_resolution_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_offbeat_unresolved_non_chord_ratio": mean(
            [float(item["objective_metrics"]["offbeat_unresolved_non_chord_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_chromatic_step_ratio": mean(
            [float(item["objective_metrics"]["chromatic_step_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_enclosure_proxy_ratio": mean(
            [float(item["objective_metrics"]["enclosure_proxy_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_dominant_altered_offbeat_ratio": mean(
            [float(item["objective_metrics"]["dominant_altered_offbeat_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_two_note_cycle_ratio": mean(
            [float(item["objective_metrics"]["two_note_cycle_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_bar_half_repeat_ratio": mean(
            [float(item["objective_metrics"]["bar_half_repeat_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_max_bar_pitch_class_jaccard": mean(
            [float(item["objective_metrics"]["max_bar_pitch_class_jaccard"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "avg_bar_pitch_shape_repeat_ratio": mean(
            [float(item["objective_metrics"]["bar_pitch_shape_repeat_ratio"]) for item in rendered]
        )
        if rendered
        else 0.0,
        "min_bar_unique_pitch_count_min": min(
            (int(item["objective_metrics"]["min_bar_unique_pitch_count"]) for item in rendered),
            default=0,
        ),
        "all_final_landings_chord_tone": all(
            bool(item["objective_metrics"]["final_landing_is_chord_tone"]) for item in rendered
        ),
        "avg_note_count": mean([int(item["objective_metrics"]["note_count"]) for item in rendered]) if rendered else 0.0,
        "max_abs_interval_max": max((int(item["objective_metrics"]["max_abs_interval"]) for item in rendered), default=0),
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "output_dir": str(output_dir),
        "source_package": str(source_package_path),
        "boundary": "stage_b_midi_to_solo_bebop_language_package",
        "generation": {
            "bars": int(bars),
            "bpm": float(bpm),
            "variants_per_progression": int(variants_per_progression),
            "selected_count": int(selected_count),
            "seed_base": int(seed_base),
            "non_chord_probability": float(non_chord_probability),
            "target_chord_tone_ratio": float(target_chord_tone_ratio),
            "target_offbeat_non_chord_ratio": float(target_offbeat_non_chord_ratio),
            "progression_count": len(progressions),
            "progressions": progressions,
        },
        "renderer": {
            "name": "fluidsynth",
            "path": render_config.renderer,
            "sample_rate": int(render_config.sample_rate),
        },
        "soundfont": {
            "path": render_config.soundfont,
            "sha256": sha256_file(Path(render_config.soundfont)),
        },
        "aggregate": aggregate_metrics,
        "listen_first": listen_first,
        "selected_candidates": rendered,
        "quality_claimed": False,
        "model_direct_claimed": False,
        "not_proven": [
            "direct_music_transformer_quality",
            "human_audio_preference",
            "artist_level_long_solo_generation",
            "production_ready_improviser",
        ],
        "decision": {
            "current_boundary": "stage_b_midi_to_solo_bebop_language_package",
            "next_boundary": "listening_review_or_model_ranking_integration",
            "critical_user_input_required": False,
            "reason": "generate a high-yield bebop-language comparison package for the current listening-quality gap",
        },
    }
    write_json(output_dir / "bebop_language_package.json", report)
    write_json(output_dir / "bebop_language_package_summary.json", validate_report(report, render_config.sample_rate))
    write_text(output_dir / "bebop_language_package.md", markdown_report(report))
    return report


def validate_report(report: dict[str, Any], expected_sample_rate: int) -> dict[str, Any]:
    selected = report.get("selected_candidates", [])
    if not selected:
        raise BebopLanguagePackageError("selected candidates missing")
    for item in selected:
        metrics = item["objective_metrics"]
        if float(metrics["strong_beat_chord_tone_ratio"]) < 0.95:
            raise BebopLanguagePackageError("strong beat chord-tone ratio below target")
        if not bool(metrics["final_landing_is_chord_tone"]):
            raise BebopLanguagePackageError("final landing must be chord tone")
        for render_key in ("solo_audio", "context_audio"):
            wav = item[render_key]["wav_file"]
            if not bool(wav["exists"]):
                raise BebopLanguagePackageError("missing wav file")
            if int(wav["sample_rate"]) != int(expected_sample_rate):
                raise BebopLanguagePackageError("unexpected sample rate")
            if int(wav["frame_count"]) <= 0:
                raise BebopLanguagePackageError("empty wav file")
    aggregate = report["aggregate"]
    return {
        "schema_version": report["schema_version"],
        "candidate_count": len(selected),
        "generated_candidate_count": int(aggregate["generated_candidate_count"]),
        "avg_gate_penalty": float(aggregate["avg_gate_penalty"]),
        "max_gate_penalty": float(aggregate["max_gate_penalty"]),
        "avg_unique_pitch_count": float(aggregate["avg_unique_pitch_count"]),
        "avg_step_motion_ratio": float(aggregate["avg_step_motion_ratio"]),
        "avg_third_fourth_motion_ratio": float(aggregate["avg_third_fourth_motion_ratio"]),
        "avg_large_leap_ratio": float(aggregate["avg_large_leap_ratio"]),
        "avg_chord_tone_ratio": float(aggregate["avg_chord_tone_ratio"]),
        "avg_tension_ratio": float(aggregate["avg_tension_ratio"]),
        "avg_strong_beat_chord_tone_ratio": float(aggregate["avg_strong_beat_chord_tone_ratio"]),
        "avg_offbeat_non_chord_ratio": float(aggregate["avg_offbeat_non_chord_ratio"]),
        "avg_offbeat_non_chord_resolution_ratio": float(aggregate["avg_offbeat_non_chord_resolution_ratio"]),
        "avg_offbeat_unresolved_non_chord_ratio": float(aggregate["avg_offbeat_unresolved_non_chord_ratio"]),
        "avg_chromatic_step_ratio": float(aggregate["avg_chromatic_step_ratio"]),
        "avg_enclosure_proxy_ratio": float(aggregate["avg_enclosure_proxy_ratio"]),
        "avg_dominant_altered_offbeat_ratio": float(aggregate["avg_dominant_altered_offbeat_ratio"]),
        "avg_two_note_cycle_ratio": float(aggregate["avg_two_note_cycle_ratio"]),
        "avg_bar_half_repeat_ratio": float(aggregate["avg_bar_half_repeat_ratio"]),
        "avg_max_bar_pitch_class_jaccard": float(aggregate["avg_max_bar_pitch_class_jaccard"]),
        "avg_bar_pitch_shape_repeat_ratio": float(aggregate["avg_bar_pitch_shape_repeat_ratio"]),
        "min_bar_unique_pitch_count_min": int(aggregate["min_bar_unique_pitch_count_min"]),
        "all_final_landings_chord_tone": bool(aggregate["all_final_landings_chord_tone"]),
        "solo_wav_count": len([item for item in selected if Path(item["solo_audio"]["wav_file"]["path"]).exists()]),
        "context_wav_count": len([item for item in selected if Path(item["context_audio"]["wav_file"]["path"]).exists()]),
        "quality_claimed": bool(report.get("quality_claimed", True)),
        "model_direct_claimed": bool(report.get("model_direct_claimed", True)),
    }


def markdown_report(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    generation = report["generation"]
    lines = [
        "# Stage B MIDI-to-Solo Bebop Language Package",
        "",
        "## Summary",
        "",
        f"- generated candidate count: `{aggregate['generated_candidate_count']}`",
        f"- selected candidate count: `{aggregate['selected_candidate_count']}`",
        f"- bars: `{generation['bars']}`",
        f"- bpm: `{generation['bpm']}`",
        f"- avg gate penalty: `{float(aggregate['avg_gate_penalty']):.4f}`",
        f"- max gate penalty: `{float(aggregate['max_gate_penalty']):.4f}`",
        f"- avg unique pitch count: `{float(aggregate['avg_unique_pitch_count']):.4f}`",
        f"- avg step motion: `{float(aggregate['avg_step_motion_ratio']):.4f}`",
        f"- avg 3rd/4th motion: `{float(aggregate['avg_third_fourth_motion_ratio']):.4f}`",
        f"- avg large leap: `{float(aggregate['avg_large_leap_ratio']):.4f}`",
        f"- avg chord-tone ratio: `{float(aggregate['avg_chord_tone_ratio']):.4f}`",
        f"- avg tension ratio: `{float(aggregate['avg_tension_ratio']):.4f}`",
        f"- avg strong-beat chord-tone ratio: `{float(aggregate['avg_strong_beat_chord_tone_ratio']):.4f}`",
        f"- avg offbeat non-chord ratio: `{float(aggregate['avg_offbeat_non_chord_ratio']):.4f}`",
        f"- avg offbeat non-chord resolution ratio: `{float(aggregate['avg_offbeat_non_chord_resolution_ratio']):.4f}`",
        f"- avg offbeat unresolved non-chord ratio: `{float(aggregate['avg_offbeat_unresolved_non_chord_ratio']):.4f}`",
        f"- avg chromatic step ratio: `{float(aggregate['avg_chromatic_step_ratio']):.4f}`",
        f"- avg enclosure proxy ratio: `{float(aggregate['avg_enclosure_proxy_ratio']):.4f}`",
        f"- avg dominant altered offbeat ratio: `{float(aggregate['avg_dominant_altered_offbeat_ratio']):.4f}`",
        f"- avg two-note cycle ratio: `{float(aggregate['avg_two_note_cycle_ratio']):.4f}`",
        f"- avg bar half-repeat ratio: `{float(aggregate['avg_bar_half_repeat_ratio']):.4f}`",
        f"- avg max bar pitch-class similarity: `{float(aggregate['avg_max_bar_pitch_class_jaccard']):.4f}`",
        f"- avg bar pitch-shape repeat ratio: `{float(aggregate['avg_bar_pitch_shape_repeat_ratio']):.4f}`",
        f"- min bar unique pitch count: `{int(aggregate['min_bar_unique_pitch_count_min'])}`",
        f"- all final landings chord tone: `{str(bool(aggregate['all_final_landings_chord_tone'])).lower()}`",
        f"- quality claimed: `{str(bool(report['quality_claimed'])).lower()}`",
        f"- model direct claimed: `{str(bool(report['model_direct_claimed'])).lower()}`",
        "",
        "## Selected Files",
        "",
        "| rank | case | chords | gate | score | unique pitch | step | 3rd/4th | leap | chord-tone | strong beat | offbeat non-chord | resolved | unresolved | chromatic | enclosure | altered | cycle | half repeat | bar sim | shape repeat | min bar unique | solo WAV | context WAV |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for item in report["selected_candidates"]:
        metrics = item["objective_metrics"]
        lines.append(
            "| "
            + " | ".join(
                [
                    str(item["rank"]),
                    str(item["case_label"]),
                    ",".join(item["chords"]),
                    f"{float(item.get('gate_penalty') or 0.0):.4f}",
                    f"{float(item['score']):.4f}",
                    str(int(metrics["unique_pitch_count"])),
                    f"{float(metrics['step_motion_ratio']):.4f}",
                    f"{float(metrics['third_fourth_motion_ratio']):.4f}",
                    f"{float(metrics['large_leap_ratio']):.4f}",
                    f"{float(metrics['chord_tone_ratio']):.4f}",
                    f"{float(metrics['strong_beat_chord_tone_ratio']):.4f}",
                    f"{float(metrics['offbeat_non_chord_ratio']):.4f}",
                    f"{float(metrics['offbeat_non_chord_resolution_ratio']):.4f}",
                    f"{float(metrics['offbeat_unresolved_non_chord_ratio']):.4f}",
                    f"{float(metrics['chromatic_step_ratio']):.4f}",
                    f"{float(metrics['enclosure_proxy_ratio']):.4f}",
                    f"{float(metrics['dominant_altered_offbeat_ratio']):.4f}",
                    f"{float(metrics['two_note_cycle_ratio']):.4f}",
                    f"{float(metrics['bar_half_repeat_ratio']):.4f}",
                    f"{float(metrics['max_bar_pitch_class_jaccard']):.4f}",
                    f"{float(metrics['bar_pitch_shape_repeat_ratio']):.4f}",
                    str(int(metrics["min_bar_unique_pitch_count"])),
                    str(item["solo_audio"]["wav_file"]["path"]),
                    str(item["context_audio"]["wav_file"]["path"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Not Proven", ""])
    for item in report["not_proven"]:
        lines.append(f"- `{item}`")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a bebop-language MIDI/WAV review package")
    parser.add_argument("--source_package", default=str(DEFAULT_SOURCE_PACKAGE))
    parser.add_argument("--output_root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run_id", default="")
    parser.add_argument("--renderer", default=shutil.which("fluidsynth") or "")
    parser.add_argument("--soundfont", default="")
    parser.add_argument("--sample_rate", type=int, default=44100)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=float, default=124.0)
    parser.add_argument("--variants_per_progression", type=int, default=10)
    parser.add_argument("--selected_count", type=int, default=12)
    parser.add_argument("--seed_base", type=int, default=41000)
    parser.add_argument("--non_chord_probability", type=float, default=DEFAULT_NON_CHORD_PROBABILITY)
    parser.add_argument("--target_chord_tone_ratio", type=float, default=DEFAULT_TARGET_CHORD_TONE_RATIO)
    parser.add_argument("--target_offbeat_non_chord_ratio", type=float, default=DEFAULT_TARGET_OFFBEAT_NON_CHORD_RATIO)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    renderer = str(args.renderer or "")
    if not renderer:
        raise BebopLanguagePackageError("renderer missing")
    if not Path(renderer).exists():
        raise BebopLanguagePackageError(f"renderer not found: {renderer}")
    soundfont = resolve_soundfont(str(args.soundfont or ""))
    if not soundfont or not Path(soundfont).exists():
        raise BebopLanguagePackageError(f"soundfont not found: {soundfont}")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_root) / run_id
    report = build_package(
        source_package_path=Path(args.source_package),
        output_dir=output_dir,
        render_config=RenderConfig(renderer=renderer, soundfont=soundfont, sample_rate=int(args.sample_rate)),
        bars=int(args.bars),
        bpm=float(args.bpm),
        variants_per_progression=int(args.variants_per_progression),
        selected_count=int(args.selected_count),
        seed_base=int(args.seed_base),
        non_chord_probability=float(args.non_chord_probability),
        target_chord_tone_ratio=float(args.target_chord_tone_ratio),
        target_offbeat_non_chord_ratio=float(args.target_offbeat_non_chord_ratio),
    )
    summary = validate_report(report, int(args.sample_rate))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
