from __future__ import annotations

import random
from pathlib import Path

import pretty_midi

from .fallback import chord_for_time, chord_pitches_in_range, parse_chord, phrase_duration_sec
from .schemas import GenerationRequest


PREFERRED_SOLO_MIN = 48
PREFERRED_SOLO_MAX = 88
HARD_PIANO_MIN = 21
HARD_PIANO_MAX = 108

TARGET_NOTES_PER_BAR = {
    "sparse": 0,
    # Medium only gets targeted large-gap repair; broad eighth-note fill raised
    # dead-air in smoke sweeps. Dense gets explicit 16th-note gap fill.
    "medium": 0,
    "dense": 12,
}


def map_pitch_to_range(
    pitch: int,
    pitch_min: int = PREFERRED_SOLO_MIN,
    pitch_max: int = PREFERRED_SOLO_MAX,
) -> int:
    pitch_class = int(pitch) % 12
    candidates = [p for p in range(pitch_min, pitch_max + 1) if p % 12 == pitch_class]
    if not candidates:
        return max(HARD_PIANO_MIN, min(HARD_PIANO_MAX, int(pitch)))
    center = (pitch_min + pitch_max) / 2.0
    return min(candidates, key=lambda p: (abs(p - center), abs(p - int(pitch))))


def has_nearby_start(notes: list[pretty_midi.Note], start_sec: float, tolerance_sec: float = 0.05) -> bool:
    return any(abs(float(note.start) - start_sec) <= tolerance_sec for note in notes)


def add_density_repair_notes(
    instrument: pretty_midi.Instrument,
    request: GenerationRequest,
    max_duration: float,
) -> None:
    target_notes = TARGET_NOTES_PER_BAR.get(request.density, TARGET_NOTES_PER_BAR["medium"]) * request.bars
    if target_notes <= 0 or len(instrument.notes) >= target_notes:
        return

    rng = random.Random(request.seed + 1009)
    sixteenth = (60.0 / float(request.bpm)) / 4.0
    duration = max(0.06, sixteenth * 0.85)
    grid_positions = []
    pos = 0.0
    while pos < max_duration:
        grid_positions.append(pos)
        pos += sixteenth

    if request.density == "medium":
        grid_positions = grid_positions[::2]

    previous_pitch = instrument.notes[-1].pitch if instrument.notes else 64
    for start in grid_positions:
        if len(instrument.notes) >= target_notes:
            break
        if has_nearby_start(instrument.notes, start):
            continue

        chord = chord_for_time(request, start)
        root_pc, intervals = parse_chord(chord)
        tones = chord_pitches_in_range(root_pc, intervals, PREFERRED_SOLO_MIN, PREFERRED_SOLO_MAX)
        nearby = [pitch for pitch in tones if abs(pitch - previous_pitch) <= 7]
        pool = nearby or tones or [previous_pitch]
        pitch = rng.choice(pool)
        previous_pitch = pitch
        end = min(max_duration, start + duration)
        if end <= start:
            continue
        instrument.notes.append(
            pretty_midi.Note(
                velocity=72 if request.energy != "high" else 88,
                pitch=int(pitch),
                start=float(start),
                end=float(end),
            )
        )


def add_medium_gap_repair_notes(
    instrument: pretty_midi.Instrument,
    request: GenerationRequest,
    max_duration: float,
) -> None:
    if request.density != "medium" or len(instrument.notes) < 2:
        return

    rng = random.Random(request.seed + 2027)
    sixteenth = (60.0 / float(request.bpm)) / 4.0
    duration = max(0.06, sixteenth * 0.85)
    max_additions = 4
    additions = 0

    notes = sorted(instrument.notes, key=lambda n: (n.start, n.pitch))
    for prev_note, next_note in zip(notes, notes[1:]):
        if additions >= max_additions:
            break
        gap = float(next_note.start) - float(prev_note.start)
        if gap < 0.36:
            continue

        start = float(prev_note.start) + sixteenth
        if start >= float(next_note.start) - 0.03 or start >= max_duration:
            continue
        if has_nearby_start(instrument.notes, start):
            continue

        chord = chord_for_time(request, start)
        root_pc, intervals = parse_chord(chord)
        tones = chord_pitches_in_range(root_pc, intervals, PREFERRED_SOLO_MIN, PREFERRED_SOLO_MAX)
        nearby = [pitch for pitch in tones if abs(pitch - int(prev_note.pitch)) <= 7]
        pool = nearby or tones or [int(prev_note.pitch)]
        pitch = rng.choice(pool)
        end = min(max_duration, start + duration)
        if end <= start:
            continue
        instrument.notes.append(
            pretty_midi.Note(
                velocity=78 if request.energy != "high" else 92,
                pitch=int(pitch),
                start=float(start),
                end=float(end),
            )
        )
        additions += 1


def repair_model_midi(
    input_path: str | Path,
    output_path: str | Path,
    request: GenerationRequest,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    source = pretty_midi.PrettyMIDI(str(input_path))
    repaired = pretty_midi.PrettyMIDI(initial_tempo=float(request.bpm))
    instrument = pretty_midi.Instrument(program=0, is_drum=False, name="model_repaired_piano_solo")

    max_duration = phrase_duration_sec(request)
    min_duration = 0.04

    notes: list[pretty_midi.Note] = []
    for source_instrument in source.instruments:
        if source_instrument.is_drum:
            continue
        notes.extend(source_instrument.notes)

    notes.sort(key=lambda n: (n.start, n.pitch))
    source_start = float(notes[0].start) if notes else 0.0
    for note in notes:
        if note.end <= note.start:
            continue
        shifted_start = max(0.0, float(note.start) - source_start)
        shifted_end = max(0.0, float(note.end) - source_start)
        if shifted_start >= max_duration:
            continue

        start = shifted_start
        end = min(shifted_end, max_duration)
        if end - start < min_duration:
            end = min(max_duration, start + min_duration)
        if end <= start:
            continue

        pitch = map_pitch_to_range(int(note.pitch))
        velocity = max(1, min(127, int(note.velocity or 80)))
        instrument.notes.append(
            pretty_midi.Note(
                velocity=velocity,
                pitch=pitch,
                start=start,
                end=end,
            )
        )

    add_density_repair_notes(instrument, request, max_duration)
    add_medium_gap_repair_notes(instrument, request, max_duration)

    # Avoid dense duplicate same-pitch notes at the exact same start.
    deduped: list[pretty_midi.Note] = []
    seen: set[tuple[int, int]] = set()
    for note in sorted(instrument.notes, key=lambda n: (n.start, n.pitch, n.end)):
        key = (int(round(note.start * 1000)), int(note.pitch))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(note)
    instrument.notes = deduped

    repaired.instruments.append(instrument)
    repaired.write(str(output_path))
    return output_path
