from __future__ import annotations

from collections import Counter
from math import ceil
from pathlib import Path

import pretty_midi

from .fallback import chord_for_time, parse_chord, phrase_duration_sec
from .schemas import GenerationMetrics, GenerationRequest


DENSITY_MIN = {
    "sparse": 0.2,
    "medium": 0.5,
    "dense": 1.0,
}

MIN_NOTES_PER_BAR = {
    "sparse": 1.5,
    "medium": 2.0,
    "dense": 4.0,
}

MIN_UNIQUE_PITCHES = {
    "sparse": 2,
    "medium": 3,
    "dense": 4,
}

MIN_PHRASE_COVERAGE = {
    "sparse": 0.25,
    "medium": 0.35,
    "dense": 0.45,
}

MAX_NOTE_DURATION_RATIO = {
    "sparse": 0.85,
    "medium": 0.55,
    "dense": 0.45,
}

LONG_NOTE_DURATION_RATIO = {
    "sparse": 0.50,
    "medium": 0.35,
    "dense": 0.25,
}

MAX_LONG_NOTE_RATIO = {
    "sparse": 0.50,
    "medium": 0.25,
    "dense": 0.20,
}

MAX_SIMULTANEOUS_NOTES = {
    "sparse": 2,
    "medium": 2,
    "dense": 3,
}

DEAD_AIR_MAX = {
    "medium": 0.8,
    "dense": 0.7,
}


def load_notes(midi_path: str | Path) -> list[pretty_midi.Note]:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[pretty_midi.Note] = []
    for instrument in pm.instruments:
        if not instrument.is_drum:
            notes.extend(instrument.notes)
    return sorted(notes, key=lambda n: (n.start, n.pitch))


def repetition_score(pitches: list[int], n: int = 4) -> float:
    if len(pitches) < n * 2:
        return 0.0
    grams = [tuple(pitches[i : i + n]) for i in range(len(pitches) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def chord_tone_metrics(
    notes: list[pretty_midi.Note],
    request: GenerationRequest | None,
) -> tuple[int, int, float | None]:
    if request is None or not notes:
        return 0, 0, None

    chord_tone_count = 0
    non_chord_tone_count = 0
    for note in notes:
        chord = chord_for_time(request, float(note.start))
        root_pc, intervals = parse_chord(chord)
        chord_pitch_classes = {(root_pc + interval) % 12 for interval in intervals}
        if int(note.pitch) % 12 in chord_pitch_classes:
            chord_tone_count += 1
        else:
            non_chord_tone_count += 1

    total = chord_tone_count + non_chord_tone_count
    ratio = chord_tone_count / total if total else None
    return chord_tone_count, non_chord_tone_count, ratio


def max_simultaneous_notes(notes: list[pretty_midi.Note]) -> int:
    events: list[tuple[float, int]] = []
    for note in notes:
        events.append((float(note.start), 1))
        events.append((float(note.end), -1))
    active = 0
    maximum = 0
    for _time, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active = max(0, active + delta)
        maximum = max(maximum, active)
    return maximum


def compute_midi_metrics(
    midi_path: str | Path,
    generation_time_ms: int,
    fallback_used: bool,
    dead_air_threshold_ms: float = 180.0,
    request: GenerationRequest | None = None,
) -> GenerationMetrics:
    notes = load_notes(midi_path)
    if not notes:
        return GenerationMetrics(
            generation_time_ms=generation_time_ms,
            note_count=0,
            duration_sec=0.0,
            note_density=0.0,
            dead_air_ratio=1.0,
            repetition_score=0.0,
            pitch_min=None,
            pitch_max=None,
            fallback_used=fallback_used,
            unique_pitch_count=0,
            unique_pitch_class_count=0,
            expected_duration_sec=phrase_duration_sec(request) if request is not None else None,
            phrase_coverage_ratio=0.0 if request is not None else None,
            avg_note_duration_sec=0.0,
            max_note_duration_sec=0.0,
            max_note_duration_ratio=0.0,
            long_note_ratio=0.0,
            max_simultaneous_notes=0,
        )

    starts = [note.start for note in notes]
    pitches = [note.pitch for note in notes]
    note_durations = [max(0.0, float(note.end) - float(note.start)) for note in notes]
    phrase_start_sec = min(note.start for note in notes)
    phrase_end_sec = max(note.end for note in notes)
    duration_sec = max(1e-6, phrase_end_sec - phrase_start_sec)
    expected_duration_sec = phrase_duration_sec(request) if request is not None else None
    phrase_coverage_ratio = (
        min(1.0, duration_sec / max(1e-6, expected_duration_sec)) if expected_duration_sec is not None else None
    )
    duration_reference_sec = expected_duration_sec if expected_duration_sec is not None else duration_sec
    max_note_duration_sec = max(note_durations)
    max_note_duration_ratio = max_note_duration_sec / max(1e-6, duration_reference_sec)
    density = request.density if request is not None else "medium"
    long_note_floor_sec = LONG_NOTE_DURATION_RATIO.get(density, LONG_NOTE_DURATION_RATIO["medium"]) * max(
        1e-6,
        duration_reference_sec,
    )
    long_note_count = sum(1 for duration in note_durations if duration >= long_note_floor_sec)
    gaps = [max(0.0, starts[i] - starts[i - 1]) for i in range(1, len(starts))]
    threshold = dead_air_threshold_ms / 1000.0
    dead_air_events = sum(1 for gap in gaps if gap >= threshold)
    chord_tone_count, non_chord_tone_count, chord_tone_ratio = chord_tone_metrics(notes, request)

    return GenerationMetrics(
        generation_time_ms=generation_time_ms,
        note_count=len(notes),
        duration_sec=duration_sec,
        note_density=len(notes) / duration_sec,
        dead_air_ratio=dead_air_events / max(1, len(gaps)),
        repetition_score=repetition_score(pitches),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        fallback_used=fallback_used,
        unique_pitch_count=len(set(pitches)),
        unique_pitch_class_count=len({pitch % 12 for pitch in pitches}),
        expected_duration_sec=expected_duration_sec,
        phrase_coverage_ratio=phrase_coverage_ratio,
        avg_note_duration_sec=sum(note_durations) / len(note_durations),
        max_note_duration_sec=max_note_duration_sec,
        max_note_duration_ratio=max_note_duration_ratio,
        long_note_ratio=long_note_count / len(note_durations),
        max_simultaneous_notes=max_simultaneous_notes(notes),
        chord_tone_count=chord_tone_count,
        non_chord_tone_count=non_chord_tone_count,
        chord_tone_ratio=chord_tone_ratio,
    )


def minimum_note_count(density: str, bars: int = 2) -> int:
    notes_per_bar = MIN_NOTES_PER_BAR.get(density, MIN_NOTES_PER_BAR["medium"])
    return max(1, ceil(notes_per_bar * max(1, int(bars))))


def minimum_unique_pitch_count(density: str) -> int:
    return MIN_UNIQUE_PITCHES.get(density, MIN_UNIQUE_PITCHES["medium"])


def minimum_phrase_coverage(density: str) -> float:
    return MIN_PHRASE_COVERAGE.get(density, MIN_PHRASE_COVERAGE["medium"])


def maximum_note_duration_ratio(density: str) -> float:
    return MAX_NOTE_DURATION_RATIO.get(density, MAX_NOTE_DURATION_RATIO["medium"])


def maximum_long_note_ratio(density: str) -> float:
    return MAX_LONG_NOTE_RATIO.get(density, MAX_LONG_NOTE_RATIO["medium"])


def maximum_simultaneous_notes(density: str) -> int:
    return MAX_SIMULTANEOUS_NOTES.get(density, MAX_SIMULTANEOUS_NOTES["medium"])


def validate_metrics(metrics: GenerationMetrics, density: str, bars: int = 2) -> tuple[bool, str | None]:
    if metrics.note_count <= 0:
        return False, "generated MIDI has no notes"
    min_notes = minimum_note_count(density, bars=bars)
    if metrics.note_count < min_notes:
        return False, f"note count too low: {metrics.note_count} < {min_notes}"
    min_unique_pitches = minimum_unique_pitch_count(density)
    if metrics.unique_pitch_count is not None and metrics.unique_pitch_count < min_unique_pitches:
        return False, f"unique pitch count too low: {metrics.unique_pitch_count} < {min_unique_pitches}"
    min_coverage = minimum_phrase_coverage(density)
    if metrics.phrase_coverage_ratio is not None and metrics.phrase_coverage_ratio < min_coverage:
        return False, f"phrase coverage too low: {metrics.phrase_coverage_ratio:.3f} < {min_coverage:.3f}"
    max_duration_ratio = maximum_note_duration_ratio(density)
    if metrics.max_note_duration_ratio is not None and metrics.max_note_duration_ratio > max_duration_ratio:
        return False, f"note duration too long: {metrics.max_note_duration_ratio:.3f} > {max_duration_ratio:.3f}"
    max_long_ratio = maximum_long_note_ratio(density)
    if metrics.long_note_ratio is not None and metrics.long_note_ratio > max_long_ratio:
        return False, f"too many long notes: {metrics.long_note_ratio:.3f} > {max_long_ratio:.3f}"
    max_simultaneous = maximum_simultaneous_notes(density)
    if metrics.max_simultaneous_notes is not None and metrics.max_simultaneous_notes > max_simultaneous:
        return False, f"too many simultaneous notes: {metrics.max_simultaneous_notes} > {max_simultaneous}"
    if metrics.duration_sec <= 0:
        return False, "generated MIDI has zero duration"
    if metrics.pitch_min is None or metrics.pitch_max is None:
        return False, "generated MIDI has no pitch range"
    if metrics.pitch_min < 21 or metrics.pitch_max > 108:
        return False, f"pitch range out of piano bounds: {metrics.pitch_min}-{metrics.pitch_max}"
    min_density = DENSITY_MIN.get(density, DENSITY_MIN["medium"])
    if metrics.note_density < min_density:
        return False, f"note density too low: {metrics.note_density:.3f} < {min_density:.3f}"
    if density != "sparse":
        max_dead_air = DEAD_AIR_MAX.get(density, DEAD_AIR_MAX["medium"])
        if metrics.dead_air_ratio >= max_dead_air:
            return False, f"dead-air ratio too high: {metrics.dead_air_ratio:.3f} >= {max_dead_air:.3f}"
    return True, None
