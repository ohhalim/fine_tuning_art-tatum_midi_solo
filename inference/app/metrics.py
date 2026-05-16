from __future__ import annotations

from collections import Counter
from pathlib import Path

import pretty_midi

from .schemas import GenerationMetrics


DENSITY_MIN = {
    "sparse": 0.2,
    "medium": 0.5,
    "dense": 1.0,
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


def compute_midi_metrics(
    midi_path: str | Path,
    generation_time_ms: int,
    fallback_used: bool,
    dead_air_threshold_ms: float = 180.0,
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
        )

    starts = [note.start for note in notes]
    pitches = [note.pitch for note in notes]
    duration_sec = max(1e-6, notes[-1].end - notes[0].start)
    gaps = [max(0.0, starts[i] - starts[i - 1]) for i in range(1, len(starts))]
    threshold = dead_air_threshold_ms / 1000.0
    dead_air_events = sum(1 for gap in gaps if gap >= threshold)

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
    )


def validate_metrics(metrics: GenerationMetrics, density: str) -> tuple[bool, str | None]:
    if metrics.note_count <= 0:
        return False, "generated MIDI has no notes"
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
