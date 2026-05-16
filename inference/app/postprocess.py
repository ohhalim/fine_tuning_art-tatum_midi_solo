from __future__ import annotations

from pathlib import Path

import pretty_midi

from .fallback import parse_time_signature
from .schemas import GenerationRequest


PREFERRED_SOLO_MIN = 48
PREFERRED_SOLO_MAX = 88
HARD_PIANO_MIN = 21
HARD_PIANO_MAX = 108


def phrase_duration_sec(request: GenerationRequest) -> float:
    beats_per_bar = parse_time_signature(request.time_signature)
    return request.bars * beats_per_bar * (60.0 / float(request.bpm))


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
