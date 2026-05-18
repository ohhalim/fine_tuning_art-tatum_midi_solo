from __future__ import annotations

from pathlib import Path

import pretty_midi

from .fallback import chord_pitches_in_range, parse_chord, phrase_duration_sec
from .schemas import GenerationRequest


CONDITIONING_PITCH_MIN = 36
CONDITIONING_PITCH_MAX = 60


def _root_pitch(root_pc: int) -> int:
    candidates = [pitch for pitch in range(CONDITIONING_PITCH_MIN, 49) if pitch % 12 == root_pc]
    if not candidates:
        return 48 + root_pc
    return min(candidates, key=lambda pitch: abs(pitch - 43))


def _voicing_for_chord(chord: str) -> list[int]:
    root_pc, intervals = parse_chord(chord)
    low_root = _root_pitch(root_pc)
    tones = chord_pitches_in_range(root_pc, intervals, 48, CONDITIONING_PITCH_MAX)
    upper = [pitch for pitch in tones if pitch != low_root][:3]
    voicing = [low_root, *upper]
    return sorted(dict.fromkeys(voicing))


def build_request_conditioning_midi(
    request: GenerationRequest,
    output_dir: str | Path,
) -> Path:
    """Create a low-register harmonic guide MIDI for the Stage A primer."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{request.job_id}_conditioning.mid"

    phrase_duration = phrase_duration_sec(request)
    segment_duration = phrase_duration / max(1, len(request.chord_progression))
    note_duration = max(0.08, segment_duration * 0.85)

    pm = pretty_midi.PrettyMIDI(initial_tempo=float(request.bpm))
    guide = pretty_midi.Instrument(program=0, is_drum=False, name="request_chord_conditioning")

    for chord_index, chord in enumerate(request.chord_progression):
        start = chord_index * segment_duration
        end = min(phrase_duration, start + note_duration)
        if end <= start:
            continue

        for pitch in _voicing_for_chord(chord):
            guide.notes.append(
                pretty_midi.Note(
                    velocity=58,
                    pitch=int(pitch),
                    start=float(start),
                    end=float(end),
                )
            )

    if not guide.notes:
        for pitch in _voicing_for_chord(request.chord_progression[0]):
            guide.notes.append(
                pretty_midi.Note(
                    velocity=58,
                    pitch=int(pitch),
                    start=0.0,
                    end=max(0.08, min(phrase_duration, 0.5)),
                )
            )

    guide.notes.sort(key=lambda note: (note.start, note.pitch))
    pm.instruments.append(guide)
    pm.write(str(output_path))
    return output_path
