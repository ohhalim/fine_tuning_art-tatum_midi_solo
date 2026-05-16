from __future__ import annotations

import random
import re
from pathlib import Path

import pretty_midi

from .schemas import GenerationRequest


ROOT_TO_PC = {
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

QUALITY_INTERVALS = {
    "maj7": [0, 4, 7, 11],
    "maj": [0, 4, 7, 11],
    "m7b5": [0, 3, 6, 10],
    "dim": [0, 3, 6, 9],
    "m7": [0, 3, 7, 10],
    "min": [0, 3, 7, 10],
    "m": [0, 3, 7, 10],
    "7": [0, 4, 7, 10],
    "": [0, 4, 7, 10],
}

DENSITY_NOTES_PER_BAR = {
    "sparse": (3, 5),
    "medium": (6, 10),
    "dense": (10, 16),
}

ENERGY_CONFIG = {
    "low": {"pitch_min": 50, "pitch_max": 76, "velocity": (55, 76), "duration_beats": (0.5, 1.0)},
    "mid": {"pitch_min": 55, "pitch_max": 84, "velocity": (65, 92), "duration_beats": (0.25, 0.75)},
    "high": {"pitch_min": 60, "pitch_max": 91, "velocity": (82, 112), "duration_beats": (0.25, 0.5)},
}


def parse_time_signature(time_signature: str) -> int:
    match = re.match(r"^\s*(\d+)\s*/\s*(\d+)\s*$", time_signature)
    if not match:
        return 4
    numerator = int(match.group(1))
    return max(1, min(12, numerator))


def parse_chord(chord: str) -> tuple[int, list[int]]:
    match = re.match(r"^\s*([A-Ga-g])([#b]?)(.*)$", chord)
    if not match:
        return ROOT_TO_PC["C"], QUALITY_INTERVALS[""]

    root = match.group(1).upper() + match.group(2)
    quality = match.group(3).strip().lower()
    root_pc = ROOT_TO_PC.get(root, ROOT_TO_PC["C"])

    for marker, intervals in QUALITY_INTERVALS.items():
        if marker and marker in quality:
            return root_pc, intervals
    return root_pc, QUALITY_INTERVALS[""]


def chord_pitches_in_range(root_pc: int, intervals: list[int], pitch_min: int, pitch_max: int) -> list[int]:
    pitch_classes = {(root_pc + interval) % 12 for interval in intervals}
    return [pitch for pitch in range(pitch_min, pitch_max + 1) if pitch % 12 in pitch_classes]


def generate_fallback_midi(request: GenerationRequest, output_path: str | Path) -> Path:
    rng = random.Random(request.seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    beats_per_bar = parse_time_signature(request.time_signature)
    seconds_per_beat = 60.0 / float(request.bpm)
    phrase_duration = request.bars * beats_per_bar * seconds_per_beat
    pm = pretty_midi.PrettyMIDI(initial_tempo=float(request.bpm))
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="fallback_piano_solo")

    density_min, density_max = DENSITY_NOTES_PER_BAR.get(request.density, DENSITY_NOTES_PER_BAR["medium"])
    energy = ENERGY_CONFIG.get(request.energy, ENERGY_CONFIG["mid"])
    pitch_min = int(energy["pitch_min"])
    pitch_max = int(energy["pitch_max"])
    velocity_min, velocity_max = energy["velocity"]
    dur_min, dur_max = energy["duration_beats"]

    for bar_idx in range(request.bars):
        chord = request.chord_progression[bar_idx % len(request.chord_progression)]
        root_pc, intervals = parse_chord(chord)
        chord_tones = chord_pitches_in_range(root_pc, intervals, pitch_min, pitch_max)
        if not chord_tones:
            chord_tones = list(range(pitch_min, pitch_max + 1))

        notes_per_bar = rng.randint(density_min, density_max)
        grid_size = 16
        positions = sorted(rng.sample(range(grid_size), k=min(notes_per_bar, grid_size)))
        bar_start = bar_idx * beats_per_bar * seconds_per_beat

        previous_pitch = None
        for pos in positions:
            beat_pos = (pos / grid_size) * beats_per_bar
            start = bar_start + beat_pos * seconds_per_beat
            duration = rng.uniform(float(dur_min), float(dur_max)) * seconds_per_beat
            end = min(start + duration, bar_start + beats_per_bar * seconds_per_beat)
            if end <= start:
                continue

            pitch = rng.choice(chord_tones)
            if previous_pitch is not None and rng.random() < 0.55:
                nearby = [p for p in chord_tones if abs(p - previous_pitch) <= 7 and p != previous_pitch]
                if nearby:
                    pitch = rng.choice(nearby)
            previous_pitch = pitch

            velocity = rng.randint(int(velocity_min), int(velocity_max))
            piano.notes.append(
                pretty_midi.Note(
                    velocity=velocity,
                    pitch=max(21, min(108, int(pitch))),
                    start=max(0.0, float(start)),
                    end=min(float(phrase_duration), float(end)),
                )
            )

    if not piano.notes:
        # Last-resort tonic pulse so fallback never silently succeeds with no notes.
        root_pc, intervals = parse_chord(request.chord_progression[0])
        pitch = chord_pitches_in_range(root_pc, intervals, 60, 72)[0]
        piano.notes.append(
            pretty_midi.Note(
                velocity=80,
                pitch=pitch,
                start=0.0,
                end=min(phrase_duration, seconds_per_beat),
            )
        )

    piano.notes.sort(key=lambda note: (note.start, note.pitch))
    pm.instruments.append(piano)
    pm.write(str(output_path))
    return output_path
