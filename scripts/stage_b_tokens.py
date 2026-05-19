"""Stage B duration-explicit symbolic MIDI tokens.

Stage B is intentionally separate from the Stage A NOTE_ON/NOTE_OFF event
stream. It uses explicit bar, position, chord, pitch, duration, and velocity
tokens so a model does not need to infer note endings from NOTE_OFF events.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pretty_midi

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from utilities.constants import TOKEN_BAR, TOKEN_CONTROL_END, TOKEN_END  # noqa: E402

try:
    from control_tokens import control_prefix_tokens
except ModuleNotFoundError:
    from scripts.control_tokens import control_prefix_tokens


SEQUENCE_FORMAT_STAGE_B_V1 = "stage_b_v1"

POSITIONS_PER_BAR = 16
MAX_DURATION_STEPS = 16
VELOCITY_BINS = 8
PIANO_PITCH_MIN = 21
PIANO_PITCH_MAX = 108
PIANO_PITCH_COUNT = PIANO_PITCH_MAX - PIANO_PITCH_MIN + 1

CHORD_ROOTS = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B", "N")
CHORD_QUALITIES = ("maj", "min", "dom7", "maj7", "min7", "dim", "halfdim", "sus", "unknown")

TOKEN_POSITION_START = TOKEN_CONTROL_END + 1
TOKEN_POSITION_END = TOKEN_POSITION_START + POSITIONS_PER_BAR - 1
TOKEN_CHORD_ROOT_START = TOKEN_POSITION_END + 1
TOKEN_CHORD_ROOT_END = TOKEN_CHORD_ROOT_START + len(CHORD_ROOTS) - 1
TOKEN_CHORD_QUALITY_START = TOKEN_CHORD_ROOT_END + 1
TOKEN_CHORD_QUALITY_END = TOKEN_CHORD_QUALITY_START + len(CHORD_QUALITIES) - 1
TOKEN_NOTE_PITCH_START = TOKEN_CHORD_QUALITY_END + 1
TOKEN_NOTE_PITCH_END = TOKEN_NOTE_PITCH_START + PIANO_PITCH_COUNT - 1
TOKEN_NOTE_DURATION_START = TOKEN_NOTE_PITCH_END + 1
TOKEN_NOTE_DURATION_END = TOKEN_NOTE_DURATION_START + MAX_DURATION_STEPS - 1
TOKEN_VELOCITY_START = TOKEN_NOTE_DURATION_END + 1
TOKEN_VELOCITY_END = TOKEN_VELOCITY_START + VELOCITY_BINS - 1

STAGE_B_TOKEN_START = TOKEN_POSITION_START
STAGE_B_TOKEN_END = TOKEN_VELOCITY_END
STAGE_B_VOCAB_SIZE = STAGE_B_TOKEN_END + 1

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

PC_TO_CANONICAL_ROOT = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}


@dataclass(frozen=True)
class StageBDecodedNote:
    pitch: int
    start: float
    end: float
    velocity: int


def step_duration_sec(tempo_bpm: float | int) -> float:
    beat_sec = 60.0 / max(1e-6, float(tempo_bpm))
    return beat_sec * 4.0 / POSITIONS_PER_BAR


def quantize_note_position(note_start_sec: float, tempo_bpm: float | int) -> tuple[int, int]:
    absolute_step = max(0, int(round(float(note_start_sec) / step_duration_sec(tempo_bpm))))
    return absolute_step // POSITIONS_PER_BAR, absolute_step % POSITIONS_PER_BAR


def quantize_note_duration(duration_sec: float, tempo_bpm: float | int) -> int:
    steps = int(round(max(0.0, float(duration_sec)) / step_duration_sec(tempo_bpm)))
    return max(1, min(MAX_DURATION_STEPS, steps))


def velocity_bin(velocity: int) -> int:
    return max(0, min(VELOCITY_BINS - 1, int(velocity) * VELOCITY_BINS // 128))


def velocity_from_bin(bin_index: int) -> int:
    clamped = max(0, min(VELOCITY_BINS - 1, int(bin_index)))
    return int(round((clamped + 0.5) * 127 / VELOCITY_BINS))


def position_token(position: int) -> int:
    if position < 0 or position >= POSITIONS_PER_BAR:
        raise ValueError(f"position out of range: {position}")
    return TOKEN_POSITION_START + int(position)


def chord_root_token(root_name: str | None) -> int:
    root = root_name if root_name in CHORD_ROOTS else "N"
    return TOKEN_CHORD_ROOT_START + CHORD_ROOTS.index(root)


def chord_quality_token(quality: str | None) -> int:
    normalized = quality if quality in CHORD_QUALITIES else "unknown"
    return TOKEN_CHORD_QUALITY_START + CHORD_QUALITIES.index(normalized)


def note_pitch_token(pitch: int) -> int:
    if pitch < PIANO_PITCH_MIN or pitch > PIANO_PITCH_MAX:
        raise ValueError(f"pitch out of piano range: {pitch}")
    return TOKEN_NOTE_PITCH_START + int(pitch) - PIANO_PITCH_MIN


def note_duration_token(duration_steps: int) -> int:
    if duration_steps < 1 or duration_steps > MAX_DURATION_STEPS:
        raise ValueError(f"duration steps out of range: {duration_steps}")
    return TOKEN_NOTE_DURATION_START + int(duration_steps) - 1


def note_velocity_token(bin_index: int) -> int:
    if bin_index < 0 or bin_index >= VELOCITY_BINS:
        raise ValueError(f"velocity bin out of range: {bin_index}")
    return TOKEN_VELOCITY_START + int(bin_index)


def is_position_token(token: int) -> bool:
    return TOKEN_POSITION_START <= int(token) <= TOKEN_POSITION_END


def is_note_pitch_token(token: int) -> bool:
    return TOKEN_NOTE_PITCH_START <= int(token) <= TOKEN_NOTE_PITCH_END


def is_note_duration_token(token: int) -> bool:
    return TOKEN_NOTE_DURATION_START <= int(token) <= TOKEN_NOTE_DURATION_END


def is_velocity_token(token: int) -> bool:
    return TOKEN_VELOCITY_START <= int(token) <= TOKEN_VELOCITY_END


def position_from_token(token: int) -> int:
    return int(token) - TOKEN_POSITION_START


def pitch_from_token(token: int) -> int:
    return PIANO_PITCH_MIN + int(token) - TOKEN_NOTE_PITCH_START


def duration_steps_from_token(token: int) -> int:
    return int(token) - TOKEN_NOTE_DURATION_START + 1


def velocity_bin_from_token(token: int) -> int:
    return int(token) - TOKEN_VELOCITY_START


def parse_chord_symbol(chord: str | None) -> tuple[str, str]:
    if not chord:
        return "N", "unknown"

    match = re.match(r"^\s*([A-Ga-g])([#b]?)(.*)$", chord)
    if not match:
        return "N", "unknown"

    root_raw = match.group(1).upper() + match.group(2)
    root_pc = ROOT_TO_PC.get(root_raw)
    if root_pc is None:
        return "N", "unknown"

    suffix = match.group(3).strip().lower().replace(" ", "")
    root = PC_TO_CANONICAL_ROOT[root_pc]

    if "m7b5" in suffix or "ø" in suffix or "half" in suffix:
        quality = "halfdim"
    elif "dim" in suffix or "o" == suffix:
        quality = "dim"
    elif "sus" in suffix:
        quality = "sus"
    elif "maj7" in suffix or "ma7" in suffix or "Δ" in suffix:
        quality = "maj7"
    elif suffix.startswith("m") and "maj" not in suffix:
        quality = "min7" if "7" in suffix else "min"
    elif "7" in suffix:
        quality = "dom7"
    elif suffix in ("", "maj", "major"):
        quality = "maj"
    else:
        quality = "unknown"

    return root, quality


def chord_tokens(chord: str | None) -> list[int]:
    root, quality = parse_chord_symbol(chord)
    return [chord_root_token(root), chord_quality_token(quality)]


def stage_b_token_name(token: int) -> str:
    value = int(token)
    if value == TOKEN_END:
        return "END"
    if value == TOKEN_BAR:
        return "BAR"
    if is_position_token(value):
        return f"POSITION_{position_from_token(value)}"
    if TOKEN_CHORD_ROOT_START <= value <= TOKEN_CHORD_ROOT_END:
        return f"CHORD_ROOT_{CHORD_ROOTS[value - TOKEN_CHORD_ROOT_START]}"
    if TOKEN_CHORD_QUALITY_START <= value <= TOKEN_CHORD_QUALITY_END:
        return f"CHORD_QUALITY_{CHORD_QUALITIES[value - TOKEN_CHORD_QUALITY_START]}"
    if is_note_pitch_token(value):
        return f"NOTE_PITCH_{pitch_from_token(value)}"
    if is_note_duration_token(value):
        return f"NOTE_DURATION_{duration_steps_from_token(value)}"
    if is_velocity_token(value):
        return f"VELOCITY_{velocity_bin_from_token(value)}"
    return str(value)


def build_stage_b_sequence(
    notes: Sequence[pretty_midi.Note],
    tempo_bpm: float | int,
    chords: Sequence[str] | None = None,
    role: str | None = "lead",
    bars: int | None = None,
) -> list[int]:
    valid_notes = [
        note
        for note in notes
        if PIANO_PITCH_MIN <= int(note.pitch) <= PIANO_PITCH_MAX and float(note.end) > float(note.start)
    ]
    quantized: list[tuple[int, int, pretty_midi.Note]] = [
        (*quantize_note_position(float(note.start), tempo_bpm), note) for note in valid_notes
    ]
    max_note_bar = max((bar for bar, _pos, _note in quantized), default=0)
    total_bars = max(1, int(bars) if bars is not None else max_note_bar + 1)

    tokens = control_prefix_tokens(role=role, tempo_bpm=tempo_bpm)[:2]
    for bar_index in range(total_bars):
        chord = chords[bar_index] if chords and bar_index < len(chords) else None
        tokens.append(TOKEN_BAR)
        tokens.extend(chord_tokens(chord))
        bar_notes = sorted(
            ((position, note) for bar, position, note in quantized if bar == bar_index),
            key=lambda item: (item[0], item[1].pitch, item[1].start),
        )
        for position, note in bar_notes:
            duration_steps = quantize_note_duration(float(note.end) - float(note.start), tempo_bpm)
            tokens.extend(
                [
                    position_token(position),
                    note_velocity_token(velocity_bin(int(note.velocity))),
                    note_pitch_token(int(note.pitch)),
                    note_duration_token(duration_steps),
                ]
            )
    tokens.append(TOKEN_END)
    return tokens


def decode_stage_b_notes(tokens: Sequence[int], tempo_bpm: float | int) -> list[StageBDecodedNote]:
    step_sec = step_duration_sec(tempo_bpm)
    bar_index = -1
    position = 0
    velocity = velocity_from_bin(VELOCITY_BINS // 2)
    pending_pitch: int | None = None
    decoded: list[StageBDecodedNote] = []

    for raw_token in tokens:
        token = int(raw_token)
        if token == TOKEN_END:
            break
        if token == TOKEN_BAR:
            bar_index += 1
            position = 0
            pending_pitch = None
            continue
        if is_position_token(token):
            position = position_from_token(token)
            pending_pitch = None
            continue
        if is_velocity_token(token):
            velocity = velocity_from_bin(velocity_bin_from_token(token))
            continue
        if is_note_pitch_token(token):
            pending_pitch = pitch_from_token(token)
            continue
        if is_note_duration_token(token) and pending_pitch is not None and bar_index >= 0:
            duration_steps = duration_steps_from_token(token)
            start = (bar_index * POSITIONS_PER_BAR + position) * step_sec
            end = start + duration_steps * step_sec
            decoded.append(
                StageBDecodedNote(
                    pitch=pending_pitch,
                    start=start,
                    end=end,
                    velocity=velocity,
                )
            )
            pending_pitch = None

    return decoded


def decode_stage_b_midi(tokens: Sequence[int], tempo_bpm: float | int = 120.0) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=float(tempo_bpm))
    piano = pretty_midi.Instrument(program=0, is_drum=False, name="stage_b_piano")
    for note in decode_stage_b_notes(tokens, tempo_bpm=tempo_bpm):
        piano.notes.append(
            pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.start,
                end=note.end,
            )
        )
    midi.instruments.append(piano)
    return midi
