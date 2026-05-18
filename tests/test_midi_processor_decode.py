from __future__ import annotations

import contextlib
import io
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "music_transformer"))
sys.path.insert(0, str(PROJECT_ROOT / "music_transformer" / "third_party"))

from midi_processor.processor import RANGE_NOTE_OFF, RANGE_NOTE_ON, RANGE_TIME_SHIFT, decode_midi


VELOCITY_START = RANGE_NOTE_ON + RANGE_NOTE_OFF + RANGE_TIME_SHIFT
TIME_SHIFT_START = RANGE_NOTE_ON + RANGE_NOTE_OFF


def velocity(value: int) -> int:
    return VELOCITY_START + value


def note_on(pitch: int) -> int:
    return pitch


def note_off(pitch: int) -> int:
    return RANGE_NOTE_ON + pitch


def time_shift(centiseconds: int) -> int:
    return TIME_SHIFT_START + centiseconds - 1


class MidiProcessorDecodeTest(unittest.TestCase):
    def decoded_notes(self, tokens: list[int]):
        with contextlib.redirect_stdout(io.StringIO()):
            midi = decode_midi(tokens)
        return midi.instruments[0].notes

    def test_stray_note_off_does_not_reuse_previous_note_on(self) -> None:
        tokens = [
            velocity(20),
            note_on(60),
            time_shift(50),
            note_off(60),
            time_shift(50),
            note_off(60),
        ]

        notes = self.decoded_notes(tokens)

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes[0].pitch, 60)
        self.assertAlmostEqual(notes[0].start, 0.0)
        self.assertAlmostEqual(notes[0].end, 0.5)

    def test_repeated_same_pitch_note_on_off_pairs_decode_as_separate_notes(self) -> None:
        tokens = [
            velocity(20),
            note_on(64),
            time_shift(25),
            note_off(64),
            time_shift(25),
            note_on(64),
            time_shift(25),
            note_off(64),
        ]

        notes = self.decoded_notes(tokens)

        self.assertEqual(len(notes), 2)
        self.assertEqual([note.pitch for note in notes], [64, 64])
        self.assertAlmostEqual(notes[0].start, 0.0)
        self.assertAlmostEqual(notes[0].end, 0.25)
        self.assertAlmostEqual(notes[1].start, 0.5)
        self.assertAlmostEqual(notes[1].end, 0.75)


if __name__ == "__main__":
    unittest.main()
