"""Computer-keyboard MIDI input (scripts/typing_keyboard.py, #1581)."""
from __future__ import annotations

import unittest

from inference.control.live_chords import recognize
from scripts.typing_keyboard import TypingKeyboard


class TypingKeyboardTest(unittest.TestCase):
    def test_keys_pressed_together_form_one_held_chord(self) -> None:
        kb = TypingKeyboard(chord_gap=0.15)
        ev = kb.press("z", 0.00) + kb.press("c", 0.03) + kb.press("b", 0.06)    # C3 E3 G3
        self.assertEqual(ev, [("note_on", 48), ("note_on", 52), ("note_on", 55)])
        self.assertEqual(recognize(kb.chord), "Cmaj7")
        self.assertTrue(all(p < 60 for p in kb.chord))                          # below the default split
        self.assertEqual(kb.due(5.0), [])                                       # chords do not time out

    def test_a_new_chord_gesture_releases_the_old_one(self) -> None:
        kb = TypingKeyboard(chord_gap=0.15)
        kb.press("x", 0.0); kb.press("v", 0.02); kb.press("n", 0.04)            # D F A
        ev = kb.press("b", 1.0)
        self.assertEqual(ev, [("note_off", 50), ("note_off", 53), ("note_off", 57), ("note_on", 55)])
        self.assertEqual(kb.press(" ", 1.5), [("note_off", 55)])

    def test_uppercase_and_repeats_inside_a_chord(self) -> None:
        kb = TypingKeyboard()
        self.assertEqual(kb.press("Z", 0.0) + kb.press("z", 0.05), [("note_on", 48)])

    def test_melody_notes_end_after_their_length_and_follow_octave_shift(self) -> None:
        kb = TypingKeyboard(melody_length=0.25)
        self.assertEqual(kb.press("q", 0.0), [("note_on", 60)])
        self.assertEqual(kb.due(0.2), [])
        self.assertEqual(kb.due(0.25), [("note_off", 60)])
        kb.press("]", 1.0)
        self.assertEqual(kb.press("w", 1.0), [("note_on", 74)])
        self.assertEqual(kb.press("w", 1.1), [("note_off", 74), ("note_on", 74)])   # restart, one off pending
        self.assertEqual(kb.due(2.0), [("note_off", 74)])

    def test_release_all_and_unknown_keys(self) -> None:
        kb = TypingKeyboard()
        kb.press("z", 0.0); kb.press("q", 0.0)
        self.assertEqual(kb.press("k", 0.1), [])
        self.assertEqual(sorted(kb.release_all()), [("note_off", 48), ("note_off", 60)])


if __name__ == "__main__":
    unittest.main()
