"""Guide dataset stream (scripts/build_guide_dataset.py)."""
from __future__ import annotations

import unittest

import pretty_midi

from inference.control.harmony_contract import guide_pitches
from scripts.build_guide_dataset import WINDOW_S, song_stream
from scripts.style_distance import tokens_to_notes


def note(p, s, e):
    return pretty_midi.Note(velocity=80, pitch=p, start=s, end=e)


class StreamTest(unittest.TestCase):
    def test_guide_before_each_window_and_carry(self) -> None:
        w = WINDOW_S
        notes = [note(36, 0.0, w), note(52, 0.0, w), note(31, 0.0, w),          # C E G under window 1 (below G3)
                 note(72, 0.0, 0.2), note(74, 0.3, 0.5),
                 note(43, w, w + 0.2),                                         # one pc only: carry
                 note(76, w + 0.1, w + 0.3)]
        toks, st = song_stream(notes)
        self.assertEqual((st["windows"], st["new_guide"], st["carried"], st["no_guide"]), (2, 1, 1, 0))
        decoded = tokens_to_notes(toks)
        guide = guide_pitches(7, {0, 4, 7})                                 # lowest note G1 is the bass
        low = [n.pitch for n in decoded if n.pitch < 60]
        self.assertEqual(low, guide + guide)                                   # the guide repeats, carried
        self.assertEqual([n.pitch for n in decoded if n.pitch >= 60], [72, 74, 76])


if __name__ == "__main__":
    unittest.main()
