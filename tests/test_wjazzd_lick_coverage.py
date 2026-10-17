"""Definitions of the #1784 WJazzD lick coverage count (synthetic notes, no database)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wjazzd_lick_coverage as wl  # noqa: E402


class LickCoverageTest(unittest.TestCase):
    def test_window_notes_take_onsets_inside_only(self):
        rows = [(0.9, 60, 0.3), (1.0, 62, 0.2), (2.99, 64, 0.5), (3.0, 65, 0.2)]
        self.assertEqual([n.pitch for n in wl.window_notes(rows, 1.0, 3.0)], [62, 64])

    def test_lick_window_fast_eighths_vs_slow_melody(self):
        fast = wl.window_notes([(i * 0.2, 60 + i % 5, 0.18) for i in range(16)], 0.0, 4.0)
        slow = wl.window_notes([(i * 0.5, 60 + i % 5, 0.45) for i in range(8)], 0.0, 4.0)
        self.assertTrue(wl.is_lick_window(fast))
        self.assertFalse(wl.is_lick_window(slow))                      # median IOI 0.5 s > 0.26 s

    def test_offbeat_positions_straight_and_swing(self):
        beats = [0.0, 0.5, 1.0, 1.5]
        straight = wl.window_notes([(b + d, 60, 0.2) for b in beats for d in (0.0, 0.25)], 0.0, 2.0)
        swing = wl.window_notes([(b + d, 60, 0.2) for b in beats for d in (0.0, 1 / 3)], 0.0, 2.0)
        self.assertEqual(wl.offbeat_positions(straight, beats, 2.0), [0.5] * 4)
        self.assertTrue(all(abs(p - 2 / 3) < 1e-9 for p in wl.offbeat_positions(swing, beats, 2.0)))


if __name__ == "__main__":
    unittest.main()
