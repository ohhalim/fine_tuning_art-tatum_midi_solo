"""Representation audit helpers (scripts/rep_audit.py): top-voice proxy, CMT frame grid, simultaneity check."""
from __future__ import annotations

import unittest

from scripts.rep_audit import grid, simultaneity_changes, top_voice


class RepAuditTest(unittest.TestCase):
    def test_top_voice_keeps_highest_pitch_per_onset_group(self) -> None:
        notes = [(48, 0.0, 1.0, 60), (72, 0.02, 0.5, 80), (50, 0.5, 1.0, 60)]
        self.assertEqual([n[0] for n in top_voice(notes)], [72, 50])

    def test_grid_keeps_first_onset_per_eighth_second_frame(self) -> None:
        tv = [(60, 0.00, 0.1, 80), (62, 0.05, 0.1, 80), (64, 0.13, 0.2, 80)]
        frames, dropped = grid(tv)
        self.assertEqual(dropped, 1)
        self.assertEqual([p for p, _, _ in frames.values()], [60, 64])

    def test_simultaneity_counts_merged_and_split_close_pairs(self) -> None:
        a, b, c = (60, 1.000, 2, 80), (64, 1.004, 2, 80), (67, 1.000, 2, 80)
        pairs = [(a, (60, 1.00, 2, 80)), (b, (64, 1.00, 2, 80)), (c, (67, 1.01, 2, 80))]
        self.assertEqual(simultaneity_changes(pairs), (1, 1))


if __name__ == "__main__":
    unittest.main()
