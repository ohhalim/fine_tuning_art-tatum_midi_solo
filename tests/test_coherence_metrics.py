"""Top-line continuity and motif reuse (docs/experiments/COHERENCE_GOAL.md)."""
from __future__ import annotations

import unittest

from scripts.coherence_metrics import generation_block_s, jumps, motif_reuse, summarize, top_line


class CoherenceTest(unittest.TestCase):
    def test_top_line_takes_the_highest_note_of_each_cluster(self) -> None:
        self.assertEqual(top_line([(0.0, 40), (0.01, 72), (0.5, 74), (0.52, 50)]), [(0.0, 72), (0.5, 74)])

    def test_jumps_split_at_block_seams(self) -> None:
        line = [(0.1, 60), (0.5, 62), (1.1, 74), (1.5, 76)]
        self.assertEqual(jumps(line, 1.0), ([12], [2, 2]))

    def test_motif_reuse_needs_an_earlier_non_overlapping_copy_within_the_horizon(self) -> None:
        motif = [0, 2, 4, 5]
        line = [(i * 0.2, 60 + p) for i, p in enumerate(motif + [9] + motif)]
        self.assertEqual(motif_reuse(line, horizon_s=8.0), (1, 6))
        self.assertEqual(motif_reuse(line, horizon_s=0.5), (0, 6))

    def test_each_line_is_cut_with_its_own_block_length(self) -> None:
        # Astra's reproduction: at 120 BPM (half bar 1.0 s) the seam is at 1.0 s.
        line = [(0.90, 60), (0.95, 72), (1.05, 73)]
        s120 = summarize([(line, 1.0)])
        self.assertEqual((s120["boundary_mean"], s120["within_mean"]), (1, 12))
        mixed = summarize([(line, 1.0), ([(0.1, 60), (0.5, 62)], 0.9375)])
        self.assertEqual(mixed["boundary_mean"], 1)

    def test_generation_block_length_from_the_report(self) -> None:
        self.assertAlmostEqual(generation_block_s({"bpm": 120, "block_beats": 2}), 1.0)
        self.assertAlmostEqual(generation_block_s({"bpm": 128, "chord_blocks_per_bar": 2}), 0.9375)
        self.assertAlmostEqual(generation_block_s({"bpm": 120}), 2.0)


if __name__ == "__main__":
    unittest.main()
