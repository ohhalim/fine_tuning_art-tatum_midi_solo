"""Top-line continuity and motif reuse (docs/experiments/COHERENCE_GOAL.md)."""
from __future__ import annotations

import unittest

from scripts.coherence_metrics import jumps, motif_reuse, top_line


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


if __name__ == "__main__":
    unittest.main()
