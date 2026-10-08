"""T4b contract v2 helpers (scripts/aria_t4b_sim.py): absolute grid, block acceptance, context normalization."""
from __future__ import annotations

import unittest

from scripts.aria_t4b_sim import accept, grid_ms, normalize_context


class GridTest(unittest.TestCase):
    def test_absolute_grid_over_40_blocks_has_no_drift_or_overlap(self) -> None:
        t0 = 8.11
        qs = [grid_ms(t0, j) for j in range(41)]
        self.assertTrue(all(q % 10 == 0 for q in qs))
        self.assertTrue(all(b - a in (930, 940) for a, b in zip(qs, qs[1:])))
        self.assertTrue(all(abs(q - (t0 * 1000 + j * 937.5)) <= 5 for j, q in enumerate(qs)))
        self.assertEqual(qs[40] - qs[0], 37500)                       # 40 blocks = 37.5 s exactly

    def test_boundary_onset_belongs_to_the_next_block(self) -> None:
        qj, qn = grid_ms(8.11, 0), grid_ms(8.11, 1)
        gen = [(60, (qn - 1) / 1000, (qn + 100) / 1000, 70), (62, qn / 1000, (qn + 100) / 1000, 70), (64, (qj - 1) / 1000, qj / 1000 + 0.05, 70)]
        acc, clipped, lost = accept(gen, qj, qn)
        self.assertEqual([n[0] for n in acc], [60])                   # one tick before the boundary stays, on it moves on
        self.assertEqual(clipped, 1)
        self.assertAlmostEqual(acc[0][2], qn / 1000)
        self.assertAlmostEqual(lost, 0.1, places=6)


class NormalizeTest(unittest.TestCase):
    def test_history_note_sounding_into_a_guide_of_the_same_pitch_ends_there(self) -> None:
        notes, log = normalize_context([(47, 7.65, 8.71, 60, "history"), (47, 8.11, 9.05, 60, "guide")])
        self.assertEqual(notes, [(47, 7.65, 8.11, 60, "history"), (47, 8.11, 9.05, 60, "guide")])
        self.assertEqual(log["counts"], {"truncated:history<-guide": 1})
        self.assertAlmostEqual(log["lost_s"]["history<-guide"], 0.6)

    def test_same_onset_same_pitch_keeps_the_guide_and_logs_the_drop(self) -> None:
        notes, log = normalize_context([(52, 9.98, 10.4, 70, "generated"), (52, 9.98, 10.92, 60, "guide")])
        self.assertEqual(notes, [(52, 9.98, 10.92, 60, "guide")])
        self.assertEqual(log["counts"], {"same_onset_drop:guide>generated": 1})

    def test_same_onset_generated_pair_keeps_the_longer_and_logs(self) -> None:
        notes, log = normalize_context([(60, 1.0, 1.2, 50, "generated"), (60, 1.0, 1.5, 80, "generated")])
        self.assertEqual(notes, [(60, 1.0, 1.5, 80, "generated")])
        self.assertEqual(log["counts"], {"same_onset_drop:generated>generated": 1})

    def test_originals_are_not_modified(self) -> None:
        tagged = [(47, 7.65, 8.71, 60, "history"), (47, 8.11, 9.05, 60, "guide")]
        normalize_context(tagged)
        self.assertEqual(tagged[0], (47, 7.65, 8.71, 60, "history"))


if __name__ == "__main__":
    unittest.main()
