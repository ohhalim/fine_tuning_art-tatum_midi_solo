"""Synthetic contrast pairs (scripts/aria_cond_synth.py)."""
from __future__ import annotations

import unittest

from scripts.aria_cond_synth import EVAL, FAMILIES, TRAIN, example, first_discriminating_index, signature, target_mask


class SynthTest(unittest.TestCase):
    def test_pair_prefix_identical_and_only_shared_tones(self) -> None:
        for fam in FAMILIES:
            n = len(FAMILIES[fam][0])
            p, cand = example(fam, "P")
            q, _ = example(fam, "Q")
            self.assertEqual(p[:n], q[:n])
            self.assertTrue(all(x[0] % 12 in (0, 7) for x in p[:n]))
            self.assertEqual((p[n][0], q[n][0]), (cand["P"], cand["Q"]))
            self.assertEqual(cand["P"] - cand["Q"], 1)

    def test_splits_are_disjoint_even_under_transposition(self) -> None:
        self.assertEqual((len(TRAIN), len(EVAL)), (8, 4))
        self.assertFalse(set(TRAIN) & set(EVAL))
        self.assertFalse({signature(f) for f in TRAIN} & {signature(f) for f in EVAL})

    def test_everything_fits_in_one_five_second_segment(self) -> None:
        for fam in FAMILIES:
            for q in "PQ":
                notes, _ = example(fam, q)
                self.assertLess(max(n[2] for n in notes), 5.0)


    def test_loss_masks_by_shifted_target_index(self) -> None:
        toks = [("prefix", "instrument", "piano"), "<S>", ("piano", 60, 80), ("onset", 0), ("dur", 200),
                ("piano", 64, 80), ("onset", 250), ("dur", 200), ("piano", 67, 80), ("onset", 500), ("dur", 200), "<E>"]
        first = first_discriminating_index(toks, {"P": 64, "Q": 63})
        self.assertEqual(first, 5)
        a, b, c = (target_mask(toks, first, arm) for arm in "ABC")
        self.assertEqual(len(a), len(toks) - 1)
        self.assertTrue(all(a))
        self.assertEqual([k + 1 for k, m in enumerate(b) if m], list(range(5, 12)))       # pitch of the first discriminating note on, <E> included
        self.assertEqual([k + 1 for k, m in enumerate(c) if m], list(range(5, 11)))       # same without the <E> target
        self.assertTrue(b[first - 1] and c[first - 1])                                     # the discriminating pitch itself counts
        self.assertFalse(any(b[: first - 1]))                                              # prefix targets dropped


if __name__ == "__main__":
    unittest.main()
