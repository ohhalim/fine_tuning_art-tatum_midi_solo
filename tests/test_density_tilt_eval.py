"""Pure helpers of the #1844 evaluation (no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import density_tilt_eval as de  # noqa: E402


class PickLamTest(unittest.TestCase):
    def rows(self, med):
        return {k: {"median_notes": m, "max_notes": mx} for k, (m, mx) in med.items()}

    def test_smallest_in_range(self):
        lam, _, _ = de.pick_lam(self.rows({"0": (10, 14), "0.1": (9, 13), "0.25": (8, 12), "0.5": (7.8, 11), "1.0": (5, 9)}))
        self.assertEqual(lam, "0.25")

    def test_fallback_closest_capped(self):
        lam, _, why = de.pick_lam(self.rows({"0": (10, 14), "0.1": (9.9, 13), "0.25": (9.5, 12), "0.5": (6, 10), "1.0": (4, 8)}))
        self.assertEqual(lam, "0.25")       # drops 5% vs 40%: 5% is closer to 20%
        self.assertIn("미달", why)


class SignTestTest(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(de.sign_test_p(0, 3), 0.25)
        self.assertEqual(de.sign_test_p(2, 4), 1.0)
        self.assertIsNone(de.sign_test_p(0, 0))


if __name__ == "__main__":
    unittest.main()


class ReviewFixesTest(unittest.TestCase):
    """Astra review of #1845: length differences, overflow-free tilt, finite CLI values."""

    def test_extra_rows_or_notes_count_as_a_difference(self):
        import density_tilt_eval as de
        a = [{"block": 1, "tokens": 5, "notes": 2, "S": 3}]
        b = a + [{"block": 2, "tokens": 6, "notes": 2, "S": 3}]
        self.assertEqual(de.first_block_diff(a, b), 2)
        self.assertEqual(de.first_note_on_diff([(60, 80, 0, 1)], [(60, 80, 0, 1), (62, 80, 10, 1)]), 1)
        self.assertIsNone(de.first_note_on_diff([(60, 80, 0, 1)], [(60, 80, 0, 1), (62, 80, 10, 5)], comparable_until=3))

    def test_large_tilt_does_not_overflow(self):
        import torch
        import rt_lh_input as li
        p = torch.tensor([0.5, 0.5], dtype=torch.float32)
        allowed = torch.tensor([True, True])
        out = li.tilted_probs(p, torch.tensor([1]), torch.tensor([20 * 1 * 468.75 / 100.0]), allowed)
        self.assertTrue(torch.isfinite(out).all())
        self.assertGreater(out[1].item(), out[0].item())
        same = li.tilted_probs(p, torch.tensor([1]), torch.tensor([0.0]), allowed)
        self.assertAlmostEqual(same[0].item(), same[1].item(), places=6)

    def test_rt_play_refuses_non_finite_tilt(self):
        import argparse
        import rt_play as rp
        for bad in ("nan", "inf", "-1"):
            with self.assertRaises(argparse.ArgumentTypeError):
                rp.non_negative_float(bad)
        self.assertEqual(rp.non_negative_float("1.5"), 1.5)
