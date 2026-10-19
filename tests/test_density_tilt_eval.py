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
