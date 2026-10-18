"""Cost fit and recompute accounting of the #1820 window-variant analysis (no model, no data)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lh_window_suffix as lw  # noqa: E402


class CostFitTest(unittest.TestCase):
    def test_exact_recovery_of_a_known_surface(self):
        a, b, c = 5.0, 0.3, 1e-5
        pts = [(p, s, a + b * s + c * s * p) for p, s in ((1000, 100), (2000, 250), (3000, 300), (3500, 120))]
        got = lw.fit_cost(pts)
        for x, y in zip(got, (a, b, c)):
            self.assertAlmostEqual(x, y, places=6)
        self.assertAlmostEqual(lw.cost_at(got, 1500, 200), a + b * 200 + c * 200 * 1500, places=6)


class RecomputeTest(unittest.TestCase):
    def test_crop_pairs_count_their_whole_request(self):
        rows = [{"cause": "expired", "S": 100, "L": 900}, {"cause": "crop", "S": 0, "L": 600}, {"cause": "none", "S": 0, "L": 500}]
        self.assertAlmostEqual(lw.recompute_per_pair(rows), (100 + 600 + 0) / 3)
        self.assertIsNone(lw.recompute_per_pair([]))


if __name__ == "__main__":
    unittest.main()
