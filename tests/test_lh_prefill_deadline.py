"""Interpolation and prediction rules of the #1791 measurement (no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lh_prefill_deadline as lp  # noqa: E402


class PredictTest(unittest.TestCase):
    def test_interp_linear_and_clamped(self):
        xs, ys = [100, 200, 400], [10.0, 20.0, 60.0]
        self.assertEqual(lp.interp(xs, ys, 150), 15.0)
        self.assertEqual(lp.interp(xs, ys, 300), 40.0)
        self.assertEqual(lp.interp(xs, ys, 50), 10.0)
        self.assertEqual(lp.interp(xs, ys, 999), 60.0)

    def test_predict_adds_first_and_rest_decode(self):
        curve = [(100, 100.0), (1000, 400.0)]
        ms, s = lp.predict([(100, 0), (100, 1), (1000, 11)], curve, decode_first_ms=20.0, decode_rest_ms=10.0, deadline=450)
        self.assertEqual(ms, [100.0, 120.0, 400.0 + 20.0 + 10 * 10.0])
        self.assertEqual(s["within_deadline"], 2)
        self.assertEqual(s["max_ms"], 520.0)


if __name__ == "__main__":
    unittest.main()
