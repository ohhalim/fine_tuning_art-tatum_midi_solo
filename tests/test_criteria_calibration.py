"""Run splitting and the separation rule of the #1762 calibration (no MIDI files)."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import criteria_calibration as cc  # noqa: E402


class CalibrationTest(unittest.TestCase):
    def test_runs_split_on_gaps_over_quarter_second(self):
        line = [SimpleNamespace(start=s, end=s + 0.1) for s in (0.0, 0.2, 0.4, 1.0, 1.2)]
        self.assertEqual([len(r) for r in cc.runs(line)], [3, 2])           # 0.4+0.1 -> 1.0 is a 0.5 s gap

    def test_separation_counts_items_outside_liked_range(self):
        res = cc.separation({"L": [1, 2, 3], "N": [2, 9, 10], "R1": [2.5, 2.0], "R2": [None]}, "x")
        self.assertEqual(res["L_range"], [1, 3])
        self.assertEqual(res["N"], {"outside": 2, "n": 3, "majority_outside": True})
        self.assertEqual(res["R1"], {"outside": 0, "n": 2, "majority_outside": False})
        self.assertNotIn("R2", res)


if __name__ == "__main__":
    unittest.main()
