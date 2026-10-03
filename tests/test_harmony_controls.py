"""U3 metrics and controls (scripts/harmony_controls.py)."""
from __future__ import annotations

import random
import unittest

from scripts.harmony_controls import controls, js_distance, window_metrics


class MetricTest(unittest.TestCase):
    def test_fit_clash_resolution(self) -> None:
        pcs = {0, 4, 7}                                          # C E G
        solo = [(73, 0.0, 0.1), (72, 0.1, 0.2), (66, 0.2, 0.3), (60, 0.9, 1.0)]   # Db->C resolves, F# leaps away
        m = window_metrics(solo, pcs)
        self.assertAlmostEqual(m["fit"], 0.5)
        self.assertAlmostEqual(m["clash"], 0.5)                 # Db and F# are a semitone from C / G
        self.assertAlmostEqual(m["resolution"], 0.5)            # F#: next note 0.7 s later, too late

    def test_enclosure_counts_as_resolution(self) -> None:
        m = window_metrics([(74, 0.0, 0.1), (71, 0.1, 0.2), (72, 0.2, 0.3)], {0, 4, 7})   # D B C
        self.assertEqual(m["resolution"], 1.0)

    def test_joint_transpose_is_invariant(self) -> None:
        w = {"solo": [(73, 0.0, 0.1), (72, 0.1, 0.2), (66, 0.2, 0.3)], "pcs": {0, 4, 7}, "bass": 0}
        c = controls(w, w, random.Random(1))
        a, b = window_metrics(*c["original"][:2]), window_metrics(*c["joint_transpose"][:2])
        self.assertEqual({k: a[k] for k in ("fit", "clash", "resolution")},
                         {k: b[k] for k in ("fit", "clash", "resolution")})

    def test_js_distance(self) -> None:
        self.assertAlmostEqual(js_distance([1, 0, 0], [1, 0, 0]), 0.0)
        self.assertGreater(js_distance([1, 0, 0], [0, 1, 0]), 0.8)


if __name__ == "__main__":
    unittest.main()
