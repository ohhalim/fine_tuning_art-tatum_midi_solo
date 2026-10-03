"""Guide adapter generation verdict (scripts/guide_gen_eval.py)."""
from __future__ import annotations

import unittest

from scripts.guide_gen_eval import judge, pooled


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        real = {"notes_per_s": 4.0}
        bebop = {"rel_js": 0.12, "fit": 0.40, "notes_per_s": 5.0}
        good = {"rel_js": 0.08, "fit": 0.45, "notes_per_s": 4.5}
        self.assertTrue(judge(real, bebop, good)["pass"])
        self.assertFalse(judge(real, bebop, {**good, "rel_js": 0.13})["pass"])
        self.assertFalse(judge(real, bebop, {**good, "rel_js": 0.11, "fit": 0.45})["pass"])
        self.assertFalse(judge(real, bebop, {**good, "fit": 0.42})["pass"])
        self.assertFalse(judge(real, bebop, {**good, "notes_per_s": 7.0})["pass"])

    def test_pooled_handles_empty_windows(self) -> None:
        ref = [1.0] * 12
        p = pooled([([], {0, 4, 7}, 0), ([(60, 0.0, 0.5), (61, 0.5, 0.9)], {0, 4, 7}, 0)], ref)
        self.assertEqual((p["windows"], p["windows_with_notes"]), (2, 1))
        self.assertAlmostEqual(p["fit"], 0.5 / 0.9, places=3)


if __name__ == "__main__":
    unittest.main()
