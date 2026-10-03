"""U2 verdict (scripts/harmony_condition_nll.py)."""
from __future__ import annotations

import unittest

from scripts.harmony_condition_nll import bootstrap_ci, judge


def m(mean, lo):
    return {"d_shuffled": {"mean": mean, "ci95": [lo, mean + 0.1]}}


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        self.assertTrue(judge({"base": m(0.2, 0.1), "bebop": m(0.0, -0.05)})["retrain_go"])     # does not read
        self.assertTrue(judge({"base": m(0.2, 0.1), "bebop": m(0.08, 0.02)})["retrain_go"])     # below half
        v = judge({"base": m(0.2, 0.1), "bebop": m(0.25, 0.15)})
        self.assertFalse(v["retrain_go"])
        self.assertTrue(v["no_go_explanation_rejected"])

    def test_bootstrap_ci_brackets_the_mean(self) -> None:
        lo, hi = bootstrap_ci([[1.0, 1.2], [0.8], [1.1, 0.9, 1.0]])
        self.assertLess(lo, 1.0)
        self.assertGreater(hi, 0.95)


if __name__ == "__main__":
    unittest.main()
