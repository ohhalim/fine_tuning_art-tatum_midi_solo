"""Repeat-weight pilot verdict (scripts/repeat_pilot_eval.py)."""
from __future__ import annotations

import unittest

from scripts.repeat_pilot_eval import judge


def summary(ir_a=0.007, ir_b=0.02, ci=(0.005, 0.02), ce=0.01, rep=0.02, copy=0.0, valid=1.0):
    arm = lambda ir: {"interval": ir, "repetition": rep, "copy_rate": copy, "valid_rate": valid}
    return {"real": {"interval": 0.052, "repetition": 0.073, "copy_rate": 0.01},
            "rollout": {"A": {"1.0": arm(ir_a)}, "B": {"1.0": arm(ir_b)}},
            "paired_ir_1.0": {"mean": ir_b - ir_a, "ci95": list(ci)},
            "ce": {"fresh12_B_minus_A": ce}}


class JudgeTest(unittest.TestCase):
    def test_passes_with_doubled_ir_and_guards(self) -> None:
        self.assertTrue(judge(summary())["pass"])

    def test_each_condition_can_fail_it(self) -> None:
        self.assertFalse(judge(summary(ir_b=0.010, ci=(0.001, 0.005)))["pass"])     # less than 2x
        self.assertFalse(judge(summary(ci=(-0.001, 0.02)))["pass"])                 # CI includes 0
        self.assertFalse(judge(summary(ce=0.03))["pass"])                            # likelihood worse
        self.assertFalse(judge(summary(rep=0.3))["pass"])                            # mechanical repetition
        self.assertFalse(judge(summary(copy=0.5))["pass"])                           # copying
        self.assertFalse(judge(summary(valid=0.8))["pass"])


if __name__ == "__main__":
    unittest.main()
