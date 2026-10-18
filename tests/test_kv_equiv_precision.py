"""Gap metrics, noise floor and judgement of the #1815 equivalence contract (numpy only, no model)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import kv_equiv_precision as ke  # noqa: E402


def rows(n=4, v=9, seed=0):
    return np.random.default_rng(seed).normal(size=(n, v))


ALL = np.ones(9, dtype=bool)


class GapsTest(unittest.TestCase):
    def test_identical_rows_have_zero_gap(self):
        x = rows()
        g = ke.gaps(x, x.copy(), ALL)
        for m in ke.METRICS:
            self.assertAlmostEqual(g[m], 0.0, places=12)
        self.assertTrue(g["top1_last_equal"])
        self.assertEqual(g["positions"], 4)

    def test_max_and_last_follow_the_changed_row(self):
        x = rows()
        y = x.copy()
        y[1] += np.linspace(0, 3, 9)                 # only position 1 changes
        g = ke.gaps(x, y, ALL)
        self.assertGreater(g["kl_max"], 0)
        self.assertAlmostEqual(g["kl_last"], 0.0, places=12)
        self.assertAlmostEqual(g["masked_kl_last"], 0.0, places=12)

    def test_mask_hides_a_change_outside_the_allowed_tokens(self):
        x = rows()
        y = x.copy()
        y[-1, 0] += 5.0                              # token 0 is not allowed at the last position
        allowed = ALL.copy()
        allowed[0] = False
        g = ke.gaps(x, y, allowed)
        self.assertGreater(g["kl_last"], 0)
        self.assertAlmostEqual(g["masked_kl_last"], 0.0, places=12)
        self.assertAlmostEqual(g["masked_tv_last"], 0.0, places=12)


class JudgeTest(unittest.TestCase):
    def gap(self, v):
        return {m: v for m in ke.METRICS}

    def case(self, refs, C, pc=None, controls=None, exact=True):
        return {"R1_exact": exact, "refs": refs, "C": C, "PC_future": pc, "controls": controls or {}}

    def test_floor_is_the_largest_present_reference(self):
        f = ke.floor_of({"R1": self.gap(0.0), "R2": self.gap(0.01), "R3": self.gap(0.002), "R4": None})
        self.assertEqual(f["kl_max"], 0.01)

    def test_c_and_future_control_within_twice_the_floor(self):
        refs = {"R1": self.gap(0.0), "R2": self.gap(0.01), "R3": None, "R4": None}
        j = ke.judge(self.case(refs, self.gap(0.019), pc=self.gap(0.0)))
        self.assertTrue(j["C_ok"])
        self.assertTrue(j["PC_future_ok"])
        j = ke.judge(self.case(refs, self.gap(0.03), pc=self.gap(0.05)))
        self.assertFalse(j["C_ok"])
        self.assertFalse(j["PC_future_ok"])

    def test_zero_floor_allows_only_eps(self):
        refs = {"R1": self.gap(0.0), "R2": None, "R3": self.gap(0.0), "R4": None}
        self.assertTrue(ke.judge(self.case(refs, self.gap(5e-8)))["C_ok"])
        self.assertFalse(ke.judge(self.case(refs, self.gap(1e-5)))["C_ok"])

    def test_controls_use_their_own_position_floor(self):
        full = {"R1": self.gap(0.0), "R2": self.gap(0.2), "R3": None, "R4": None}       # large floor elsewhere
        tail = {"R1": self.gap(0.0), "R2": self.gap(0.001), "R3": None, "R4": None}     # small floor on the control's positions
        j = ke.judge(self.case(full, self.gap(0.0), controls={
            "NC_skip": {"gap": self.gap(0.5), "refs": tail},
            "NC_unfed": {"gap": self.gap(0.005), "refs": tail}, "NC_crop": None}))
        self.assertEqual(j["controls_detected"], {"NC_skip": True, "NC_unfed": False})
        self.assertFalse(j["controls_all_detected"])

    def test_inexact_repeat_marks_the_case_invalid(self):
        refs = {"R1": self.gap(0.0), "R2": None, "R3": None, "R4": None}
        self.assertFalse(ke.judge(self.case(refs, self.gap(0.0), exact=False))["valid"])


class FailureInputTest(unittest.TestCase):
    def test_nonfinite_logits_and_empty_mask_raise(self):
        x = rows()
        y = x.copy()
        y[0, 0] = np.nan
        with self.assertRaises(FloatingPointError):
            ke.gaps(x, y, ALL)
        with self.assertRaises(ValueError):
            ke.gaps(x, x, np.zeros(9, dtype=bool))


class ControlPointsTest(unittest.TestCase):
    def test_skip_needs_a_differing_token_before_the_restart(self):
        P, N = [1, 2, 3, 4, 5, 6], [1, 2, 9, 4, 5, 7]
        self.assertEqual(ke.control_points(P, N, 2, 2, False)["NC_skip"], 5)
        same_tail = ke.control_points([1, 2, 3], [1, 2, 3, 4], 3, 3, False)
        self.assertIsNone(same_tail["NC_skip"])           # nothing wrong would be kept

    def test_unfed_needs_enough_shared_prefix(self):
        P = list(range(20))
        N = list(range(12)) + [99, 98]
        cp = ke.control_points(P, N, 12, 12, False)
        self.assertEqual(cp["NC_unfed"], 12)
        self.assertIsNone(ke.control_points(P[:5] + [7], P[:5] + [8], 5, 5, False)["NC_unfed"])

    def test_crop_pair_uses_the_crop_control_only(self):
        P, N = [1, 2, 3, 4, 5], [1, 2, 8, 9]
        cp = ke.control_points(P, N, 2, 0, True)
        self.assertEqual(cp["NC_crop"], 3)
        self.assertIsNone(cp["NC_skip"])
        self.assertIsNone(cp["NC_unfed"])


if __name__ == "__main__":
    unittest.main()
