"""Arm order and divergence math of the #1811 precision probe (numpy only, no model)."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import aria_precision_probe as ap  # noqa: E402


class ArmOrderTest(unittest.TestCase):
    def test_each_arm_takes_each_slot_once_over_four_reps(self):
        orders = [ap.arm_order(r) for r in range(4)]
        for o in orders:
            self.assertEqual(sorted(o), sorted(a for a, _ in ap.ARMS))
        for slot in range(4):
            self.assertEqual(len({o[slot] for o in orders}), 4)


class DivergenceTest(unittest.TestCase):
    def test_identical_logits_give_zero_kl_and_tv(self):
        x = np.random.default_rng(0).normal(size=(5, 11)).astype(np.float32)
        d = ap.divergence(x, x.copy(), np.arange(5))
        self.assertTrue(np.allclose(d["kl"], 0, atol=1e-12))
        self.assertTrue(np.allclose(d["tv"], 0, atol=1e-12))
        self.assertTrue(d["top1"].all())
        self.assertTrue(np.allclose(d["nll"], d["nll_ref"]))

    def test_known_two_way_split(self):
        ref = np.log(np.array([[0.5, 0.5]]))
        arm = np.log(np.array([[0.9, 0.1]]))
        d = ap.divergence(ref, arm, np.array([1]))
        self.assertAlmostEqual(float(d["kl"][0]), 0.5 * np.log(0.5 / 0.9) + 0.5 * np.log(0.5 / 0.1), places=9)
        self.assertAlmostEqual(float(d["tv"][0]), 0.4, places=9)
        self.assertAlmostEqual(float(d["nll"][0]), -np.log(0.1), places=9)
        self.assertAlmostEqual(float(d["entropy_ref"][0]), np.log(2), places=9)

    def test_shift_invariance_of_logits(self):
        x = np.random.default_rng(1).normal(size=(3, 7))
        d = ap.divergence(x, x + 100.0, np.arange(3))
        self.assertTrue(np.allclose(d["kl"], 0, atol=1e-9))


if __name__ == "__main__":
    unittest.main()
