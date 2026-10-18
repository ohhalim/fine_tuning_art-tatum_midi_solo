"""Reuse rule and case selection of #1795 (no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lh_cache_replay as cr  # noqa: E402


class ReuseRuleTest(unittest.TestCase):
    def test_reuse_start(self):
        self.assertEqual(cr.reuse_start(valid_end=99, lcp=50, n_len=120, crop_changed=False), 50)    # change inside the KV
        self.assertEqual(cr.reuse_start(valid_end=99, lcp=100, n_len=120, crop_changed=False), 99)   # unfed last token: stop at V
        self.assertEqual(cr.reuse_start(valid_end=100, lcp=80, n_len=80, crop_changed=False), 79)    # S = 0: recompute the last token
        self.assertEqual(cr.reuse_start(valid_end=100, lcp=80, n_len=120, crop_changed=True), 0)     # crop: full reset

    def test_pick_cases(self):
        rows = [{"song": 44, "k": k, "P": P, "S": S, "cause": c} for k, (P, S, c) in enumerate([
            (1700, 120, "expired"), (2100, 250, "held_update"), (2900, 300, "expired"), (3300, 240, "insert"),
            (0, 900, "crop"), (500, 0, "none"), (900, 0, "none")], start=2)]
        cases = cr.pick_cases(rows)
        self.assertEqual(cases["longP_1692"]["P"], 1700)
        self.assertEqual(cases["longP_3343"]["P"], 3300)
        self.assertEqual(cases["crop"]["cause"], "crop")
        self.assertEqual(cases["s0"]["P"], 900)                              # largest P among no-change pairs


if __name__ == "__main__":
    unittest.main()
