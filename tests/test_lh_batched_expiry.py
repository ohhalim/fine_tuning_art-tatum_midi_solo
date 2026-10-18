"""Window-start grids, expiry tagging and the verdict rule of the #1831 batched-expiry analysis (no model, no data)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lh_batched_expiry as be  # noqa: E402

B = be.BLOCK


class GridStartTest(unittest.TestCase):
    def test_window_always_covers_w_and_start_moves_every_n_blocks(self):
        t0 = 1234.0
        rule = be.block_grid(4)
        starts = []
        for k in range(1, 40):
            s = t0 + k * B
            st = rule({"s": s, "k": k}, -1e9)
            self.assertLessEqual(st, s - be.W_MS)                 # at least 7.5 s observed
            self.assertGreater(st, s - be.W_MS - 4 * B)          # less than one grid step more
            starts.append(st)
        changes = sum(1 for a, b in zip(starts, starts[1:]) if a != b)
        self.assertLessEqual(changes, len(starts) // 4 + 1)

    def test_base_clamps_and_segment_grid(self):
        self.assertEqual(be.grid_start(20000.0, 15000.0, 0.0, 5000.0), 15000.0)    # base later than the grid point
        self.assertEqual(be.segment_grid({"s": 18000.0, "k": 5}, 0.0), 10000.0)    # 18000 - 7500 = 10500 -> 10000
        self.assertEqual(be.segment_grid({"s": 17500.0, "k": 5}, 0.0), 10000.0)    # exactly on the grid


class ExpiryTagTest(unittest.TestCase):
    def test_expiry_is_a_moved_start(self):
        reqs = [{"k": 1, "s": 10000.0, "window_ms": 7500.0}, {"k": 2, "s": 10937.5, "window_ms": 8437.5},
                {"k": 3, "s": 11875.0, "window_ms": 7500.0}]
        rows = [{"k": 2, "cause": "none", "S": 0}, {"k": 3, "cause": "expired", "S": 120}]
        out = be.tag_expiry(rows, reqs)
        self.assertEqual([x["expiry"] for x in out], [False, True])
        self.assertEqual(out[1]["start"], 4375.0)


class VerdictTest(unittest.TestCase):
    def cur(self):
        return {"S": {"p90": 248, "max": 353}, "crop_pairs": 8, "large_fraction": 0.2}

    def test_all_conditions_needed(self):
        good = {"S_non_expiry": {"p90": 100}, "S": {"max": 400}, "crop_pairs": 12, "large_fraction": 0.1}
        self.assertTrue(be.verdict(good, self.cur())["worth_contract_experiment"])
        for key, bad in (("S_non_expiry", {"p90": 130}), ("S", {"max": 450}), ("crop_pairs", 13), ("large_fraction", 0.11)):
            v = dict(good, **{key: bad})
            self.assertFalse(be.verdict(v, self.cur())["worth_contract_experiment"], key)


if __name__ == "__main__":
    unittest.main()
