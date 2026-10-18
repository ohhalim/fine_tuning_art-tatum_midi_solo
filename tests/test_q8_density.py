"""Hodges–Lehmann interval and the preregistered verdict of the #1832 density unit (pure Python, no model)."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import q8_density as qd  # noqa: E402


class HodgesLehmannTest(unittest.TestCase):
    def test_small_known_case(self):
        hl, (lo, hi) = qd.hodges_lehmann([1.0, 2.0, 3.0], z=0.0)
        self.assertEqual(hl, 2.0)                                     # Walsh averages 1 1.5 2 2 2.5 3
        self.assertEqual((lo, hi), (2.0, 2.0))                        # z = 0: k = floor(3) -> middle pair

    def test_interval_contains_estimate_and_shifts(self):
        rng = random.Random(0)
        d = [rng.gauss(0, 1) for _ in range(200)]
        hl, (lo, hi) = qd.hodges_lehmann(d)
        self.assertTrue(lo <= hl <= hi)
        hl2, (lo2, hi2) = qd.hodges_lehmann([x - 0.5 for x in d])
        self.assertAlmostEqual(hl2, hl - 0.5, places=9)
        self.assertAlmostEqual(lo2, lo - 0.5, places=9)
        self.assertAlmostEqual(hi2, hi - 0.5, places=9)

    def test_coverage_of_99_interval_under_null(self):
        rng = random.Random(1)
        miss = 0
        for _ in range(300):
            _, (lo, hi) = qd.hodges_lehmann([rng.gauss(0, 1) for _ in range(60)])
            miss += not (lo <= 0 <= hi)
        self.assertLessEqual(miss, 12)                                # expected about 3 of 300

    def test_empty(self):
        self.assertEqual(qd.hodges_lehmann([]), (None, (None, None)))


class VerdictTest(unittest.TestCase):
    def test_rule(self):
        self.assertEqual(qd.verdict(-0.2, -0.05), "q8 성김 검출, 실용 차이 범위 안")
        self.assertEqual(qd.verdict(-0.6, -0.1), "q8 성김 검출, 실용 차이 범위 밖 가능")
        self.assertEqual(qd.verdict(0.05, 0.2), "q8 조밀함 검출, 실용 차이 범위 안")
        self.assertEqual(qd.verdict(-0.1, 0.2), "실용 차이 없음")
        self.assertEqual(qd.verdict(-0.5, 0.1), "미결")
        self.assertEqual(qd.verdict(-0.3, 0.3), "실용 차이 없음")       # margin endpoints included


class SeedsTest(unittest.TestCase):
    def test_new_seeds_are_disjoint_from_1823(self):
        import q8_free_gen as fg
        self.assertFalse(set(qd.SEEDS) & set(fg.SEEDS))
        self.assertEqual(len(qd.SEEDS) * len(qd.PROMPTS), 768)


class SeedClusterBlockTest(unittest.TestCase):
    def test_seed_means_and_rule(self):
        import q8_density as qd
        A = {(p, s): {"stats": {"x": 5.0}} for p in (0, 1) for s in range(1, 11)}
        B = {(p, s): {"stats": {"x": 5.0 + (0.01 if p == 0 else -0.01)}} for p in (0, 1) for s in range(1, 11)}
        out = qd.seed_cluster_block(A, B, sorted(A), "x")
        self.assertEqual(out["seeds"], 10)
        self.assertAlmostEqual(out["hl_diff"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
