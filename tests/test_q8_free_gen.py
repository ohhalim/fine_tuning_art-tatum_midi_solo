"""Run statistics and the paired comparison of the #1823 free-generation unit (pure Python, no model)."""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import q8_free_gen as fg  # noqa: E402


class RunStatsTest(unittest.TestCase):
    def test_known_run(self):
        t0 = 1000
        notes = [(1000, 1500, 60, 70), (1010, 1400, 64, 70),        # a two-note chord (onsets 10 ms apart)
                 (2000, 2200, 67, 80), (3000, 3100, 72, 90),
                 (40_000, 40_100, 50, 60)]                            # outside the 30 s span: ignored
        s = fg.run_stats(notes, t0)
        self.assertEqual(s["notes"], 4)
        self.assertAlmostEqual(s["notes_per_s"], 4 / 30)
        self.assertEqual(s["pitch_median"], 65.5)
        self.assertEqual(s["pitch_range"], 12)
        self.assertAlmostEqual(s["chord_share"], 0.5)
        self.assertAlmostEqual(s["ioi_median_s"], 1.0)                 # cluster starts 1000, 2000, 3000
        self.assertAlmostEqual(s["dur_median_s"], 0.295)               # durations .5 .39 .2 .1 -> (.2 + .39) / 2
        self.assertEqual(s["vel_median"], 75)
        self.assertAlmostEqual(s["rest_share"], 1 - (500 + 200 + 100) / 30_000)
        self.assertEqual(s["win_max"], 2)
        self.assertEqual(s["win_median"], 0)

    def test_empty_run(self):
        s = fg.run_stats([], 0)
        self.assertEqual(s["notes"], 0)
        self.assertIsNone(s["pitch_median"])


class TestsTest(unittest.TestCase):
    def test_ranks_with_ties(self):
        self.assertEqual(fg.ranks([3, 1, 3, 2]), [3.5, 1, 3.5, 2])

    def test_wilcoxon_all_positive(self):
        w = fg.wilcoxon([1, 2, 3, 4, 5])
        self.assertEqual(w["n"], 5)
        self.assertEqual(w["w_plus"], 15)
        self.assertAlmostEqual(w["z"], 7.5 / math.sqrt(13.75))
        self.assertAlmostEqual(w["p"], 0.0431, places=3)
        self.assertEqual(w["r"], 1.0)

    def test_wilcoxon_symmetric_and_zero(self):
        self.assertAlmostEqual(fg.wilcoxon([1, -1, 2, -2])["r"], 0.0)
        self.assertEqual(fg.wilcoxon([0, 0])["p"], 1.0)

    def test_cliffs_delta(self):
        self.assertEqual(fg.cliffs_delta([1, 2], [3, 4]), 1.0)
        self.assertEqual(fg.cliffs_delta([1, 2], [1, 2]), 0.0)

    def test_detection_needs_p_and_effect(self):
        big = fg.compare_stat([(i, i + 1 + (i % 3)) for i in range(64)])
        self.assertTrue(big["detected"])
        none = fg.compare_stat([(i, i + (1 if i % 2 else -1)) for i in range(64)])
        self.assertFalse(none["detected"])

    def test_first_divergence(self):
        self.assertEqual(fg.first_divergence([1, 2, 3], [1, 2, 4]), 2)
        self.assertIsNone(fg.first_divergence([1, 2], [1, 2]))
        self.assertEqual(fg.first_divergence([1, 2], [1, 2, 3]), 2)


class InputChecksTest(unittest.TestCase):
    """Astra review of #1830: empty, missing, duplicate, invalid or non-duration runs give no verdict."""

    def write(self, d, name, rows):
        import json
        p = os.path.join(d, name)
        with open(p, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        return p

    def row(self, prec, prompt, seed, valid=True, stop="duration"):
        return {"precision": prec, "prompt": prompt, "seed": seed, "valid": valid, "stop": stop}

    def test_empty_files_fail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            ok, info = fg.input_checks(self.write(d, "a", []), self.write(d, "b", []), [(0, 1)])
            self.assertFalse(ok)
            self.assertEqual(info["fp32"]["missing"], [(0, 1)])

    def test_duplicate_invalid_and_max_tokens_fail(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            a = self.write(d, "a", [self.row("fp32", 0, 1), self.row("fp32", 0, 1)])
            b = self.write(d, "b", [self.row("q8", 0, 1, stop="max_tokens")])
            ok, info = fg.input_checks(a, b, [(0, 1)])
            self.assertFalse(ok)
            self.assertEqual(info["fp32"]["duplicates"], [(0, 1)])
            self.assertEqual(info["q8"]["not_duration"], [(0, 1)])

    def test_complete_inputs_pass(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            exp = [(p, s) for p in (0, 1) for s in (1, 2)]
            ok, _ = fg.input_checks(self.write(d, "a", [self.row("fp32", p, s) for p, s in exp]),
                                    self.write(d, "b", [self.row("q8", p, s) for p, s in exp]), exp)
            self.assertTrue(ok)


class SeedClusterTest(unittest.TestCase):
    def test_per_seed_mean_over_prompts(self):
        A = {(p, s): {"stats": {"x": 1.0}} for p in (0, 1) for s in (1, 2, 3)}
        B = {(p, s): {"stats": {"x": 1.0 + s}} for p in (0, 1) for s in (1, 2, 3)}
        out = fg.seed_cluster(A, B, sorted(A), "x")
        self.assertEqual(out["seeds"], 3)
        self.assertEqual(out["median_of_seed_means"], 2.0)


if __name__ == "__main__":
    unittest.main()
