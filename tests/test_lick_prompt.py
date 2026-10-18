"""Pure helpers of the #1827 liked-lick prompt comparison (no model)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import lick_prompt as lp  # noqa: E402

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None


class HelperTest(unittest.TestCase):
    def test_prefix_within_budget(self):
        self.assertEqual(lp.prefix_within([5, 8, 11, 14], 11), 3)
        self.assertEqual(lp.prefix_within([12], 11), 0)

    def test_window_max_half_open(self):
        self.assertEqual(lp.window_max([0.0, 0.5, 1.0], 1.0), 2)             # 1.0 is outside [0, 1)
        self.assertEqual(lp.window_max([0.0, 0.5, 0.9, 1.0, 1.2], 1.0), 4)   # [0.5, 1.5) holds four
        self.assertEqual(lp.window_max([], 1.0), 0)

    def test_max_shared_ngram(self):
        self.assertEqual(lp.max_shared_ngram([2, 2, -1, 3, 5], [9, 2, -1, 3, 0]), 3)
        self.assertEqual(lp.max_shared_ngram([1, 1], [2, 2]), 0)

    def test_median_diff_stats_detects_a_clear_shift_and_not_noise(self):
        a = [1.0, 1.1, 0.9, 1.0, 1.2, 0.8, 1.05, 0.95, 1.0, 1.1, 0.9, 1.0]
        b = [x + 3.0 for x in a]
        s = lp.median_diff_stats(a, b, n=2000)
        self.assertAlmostEqual(s["diff"], 3.0)
        self.assertLess(s["p"], 0.01)
        self.assertLessEqual(s["ci95"][0], 3.0)
        same = lp.median_diff_stats(a, list(a), n=2000)
        self.assertGreater(same["p"], 0.5)

    def test_verdict_needs_significance_and_effect(self):
        t = lambda d, p: {"diff": d, "p": p, "ci95": [0, 0]}  # noqa: E731
        self.assertEqual(lp.verdict({"licks_per_min": t(4.0, 0.001), "liked_speed_run_share": t(0.0, 0.9)}), "가까워짐")
        self.assertEqual(lp.verdict({"licks_per_min": t(1.0, 0.001), "liked_speed_run_share": t(0.0, 0.9)}), "차이 미검출")
        self.assertEqual(lp.verdict({"licks_per_min": t(4.0, 0.02), "liked_speed_run_share": None}), "차이 미검출")
        self.assertEqual(lp.verdict({"licks_per_min": t(-4.0, 0.001), "liked_speed_run_share": t(0.2, 0.001)}), "섞임")
        self.assertEqual(lp.verdict({"licks_per_min": t(-4.0, 0.001), "liked_speed_run_share": None}), "멀어짐")


@unittest.skipIf(pretty_midi is None, "pretty_midi not installed")
class MetricsTest(unittest.TestCase):
    def test_a_fast_top_line_run_counts_as_one_lick(self):
        # 12-note top-line run at 0.12 s IOI above G3, plus a held bass note
        notes = [[72 + (i % 5), 80, 1000 + i * 120, 1000 + i * 120 + 110] for i in range(12)] + [[40, 70, 1000, 3000]]
        m = lp.run_metrics(notes, 1000, 15.0)
        self.assertEqual(m["licks"], 1)
        self.assertEqual(m["runs_8plus"], 1)
        self.assertAlmostEqual(m["liked_speed_run_share"], 1.0)        # 11 / 1.32 s = 8.3 notes/s
        self.assertEqual(m["melody_window_max"], 8)                    # 0.9375 s window at 0.12 s IOI

    def test_empty_generation_has_zero_rates(self):
        m = lp.run_metrics([], 0, 15.0)
        self.assertEqual(m["licks"], 0)
        self.assertEqual(m["melody_notes_per_s"], 0.0)
        self.assertIsNone(m["liked_speed_run_share"])


class PairedSignflipTest(unittest.TestCase):
    def test_all_positive_pairs(self):
        import lick_prompt as lp
        out = lp.paired_signflip([(0.0, 1.0)] * 6)
        self.assertEqual(out["n"], 6)
        self.assertEqual(out["diff"], 1.0)
        self.assertAlmostEqual(out["p"], 2 / 64)          # only the all-plus and all-minus patterns reach |sum| 6

    def test_zero_and_empty(self):
        import lick_prompt as lp
        self.assertEqual(lp.paired_signflip([(1.0, 1.0)] * 3)["p"], 1.0)
        self.assertIsNone(lp.paired_signflip([]))


if __name__ == "__main__":
    unittest.main()
