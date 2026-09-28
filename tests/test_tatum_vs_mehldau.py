from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))


class NearDuplicateHashTest(unittest.TestCase):
    def test_hash_matches_window_identity(self) -> None:
        from scripts.check_near_duplicates import ngram_hashes

        a = np.arange(40) % 7
        h = ngram_hashes(a, 16)
        self.assertEqual(len(h), 25)
        # identical windows (period 7) hash identically, different windows differ
        self.assertEqual(h[0], h[7])
        self.assertNotEqual(h[0], h[1])
        self.assertEqual(len(ngram_hashes(np.arange(5), 16)), 0)

    def test_shared_ngrams_detected(self) -> None:
        from scripts.check_near_duplicates import ngram_hashes

        song = np.random.default_rng(0).integers(0, 388, 200)
        other = np.concatenate([np.random.default_rng(1).integers(0, 388, 100), song[50:150]])
        share = np.isin(np.unique(ngram_hashes(song)), np.unique(ngram_hashes(other))).mean()
        self.assertAlmostEqual(share, 85 / 185, places=6)   # 100-token copy -> 85 of 185 windows


class CrossArtistStatsTest(unittest.TestCase):
    def test_token_vs_macro_ce(self) -> None:
        from scripts.eval_cross_artist import column_stats

        out = column_stats([(10.0, 10), (30.0, 10), (4.0, 1)])
        self.assertAlmostEqual(out["token_ce"], 44 / 21)
        self.assertAlmostEqual(out["macro_ce"], (1 + 3 + 4) / 3)

    def test_bootstrap_ci_contains_mean(self) -> None:
        from scripts.eval_cross_artist import bootstrap_mean_ci

        lo, hi = bootstrap_mean_ci([-0.05, -0.04, -0.06, -0.03])
        self.assertLess(lo, -0.045 + 1e-9)
        self.assertGreater(hi, -0.045 - 1e-9)
        self.assertEqual(bootstrap_mean_ci([0.1]), [0.1, 0.1])


class SplitSelectionTest(unittest.TestCase):
    def test_pick_is_deterministic_and_disjoint(self) -> None:
        from scripts.build_tvm_splits import pick

        names = [f"{i:05d}.npy" for i in range(110)]
        train = pick(names, 16, 0)
        self.assertEqual(train, pick(list(reversed(names)), 16, 0))
        fresh = pick([n for n in names if n not in set(train)], 12, 1)
        self.assertFalse(set(train) & set(fresh))


class DescribeGenerationTest(unittest.TestCase):
    def test_descriptors_and_copy(self) -> None:
        from scripts.describe_generations import describe_tokens
        from scripts.eval_mehldau_snapshots import ngram_set

        vel, shift = 372, 256 + 20
        toks = []
        for p in (60, 64, 67, 72):
            toks += [vel, p, shift, 128 + p]
        out = describe_tokens(toks, {"self": ngram_set(toks, 16), "other": set()})
        self.assertTrue(out["grammar_valid"])
        self.assertEqual(out["notes"], 4)
        self.assertEqual(out["pitch_range"], 12)
        self.assertEqual(out["copy16_self"], 1.0)
        self.assertEqual(out["copy16_other"], 0.0)
        self.assertFalse(out["empty"])


class CvSelectionRulesTest(unittest.TestCase):
    @staticmethod
    def _folds(values):
        rows = [{"update": 0, "specialisation_val": 0, "d_ce_target_val": 0, "d_ce_generic": 0}]
        rows += [{"update": u, "specialisation_val": sp, "d_ce_target_val": held, "d_ce_generic": gen}
                 for u, sp, held, gen in values]
        return [{"rows": rows}] * 4

    def test_tie_prefers_smaller_update(self) -> None:
        from scripts.aggregate_song_cv import aggregate

        folds = self._folds([(32, -0.0300, -0.05, -0.02), (64, -0.0305, -0.05, -0.02)])
        self.assertEqual(aggregate(folds, tie_tolerance=0.001)["chosen"]["update"], 32)
        self.assertEqual(aggregate(folds)["chosen"]["update"], 64)   # default: no tie rule

    def test_target_drop_required_for_eligibility(self) -> None:
        from scripts.aggregate_song_cv import aggregate

        folds = self._folds([(32, -0.02, -0.01, 0.01), (64, -0.04, 0.005, 0.015)])
        self.assertEqual(aggregate(folds)["chosen"]["update"], 64)
        self.assertEqual(aggregate(folds, require_target_drop=True)["chosen"]["update"], 32)


if __name__ == "__main__":
    unittest.main()
