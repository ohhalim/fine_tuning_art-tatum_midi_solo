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


class ReviewFixesTest(unittest.TestCase):
    """PR #1500 review: chord metric naming, empty first sample, missing budget."""

    def test_chord_onset_share_counts_clusters(self) -> None:
        from scripts.describe_generations import cluster_sizes, describe_tokens

        self.assertEqual(cluster_sizes([0.0, 0.01, 0.02, 0.5, 1.0, 1.01], 0.03), [3, 1, 2])
        vel, s20, s1 = 372, 256 + 20, 256 + 0   # time shift 256+k = (k+1) x 10 ms
        # chord of 3 notes (10 ms apart, all within 30 ms), then 2 single notes
        toks = [vel, 60, s1, 64, s1, 67, s20, 128 + 60, 128 + 64, 128 + 67,
                vel, 72, s20, 128 + 72, vel, 74, s20, 128 + 74]
        out = describe_tokens(toks, {})
        self.assertAlmostEqual(out["chord_onset_share"], 1 / 3)          # 1 of 3 clusters
        self.assertAlmostEqual(out["simultaneous_note_share"], 2 / 5)    # 2 of 5 notes joined

    def test_empty_first_sample_keeps_later_statistics(self) -> None:
        import json
        import tempfile

        from scripts.describe_generations import main

        vel, s20 = 372, 256 + 20
        good = [vel, 60, s20, 128 + 60, vel, 67, s20, 128 + 67]
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "t.json").write_text(json.dumps({"5": [[s20, s20], good]}))
            (tmp / "train").mkdir()
            np.save(tmp / "train" / "a.npy", np.array(good))
            main(["--model", f"m={tmp / 't.json'}:5", "--train-set", f"x={tmp / 'train'}",
                  "--output", str(tmp / "o.json")])
            summary = json.loads((tmp / "o.json").read_text())["models"]["m"]["summary"]
        self.assertEqual(summary["empty"], 1)
        self.assertEqual(summary["pitch_range"], 7)          # from the valid second sample
        self.assertIn("ioi_median_ms", summary)

    def _check(self, budget_lines):
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            budget = tmp / "chosen_budgets.txt"
            if budget_lines is not None:
                budget.write_text("".join(f"{line}\n" for line in budget_lines))
            proc = subprocess.run(["bash", str(ROOT / "scripts" / "tvm_check_budgets.sh"),
                                   str(budget), str(tmp)], capture_output=True, text=True)
            failed = tmp / "adaptation_failed.txt"
            return proc, failed.read_text() if failed.exists() else None

    def test_budget_check_both_present(self) -> None:
        proc, failed = self._check(["tatum 128", "mehldau 64"])
        self.assertEqual(proc.returncode, 0)
        self.assertEqual(proc.stdout.split(), ["128", "64"])
        self.assertIsNone(failed)

    def test_budget_check_one_missing(self) -> None:
        proc, failed = self._check(["tatum 128"])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("ADAPTATION_FAILED tatum='128' mehldau=''", failed)

    def test_budget_check_both_missing_empty_file(self) -> None:
        proc, failed = self._check([])
        self.assertEqual(proc.returncode, 2)
        self.assertIn("ADAPTATION_FAILED tatum='' mehldau=''", failed)

    def test_budget_check_file_absent(self) -> None:
        proc, failed = self._check(None)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("ADAPTATION_FAILED", failed)
        self.assertNotIn("awk", proc.stderr)

    def test_pipeline_initialises_budget_file_and_uses_check(self) -> None:
        text = (ROOT / "scripts" / "run_tvm_pipeline.sh").read_text()
        self.assertLess(text.index(': > "$O/chosen_budgets.txt"'), text.index("for artist in tatum mehldau; do\n  B="))
        self.assertLess(text.index("tvm_check_budgets.sh"), text.index("scripts/eval_cross_artist.py"))


if __name__ == "__main__":
    unittest.main()
