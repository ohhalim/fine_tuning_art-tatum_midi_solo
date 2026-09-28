from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

from scripts.eval_mehldau_snapshots import bootstrap_diff, copy_rate, ngram_set
from scripts.style_distance import (FEATURES, LogisticModel, distance, feature_counts,
                                    feature_vector, js_divergence, pool, tokens_to_notes)
from scripts.validate_style_distance import middle_chunk


def note(pitch, start, end=None, velocity=64):
    return SimpleNamespace(pitch=pitch, start=start, end=end if end is not None else start + 0.2,
                           velocity=velocity)


class StyleDistanceTest(unittest.TestCase):
    def test_js_is_zero_for_identical_and_symmetric(self) -> None:
        p, q = np.array([1.0, 2.0, 3.0]), np.array([3.0, 1.0, 0.0])
        self.assertAlmostEqual(js_divergence(p, p), 0.0, places=6)
        self.assertAlmostEqual(js_divergence(p, q), js_divergence(q, p), places=9)
        self.assertLessEqual(js_divergence(np.array([1.0, 0]), np.array([0, 1.0])), 1.0 + 1e-6)

    def test_chord_clusters_and_intervals(self) -> None:
        notes = [note(60, 0.0), note(64, 0.01), note(67, 0.02), note(72, 0.5)]
        c = feature_counts(notes)
        self.assertEqual(c["chord"][2], 1)  # one 3-note cluster
        self.assertEqual(c["chord"][0], 1)  # one single note
        self.assertEqual(c["interval"][24 + 4], 1)
        self.assertEqual(c["interval"][24 + 5], 1)
        self.assertEqual(c["register"].sum(), 4)

    def test_interval_is_transposition_invariant(self) -> None:
        a = [note(60, 0.0), note(62, 0.3), note(65, 0.6)]
        b = [note(p.pitch + 5, p.start) for p in a]
        self.assertTrue(np.array_equal(feature_counts(a)["interval"], feature_counts(b)["interval"]))

    def test_distance_zero_to_self_reference(self) -> None:
        c = feature_counts([note(60, 0.0), note(67, 0.25), note(64, 0.5)])
        self.assertAlmostEqual(distance(c, pool([c])), 0.0, places=6)

    def test_feature_vector_normalised_per_feature(self) -> None:
        v = feature_vector(feature_counts([note(60, 0.0), note(62, 0.3)]))
        self.assertEqual(len(v), 119)

    def test_empty_notes(self) -> None:
        c = feature_counts([])
        self.assertTrue(all(c[f].sum() == 0 for f in FEATURES))
        self.assertEqual(tokens_to_notes([]), [])
        self.assertEqual(tokens_to_notes([500, 600]), [])  # control tokens dropped

    def test_logistic_model_separates_and_roundtrips(self) -> None:
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (40, 3)), rng.normal(3, 1, (40, 3))])
        y = np.array([0] * 40 + [1] * 40)
        m = LogisticModel().fit(X, y)
        acc = np.mean((m.predict_proba(X) >= 0.5) == y)
        self.assertGreater(acc, 0.9)
        m2 = LogisticModel.from_dict(m.to_dict())
        self.assertTrue(np.allclose(m.predict_proba(X), m2.predict_proba(X)))


class SnapshotEvalHelpersTest(unittest.TestCase):
    def test_copy_rate(self) -> None:
        ref = ngram_set([1, 2, 3, 4, 5], 3)
        self.assertEqual(copy_rate([1, 2, 3, 9], ref, 3), 0.5)
        self.assertIsNone(copy_rate([1, 2], ref, 3))

    def test_bootstrap_diff_contains_true_shift(self) -> None:
        a = [0.0, 0.1, -0.1, 0.05]
        lo, hi = bootstrap_diff(a, [x + 1.0 for x in a])
        self.assertAlmostEqual(lo, 1.0, places=6)
        self.assertAlmostEqual(hi, 1.0, places=6)

    def test_middle_chunk(self) -> None:
        self.assertEqual(middle_chunk(np.arange(10), 4).tolist(), [3, 4, 5, 6])
        self.assertEqual(middle_chunk(np.arange(3), 4).tolist(), [0, 1, 2])


class RunDensityTest(unittest.TestCase):
    def test_clusters_and_run_ratio(self) -> None:
        from scripts.measure_run_density import cluster_onsets, run_stats

        # chord at 0 (3 notes within 30 ms), then 80 ms run notes, then a 500 ms gap
        starts = [0.0, 0.01, 0.02, 0.08, 0.16, 0.24, 0.74]
        self.assertEqual(cluster_onsets(starts), [0.0, 0.08, 0.16, 0.24, 0.74])
        stats = run_stats(starts)
        self.assertAlmostEqual(stats["run_ratio"], 3 / 4)
        self.assertEqual(stats["onsets"], 5)

    def test_too_few_onsets(self) -> None:
        from scripts.measure_run_density import run_stats

        self.assertIsNone(run_stats([0.0])["run_ratio"])


class FreeGenerationValidityTest(unittest.TestCase):
    def test_grammar_checks_and_cut_phrase_allowed(self) -> None:
        from scripts.eval_mehldau_snapshots import free_generation_validity

        vel, shift = 356 + 16, 256 + 50   # velocity 64, time shift
        ok = [vel, 60, shift, 128 + 60, vel, 64, shift]   # second note still open: allowed
        out = free_generation_validity(ok)
        self.assertTrue(out["grammar_valid"], out)
        self.assertEqual(out["open_at_end"], 1)
        orphan = [vel, 60, shift, 128 + 61, shift, 128 + 60]
        self.assertFalse(free_generation_validity(orphan)["grammar_valid"])
        self.assertFalse(free_generation_validity([shift, shift])["grammar_valid"])


if __name__ == "__main__":
    unittest.main()
