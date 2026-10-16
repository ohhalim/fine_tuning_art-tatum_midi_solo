"""Examples for the minimal condition-adapter run (issue #1760). Pure Python, synthetic streams."""
from __future__ import annotations

import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.dirname(__file__))
from aria_mlx_cond import cond_vector  # noqa: E402
from cond_adapter_data import pick_evenly, pick_training, song_examples, token_kind, with_current_condition, wrong_payload  # noqa: E402
from lh_role_proxy import Note  # noqa: E402
from test_block_request import aria_tokens  # noqa: E402


def synthetic(seed=0, n=150, span=60000):
    rng = random.Random(seed)
    notes = [(rng.randint(55, 95), 10 * rng.randint(0, span // 10), 10 * rng.randint(1, 80), 64) for _ in range(n)]
    toks = aria_tokens(notes)
    ids = list(range(len(toks)))                                              # stand-in ids, one per token
    hum = [Note(i, rng.randint(28, 54), on, on + 700) for i, on in enumerate(range(0, span, 400))]
    return toks, ids, hum


class ExampleTest(unittest.TestCase):
    def test_examples_line_up_rows_labels_and_kinds(self):
        toks, ids, hum = synthetic()
        exs, counts = song_examples(toks, ids, hum, 0.0)
        self.assertEqual(counts["requests"], counts["empty_window"] + counts["over_cap"] + len(exs))
        for ex in exs:
            self.assertEqual(len(ex["rows"]), len(ex["ids"]) - 1)               # one row per input position
            self.assertEqual(len(ex["label_kinds"]), len(ex["ids"]) - ex["first_label"])
            self.assertGreater(len(ex["label_kinds"]), 0)
            self.assertEqual(ex["rows"][ex["first_label"] - 1], cond_vector(ex["payload"]))   # p0 carries the current request
            self.assertTrue(all(len(r) == 13 for r in ex["rows"]))

    def test_token_kinds(self):
        self.assertEqual([token_kind(t) for t in ("<T>", ("piano", 60, 64), ("onset", 10), ("dur", 20))], ["T", "pitch", "onset", "dur"])


class WrongConditionTest(unittest.TestCase):
    def test_same_size_same_available_different_set(self):
        c_e = tuple(int(c in (0, 4)) for c in range(12))
        w = wrong_payload((c_e, 1))
        self.assertEqual(w, (tuple(int(c in (6, 10)) for c in range(12)), 1))
        tritone = tuple(int(c in (0, 6)) for c in range(12))                  # symmetric under a tritone
        self.assertEqual(wrong_payload((tritone, 1)), (tuple(int(c in (1, 7)) for c in range(12)), 1))
        self.assertIsNone(wrong_payload(((0,) * 12, 1)))                      # warm-empty
        self.assertIsNone(wrong_payload(((0,) * 12, 0)))                      # cold

    def test_only_the_current_request_rows_change(self):
        toks, ids, hum = synthetic(1)
        exs, _ = song_examples(toks, ids, hum, 0.0)
        ex = next(e for e in exs if wrong_payload(e["payload"]) and e["first_label"] > 5)
        rows = with_current_condition(ex, wrong_payload(ex["payload"]))
        p0 = ex["first_label"] - 1
        self.assertEqual(rows[:p0], ex["rows"][:p0])
        self.assertTrue(all(r == cond_vector(wrong_payload(ex["payload"])) for r in rows[p0:]))


class PickTest(unittest.TestCase):
    def test_fixed_picks(self):
        xs = list(range(100))
        self.assertEqual(pick_training(xs, 5), pick_training(xs, 5))
        self.assertEqual(len(set(pick_training(xs, 32))), 32)
        self.assertEqual(pick_evenly(xs, 5), [0, 25, 50, 74, 99])
        self.assertEqual(pick_evenly(xs[:3], 5), [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
