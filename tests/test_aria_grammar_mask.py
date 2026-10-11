"""Regression for the format-only grammar mask (issue #1731). CPU, small fake vocab."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aria_grammar_mask import GrammarMask, masked_pick  # noqa: E402
from aria_token_validity import check  # noqa: E402

VOCAB = [("piano", 60, 80), ("piano", 64, 70), ("onset", 0), ("onset", 10), ("onset", 20),
         ("dur", 100), ("dur", 200), "<T>", "<D>", "<E>", ("organ", 60, 80), ("prefix", "instrument", "piano"), "<S>"]
IDX = {t: i for i, t in enumerate(VOCAB)}
PROMPT = [("prefix", "instrument", "piano"), "<S>", ("piano", 60, 80), ("onset", 10), ("dur", 100)]


def allowed_tokens(gm):
    return {VOCAB[i] for i in torch.nonzero(gm.allowed()).flatten().tolist()}


class GrammarMaskTest(unittest.TestCase):
    def setUp(self):
        self.gm = GrammarMask(VOCAB)
        self.gm.start(PROMPT)

    def test_note_start_then_onset_then_dur(self):
        self.assertEqual(allowed_tokens(self.gm), {("piano", 60, 80), ("piano", 64, 70), "<T>", "<D>", "<E>"})
        self.gm.advance(("piano", 64, 70))
        self.assertEqual(allowed_tokens(self.gm), {("onset", 10), ("onset", 20)})
        self.gm.advance(("onset", 20))
        self.assertEqual(allowed_tokens(self.gm), {("dur", 100), ("dur", 200)})
        self.gm.advance(("dur", 200))
        self.assertIn("<E>", allowed_tokens(self.gm))

    def test_same_onset_allowed_for_chords_but_not_earlier(self):
        self.gm.advance(("piano", 64, 70))
        self.assertIn(("onset", 10), allowed_tokens(self.gm))
        self.assertNotIn(("onset", 0), allowed_tokens(self.gm))
        with self.assertRaises(ValueError):
            self.gm.advance(("onset", 0))

    def test_time_token_resets_segment(self):
        self.gm.advance("<T>")
        self.gm.advance(("piano", 60, 80))
        self.assertIn(("onset", 0), allowed_tokens(self.gm))

    def test_eos_and_non_piano_only_where_allowed(self):
        self.assertNotIn(("organ", 60, 80), allowed_tokens(self.gm))
        self.gm.advance(("piano", 60, 80))
        self.assertNotIn("<E>", allowed_tokens(self.gm))
        with self.assertRaises(ValueError):
            self.gm.advance("<E>")

    def test_empty_allowed_set_is_visible(self):
        self.gm.advance(("piano", 60, 80))
        self.gm.last = 30          # later than every onset in this vocab
        self.assertFalse(bool(self.gm.allowed().any()))

    def test_masked_pick_keeps_free_pick_when_allowed(self):
        g = torch.Generator().manual_seed(0)
        ok = self.gm.allowed()
        for _ in range(50):
            p = torch.softmax(torch.randn(len(VOCAB), generator=g) * 2, -1)
            q = torch.empty(len(VOCAB)).exponential_(1, generator=g)
            pick, free = masked_pick(p, ok, q)
            self.assertTrue(bool(ok[pick]))
            if ok[free]:
                self.assertEqual(pick, free)

    def test_masked_streams_always_pass_validity_and_cap_tail_is_incomplete_not_error(self):
        g = torch.Generator().manual_seed(1)
        tails = set()
        for run in range(40):
            gm = GrammarMask(VOCAB)
            gm.start(PROMPT)
            new = []
            for _ in range(7 + run % 5):
                p = torch.softmax(torch.randn(len(VOCAB), generator=g) * 3, -1)
                q = torch.empty(len(VOCAB)).exponential_(1, generator=g)
                ok = gm.allowed()
                self.assertGreater(p[ok].sum().item(), 0)
                pick, _ = masked_pick(p, ok, q)
                new.append(VOCAB[pick])
                if VOCAB[pick] == "<E>":
                    break
                gm.advance(VOCAB[pick])
            v = check(PROMPT, new)
            self.assertTrue(v["valid"], (new, v))
            tails.add(v["tail"])
        self.assertIn("incomplete_tail", tails)


if __name__ == "__main__":
    unittest.main()
