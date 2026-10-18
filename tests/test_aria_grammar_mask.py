"""Regression for the format-only grammar mask (issue #1731). CPU, small fake vocab."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aria_grammar_mask import GrammarMask, MaskStop, masked_generate, masked_pick, run_streams  # noqa: E402
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

    def test_commit_floor_raises_min_onset_and_forces_time_token(self):
        self.gm.advance(("piano", 60, 80))
        self.gm.min_onset = 20
        self.assertEqual({t for t in allowed_tokens(self.gm) if t[0] == "onset"}, {("onset", 20)})
        self.gm.advance(("onset", 20))
        self.gm.advance(("dur", 100))
        self.gm.reset_first = True
        self.assertEqual(allowed_tokens(self.gm), {"<T>"})

    def test_sync_takes_state_and_floor_from_a_stream(self):
        from stream_contract import TokenStream
        s = TokenStream([("piano", 60, 80), ("onset", 0), ("dur", 100)])
        for t in [("piano", 64, 70), ("onset", 20), ("dur", 100)]:
            s.append_sampled(t)
        s.commit_until(15)                           # stops at the note at 20: the floor is 15
        s.rollback_to_committed()
        s.append_sampled(("piano", 64, 70))
        st = s.grammar_state()
        self.gm.sync(st, s.onset_floor(st))
        self.assertEqual({t for t in allowed_tokens(self.gm) if t[0] == "onset"}, {("onset", 20)})

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


class Boom(Exception):
    pass


def fake_probs(seed, fail_after=None):
    g, calls = torch.Generator().manual_seed(seed), [0]

    def next_probs(seq):
        calls[0] += 1
        if fail_after is not None and calls[0] > fail_after:
            raise Boom("wall limit")
        return torch.softmax(torch.randn(len(VOCAB), generator=g) * 3, -1)
    return next_probs


class GenerateAndPreserveTest(unittest.TestCase):
    def test_masked_generate_valid_and_same_noise_as_unmasked_sampler(self):
        for seed in range(10):
            buf = {"ids": [], "pre_mask_allowed": [], "mask_changed": []}
            masked_generate(fake_probs(seed), PROMPT, [IDX[t] for t in PROMPT], VOCAB, seed, 12, IDX["<E>"], buf)
            self.assertTrue(check(PROMPT, [VOCAB[i] for i in buf["ids"]])["valid"])
            self.assertEqual(len(buf["pre_mask_allowed"]), len(buf["ids"]))
            for c in buf["mask_changed"]:
                self.assertNotEqual(c["unmasked_pick"], c["masked_pick"])

    def test_zero_allowed_mass_stops_and_keeps_steps_so_far(self):
        calls = [0]

        def next_probs(seq):
            calls[0] += 1
            p = torch.zeros(len(VOCAB))
            p[IDX[("organ", 60, 80)] if calls[0] > 3 else IDX[("piano", 60, 80)] if calls[0] == 1 else IDX[("onset", 20)] if calls[0] == 2 else IDX[("dur", 100)]] = 1.0
            return p
        buf = {"ids": [], "pre_mask_allowed": [], "mask_changed": []}
        with self.assertRaises(MaskStop):
            masked_generate(next_probs, PROMPT, [IDX[t] for t in PROMPT], VOCAB, 0, 12, IDX["<E>"], buf)
        self.assertEqual(len(buf["ids"]), 3)

    def test_stop_in_the_middle_keeps_incomplete_stream_apart_from_completed(self):
        saved = []
        specs = [{"n": 0}, {"n": 1}, {"n": 2}]
        probs = {0: fake_probs(0), 1: fake_probs(1, fail_after=5), 2: fake_probs(2)}

        def run_one(spec, buf):
            masked_generate(probs[spec["n"]], PROMPT, [IDX[t] for t in PROMPT], VOCAB, spec["n"], 12, IDX["<E>"], buf)
        with self.assertRaises(Boom):
            run_streams(specs, run_one, lambda buf: None, saved.append)
        self.assertEqual(len(saved), 1)
        done, inc = saved[0]["completed"], saved[0]["incomplete"]
        self.assertEqual([b["n"] for b in done], [0])
        self.assertEqual(inc["n"], 1)
        self.assertEqual(inc["stop_step"], 5)
        self.assertEqual(len(inc["ids"]), 5)
        self.assertEqual(len(inc["pre_mask_allowed"]), 5)
        self.assertIn("wall limit", inc["stop_reason"])

    def test_after_check_failure_keeps_stream_in_completed(self):
        saved = []

        def after(buf):
            raise Boom("format invalid")
        with self.assertRaises(Boom):
            run_streams([{"n": 0}], lambda spec, buf: buf["ids"].append(1), after, saved.append)
        self.assertEqual([b["n"] for b in saved[0]["completed"]], [0])
        self.assertIsNone(saved[0]["incomplete"])


if __name__ == "__main__":
    unittest.main()
