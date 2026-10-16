"""Stop rule and note handling of the #1774 comparison runner. Synthetic vocab and logits, no model."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import torch  # noqa: E402

import jps_compare as jc  # noqa: E402

VOCAB = ["<S>", "<E>", "<D>", "<T>", ("piano", 60, 60), ("piano", 62, 60)] + \
    [("onset", x) for x in range(0, 5000, 10)] + [("dur", x) for x in (10, 100, 500)]
IDX = {t: i for i, t in enumerate(VOCAB)}


def peaked(pref):
    """Logits that put almost all mass on pref(seq tokens)."""
    def f(seq):
        lg = torch.full((len(VOCAB),), -50.0)
        lg[IDX[pref([VOCAB[i] for i in seq])]] = 50.0
        return lg
    return f


def run(prompt, pref, seed=1):
    buf = {"ids": [], "pre_mask_allowed": [], "mask_changed": [], "stop": None}
    jc.generate_cell(peaked(pref), prompt, [IDX[t] for t in prompt], VOCAB, seed, buf)
    return buf, [VOCAB[i] for i in buf["ids"]]


class StopRuleTest(unittest.TestCase):
    PROMPT = ["<T>", ("piano", 60, 60), ("onset", 1000), ("dur", 100)]      # last onset at 6,000 ms

    def test_segment_markers_stop_at_window_end(self):
        buf, toks = run(self.PROMPT, lambda s: "<T>")
        self.assertEqual(buf["stop"], "window_end_by_T")
        self.assertEqual(toks, ["<T>"] * 4)                                 # segments 2..5; the 6th would reach 30,000

    def test_note_past_window_is_dropped(self):
        def pref(s):
            last = s[-1]
            if isinstance(last, tuple) and last[0] == "piano":
                return ("onset", 4990)
            if isinstance(last, tuple) and last[0] == "onset":
                return ("dur", 100)
            return "<T>" if isinstance(last, tuple) and last[0] == "dur" else ("piano", 62, 60)
        buf, toks = run(self.PROMPT, pref)
        self.assertEqual(buf["stop"], "window_end")
        notes = jc.note_list(jc.complete_prefix(toks), seg0=1)
        self.assertTrue(notes)
        self.assertTrue(all(on < jc.WIN_END_MS for on, _, _, _ in notes))
        self.assertEqual(toks[-1], "<T>")                                   # segment 5 opened, its note at 29,990 ms dropped
        self.assertNotEqual(jc.kind(toks[-1]), "piano")

    def test_D_and_E_never_picked(self):
        buf, toks = run(self.PROMPT[:-1] + [("dur", 100)], lambda s: "<E>")
        self.assertNotIn("<E>", toks)
        self.assertNotIn("<D>", toks)
        self.assertGreater(len(buf["mask_changed"]), 0)

    def test_token_cap_marks_short_and_drops_tail(self):
        old = jc.MAX_NEW
        jc.MAX_NEW = 5
        try:
            def pref(s):
                last = s[-1]
                if isinstance(last, tuple) and last[0] == "piano":
                    return ("onset", 1200)
                if isinstance(last, tuple) and last[0] == "onset":
                    return ("dur", 10)
                return ("piano", 62, 60)
            buf, toks = run(self.PROMPT, pref)
        finally:
            jc.MAX_NEW = old
        self.assertEqual(buf["stop"], "token_cap")
        self.assertEqual(len(toks), 5)
        self.assertEqual(len(jc.complete_prefix(toks)), 3)                  # piano, onset, dur | piano, onset dropped

    def test_same_seed_same_stream(self):
        flat = lambda seq: torch.zeros(len(VOCAB))                           # noqa: E731 - uniform: the noise decides
        a = {"ids": [], "pre_mask_allowed": [], "mask_changed": [], "stop": None}
        b = {"ids": [], "pre_mask_allowed": [], "mask_changed": [], "stop": None}
        ids = [IDX[t] for t in self.PROMPT]
        jc.generate_cell(flat, self.PROMPT, ids, VOCAB, 7, a)
        jc.generate_cell(flat, self.PROMPT, ids, VOCAB, 7, b)
        self.assertEqual(a["ids"], b["ids"])


class NotesTest(unittest.TestCase):
    def test_note_list_absolute_time_and_velocity(self):
        toks = [("piano", 60, 70), ("onset", 100), ("dur", 500), "<T>", ("piano", 62, 40), ("onset", 20), ("dur", 10)]
        self.assertEqual(jc.note_list(toks), [(100, 600, 60, 70), (5020, 5030, 62, 40)])
        self.assertEqual(jc.note_list(toks[3:], seg0=1)[0][0], 10020)

    def test_complete_prefix(self):
        toks = [("piano", 60, 70), ("onset", 100), ("dur", 500), ("piano", 62, 40), ("onset", 20)]
        self.assertEqual(jc.complete_prefix(toks), toks[:3])
        self.assertEqual(jc.complete_prefix(toks[:3] + ["<T>"]), toks[:3] + ["<T>"])


if __name__ == "__main__":
    unittest.main()
