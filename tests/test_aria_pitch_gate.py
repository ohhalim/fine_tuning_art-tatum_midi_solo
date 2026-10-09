"""Pitch-position gate (scripts/aria_pitch_gate.py)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from scripts.aria_pitch_gate import gate_step, gates

HEAD = [("prefix", "instrument", "piano"), "<S>"]


def n(p, onset, dur=100):
    return [("piano", p, 80), ("onset", onset), ("dur", dur)]


SEQ = HEAD + n(60, 0) + n(64, 250) + ["<T>"] + n(67, 100) + ["<D>"] + n(72, 200) + ["<E>"]


class PitchGateTest(unittest.TestCase):
    def test_on_only_between_notes(self) -> None:
        g = gates(SEQ)
        expect = [0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0]
        self.assertEqual(g, expect)

    def test_each_on_position_is_followed_by_a_note_start_or_boundary(self) -> None:
        g = gates(SEQ)
        nxt = [SEQ[i + 1] for i in range(len(SEQ) - 1) if g[i]]
        self.assertTrue(all((isinstance(t, tuple) and t[0] == "piano") or t in ("<T>", "<D>", "<E>") for t in nxt))
        off_nxt = [SEQ[i + 1] for i in range(len(SEQ) - 1) if not g[i] and i >= 1]
        self.assertTrue(all(isinstance(t, tuple) and t[0] in ("onset", "dur", "piano") for t in off_nxt))

    def test_prefix_invariance(self) -> None:
        a = gates(HEAD + n(60, 0) + [("piano", 62, 80)] + [("onset", 300), ("dur", 50)])
        b = gates(HEAD + n(60, 0) + [("piano", 62, 80)] + [("organ", 50, 80), "<E>"])
        k = len(HEAD) + 4
        self.assertEqual(a[:k], b[:k])

    def test_stepwise_equals_teacher_forcing(self) -> None:
        state, step = "header", []
        for t in SEQ:
            state, g = gate_step(state, t)
            step.append(g)
        self.assertEqual(step, gates(SEQ))

    def test_unexpected_token_turns_off_until_resync(self) -> None:
        g = gates(HEAD + [("piano", 60, 80), ("organ", 96, 120), ("dur", 100)] + n(62, 300))
        self.assertEqual(g[2:], [0, 0, 1, 0, 0, 1])


if __name__ == "__main__":
    unittest.main()
