"""Pattern-cache logit bias (scripts/pattern_cache.py) and its verdict."""
from __future__ import annotations

import math
import unittest

import pretty_midi
import torch

from scripts.generate import encode_notes_simple
from scripts.pattern_cache import PatternCacheBias, candidate_pitches
from scripts.pattern_cache_diag import verdict


def melody(pitches, ioi=0.2):
    return encode_notes_simple([pretty_midi.Note(velocity=80, pitch=p, start=k * ioi, end=k * ioi + 0.1)
                                for k, p in enumerate(pitches)])


class CandidateTest(unittest.TestCase):
    def test_completes_an_earlier_pattern_transposed(self) -> None:
        # earlier 60 62 64 65 (intervals 2,2,1); now 67 69 71 -> 72 completes (2,2,1)
        self.assertEqual(candidate_pitches(melody([60, 62, 64, 65, 50, 67, 69, 71])), {72})

    def test_no_candidate_without_a_match_or_for_same_notes(self) -> None:
        self.assertEqual(candidate_pitches(melody([60, 61, 63, 66, 70, 75])), set())
        self.assertEqual(candidate_pitches(melody([60, 60, 60, 60, 60, 60, 60])), set())

    def test_an_overlapping_window_is_not_used(self) -> None:
        # the only (2,2,x) window overlaps the current one: 60 62 64 | 66 -> nothing earlier
        self.assertEqual(candidate_pitches(melody([60, 62, 64, 66])), set())

    def test_outside_the_horizon_is_ignored(self) -> None:
        # first pattern starts 10 s before the current one: outside the 8 s horizon
        self.assertEqual(candidate_pitches(melody([60, 62, 64, 65, 50, 67, 69, 71], ioi=2.0)), set())


class BiasTest(unittest.TestCase):
    def test_adds_bias_only_to_candidate_note_on(self) -> None:
        seq = torch.tensor(melody([60, 62, 64, 65, 50, 67, 69, 71]))
        logits = torch.zeros(1, 390)
        proc = PatternCacheBias(math.log(3.0))
        out = proc(logits, seq)
        self.assertAlmostEqual(float(out[0, 72]), math.log(3.0), places=6)
        self.assertEqual(float(out.abs().sum()), float(out[0, 72]))
        self.assertEqual((proc.steps, proc.fired), (1, 1))
        masked = torch.full((1, 390), float("-inf"))
        self.assertTrue(torch.isinf(proc(masked, seq)[0, 72]))          # grammar-masked stays masked


class VerdictTest(unittest.TestCase):
    def s(self, on_ir=0.03, ci=(0.01, 0.03), rep=0.05, copy=0.0, valid=1.0):
        arm = lambda ir: {"interval": ir, "repetition": rep, "copy_rate": copy, "valid_rate": valid}
        return {"real": {"interval": 0.052, "repetition": 0.073, "copy_rate": 0.01},
                "arms": {"off": arm(0.007), "on": arm(on_ir)}, "ir_on_minus_off": {"ci95": list(ci)}}

    def test_rules(self) -> None:
        self.assertTrue(verdict(self.s())["pass"])
        self.assertFalse(verdict(self.s(on_ir=0.02))["pass"])
        self.assertFalse(verdict(self.s(ci=(-0.001, 0.03)))["pass"])
        self.assertFalse(verdict(self.s(rep=0.3))["pass"])
        self.assertFalse(verdict(self.s(copy=0.3))["pass"])


if __name__ == "__main__":
    unittest.main()
