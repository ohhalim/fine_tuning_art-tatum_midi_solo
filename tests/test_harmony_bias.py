"""Avoid-note logit penalty (inference/control/harmony_bias.py)."""
from __future__ import annotations

import unittest

import torch

from inference.control.harmony_bias import HarmonyBias, avoid_pitch_classes


class BiasTest(unittest.TestCase):
    def test_avoid_sets(self) -> None:
        self.assertEqual(avoid_pitch_classes("C7"), {5, 11})                 # F, B over C7
        self.assertEqual(avoid_pitch_classes("Cmaj7"), {1, 3, 5, 8, 10})
        self.assertEqual(avoid_pitch_classes("Dm7"), {3, 6, 8, 10, 1})       # Eb F# Bb C... relative to D
        self.assertIn(5, avoid_pitch_classes("Cmaj7"))                        # the 11th

    def test_only_avoid_note_on_tokens_move_and_masked_stay_masked(self) -> None:
        proc = HarmonyBias("C7", strength=2.0)
        logits = torch.zeros(1, 390)
        logits[0, 200] = float("-inf")
        out = proc(logits)
        self.assertAlmostEqual(float(out[0, 65]), -2.0)                      # F4: avoid over C7
        self.assertAlmostEqual(float(out[0, 64]), 0.0)                       # E4: chord tone
        self.assertAlmostEqual(float(out[0, 61]), 0.0)                       # Db: b9 allowed on a dominant
        self.assertTrue(torch.isinf(out[0, 200]))
        self.assertEqual(float(out[0, 128:].clamp(min=-1).abs().sum()), 1.0)  # only the masked one changed below


class RepeatTest(unittest.TestCase):
    def test_last_note_on_is_penalised(self) -> None:
        proc = HarmonyBias("C7", strength=2.0, repeat=1.5)
        seq = torch.tensor([300, 64, 260, 192, 67, 270])          # ... note_on 64, ..., note_on 67, time shift
        out = proc(torch.zeros(1, 390), seq)
        self.assertAlmostEqual(float(out[0, 67]), -1.5)            # G: last struck, chord tone
        self.assertAlmostEqual(float(out[0, 64]), 0.0)
        self.assertAlmostEqual(float(out[0, 65]), -2.0)            # F: avoid
        self.assertAlmostEqual(float(HarmonyBias("C7", 2.0)(torch.zeros(1, 390), seq)[0, 67]), 0.0)


if __name__ == "__main__":
    unittest.main()
