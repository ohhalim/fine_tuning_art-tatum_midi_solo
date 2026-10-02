"""Repeat-weight marking and weighted loss (scripts/repeat_weights.py)."""
from __future__ import annotations

import unittest

import pretty_midi
import torch

from scripts.coherence_metrics import motif_reuse, top_line
from scripts.generate import encode_notes_simple
from scripts.repeat_weights import batch_weights, repeat_note_on_mask, top_line_tokens, weighted_smooth_ce
from scripts.style_distance import tokens_to_notes


def melody(pitches, ioi=0.2, chord_every=0):
    notes = []
    for k, p in enumerate(pitches):
        notes.append(pretty_midi.Note(velocity=80, pitch=p, start=k * ioi, end=k * ioi + 0.1))
        if chord_every and k % chord_every == 0:
            notes.append(pretty_midi.Note(velocity=60, pitch=40, start=k * ioi, end=k * ioi + 0.15))
    return encode_notes_simple(sorted(notes, key=lambda n: (n.start, n.pitch)))


class MarkTest(unittest.TestCase):
    def test_top_line_matches_the_metric(self) -> None:
        toks = melody([60, 62, 64, 65, 67, 69], chord_every=2)
        ours = [(round(t, 3), p) for t, p, _ in top_line_tokens(toks)]
        ref = [(round(t, 3), p) for t, p in top_line([(n.start, n.pitch) for n in tokens_to_notes(toks)])]
        self.assertEqual(ours, ref)

    def test_marks_the_completing_note_on_of_a_repeat_only(self) -> None:
        motif = [0, 2, 4, 5]
        toks = melody([60 + p for p in motif + [9] + motif], chord_every=3)
        mask = repeat_note_on_mask(toks)
        marked = [toks[i] for i, m in enumerate(mask) if m]
        self.assertEqual(marked, [65])                                   # 4th note of the second motif
        line = top_line([(n.start, n.pitch) for n in tokens_to_notes(toks)])
        self.assertEqual(sum(mask), motif_reuse(line)[0])                # same count as the metric

    def test_same_note_and_out_of_horizon_are_not_marked(self) -> None:
        self.assertFalse(any(repeat_note_on_mask(melody([60] * 12))))
        motif = [60, 62, 64, 65]
        far = melody(motif + [50] * 0 + motif, ioi=1.6)                  # second copy starts 6.4 s later: in
        self.assertTrue(any(repeat_note_on_mask(far)))
        farther = melody(motif + motif, ioi=2.5)                        # starts 10 s later: out of 8 s
        self.assertFalse(any(repeat_note_on_mask(farther)))


class LossTest(unittest.TestCase):
    def test_weight_one_equals_the_smoothed_loss(self) -> None:
        import sys
        sys.path.insert(0, "music_transformer")
        from model.loss import SmoothCrossEntropyLoss

        torch.manual_seed(0)
        logits, target = torch.randn(6, 390), torch.tensor([1, 5, 7, 389, 3, 0])
        w = torch.ones(6)
        ref = SmoothCrossEntropyLoss(0.1, 390, ignore_index=-100)(logits, target)
        self.assertAlmostEqual(float(weighted_smooth_ce(logits, target, w, 0.1, 390)), float(ref), places=5)

    def test_pad_gets_zero_weight_and_marked_tokens_weight(self) -> None:
        motif = [0, 2, 4, 5]
        toks = melody([60 + p for p in motif + [9] + motif])
        seq = torch.tensor(toks + [999, 999])
        x, y = seq[:-1].unsqueeze(0), seq[1:].unsqueeze(0)
        w = batch_weights(x, y, 3.0, pad=999)
        self.assertEqual(float(w[0, -1]), 0.0)
        self.assertEqual(int((w == 3.0).sum()), 1)


if __name__ == "__main__":
    unittest.main()
