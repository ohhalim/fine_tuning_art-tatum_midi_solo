from __future__ import annotations

import unittest

import pretty_midi

from scripts.run_chord_ablation import (
    MIN_NOTES_FOR_RATIO,
    PERMUTATION,
    PROGRESSION,
    fit,
)
from inference.control.chord_primer import chord_tone_pitch_classes


class PermutationDesignTests(unittest.TestCase):
    """The permutation must hold key fit constant and still separate."""

    def test_same_multiset_in_a_different_order(self):
        self.assertEqual(sorted(PROGRESSION), sorted(PERMUTATION))
        self.assertNotEqual(PROGRESSION, PERMUTATION)

    def test_no_bar_keeps_its_own_chord(self):
        """A bar where the permutation repeats the chord measures nothing."""
        for true_chord, permuted in zip(PROGRESSION, PERMUTATION):
            with self.subTest(bar=true_chord):
                self.assertNotEqual(true_chord, permuted)

    def test_pairs_share_few_chord_tones(self):
        """Diatonic sevenths overlap; the design has to keep that low."""
        for true_chord, permuted in zip(PROGRESSION, PERMUTATION):
            shared = chord_tone_pitch_classes(true_chord) & chord_tone_pitch_classes(permuted)
            with self.subTest(pair=(true_chord, permuted)):
                self.assertLessEqual(len(shared), 2, f"{true_chord}/{permuted} share {shared}")


class FitTests(unittest.TestCase):
    def test_fit_is_read_only(self):
        notes = [pretty_midi.Note(90, 62, 0.0, 0.2), pretty_midi.Note(90, 61, 0.2, 0.4)]
        before = [(n.pitch, n.start, n.end, n.velocity) for n in notes]

        self.assertAlmostEqual(0.5, fit(notes, "Dm7"))
        self.assertEqual(before, [(n.pitch, n.start, n.end, n.velocity) for n in notes])

    def test_empty_bar_is_none_not_zero(self):
        self.assertIsNone(fit([], "Dm7"))

    def test_min_notes_threshold_is_above_one(self):
        """A single-note bar can only score 0.0 or 1.0, so it must be excluded."""
        self.assertGreater(MIN_NOTES_FOR_RATIO, 1)


class PrimerLengthTests(unittest.TestCase):
    def test_prefix_costs_slots_from_the_body(self):
        """Adding the prefix must not push the primer over the budget."""
        from scripts.run_chord_ablation import PRIMER_MAX_TOKENS, build_primer

        with_prefix = build_primer("chord", "Dm7", bpm=128, conditioning_midi=None,
                                   with_prefix=True)
        without = build_primer("chord", "Dm7", bpm=128, conditioning_midi=None,
                               with_prefix=False)

        self.assertLessEqual(len(with_prefix), PRIMER_MAX_TOKENS)
        self.assertLessEqual(len(without), PRIMER_MAX_TOKENS)
        self.assertGreater(len(with_prefix), len(without))

    def test_default_arm_requires_a_conditioning_midi(self):
        from scripts.run_chord_ablation import build_primer

        with self.assertRaisesRegex(ValueError, "conditioning-midi"):
            build_primer("default", "Dm7", bpm=128, conditioning_midi=None, with_prefix=False)


class RefreshModeTests(unittest.TestCase):
    """The control must remove harmony without also removing the primer."""

    def test_first_mode_second_block_carries_played_notes_not_a_chord(self):
        from scripts.run_chord_ablation import build_block_primer
        from inference.control.chord_primer import chord_tone_pitch_classes

        played = [pretty_midi.Note(88, 75, 0.0, 0.2), pretty_midi.Note(88, 77, 0.2, 0.4)]
        primer = build_block_primer(None, bpm=128, seconds=0.9375, carry_notes=played)

        tokens = primer.tolist()
        self.assertIn(75, tokens)
        self.assertIn(77, tokens)
        # The bare fallback would be a single [60]; this is a real primer.
        self.assertGreater(len(tokens), 1)

    def test_chord_mode_states_the_harmony(self):
        from scripts.run_chord_ablation import build_block_primer
        from inference.control.chord_primer import chord_tone_pitch_classes

        primer = build_block_primer("Dm7", bpm=128, seconds=0.9375)
        pitch_classes = {t % 12 for t in primer.tolist() if 0 <= t < 128}

        self.assertTrue(pitch_classes <= chord_tone_pitch_classes("Dm7"),
                        f"{pitch_classes} outside Dm7")

    def test_carry_tail_is_retimed_into_the_block(self):
        from scripts.run_chord_ablation import _carry_tail

        notes = [pretty_midi.Note(80, 60 + i, 1.0 + i * 0.1, 1.1 + i * 0.1) for i in range(6)]
        tail = _carry_tail(notes, seconds=0.5, keep=3)

        self.assertEqual(3, len(tail))
        self.assertAlmostEqual(0.0, min(n.start for n in tail))
        self.assertLessEqual(max(n.end for n in tail), 0.5)
        self.assertTrue(all(n.end > n.start for n in tail))

    def test_carry_tail_of_nothing_is_empty(self):
        from scripts.run_chord_ablation import _carry_tail

        self.assertEqual([], _carry_tail([], seconds=0.5))
