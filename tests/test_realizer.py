"""Phrase realizer (inference/control/realizer.py)."""
from __future__ import annotations

import unittest

import pretty_midi

from inference.control.realizer import HIGH, LOW, abstract, realize


def note(p, s, e):
    return pretty_midi.Note(velocity=80, pitch=p, start=s, end=e)


def c7_at(t):
    return {0, 4, 7, 10}, {0, 2, 4, 7, 9, 10}


class RealizerTest(unittest.TestCase):
    def test_abstract_keeps_rhythm_and_contour(self) -> None:
        ev = abstract([note(70, 0.0, 0.2), note(72, 0.2, 0.4), note(69, 0.4, 0.6), note(75, 1.2, 1.4)])
        self.assertEqual([e["interval"] for e in ev], [None, 2, -3, None])
        self.assertEqual([e["phrase_end"] for e in ev], [False, False, True, True])

    def test_anchors_are_chord_tones_and_notes_stay_in_range_and_scale(self) -> None:
        beat = 60 / 128
        src = [note(60 + (k * 5) % 30, k * beat / 2, k * beat / 2 + 0.1) for k in range(40)]   # climbing contour
        out = realize(abstract(src), chord_at=c7_at, bpm=128)
        for p, s, e, _ in out:
            self.assertTrue(LOW <= p <= HIGH)
            on_beat = abs(s - round(s / beat) * beat) <= 0.03
            if on_beat:
                self.assertIn(p % 12, {0, 4, 7, 10})
        outside = [(i, p) for i, (p, *_rest) in enumerate(out) if p % 12 not in {0, 2, 4, 7, 9, 10}]
        for i, p in outside:                                   # chromatic only as an approach into a chord tone
            self.assertLessEqual(abs(out[i + 1][0] - p), 2)
            self.assertIn(out[i + 1][0] % 12, {0, 4, 7, 10})

    def test_contour_direction_is_kept(self) -> None:
        src = [note(p, k * 0.2, k * 0.2 + 0.15) for k, p in enumerate([67, 69, 71, 72, 71, 69, 67, 65])]
        out = realize(abstract(src), chord_at=c7_at, bpm=128)
        sign = lambda x: (x > 0) - (x < 0)
        self.assertEqual([sign(b[0] - a[0]) for a, b in zip(out, out[1:])], [1, 1, 1, -1, -1, -1, -1])


class ApproachResolvesTest(unittest.TestCase):
    def test_an_approach_note_is_followed_by_its_goal(self) -> None:
        beat = 60 / 128
        # weak short note a step below a beat-1 note: realized as a chromatic approach that resolves up by a half step
        src = [note(70, 0.0, 0.2), note(72, beat * 1.5, beat * 1.5 + 0.1), note(74, beat * 2, beat * 2 + 0.3)]
        out = realize(abstract(src), chord_at=c7_at, bpm=128)
        for (p, s, e, _), nxt in zip(out, out[1:]):
            if p % 12 not in {0, 2, 4, 7, 9, 10}:
                self.assertEqual(nxt[0] - p, 1)

    def test_comp_aware_candidates_avoid_semitones(self) -> None:
        from inference.control.realizer import clear_comp
        solo = [(65, 0.0, 0.5, 80)]                  # F over a sounding E: semitone
        comp = [(52, 0.0, 1.0, 50)]
        fixed = clear_comp(solo, comp, c7_at)
        self.assertNotIn((fixed[0][0] - 52) % 12, (1, 11))


if __name__ == "__main__":
    unittest.main()
