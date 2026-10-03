"""Varied comping (inference/control/comping.py)."""
from __future__ import annotations

import unittest

from inference.control.comping import FIGURES, chord_tones, closest, comp_half, voicings
from scripts.comping_eval import comp_run, judge, metrics


class CompTest(unittest.TestCase):
    def test_voicings_are_rootless_3rd_7th_plus_one(self) -> None:
        root, (third, fifth, seventh, ninth) = chord_tones("G7")
        for v in voicings("G7"):
            pcs = {p % 12 for p in v}
            self.assertTrue({third, seventh} <= pcs and root not in pcs and (fifth in pcs or ninth in pcs))
            self.assertTrue(all(52 <= p <= 64 for p in v))

    def test_voice_leading_prefers_small_motion(self) -> None:
        prev = (53, 60, 64)                                   # Dm7: F C E
        self.assertEqual(closest(voicings("G7"), prev), (53, 59, 62))       # F B D

    def test_new_chord_stated_in_first_beat_with_root_and_never_rest(self) -> None:
        state = {}
        chords = ["Dm7", "Dm7", "G7", "G7", "Cmaj7", "Cmaj7"] * 10
        figures = []
        for block, chord in enumerate(chords):
            changed = chord != state.get("chord")
            notes, fig = comp_half(chord, block=block, bpm=128, seed=3, state=state)
            figures.append(fig)
            if changed:
                self.assertNotEqual(fig, "rest")
                first = min(s for _, s, _, _ in notes)
                self.assertLess(first, 60 / 128)
                self.assertIn(36 + chord_tones(chord)[0], [p for p, s, _, _ in notes if s == first])
        self.assertTrue(all(a != b for a, b in zip(figures, figures[1:])))
        self.assertEqual(set(figures) <= set(FIGURES), True)

    def test_deterministic(self) -> None:
        a = comp_half("F7", block=5, bpm=120, seed=9, state={})
        b = comp_half("F7", block=5, bpm=120, seed=9, state={})
        self.assertEqual(a, b)


class EvalTest(unittest.TestCase):
    def test_random_control_fails_and_fixed_comp_repeats(self) -> None:
        chords, bars = ["Gm7", "C7", "Fmaj7", "Fmaj7"], 16
        m = {arm: metrics(comp_run(chords, bars, arm, 42), chords, bars, []) for arm in "FVR"}
        self.assertEqual(m["F"]["max_identical_bar_rhythm_run"], bars)
        self.assertEqual(m["F"]["velocity_std"], 0.0)
        self.assertEqual(m["V"]["chord_alignment"], 1.0)
        self.assertTrue(m["R"]["chord_alignment"] < 1.0 or m["R"]["voice_motion_p50"] > 3.0)


if __name__ == "__main__":
    unittest.main()
