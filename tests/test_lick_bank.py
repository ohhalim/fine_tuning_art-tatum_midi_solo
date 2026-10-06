"""Lick bank (inference/control/lick_bank.py)."""
from __future__ import annotations

import unittest

import pretty_midi

from inference.control.lick_bank import build, lick_from_run, plan, swing


def run(n, ioi=0.15, start=0.0, base=70):
    return [pretty_midi.Note(velocity=80, pitch=base + (k % 5), start=start + k * ioi, end=start + k * ioi + ioi * 0.9)
            for k in range(n)]


class LickTest(unittest.TestCase):
    def test_runs_in_eighths_whatever_the_source_tempo(self) -> None:
        a, b = lick_from_run(run(10, ioi=0.15)), lick_from_run(run(10, ioi=0.22))
        self.assertEqual(a["onset_8ths"], [float(k) for k in range(10)])
        self.assertEqual(a["onset_8ths"], b["onset_8ths"])            # the same lick at two tempi
        self.assertEqual(a["intervals"], [1, 1, 1, 1, -4, 1, 1, 1, 1])

    def test_short_or_slow_runs_are_not_licks(self) -> None:
        self.assertIsNone(lick_from_run(run(5)))
        self.assertIsNone(lick_from_run(run(10, ioi=0.5)))

    def test_plan_fills_the_bars_without_stopping_early(self) -> None:
        bank = build([run(12) + run(9, start=3.0)])
        ev = plan(bank, bars=8, bpm=128, seed=1)
        self.assertGreater(ev[-1]["onset"], 8 * 4 * 60 / 128 * 0.75)
        self.assertTrue(all(e["interval"] is None for e in ev if e is ev[0]))

    def test_swing_never_overlaps_the_next_onset(self) -> None:
        # Astra's repro on #1669: min_s=0.12 used to push the first end past the second onset (42 ms same-pitch overlap)
        out = swing([(60, 0.234375, 0.344375, 70), (60, 0.3515625, 0.4615625, 70)], 128)
        self.assertLess(out[0][2], out[1][1])
        self.assertGreater(out[0][2], out[0][1])

    def test_swing_keeps_min_length_when_there_is_room(self) -> None:
        out = swing([(60, 0.0, 0.05, 70), (62, 0.46875, 0.6, 70)], 128)
        self.assertAlmostEqual(out[0][2] - out[0][1], 0.12)

    def test_swing_chord_tones_do_not_cut_each_other(self) -> None:
        chord = [(48, 0.0, 0.5, 46), (52, 0.0, 0.5, 46), (48, 0.46875, 0.8, 46), (52, 0.46875, 0.8, 46)]
        out = swing(chord, 128)
        self.assertEqual(out[0][2], out[1][2])                       # same onset, same end
        self.assertGreater(out[0][2] - out[0][1], 0.4)
        for p, s, e, _ in out[:2]:
            self.assertLess(e, out[2][1])                             # nor into the next chord's same pitches

    def test_swing_same_pitch_bound_holds_for_sub_10ms_gaps(self) -> None:
        # Astra's repro (10/6): the 10 ms minimum used to push the first end past a 5.3 ms-later onset
        out = swing([(60, 0.0, 0.004, 70), (60, 0.004, 0.008, 70)], 128)
        self.assertLess(out[0][2], out[1][1])
        self.assertGreater(out[0][2], out[0][1])

    def test_swing_keeps_a_held_comp_note_over_another_voice(self) -> None:
        # a sustained bass under an off-beat upper note is held, not cut at the other voice's onset (Astra on #1674)
        out = swing([(36, 0.0, 0.9, 46), (60, 0.234375, 0.3, 70)], 128)
        self.assertGreater(out[0][2], out[1][1] + 0.3)

if __name__ == "__main__":
    unittest.main()
