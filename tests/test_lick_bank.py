"""Lick bank (inference/control/lick_bank.py)."""
from __future__ import annotations

import unittest

import pretty_midi

from inference.control.lick_bank import build, lick_from_run, plan


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


if __name__ == "__main__":
    unittest.main()
