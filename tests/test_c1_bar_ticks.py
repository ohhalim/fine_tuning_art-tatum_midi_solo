"""Bar attribution from MIDI ticks in scripts/c1_comp_ab.py (#1678 review)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.c1_comp_ab import bar_of


def roundtrip(times, bpm):
    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm)
    inst = pretty_midi.Instrument(program=0)
    inst.notes = [pretty_midi.Note(velocity=60, pitch=60 + k, start=t, end=t + 0.1) for k, t in enumerate(times)]
    pm.instruments.append(inst)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "x.mid"
        pm.write(str(path))
        back = pretty_midi.PrettyMIDI(str(path))
    return back, sorted(n.start for n in back.instruments[0].notes)


class BarTicksTest(unittest.TestCase):
    def test_bar_line_strike_read_back_early_stays_in_the_new_bar(self) -> None:
        back, (t,) = roundtrip([5 * 60 / 128 * 4], 128)          # 9.375 s, read back slightly early
        self.assertLess(t, 9.375)
        self.assertEqual(bar_of(back, t), 5)

    def test_anticipation_before_the_bar_line_stays_in_the_old_bar(self) -> None:
        back, (t,) = roundtrip([9.375 - 0.0015], 128)             # 1.5 ms early, deliberate
        self.assertEqual(bar_of(back, t), 4)

    def test_other_tempo(self) -> None:
        bar = 60 / 100 * 4
        back, times = roundtrip([3 * bar, 3 * bar - 0.003], 100)
        self.assertEqual(sorted(bar_of(back, t) for t in times), [2, 3])


if __name__ == "__main__":
    unittest.main()
