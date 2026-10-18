"""Run splitting and the separation rule of the #1762 calibration (no MIDI files)."""
from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import criteria_calibration as cc  # noqa: E402


class CalibrationTest(unittest.TestCase):
    def test_runs_split_on_gaps_over_quarter_second(self):
        line = [SimpleNamespace(start=s, end=s + 0.1) for s in (0.0, 0.2, 0.4, 1.0, 1.2)]
        self.assertEqual([len(r) for r in cc.runs(line)], [3, 2])           # 0.4+0.1 -> 1.0 is a 0.5 s gap

    def test_separation_counts_items_outside_liked_range(self):
        res = cc.separation({"L": [1, 2, 3], "N": [2, 9, 10], "R1": [2.5, 2.0], "R2": [None]}, "x")
        self.assertEqual(res["L_range"], [1, 3])
        self.assertEqual(res["N"], {"outside": 2, "n": 3, "majority_outside": True})
        self.assertEqual(res["R1"], {"outside": 0, "n": 2, "majority_outside": False})
        self.assertNotIn("R2", res)

    def test_notes_before_from_s_are_ignored(self):
        import tempfile
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(initial_tempo=128)
        inst = pretty_midi.Instrument(program=0)
        inst.notes = [pretty_midi.Note(velocity=90, pitch=60 + (i % 3), start=i * 0.1171875, end=i * 0.1171875 + 0.11)
                      for i in range(16)]                                     # bar 1: even 16ths (primer)
        inst.notes += [pretty_midi.Note(velocity=90, pitch=67 + (i % 2), start=1.875 + i * 0.3, end=1.875 + i * 0.3 + 0.2)
                       for i in range(8)]                                    # generated bars: one slower run
        pm.instruments.append(inst)
        with tempfile.NamedTemporaryFile(suffix=".mid") as f:
            pm.write(f.name)
            full = cc.metrics(f.name, False)
            gen = cc.metrics(f.name, False, from_s=1.875)
        self.assertEqual(gen["line_notes"], 8)
        self.assertEqual(full["line_notes"], 24)
        self.assertNotEqual(full["P4_even_ioi_share"], gen["P4_even_ioi_share"])


if __name__ == "__main__":
    unittest.main()
