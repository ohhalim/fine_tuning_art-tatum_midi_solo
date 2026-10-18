"""Selection and output rules of the #1786 clip lick baseline (synthetic notes)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import clip18_lick as cl  # noqa: E402

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None


def make(runs, lh=()):
    """runs: [(start, n, step, dur)] right-hand runs; lh: [(start, end, pitches)]."""
    notes = []
    for s0, n, step, dur in runs:
        notes += [pretty_midi.Note(velocity=70 + i % 3, pitch=72 + i % 5, start=s0 + i * step, end=s0 + i * step + dur) for i in range(n)]
    notes += [pretty_midi.Note(velocity=55, pitch=p, start=s, end=e) for s, e, ps in lh for p in ps]
    out = [(f"i0_n{j}", n) for j, n in enumerate(notes)]
    return sorted(out, key=lambda x: (x[1].start, x[1].pitch, x[0]))


@unittest.skipIf(pretty_midi is None, "pretty_midi not installed")
class Clip18LickTest(unittest.TestCase):
    def test_first_lick_inside_clip_skips_slow_and_outside_runs(self):
        notes = make([(68.0, 12, 0.2, 0.18),        # a lick, but it starts before the clip
                      (70.0, 8, 0.5, 0.45),          # inside, but slow (median IOI 0.5 s)
                      (75.0, 12, 0.2, 0.3)])         # first lick inside: picked
        run, *_ = cl.pick_lick(notes)
        self.assertAlmostEqual(run[0][1].start, 75.0)
        self.assertEqual(len(run), 12)

    def test_no_lick_returns_none(self):
        run, *_ = cl.pick_lick(make([(70.0, 8, 0.5, 0.45)]))
        self.assertIsNone(run)

    def test_outputs_keep_original_lengths_and_held_notes(self):
        notes = make([(75.0, 12, 0.2, 0.3)], lh=[(74.5, 77.0, (41, 45, 52))])
        run, line, trunc, acc = cl.pick_lick(notes)
        out = cl.build(notes, run, trunc, acc)
        self.assertEqual(len(out["melody"]), 12)
        self.assertAlmostEqual(out["melody"][0]["end"] - out["melody"][0]["start"], 0.3)      # original length, not cut
        self.assertAlmostEqual(out["melody"][0]["trunc_end"], 0.2)
        held = [r for r in out["twohand"] if r["cut_start"]]
        self.assertEqual(sorted(r["pitch"] for r in held), [41, 45, 52])                       # left hand struck before t0
        self.assertAlmostEqual(out["context"]["implied_tempo_bpm_if_ioi_is_eighth"], 150.0)


if __name__ == "__main__":
    unittest.main()
