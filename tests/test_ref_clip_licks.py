"""Every-lick selection and speed of the #1809 reference clip licks (synthetic notes)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import ref_clip_licks as rl  # noqa: E402

try:
    import pretty_midi
except ImportError:  # pragma: no cover
    pretty_midi = None


def make(runs):
    """runs: [(start, n, step, dur)] right-hand runs above G3."""
    notes = []
    for s0, n, step, dur in runs:
        notes += [pretty_midi.Note(velocity=70, pitch=72 + i % 5, start=s0 + i * step, end=s0 + i * step + dur) for i in range(n)]
    out = [(f"i0_n{j}", n) for j, n in enumerate(notes)]
    return sorted(out, key=lambda x: (x[1].start, x[1].pitch, x[0]))


@unittest.skipIf(pretty_midi is None, "pretty_midi not installed")
class RefClipLicksTest(unittest.TestCase):
    def test_every_lick_inside_the_window_is_kept_and_rejects_are_counted(self):
        notes = make([(68.0, 12, 0.2, 0.18),        # starts before the window: not counted
                      (71.0, 12, 0.2, 0.18),        # lick 1
                      (75.0, 30, 0.2, 0.18),        # 30 notes > 24: too long
                      (83.0, 8, 0.5, 0.45),         # slow: other reject
                      (90.0, 10, 0.15, 0.12)])      # lick 2
        found, counted = rl.clip_licks(notes, (70.0, 100.0))
        self.assertEqual([run[0][1].start for run, _ in found], [71.0, 90.0])
        self.assertEqual(counted, {"runs_8plus": 4, "too_long": 1, "other_reject": 1})

    def test_speed_is_notes_per_second_over_the_onset_span(self):
        found, _ = rl.clip_licks(make([(71.0, 11, 0.1, 0.08)]), (70.0, 100.0))
        sp = rl.speed(found[0][0])
        self.assertEqual((sp["notes"], sp["span_s"], sp["notes_per_s"], sp["median_ioi_s"]), (11, 1.0, 10.0, 0.1))


if __name__ == "__main__":
    unittest.main()
