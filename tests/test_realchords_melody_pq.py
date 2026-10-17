"""Input contract and scoring of the #1780 runner. No model, no realchords import."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import realchords_melody_pq as rc  # noqa: E402


class InputContractTest(unittest.TestCase):
    def test_chord_frames_onset_on_change_hold_on_repeat(self):
        f = rc.chord_frames(rc.PROG["P"])
        self.assertEqual(len(f), 128)
        onsets = [i for i, n in enumerate(f) if n.startswith("CHORD_ON_")]
        self.assertEqual(onsets, [0, 16, 32, 64, 80, 96])                 # bars 4 and 8 repeat Fmaj7: no new onset
        self.assertEqual(f[48], "CHORD_Fmaj7")
        self.assertEqual(f[64], "CHORD_ON_Gm7")
        self.assertEqual(rc.chord_frames(rc.PROG["Q"])[0], "CHORD_ON_Gm7b5")

    def test_primer_is_eighths_on_16th_grid(self):
        p = rc.primer_frames()
        self.assertEqual(len(p), 16)
        self.assertEqual(p[:4], ["NOTE_ON_67", "NOTE_67", "NOTE_ON_69", "NOTE_69"])
        self.assertEqual(rc.frames_to_notes(p), [(2 * i, 2 * i + 2, q) for i, q in enumerate(rc.HEAD_BAR1)])

    def test_frames_to_notes_silence_and_repeated_onset(self):
        names = ["NOTE_ON_60", "NOTE_60", "SILENCE", "NOTE_ON_62", "NOTE_ON_62", "NOTE_62", "NOTE_62"]
        self.assertEqual(rc.frames_to_notes(names), [(0, 2, 60), (3, 4, 62), (4, 7, 62)])


class ScoringTest(unittest.TestCase):
    def test_bar_scores_split_first16_and_pair_diff(self):
        # bar 3 (frames 32-47): A (P tone) on the first 16th and C later; bar 5 (frames 64-79): Db (Q tone) on the first 16th
        p_notes = [(32, 34, 69), (36, 38, 72)]
        q_notes = [(32, 34, 68), (64, 66, 61)]
        bp, bq = rc.bar_scores(p_notes), rc.bar_scores(q_notes)
        self.assertEqual(bp[3]["all"], {"n": 2, "P": 1, "Q": 0})
        self.assertEqual(bp[3]["first16"], {"n": 1, "P": 1, "Q": 0})
        self.assertEqual(bp[3]["rest"], {"n": 1, "P": 0, "Q": 0})
        self.assertEqual(bq[5]["first16"], {"n": 1, "P": 0, "Q": 1})
        self.assertTrue(bp[3]["transition"] and bp[5]["transition"] and bp[7]["transition"])
        self.assertFalse(bp[4]["transition"] or bp[8]["transition"])        # 4 and 8 hold Fmaj7/Fm7
        self.assertEqual(rc.pair_diff(bp, bq)[3], round(1 / 2 - (-1 / 1), 3))
        self.assertEqual(rc.pair_diff(bp, bq, "first16")[3], 2.0)
        self.assertEqual(rc.pair_diff(bp, bq, "rest")[3], None)             # Q has no later onset in bar 3
        self.assertIsNone(rc.pair_diff(bp, bq)[4])

    def test_phrase_metrics_seconds(self):
        notes = [(16, 18, 60), (18, 20, 62), (24, 26, 64)]                   # gap 4 frames = 0.469 s
        m = rc.phrase_metrics(notes)
        self.assertEqual(m["gen_notes"], 3)
        self.assertEqual(m["rests_ge_0.3s"], 1)
        self.assertAlmostEqual(m["dur_median_s"], round(2 * rc.FRAME_S, 3))


class FailedCellTest(unittest.TestCase):
    def test_short_failed_cell_is_not_a_normal_result(self):
        full = {"status": "done", "melody": [3] * 128, "filter_violations": 0}
        short = {"status": "failed", "stop": "cell wall 120 s", "melody": [3] * 40, "filter_violations": 0}
        done_but_short = {"status": "done", "melody": [3] * 127, "filter_violations": 0}
        violating = {"status": "done", "melody": [3] * 128, "filter_violations": 1}
        cells = {("P", 1): full, ("Q", 1): short, ("P", 2): done_but_short, ("Q", 2): violating}
        ok, bad = rc.split_cells(cells)
        self.assertEqual(list(ok), [("P", 1)])
        self.assertEqual(set(bad), {("Q", 1), ("P", 2), ("Q", 2)})
        self.assertFalse(rc.cell_complete(None))


class GenerateCellTest(unittest.TestCase):
    def test_primer_kept_and_one_token_per_frame(self):
        import torch
        V = 10
        calls = []

        def logits(x):
            calls.append(x.shape[1])
            lg = torch.full((V,), -50.0)
            lg[5] = 50.0
            return lg

        buf = {}
        seq = rc.generate_cell(logits, lambda lg, x, t: lg, 1, [3, 4], [7, 8, 9, 9], 1, buf)
        self.assertEqual(buf["melody"], [3, 4, 5, 5])
        self.assertEqual(seq, [1, 3, 7, 4, 8, 5, 9, 5, 9])                  # melody before chord in each frame
        self.assertEqual(calls, [5, 7])                                     # frame t sees chords up to t-1 only


if __name__ == "__main__":
    unittest.main()
