"""Retrieval rules (scripts/wjazzd_retrieval.py) on synthetic beats and notes (issue #1718)."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from scripts.wjazzd_retrieval import boundary_errors, build_windows, match, place, rank, transpose_amount


def beats(chords_per_beat, start=0.0, step=0.5, first_bar=0, steps=None):
    """chords_per_beat: chord written at each beat ('' = none). 4/4, bars from first_bar."""
    out, t = [], start
    for i, ch in enumerate(chords_per_beat):
        out.append((t, first_bar + i // 4, i % 4 + 1, ch))
        t += steps[i] if steps else step
    return out


II_V = ["D-7", "", "", "", "G7", "", "", "", ""]          # 2 bars + the 9th beat for the window end


class RetrievalRulesTest(unittest.TestCase):
    def test_chord_sounding_at_window_start_is_carried_in(self) -> None:
        b = beats(["C-7", "", "", "", "", "", "", "", "F7", "", "", "", ""])
        w = [x for x in build_windows(b) if x["bar"] == 1][0]
        self.assertTrue(w["eligible"])
        self.assertEqual(w["rel"], ((0, 0, "m7"), (4, 5, "7")))   # C-7 written in bar 0, still sounding at bar 1

    def test_repeated_chord_marks_are_merged(self) -> None:
        w = build_windows(beats(["C-7", "", "C-7", "", "F7", "", "", "", ""]))[0]
        self.assertEqual(w["rel"], ((0, 0, "m7"), (4, 5, "7")))

    def test_window_needs_the_next_beat_and_full_bars(self) -> None:
        self.assertEqual(build_windows(beats(II_V[:8])), [])       # no 9th beat, no end time
        self.assertEqual(len(build_windows(beats(II_V))), 1)

    def test_lossy_or_unmapped_chords_make_the_window_ineligible(self) -> None:
        w = build_windows(beats(["D-7", "", "", "", "G79b", "", "", "", ""]))[0]
        self.assertFalse(w["eligible"])

    def test_transposition_keeps_the_relative_progression(self) -> None:
        a = build_windows(beats(II_V))[0]
        b = build_windows(beats(["C-7", "", "", "", "F7", "", "", "", ""]))[0]
        self.assertEqual(a["rel"], b["rel"])
        self.assertEqual(transpose_amount(a["root0"], b["root0"]), 2)

    def test_uneven_source_beats_give_a_recorded_boundary_error(self) -> None:
        tgt = build_windows(beats(II_V))[0]                                     # even 0.5 s beats
        src = build_windows(beats(II_V, steps=[0.6, 0.6, 0.6, 0.6, 0.4, 0.4, 0.4, 0.4, 0.5]))[0]
        pl = place([(0.0, 60, 0.3)], src, tgt, 0)
        self.assertAlmostEqual(pl["scale"], 4.0 / 4.0)
        err = boundary_errors(src, tgt, pl["scale"])
        self.assertAlmostEqual(err[0]["error_s"], 0.4)                          # change at 2.4 s vs 2.0 s

    def test_boundary_notes(self) -> None:
        w = build_windows(beats(II_V))[0]                                       # [0, 4) s
        notes = [(-0.2, 60, 0.5), (0.5, 62, 0.2), (3.8, 64, 0.6), (4.1, 65, 0.2)]
        pl = place(notes, w, w, 0)
        self.assertEqual(pl["carry_in_excluded"], 1)
        self.assertEqual([p for p, _, _ in pl["notes"]], [62, 64])
        self.assertEqual(pl["cropped_at_end"], 1)
        self.assertAlmostEqual(pl["cropped_seconds"], 0.4)
        self.assertAlmostEqual(pl["notes"][-1][2], 4.0)

    def test_empty_and_out_of_range_windows_are_not_candidates_and_no_match(self) -> None:
        tgt = build_windows(beats(II_V))[0]
        empty = dict(build_windows(beats(II_V))[0], melid=1, bar=0)
        high = dict(build_windows(beats(["C-7", "", "", "", "F7", "", "", "", ""]))[0], melid=2, bar=0)
        notes = {1: [], 2: [(0.0, 126, 0.2)]}
        bank = {tgt["rel"]: [empty, high]}
        ranked, excluded = match(tgt, bank, lambda m, s, e: notes[m])
        self.assertEqual(ranked, [])
        self.assertEqual(dict(excluded), {"empty_window": 1, "out_of_midi_range": 1})   # C -> D is +2: 126 + 2 > 127
        self.assertEqual(match(tgt, {}, lambda m, s, e: [])[0], [])

    def test_rank_is_deterministic(self) -> None:
        cands = [{"melid": 9, "bar": 3, "scale": 1.0, "transpose": 4}, {"melid": 7, "bar": 2, "scale": 1.0, "transpose": -1},
                 {"melid": 3, "bar": 8, "scale": 1.5, "transpose": 0}]
        order = [(c["melid"], c["bar"]) for c in rank(cands)]
        self.assertEqual(order, [(7, 2), (9, 3), (3, 8)])
        self.assertEqual(order, [(c["melid"], c["bar"]) for c in rank(list(reversed(cands)))])


if __name__ == "__main__":
    unittest.main()


class ExportTest(unittest.TestCase):
    def test_sub_tick_note_is_excluded_not_lengthened(self) -> None:
        from scripts.wjazzd_retrieval import quantize_for_export
        kept, excluded, _ = quantize_for_export([(77, 1.9431, 1.9433)], 1.9433)   # the 0.2 ms note of #1718
        self.assertEqual(kept, [])
        self.assertEqual(len(excluded), 1)

    def test_one_tick_note_is_kept(self) -> None:
        from scripts.wjazzd_retrieval import quantize_for_export
        kept, excluded, _ = quantize_for_export([(60, 0.010, 0.011)], 1.0)
        self.assertEqual((kept, excluded), ([(0, 60, 10, 11)], []))

    def test_rounding_ties_follow_the_writer(self) -> None:
        from scripts.wjazzd_retrieval import quantize_for_export
        kept, excluded, _ = quantize_for_export([(60, 0.0015, 0.0025), (62, 0.0025, 0.0045)], 1.0)
        self.assertEqual(round(1.5), 2)
        self.assertEqual(round(2.5), 2)                    # Python rounds half to even, as write_notes does
        self.assertEqual([e["pitch"] for e in excluded], [60])   # 2 >= 2: excluded
        self.assertEqual(kept, [(1, 62, 2, 4)])

    def test_end_never_passes_the_window_end_tick(self) -> None:
        from scripts.wjazzd_retrieval import quantize_for_export
        kept, _, checks = quantize_for_export([(60, 0.5, 1.0004)], 1.0)
        self.assertEqual(kept[0][3], 1000)
        self.assertEqual(checks["past_window_end"], 0)

    def test_same_pitch_overlap_is_counted(self) -> None:
        from scripts.wjazzd_retrieval import quantize_for_export
        _, _, checks = quantize_for_export([(60, 0.0, 0.5), (60, 0.3, 0.6), (62, 0.0, 0.5)], 1.0)
        self.assertEqual(checks["same_pitch_overlaps"], 1)

    def test_kept_notes_survive_a_midi_round_trip_exactly(self) -> None:
        import tempfile
        from scripts.aria_t4b_sim import read_notes, write_notes
        from scripts.wjazzd_retrieval import quantize_for_export
        kept, _, _ = quantize_for_export([(60, 0.0123, 0.2), (64, 0.2, 0.2004), (67, 0.21, 0.5)], 0.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.mid")
            write_notes([(p, on / 1000, off / 1000, 80) for _, p, on, off in kept], path)
            back = [(p, round(s * 1000), round(e * 1000)) for p, s, e, _ in read_notes(path)]
        self.assertEqual(sorted(back), sorted((p, on, off) for _, p, on, off in kept))
