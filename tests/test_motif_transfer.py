"""Motif transfer A/B/C: selection, control, metrics and gate (docs/experiments/MOTIF_TRANSFER.md)."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi

from scripts.coherence_metrics import top_line
from scripts.generate import encode_notes_simple
from scripts.motif_transfer_ab import (
    LineNote,
    arm_rates,
    comparable,
    gate,
    grams,
    has_copy,
    intervals,
    motif_notes,
    prepare_case,
    primer_for,
    reappearance,
    select_motif,
    session_notes,
    shuffled_motif,
    top_line_notes,
)
from scripts.run_resident_model_probe import validate_generated_token_block


def line_of(pitches, ioi=0.2, t0=0.0):
    return [(t0 + i * ioi, p) for i, p in enumerate(pitches)]


def line_notes(pitches, iois, t0=0.0, vel=80):
    out, t = [], t0
    for i, p in enumerate(pitches):
        out.append(LineNote(t, p, t + 0.5, vel))
        if i < len(iois):
            t += iois[i]
    return out


def record(valid=True, windows=2, variant=1, exact=0, interval=1, distinct=2, copy=False, ct=(1, 2)):
    return {"valid": valid, "windows": windows, "variant": variant, "exact": exact, "interval": interval,
            "distinct_windows": distinct, "copy": copy, "chord_tone": ct}


def session(vr_b, vr_a, vr_c=None, cases=4, ct=(0.5, 0.5), valid=(1.0, 1.0), copies=0, b_valid=4):
    def arm(vr, ctv=0.5, vrate=1.0, cp=0, nv=4):
        return {"VR": vr, "chord_tone": ctv, "valid_rate": vrate, "copies": cp, "valid": nv}
    return {"BA": {"case_count": cases, "B": arm(vr_b, ct[0], valid[0], copies, b_valid), "A": arm(vr_a, ct[1], valid[1])},
            "BC": {"case_count": cases, "B": arm(vr_b), "C": arm(vr_c if vr_c is not None else vr_a)}}


class SelectionTest(unittest.TestCase):
    def test_top_line_notes_agree_with_coherence_top_line(self) -> None:
        notes = [(0.0, 0.3, 40, 50), (0.01, 0.2, 72, 90), (0.5, 0.7, 74, 60), (0.52, 0.9, 50, 70), (0.9, 1.0, 60, 64)]
        line = top_line_notes(notes)
        self.assertEqual([(x.onset, x.pitch) for x in line], top_line([(s, p) for s, _, p, _ in notes]))
        self.assertEqual((line[0].end, line[0].velocity), (0.2, 90))   # the highest note's own end and velocity

    def test_select_motif_takes_the_latest_run_with_every_ioi_in_range(self) -> None:
        line = line_notes([60, 62, 64, 65, 67, 69, 71], [0.2, 0.2, 0.2, 0.2, 0.7, 0.2])
        run = select_motif(line, 0.0, 2.0)
        self.assertEqual([x.pitch for x in run], [60, 62, 64, 65, 67])   # every later run spans the 0.7 s gap
        later = line_notes([60, 62, 64, 65, 67, 69], [0.2] * 5)
        self.assertEqual([x.pitch for x in select_motif(later, 0.0, 2.0)], [62, 64, 65, 67, 69])

    def test_no_motif_when_too_few_notes_or_out_of_window(self) -> None:
        line = line_notes([60, 62, 64, 65, 67], [0.2, 0.03, 0.2, 0.2])   # 30 ms IOI
        self.assertIsNone(select_motif(line, 0.0, 2.0))
        self.assertIsNone(select_motif(line_notes([60, 62, 64, 65], [0.2] * 3), 0.0, 2.0))
        self.assertIsNone(select_motif(line_notes([60, 62, 64, 65, 67], [0.2] * 4), 0.5, 2.0))

    def test_motif_is_monophonic_from_zero_and_encodes_cleanly(self) -> None:
        run = [LineNote(1.0, 60, 1.9, 80), LineNote(1.2, 60, 1.21, 80), LineNote(1.5, 64, 1.6, 70),
               LineNote(1.7, 62, 2.5, 90), LineNote(1.9, 65, 2.5, 90)]
        notes = motif_notes(run, t_end=2.0)
        self.assertEqual([n.start for n in notes], [0.0, 0.2, 0.5, 0.7, 0.9])
        self.assertTrue(all(a.end <= b.start for a, b in zip(notes, notes[1:])))
        self.assertTrue(all(n.end - n.start >= 0.02 - 1e-9 for n in notes))
        self.assertAlmostEqual(notes[-1].end, 1.0)                        # clipped at the block start
        tokens = encode_notes_simple(notes)
        self.assertLessEqual(len(tokens), 24)
        self.assertTrue(validate_generated_token_block(tokens, lookahead_ms=1000, allow_rest_bar=True)["valid"])

    def test_session_velocity_comes_from_played_mid(self) -> None:
        report = {"bpm": 120, "beats_per_bar": 4,
                  "played_bars": [{"notes": [[60, 0.5, 0.7], [64, 1.0, 1.2]]}, {"notes": [[67, 0.25, 0.5]]}]}
        midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
        inst = pretty_midi.Instrument(program=0)
        for p, s, e, v in ((60, 0.0, 0.2, 70), (64, 0.5, 0.7, 90)):          # starts at the first note
            inst.notes.append(pretty_midi.Note(velocity=v, pitch=p, start=s, end=e))
        midi.instruments = [inst]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "played.mid"
            midi.write(str(path))
            notes = session_notes(report, path)
        self.assertEqual([(round(s, 3), p, v) for s, _, p, v in notes], [(0.5, 60, 70), (1.0, 64, 90), (2.25, 67, None)])


class ShuffledControlTest(unittest.TestCase):
    def motif(self, pitches):
        return [pretty_midi.Note(velocity=60 + 4 * i, pitch=p, start=0.2 * i, end=0.2 * i + 0.15)
                for i, p in enumerate(pitches)]

    def test_control_keeps_rhythm_ends_and_token_count_but_no_shared_trigram(self) -> None:
        b = self.motif([60, 62, 65, 64, 69])
        c, perm = shuffled_motif(b, seed=7)
        self.assertEqual([(n.start, n.end, n.velocity) for n in c], [(n.start, n.end, n.velocity) for n in b])
        self.assertEqual((c[0].pitch, c[-1].pitch), (b[0].pitch, b[-1].pitch))
        self.assertNotEqual(perm, intervals([n.pitch for n in b]))
        self.assertFalse(grams(perm) & grams(intervals([n.pitch for n in b])))
        self.assertEqual(len(encode_notes_simple(c)), len(encode_notes_simple(b)))
        self.assertEqual(shuffled_motif(b, seed=7)[1], perm)                  # deterministic

    def test_no_control_when_every_permutation_shares_a_trigram(self) -> None:
        self.assertIsNone(shuffled_motif(self.motif([60, 62, 64, 66, 68]), seed=1))   # all intervals equal
        # one differing interval is enough: (0,0,0,2) -> (0,2,0,0) shares no 3-gram
        self.assertEqual(shuffled_motif(self.motif([60, 60, 60, 60, 62]), seed=1)[1], (0, 2, 0, 0))

    def test_control_stays_in_the_piano_range(self) -> None:
        res = shuffled_motif(self.motif([100, 107, 108, 96, 103]), seed=3)
        if res is not None:
            self.assertTrue(all(21 <= n.pitch <= 108 for n in res[0]))


class ReappearanceTest(unittest.TestCase):
    SESSION = line_of([60, 62, 64, 65, 67], ioi=0.2)            # patterns (2,2,1) and (2,1,2)

    def test_exact_variant_and_interval_are_separate(self) -> None:
        exact = reappearance(line_of([60, 62, 64, 65], ioi=0.2), self.SESSION)
        self.assertEqual((exact["exact"], exact["variant"], exact["interval"], exact["windows"]), (1, 0, 1, 1))
        transposed = reappearance(line_of([67, 69, 71, 72], ioi=0.2), self.SESSION)
        self.assertEqual((transposed["exact"], transposed["variant"]), (0, 1))
        slower = reappearance(line_of([60, 62, 64, 65], ioi=0.23), self.SESSION)    # ratio 1.15
        self.assertEqual((slower["exact"], slower["variant"]), (0, 1))
        far = reappearance(line_of([67, 69, 71, 72], ioi=0.4), self.SESSION)        # ratio 2.0
        self.assertEqual((far["variant"], far["interval"]), (0, 1))

    def test_same_note_and_too_short_lines_have_no_windows(self) -> None:
        same = reappearance(line_of([60] * 8), self.SESSION)
        self.assertEqual((same["windows"], same["variant"], same["exact"]), (0, 0, 0))
        self.assertEqual(reappearance(line_of([60, 62, 64]), self.SESSION)["windows"], 0)
        # a same-note session gives nothing to match either
        self.assertEqual(reappearance(line_of([67, 67, 67, 67]), line_of([60] * 6))["interval"], 0)

    def test_mechanical_repetition_does_not_raise_the_rate(self) -> None:
        once = reappearance(line_of([67, 69, 71, 72]), self.SESSION)
        looped = reappearance(line_of([67, 69, 71, 72] * 4), self.SESSION)
        self.assertEqual(once["variant"] / once["windows"], 1.0)
        self.assertLess(looped["variant"] / looped["windows"], once["variant"] / once["windows"])
        self.assertLessEqual(looped["variant"], 3)          # distinct session patterns, not occurrences

    def test_copy_flag_needs_five_matching_notes(self) -> None:
        ref = line_of([60, 62, 64, 65, 67, 69])
        self.assertTrue(has_copy(line_of([62, 64, 65, 67, 69], t0=3.0), [ref]))
        self.assertFalse(has_copy(line_of([62, 64, 65, 67], t0=3.0), [ref]))
        self.assertFalse(has_copy(line_of([63, 65, 66, 68, 70], t0=3.0), [ref]))   # transposed is not a copy


class AggregationTest(unittest.TestCase):
    def test_invalid_outputs_leave_reappearance_but_stay_in_validity(self) -> None:
        r = arm_rates([record(), record(valid=False, windows=10, variant=10)])
        self.assertEqual((r["cases"], r["valid"], r["valid_rate"]), (2, 1, 0.5))
        self.assertEqual((r["windows"], r["VR"]), (2, 0.5))

    def test_zero_windows_give_no_rate(self) -> None:
        r = arm_rates([record(windows=0, variant=0, interval=0, distinct=0)])
        self.assertIsNone(r["VR"])
        self.assertEqual(r["zero_window_valid"], 1)
        self.assertIsNone(arm_rates([])["valid_rate"])

    def test_comparison_case_lists(self) -> None:
        self.assertTrue(comparable("c_unavailable", "BA"))
        self.assertFalse(comparable("c_unavailable", "BC"))
        self.assertFalse(comparable("no_motif", "BA"))

    def test_gate_passes_only_with_enough_positive_sessions(self) -> None:
        good = {f"s{i}": session(0.10, 0.05, 0.05) for i in range(6)}
        self.assertTrue(gate(good, 5)["pass"])
        two_bad = dict(good, s0=session(0.0, 0.05, 0.05), s1=session(0.0, 0.05, 0.05))
        self.assertFalse(gate(two_bad, 5)["VR_BC"]["pass"])

    def test_undefined_sessions_count_as_not_positive(self) -> None:
        sessions = {f"s{i}": session(0.10, 0.05, 0.05) for i in range(4)}
        sessions.update({"s4": session(None, None, None, cases=0), "s5": session(None, 0.05, 0.05)})
        g = gate(sessions, 5)
        self.assertEqual((g["VR_BC"]["positive"], g["VR_BC"]["defined"]), (4, 4))
        self.assertFalse(g["pass"])
        none = gate({"s0": session(None, None, None, cases=0)}, 1)
        self.assertFalse(none["VR_BA"]["pass"])
        self.assertIsNone(none["VR_BA"]["mean"])

    def test_tolerance_and_copy_cap(self) -> None:
        base = {f"s{i}": session(0.10, 0.05, 0.05) for i in range(6)}
        worse_ct = {k: session(0.10, 0.05, 0.05, ct=(0.40, 0.50)) for k in base}
        self.assertFalse(gate(worse_ct, 5)["chord_tone_BA"]["pass"])
        copies = {k: session(0.10, 0.05, 0.05, copies=2, b_valid=4) for k in base}
        self.assertFalse(gate(copies, 5)["copy_B"]["pass"])


class CaseTest(unittest.TestCase):
    def report(self, pitches, ioi=0.2):
        notes = [[p, round(i * ioi, 4), round(i * ioi + 0.15, 4)] for i, p in enumerate(pitches)]
        return {"bpm": 120, "beats_per_bar": 4, "bars": 4, "chords": ["Dm7", "G7"],
                "played_bars": [{"notes": notes}, {"notes": []}, {"notes": []}, {"notes": []}]}

    def case(self, pitches, k=2):
        report = self.report(pitches)
        notes = [(s, e, p, 80) for p, s, e in report["played_bars"][0]["notes"]]
        return prepare_case("m", "sess", report, notes, top_line_notes(notes), k)

    def test_primers_share_the_chord_statement_and_b_c_lengths(self) -> None:
        c = self.case([60, 62, 65, 64, 69, 67, 71])
        self.assertEqual(c["status"], "ok")
        self.assertEqual(c["chord"], "G7")                       # block 2 = bar 1
        a, b, cc = (primer_for(c, arm) for arm in "ABC")
        self.assertEqual(b[-len(a):], a)
        self.assertEqual(cc[-len(a):], a)
        self.assertEqual(len(b), len(cc))
        self.assertEqual(c["motif_B"], [65, 64, 69, 67, 71])     # the latest run
        self.assertTrue(all(t < 0.0 for t, _ in c["session_line"]))

    def test_statuses_are_kept(self) -> None:
        self.assertEqual(self.case([60, 62])["status"], "no_motif")
        self.assertEqual(self.case([60, 62, 64, 66, 68, 70])["status"], "c_unavailable")


if __name__ == "__main__":
    unittest.main()
