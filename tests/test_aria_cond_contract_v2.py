"""Condition contract v2 (scripts/aria_cond_contract_v2.py) on synthetic token lists and plans."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from scripts.aria_cond_contract_v2 import CAP_S, FEATURE_DIM, NC, beats_to_ms, features, plan_from_beats, vector

CMAJ7, CM7, G7, F7 = frozenset({0, 4, 7, 11}), frozenset({0, 3, 7, 10}), frozenset({7, 11, 2, 5}), frozenset({5, 9, 0, 3})
# 120 BPM: one beat = 500 ms. Cmaj7 beats 0-4, Cm7 4-8, G7 8-16
PLAN = plan_from_beats([(0, 4, CMAJ7), (4, 8, CM7), (8, 16, G7)], [(0, 120)])


def note(p, onset):
    return [("piano", p, 80), ("onset", onset), ("dur", 200)]


HEAD = [("prefix", "instrument", "piano"), "<S>"]


class ContractV2Test(unittest.TestCase):
    def test_before_at_and_after_a_change(self) -> None:
        toks = HEAD + note(64, 1900) + note(63, 2000) + note(63, 2100)
        f = features(toks, PLAN)
        before, at, after = f[3], f[6], f[9]                       # onset positions: 1900, 2000, 2100 ms
        self.assertEqual((before["current"], before["next"], round(before["delta_sec"], 3)), (CMAJ7, CM7, 0.1))
        self.assertEqual((at["current"], at["next"], round(at["delta_sec"], 3)), (CM7, G7, 2.0))   # change at t is current
        self.assertEqual((after["current"], after["next"], round(after["delta_sec"], 3)), (CM7, G7, 1.9))

    def test_pitch_position_before_a_crossing_onset_keeps_the_old_time(self) -> None:
        toks = HEAD + note(64, 1900) + note(63, 2100)
        f = features(toks, PLAN)
        self.assertEqual((f[5]["current"], f[5]["next"]), (CMAJ7, CM7))   # (piano, 63): its onset is not read yet
        self.assertAlmostEqual(f[5]["delta_sec"], 0.1)

    def test_simultaneous_notes_share_the_time(self) -> None:
        toks = HEAD + note(60, 1000) + note(64, 1000) + note(67, 1000)
        f = features(toks, PLAN)
        self.assertEqual({(x["t_ms"], x["current"]) for x in f[3:]}, {(1000, CMAJ7)})

    def test_rest_across_several_changes_points_to_the_first_one(self) -> None:
        toks = HEAD + note(60, 500)                                  # then silence over the 2 s and 4 s changes
        last = features(toks, PLAN)[-1]
        self.assertEqual((last["current"], last["next"], last["delta_sec"]), (CMAJ7, CM7, 1.5))

    def test_five_second_boundary(self) -> None:
        toks = HEAD + note(60, 4900) + ["<T>"]
        f = features(toks, PLAN)
        self.assertEqual((f[-1]["t_ms"], f[-1]["current"], f[-1]["next"], f[-1]["delta_sec"]), (5000, G7, None, None))
        self.assertEqual((f[-1]["has_next"], f[-1]["delta_norm"], f[-1]["delta_clamped"]), (0, 0.0, 0))

    def test_tempo_change_converts_beats_with_the_plan_map(self) -> None:
        tm = [(0, 120), (4, 60)]                                     # beats 0-4 at 500 ms, then 1000 ms per beat
        self.assertEqual((beats_to_ms(4, tm), beats_to_ms(6, tm)), (2000.0, 4000.0))
        plan = plan_from_beats([(0, 4, CMAJ7), (4, 6, CM7), (6, 8, G7)], tm)
        f = features(HEAD + note(60, 2500), plan)[-1]
        self.assertEqual((f["current"], f["next"], f["delta_sec"]), (CM7, G7, 1.5))

    def test_far_change_is_clamped_and_differs_from_no_change(self) -> None:
        plan = plan_from_beats([(0, 20, CMAJ7), (20, 24, CM7)], [(0, 120)])      # change at 10 s
        f = features(HEAD + note(60, 1000), plan)[-1]
        self.assertEqual((f["has_next"], f["delta_norm"], f["delta_clamped"]), (1, 1.0, 1))
        self.assertEqual(f["delta_sec"], 9.0)
        self.assertGreater(9.0, CAP_S)

    def test_unknown_and_no_chord_are_different(self) -> None:
        plan = plan_from_beats([(0, 4, CMAJ7), (4, 8, NC), (12, 16, G7)], [(0, 120)])   # gap 8-12 beats is unknown
        nc = features(HEAD + note(60, 2500), plan)[-1]
        gap = features(HEAD + note(60, 4500), plan)[-1]
        end = features(HEAD + note(60, 8500), plan)[-1]
        self.assertEqual((nc["current"], nc["current_known"], vector(nc)[:12]), (NC, 1, [0.0] * 12))
        self.assertEqual((gap["current"], gap["current_known"], gap["next"], gap["next_known"]), (None, 0, G7, 1))
        self.assertEqual((end["current_known"], end["has_next"]), (0, 0))
        self.assertNotEqual(vector(nc), vector(gap))

    def test_repeated_chord_segments_are_merged(self) -> None:
        plan = plan_from_beats([(0, 4, CM7), (4, 8, CM7), (8, 12, G7)], [(0, 120)])
        f = features(HEAD + note(60, 500), plan)[-1]
        self.assertEqual((f["next"], f["delta_sec"]), (G7, 3.5))

    def test_suffix_with_different_next_onsets_does_not_change_prefix_features(self) -> None:
        prefix = HEAD + note(64, 1500)
        a = features(prefix + note(63, 2100), PLAN)
        b = features(prefix + note(62, 4700), PLAN)
        self.assertEqual(a[: len(prefix) + 1], b[: len(prefix) + 1])  # up to and including the next pitch token

    def test_swapping_only_the_next_planned_chord_changes_only_next(self) -> None:
        other = plan_from_beats([(0, 4, CMAJ7), (4, 8, F7), (8, 16, G7)], [(0, 120)])
        toks = HEAD + note(64, 1000)
        x, y = features(toks, PLAN)[-1], features(toks, other)[-1]
        vx, vy = vector(x), vector(y)
        self.assertEqual(len(vx), FEATURE_DIM)
        self.assertEqual(vx[:12], vy[:12])
        self.assertNotEqual(vx[12:24], vy[12:24])
        self.assertEqual(vx[24:], vy[24:])


if __name__ == "__main__":
    unittest.main()
