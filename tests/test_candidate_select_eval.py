"""Candidate selection ranker and verdict (scripts/candidate_select_eval.py)."""
from __future__ import annotations

import unittest

from scripts.candidate_select_eval import judge, rank_pick


class RankTest(unittest.TestCase):
    def test_picks_best_valid_candidate_with_enough_notes(self) -> None:
        pcs = {0, 4, 7}
        wrong = [(61, 0.0, 0.2), (66, 0.2, 0.4), (63, 0.4, 0.6)]
        right = [(60, 0.0, 0.2), (64, 0.2, 0.4), (67, 0.4, 0.6)]
        cands = [{"solo": wrong, "valid": True}, {"solo": right, "valid": False},
                 {"solo": right[:2], "valid": True}, {"solo": right, "valid": True}]
        self.assertEqual(rank_pick(cands, pcs), 3)            # invalid and 2-note candidates are skipped
        self.assertEqual(rank_pick([{"solo": [], "valid": True}], pcs), 0)


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        a = {"notes_per_s": 4.6, "same_note_share": 0.05, "empty_rate": 0.02, "leap_share": 0.1,
             "distinct_opening_intervals": 0.8, "on_beat_clash": 0.4}
        b = dict(a)
        c = {**a, "on_beat_clash": 0.3}
        self.assertTrue(judge(a, b, c, 0.1)["pass"])
        self.assertFalse(judge(a, b, c, 0.04)["pass"])
        self.assertFalse(judge(a, b, {**c, "notes_per_s": 6.0}, 0.1)["pass"])
        self.assertFalse(judge(a, b, {**c, "same_note_share": 0.11}, 0.1)["pass"])
        self.assertFalse(judge(a, b, {**c, "distinct_opening_intervals": 0.7}, 0.1)["pass"])
        self.assertFalse(judge(a, b, {**c, "on_beat_clash": 0.4}, 0.1)["pass"])


if __name__ == "__main__":
    unittest.main()
