"""Harmony bias verdict (scripts/dissonance_check.py)."""
from __future__ import annotations

import unittest

from scripts.dissonance_check import judge, judge_repeat


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        a = {"avoid_share": 0.16, "clash_share": 0.12, "on_beat_clash": 0.28, "distinct_openings": 0.6,
             "same_note_share": 0.11}
        b = {"avoid_share": 0.07, "clash_share": 0.07, "on_beat_clash": 0.25, "across_ge9": 0.15, "across_p50": 3,
             "distinct_openings": 0.6, "same_note_share": 0.12, "copied_block_share": 0.0, "notes_per_s": 4.6,
             "rest_share": 0.12, "fallback_total": 0, "misses_total": 0, "invalid": 0, "gen_ms_p99": 400}
        self.assertTrue(judge(a, b)["pass"])
        for k, v in (("avoid_share", 0.1), ("clash_share", 0.1), ("on_beat_clash", 0.3), ("across_p50", 6),
                     ("distinct_openings", 0.5), ("notes_per_s", 6.0), ("gen_ms_p99", 430)):
            self.assertFalse(judge(a, {**b, k: v})["pass"], k)


class RepeatJudgeTest(unittest.TestCase):
    def test_repetition_must_not_exceed_baseline(self) -> None:
        a = {"avoid_share": 0.16, "clash_share": 0.12, "on_beat_clash": 0.28, "distinct_openings": 0.6,
             "same_note_share": 0.11}
        b = {"avoid_share": 0.05, "clash_share": 0.08, "on_beat_clash": 0.25, "across_ge9": 0.15, "across_p50": 3,
             "distinct_openings": 0.6, "same_note_share": 0.10, "copied_block_share": 0.0, "notes_per_s": 4.6,
             "rest_share": 0.12, "fallback_total": 0, "misses_total": 0, "invalid": 0, "gen_ms_p99": 400}
        self.assertTrue(judge_repeat(a, b)["pass"])
        self.assertFalse(judge_repeat(a, {**b, "same_note_share": 0.12})["pass"])
        self.assertFalse(judge_repeat(a, {**b, "avoid_share": 0.1})["pass"])


if __name__ == "__main__":
    unittest.main()
