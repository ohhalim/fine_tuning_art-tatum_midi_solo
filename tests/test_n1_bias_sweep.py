"""N1 bias sweep selection rule and confirmation verdict (scripts/n1_bias_sweep.py)."""
from __future__ import annotations

import unittest

from scripts.n1_bias_sweep import judge, select


class SelectTest(unittest.TestCase):
    def test_smallest_r_meeting_both_conditions(self) -> None:
        s = {0.0: {"same_note_share": 0.17, "within_p90": 9}, 0.5: {"same_note_share": 0.10, "within_p90": 11},
             1.0: {"same_note_share": 0.06, "within_p90": 12}}
        self.assertEqual(select(s), 0.5)
        s[0.5]["within_p90"] = 15
        self.assertEqual(select(s), 1.0)
        s[1.0]["same_note_share"] = 0.2
        self.assertIsNone(select(s))


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        b = {"avoid_share": 0.03, "clash_share": 0.08, "on_beat_clash": 0.25, "across_ge9": 0.12, "across_p50": 3,
             "same_note_share": 0.1, "within_p90": 11, "distinct_openings": 0.9, "copied_block_share": 0.0,
             "notes_per_s": 4.6, "rest_share": 0.13, "fallback_total": 0, "misses_total": 0, "invalid": 0,
             "gen_ms_p99": 280}
        self.assertTrue(judge(b)["pass"])
        for k, v in (("avoid_share", 0.09), ("on_beat_clash", 0.3), ("same_note_share", 0.12), ("within_p90", 13),
                     ("gen_ms_p99", 430), ("fallback_total", 1)):
            self.assertFalse(judge({**b, k: v})["pass"], k)


if __name__ == "__main__":
    unittest.main()
