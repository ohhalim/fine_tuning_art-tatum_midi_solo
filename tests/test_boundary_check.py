"""Boundary continuity verdict (scripts/boundary_check.py)."""
from __future__ import annotations

import unittest

from scripts.boundary_check import judge


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        a = {"within_p90": 7, "on_beat_clash": 0.35, "same_note_share": 0.1}
        b = {"across_ge9": 0.15, "across_p50": 3, "within_p90": 8, "on_beat_clash": 0.36, "same_note_share": 0.12,
             "copied_block_share": 0.0, "notes_per_s": 4.5, "rest_share": 0.08, "fallback_total": 0,
             "misses_total": 0, "invalid": 0, "gen_ms_p99": 300}
        self.assertTrue(judge(a, b)["pass"])
        for k, v in (("across_ge9", 0.3), ("across_p50", 6), ("within_p90", 10), ("on_beat_clash", 0.4),
                     ("same_note_share", 0.2), ("copied_block_share", 0.1), ("notes_per_s", 7.0),
                     ("rest_share", 0.01), ("fallback_total", 1), ("gen_ms_p99", 500)):
            self.assertFalse(judge(a, {**b, k: v})["pass"], k)


if __name__ == "__main__":
    unittest.main()
