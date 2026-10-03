"""Combined listening-config verdict (scripts/combined_check.py)."""
from __future__ import annotations

import unittest

from scripts.combined_check import judge


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        p = {"fallback_total": 0, "misses_total": 1, "invalid": 0, "gen_ms_p99": 350, "across_ge9": 0.1,
             "across_p50": 2, "on_beat_clash": 0.35, "notes_per_s": 4.5, "rest_share": 0.12, "copied_block_share": 0.0}
        c = {"delivery": 1.0, "composite_alignment": 1.0}
        self.assertTrue(judge(p, c)["pass"])
        for k, v in (("gen_ms_p99", 450), ("across_ge9", 0.3), ("across_p50", 5), ("on_beat_clash", 0.42),
                     ("notes_per_s", 6.0), ("copied_block_share", 0.1), ("misses_total", 3)):
            self.assertFalse(judge({**p, k: v}, c)["pass"], k)
        self.assertFalse(judge(p, {**c, "delivery": 0.95})["pass"])
        self.assertFalse(judge(p, {**c, "composite_alignment": 0.9})["pass"])


if __name__ == "__main__":
    unittest.main()
