"""Phrase breath A/B verdict (scripts/phrase_breath_check.py)."""
from __future__ import annotations

import unittest

from scripts.phrase_breath_check import judge


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        a = {"notes_per_s": 5.3, "rest_share": 0.045, "phrase_median": 26, "solo_chord_tone": 0.52}
        b = {"notes_per_s": 4.6, "rest_share": 0.08, "phrase_median": 15, "solo_chord_tone": 0.51}
        ax, bx = {"gen_ms_max": 230, "rendered_invalid": 0}, {"gen_ms_max": 240, "rendered_invalid": 0}
        self.assertTrue(judge(a, b, ax, bx)["pass"])
        self.assertFalse(judge(a, {**b, "phrase_median": 25}, ax, bx)["pass"])
        self.assertFalse(judge(a, {**b, "rest_share": 0.04}, ax, bx)["pass"])          # not more than A
        self.assertFalse(judge({**a, "rest_share": 0.01}, {**b, "rest_share": 0.03}, ax, bx)["pass"])  # < half data
        self.assertFalse(judge(a, {**b, "notes_per_s": 2.0}, ax, bx)["pass"])
        self.assertFalse(judge(a, {**b, "solo_chord_tone": 0.48}, ax, bx)["pass"])
        self.assertFalse(judge(a, b, ax, {**bx, "rendered_invalid": 1})["pass"])
        self.assertFalse(judge(a, b, ax, {**bx, "gen_ms_max": 260})["pass"])


if __name__ == "__main__":
    unittest.main()
