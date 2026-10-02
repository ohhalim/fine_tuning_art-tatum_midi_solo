"""Runtime solo-line verdict (scripts/runtime_rh_check.py)."""
from __future__ import annotations

import unittest

from scripts.runtime_rh_check import judge


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        good = {"notes_per_s": 4.0, "rest_share": 0.06, "phrase_median": 14, "solo_chord_tone": 0.5, "fallback_total": 0}
        base = {"solo_chord_tone": 0.5}
        self.assertTrue(judge(good, base)["pass"])
        self.assertFalse(judge({**good, "notes_per_s": 8.0}, base)["pass"])
        self.assertFalse(judge({**good, "rest_share": 0.01}, base)["pass"])
        self.assertFalse(judge({**good, "phrase_median": 40}, base)["pass"])
        self.assertFalse(judge({**good, "solo_chord_tone": 0.4}, base)["pass"])
        self.assertFalse(judge({**good, "fallback_total": 1}, base)["pass"])


if __name__ == "__main__":
    unittest.main()
