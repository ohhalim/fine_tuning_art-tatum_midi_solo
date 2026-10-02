"""Phrase breath A/B verdict (scripts/phrase_breath_check.py)."""
from __future__ import annotations

import unittest

import json
import tempfile
from pathlib import Path

from scripts.phrase_breath_check import breath_problems, judge


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


class BreathSettingTest(unittest.TestCase):
    def test_max_notes_and_rest_must_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            def run(name, pb):
                d = Path(tmp) / name
                d.mkdir()
                (d / "continuous_report.json").write_text(json.dumps({"phrase_breath": pb}))
                return str(d)
            ok = run("ok", {"max_notes": 24, "rest_s": 0.4, "dropped_notes": 3})
            bad_rest = run("bad_rest", {"max_notes": 24, "rest_s": 0.8, "dropped_notes": 3})
            off = run("off", None)
            self.assertEqual(breath_problems([ok], 24), [])
            self.assertTrue(breath_problems([bad_rest], 24))
            self.assertTrue(breath_problems([off], 24))
            self.assertEqual(breath_problems([off], None), [])
            self.assertTrue(breath_problems([ok], None))


if __name__ == "__main__":
    unittest.main()
