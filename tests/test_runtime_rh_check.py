"""Runtime solo-line verdict (scripts/runtime_rh_check.py)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.runtime_rh_check import PROGRESSIONS, SEEDS, check_run_set, judge


def write_run(root: Path, name: str, **over) -> str:
    tag = name.rsplit("_", 2)[1]
    chords, bars, bpm = PROGRESSIONS[tag]
    r = {"chords": chords.split(","), "bars": bars, "bpm": bpm, "run_completed": True,
         "completed_bars": bars, "solo_line": True, "comp": False, "temperature": 1.0,
         "context_carry_tokens": 0, "context_history": False, "pattern_cache": False, **over}
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "continuous_report.json").write_text(json.dumps(r))
    return str(d)


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


class RunSetTest(unittest.TestCase):
    def full(self, root: Path) -> list[str]:
        return [write_run(root, f"bebop_{t}_s{s}") for t in PROGRESSIONS for s in SEEDS]

    def test_the_full_preregistered_set_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(check_run_set(self.full(Path(tmp)), "bebop"), [])

    def test_a_single_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = check_run_set([write_run(Path(tmp), "bebop_iiVI_s42")], "bebop")
            self.assertEqual(sum("missing" in p for p in problems), 5)

    def test_incomplete_comp_or_wrong_settings_are_refused(self) -> None:
        for over in ({"run_completed": False}, {"completed_bars": 3}, {"comp": True}, {"solo_line": False},
                     {"bpm": 90}, {"seed": 7}, {"temperature": 0.6}, {"context_carry_tokens": 256},
                     {"context_history": True}, {"pattern_cache": True}, {"checkpoint": "outputs/final_tatum/export/checkpoint_update518.pt"}):
            with tempfile.TemporaryDirectory() as tmp:
                dirs = self.full(Path(tmp))
                write_run(Path(tmp), "bebop_iiVI_s42", **over)
                self.assertTrue(check_run_set(dirs, "bebop"), over)

    def test_other_presets_or_unregistered_runs_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dirs = self.full(Path(tmp)) + [write_run(Path(tmp) / "other", "tatum_iiVI_s42")]
            self.assertTrue(any("not in the preregistered set" in p for p in check_run_set(dirs, "bebop")))


if __name__ == "__main__":
    unittest.main()
