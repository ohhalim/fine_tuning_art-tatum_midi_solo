"""--candidates runtime check (scripts/runtime_candidates_check.py)."""
from __future__ import annotations

import unittest

import json
import tempfile
from pathlib import Path

from scripts.runtime_candidates_check import EXPECT, PROGRESSIONS, SEEDS, check_set, pctl, run_stats


def write(root: Path, n: int, tag: str, sd: int, **over) -> str:
    r = {"chords": PROGRESSIONS[tag].split(","), "candidates": n, "solo_line": True, "comp": False,
         "phrase_breath": None, "run_completed": True, "completed_bars": 16, "seed": sd,
         "checkpoint": "/x/outputs/bebop_rh/export/checkpoint_update516.pt", **EXPECT, **over}
    d = root / f"bebop_{tag}_s{sd}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "continuous_report.json").write_text(json.dumps(r))
    return str(d)


class StatsTest(unittest.TestCase):
    def test_on_beat_clash_against_the_bar_chord(self) -> None:
        # 128 BPM: beat 0.46875 s. C7 bar: C# on beat 1 (clash), E on beat 2, D off the beat
        r = {"bpm": 128, "beats_per_bar": 4, "chords": ["C7"], "bars": 1, "completed_bars": 1, "run_completed": True,
             "scheduler_dispatch_deadline_miss_count": 0, "production": {"fallback_bar_count": 0},
             "played_bars": [{"bar": 0, "notes": [[73, 0.0, 0.2], [74, 0.25, 0.4], [76, 0.469, 0.6]]}]}
        s = run_stats(r)
        self.assertEqual((s["on_beat"], s["on_beat_clash"], s["notes"]), (2, 1, 3))

    def test_pctl(self) -> None:
        self.assertEqual(pctl([5, 1, 3, 2, 4], 0.5), 3)


class SetTest(unittest.TestCase):
    def test_full_set_ok_and_each_defect_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dirs = [write(Path(tmp), 3, t, sd) for t in PROGRESSIONS for sd in SEEDS]
            self.assertEqual(check_set(dirs, 3), [])
            self.assertTrue(check_set(dirs[:1], 3))                                  # missing runs
        for over in ({"run_completed": False}, {"completed_bars": 9}, {"bpm": 120}, {"seed": 1},
                     {"temperature": 0.6}, {"context_history": True}, {"start_budget_bars": 0.9},
                     {"checkpoint": "outputs/final_tatum/export/checkpoint_update518.pt"}, {"candidates": 1}):
            with tempfile.TemporaryDirectory() as tmp:
                dirs = [write(Path(tmp), 3, t, sd) for t in PROGRESSIONS for sd in SEEDS]
                write(Path(tmp), 3, "iiVIF", 42, **over)
                self.assertTrue(check_set(dirs, 3), over)


if __name__ == "__main__":
    unittest.main()
