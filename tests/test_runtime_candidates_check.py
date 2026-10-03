"""--candidates runtime check (scripts/runtime_candidates_check.py)."""
from __future__ import annotations

import unittest

from scripts.runtime_candidates_check import pctl, run_stats


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


if __name__ == "__main__":
    unittest.main()
