"""Source tagging of played notes (scripts/comp_source_check.py)."""
from __future__ import annotations

import unittest

from scripts.comp_source_check import solo_metrics, tag_run

HALF = 60 / 128 * 2


def report():
    # 1 bar, 2 half blocks; block 1 is a fallback. G7: 3rd B (11), 7th F (5)
    return {"bpm": 128, "beats_per_bar": 4, "bars": 1,
            "production": {"fallback_bar_count": 1}, "scheduler_dispatch_deadline_miss_count": 0,
            "bars_detail": [{"source": "model", "generation_ms": 100}, {"source": "fallback_not_ready", "generation_ms": 500}],
            "comp_trace": [
                {"chord": "G7", "planned": [[53, 0.0, 0.2, 50], [59, 0.0, 0.2, 50], [71, 0.0, 0.2, 50]],
                 "emitted": [[53, 0.0, 0.2, 50], [59, 0.0, 0.2, 50]], "dropped": {"same_pitch_overlap": 1},
                 "outcome": "rendered"},
                {"chord": "Cmaj7", "planned": [[52, 0.0, 0.2, 50]], "emitted": [[52, 0.0, 0.2, 50]], "dropped": {},
                 "outcome": "rendered"}],
            "played_bars": [{"bar": 0, "notes": [
                [53, 0.0, 0.2], [59, 0.0, 0.2],                # comp, exact
                [71, 0.0, 0.3], [74, 0.3, 0.5],                # solo (71 is the B the comp dropped)
                [59, 0.32, 0.4],                               # same pitch as comp, not at its time: solo
                [60, HALF + 0.0, HALF + 0.2]]}]}               # in the fallback block


class TagTest(unittest.TestCase):
    def test_tagging(self) -> None:
        t = tag_run(report())
        self.assertEqual((t["emitted"], t["played_comp"], t["fallback_notes"], t["unknown"]), (2, 2, 1, 0))
        self.assertEqual(sorted(n[0] for n in t["solo"]), [59, 71, 74])
        self.assertEqual(t["dropped"], {"same_pitch_overlap": 1})            # fallback block's trace not counted
        self.assertEqual((t["align_ok"], t["align_n"]), (1, 1))               # comp-only: F + B in the first hit
        self.assertEqual((t["union_ok"], t["union_n"]), (1, 1))

    def test_trailing_rest_counts(self) -> None:
        m = solo_metrics([(72, 0.0, 0.2)], 1.0)
        self.assertAlmostEqual(m["rest_time"], 0.8)


if __name__ == "__main__":
    unittest.main()
