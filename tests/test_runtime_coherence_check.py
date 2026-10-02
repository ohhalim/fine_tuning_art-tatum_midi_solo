"""Runtime coherence verdict and --pattern-cache flag."""
from __future__ import annotations

import tempfile
import unittest

from scripts.run_continuous_jazz import main
from scripts.runtime_coherence_check import judge


class JudgeTest(unittest.TestCase):
    def test_rules(self) -> None:
        good = {"motif_reuse": 0.04, "chord_tone_diff_mean": -0.01, "fallback_total": 0}
        self.assertTrue(judge(good, {}, "tatum")["pass"])
        self.assertFalse(judge({**good, "motif_reuse": 0.03}, {}, "tatum")["pass"])
        self.assertFalse(judge({**good, "chord_tone_diff_mean": -0.05}, {}, "tatum")["pass"])
        self.assertFalse(judge({**good, "fallback_total": 1}, {}, "tatum")["pass"])


class SameNoteTest(unittest.TestCase):
    def test_same_note_hammering_does_not_count_as_reuse(self) -> None:
        from scripts.coherence_metrics import motif_reuse, same_note_share
        line = [(i * 0.2, 60) for i in range(20)]
        self.assertGreater(motif_reuse(line)[0], 0)                       # original definition counts it
        self.assertEqual(motif_reuse(line, skip_same_note=True), (0, 0))
        self.assertEqual(same_note_share(line), 1.0)

    def test_guarded_verdict_uses_the_clean_reuse_and_caps_same_note_share(self) -> None:
        good = {"motif_reuse": 0.38, "motif_reuse_no_same_note": 0.088, "same_note_share": 0.34,
                "chord_tone_diff_mean": 0.1, "fallback_total": 0}
        self.assertTrue(judge(good, {}, "mehldau")["pass"])                       # old rule: fooled
        v = judge(good, {}, "mehldau", same_note_guard=True)
        self.assertFalse(v["pass"])
        self.assertFalse(v["same_note_guard"])
        self.assertTrue(judge({**good, "same_note_share": 0.01}, {}, "mehldau", same_note_guard=True)["pass"])


class FlagTest(unittest.TestCase):
    def test_pattern_cache_needs_the_sub_block_path(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid", "--pattern-cache"])


if __name__ == "__main__":
    unittest.main()
