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


class FlagTest(unittest.TestCase):
    def test_pattern_cache_needs_the_sub_block_path(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid", "--pattern-cache"])


if __name__ == "__main__":
    unittest.main()
