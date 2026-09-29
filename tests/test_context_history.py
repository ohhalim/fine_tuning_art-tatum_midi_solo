"""Multi-block history carried into the primer (docs/experiments/CONTEXT_HISTORY.md)."""
from __future__ import annotations

import tempfile
import unittest

from scripts.run_continuous_jazz import main, pad_to_duration


class PadTest(unittest.TestCase):
    def test_short_block_is_padded_to_its_window(self) -> None:
        out = pad_to_duration([60, 256 + 49, 188], 0.9375)          # 500 ms used
        self.assertEqual(out[:3], [60, 305, 188])
        self.assertEqual(sum((t - 255) * 10 for t in out if 256 <= t <= 355), 930)

    def test_full_or_overfull_block_is_left_alone(self) -> None:
        self.assertEqual(pad_to_duration([355, 355], 1.5), [355, 355])


class FlagTest(unittest.TestCase):
    def test_history_needs_carry_tokens(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid",
                  "--chord-primer", "--context-history"])


if __name__ == "__main__":
    unittest.main()
