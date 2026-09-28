"""Half-bar scheduler blocks (docs/experiments/HALF_BAR_BLOCKS.md)."""
from __future__ import annotations

import tempfile
import unittest

from scripts.run_continuous_jazz import main, merge_half_bars


class MergeHalfBarsTest(unittest.TestCase):
    def test_second_half_is_offset(self) -> None:
        halves = [{"bar": 0, "notes": [[60, 0.0, 0.2]]}, {"bar": 1, "notes": [[62, 0.1, 0.3]]},
                  {"bar": 2, "notes": []}, {"bar": 3, "notes": [[64, 0.0, 0.5]]}]
        self.assertEqual(merge_half_bars(halves, half_seconds=1.0),
                         [{"bar": 0, "notes": [[60, 0.0, 0.2], [62, 1.1, 1.3]]},
                          {"bar": 1, "notes": [[64, 1.0, 1.5]]}])


class FlagsTest(unittest.TestCase):
    def test_needs_two_sub_block_chord_primer(self) -> None:
        for extra in ([], ["--chord-primer"], ["--chord-primer", "--chord-blocks-per-bar", "2",
                                                "--fallback-only"]):
            with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
                main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid",
                      "--half-bar-blocks", *extra])


if __name__ == "__main__":
    unittest.main()
