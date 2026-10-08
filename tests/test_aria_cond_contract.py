"""Chord condition timing contract (scripts/aria_cond_contract.py) on synthetic token lists."""
from __future__ import annotations

import unittest

from scripts.aria_cond_contract import chroma_per_position, prefix_times

CMAJ7, CM7 = {0, 4, 7, 11}, {0, 3, 7, 10}
PLAN = [(0, 2000, CMAJ7), (2000, 5000, CM7), (5000, 9000, CMAJ7)]
TOKENS = [("prefix", "instrument", "piano"), "<S>",
          ("piano", 64, 80), ("onset", 1750), ("dur", 200),
          ("piano", 63, 80), ("onset", 2250), ("dur", 200),     # first note after the 2 s change
          "<T>",
          ("piano", 64, 80), ("onset", 250), ("dur", 200), "<E>"]


def pcs(row):
    return {k for k, v in enumerate(row) if v}


class ConditionContractTest(unittest.TestCase):
    def test_prefix_times_follow_the_table(self) -> None:
        self.assertEqual(prefix_times(TOKENS), [0, 0, 0, 1750, 1750, 1750, 2250, 2250, 5000, 5000, 5250, 5250, 5250])

    def test_chord_change_is_seen_only_after_the_onset_that_crosses_it(self) -> None:
        rows = [pcs(r) for r in chroma_per_position(TOKENS, PLAN)]
        self.assertEqual(rows[5], CMAJ7)       # (piano, 63): its onset 2250 is not in the input yet
        self.assertEqual(rows[6], CM7)         # (onset, 2250) is in the input now
        self.assertEqual(rows[8], CMAJ7)       # <T>: 5 s boundary, next plan segment

    def test_no_position_reads_later_tokens(self) -> None:
        base = chroma_per_position(TOKENS, PLAN)
        for i in range(len(TOKENS) - 1):
            changed = TOKENS[: i + 1] + [("piano", 70, 80), ("onset", 4999), ("dur", 100), "<T>", "<T>"]
            self.assertEqual(chroma_per_position(changed, PLAN)[: i + 1], base[: i + 1])

    def test_outside_the_plan_is_no_chord(self) -> None:
        self.assertEqual(chroma_per_position([("onset", 100)], [(200, 300, CMAJ7)]), [[0.0] * 12])


if __name__ == "__main__":
    unittest.main()
