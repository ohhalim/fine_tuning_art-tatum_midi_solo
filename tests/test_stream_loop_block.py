"""Regression for fill_block in the MLX stream loop (issue #1743, Astra review of #1744). Fake sampler, fake clock."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aria_mlx_stream_loop import fill_block  # noqa: E402


class Fake:
    def __init__(self, steps):
        self.steps, self.i, self.ready_calls, self.t = steps, 0, [], 0.0

    def coverage(self):
        return self.steps[self.i]

    def sample(self):
        self.i += 1
        self.t += 1.0                    # each sample costs one fake second

    def on_ready(self):
        self.ready_calls.append(self.i)

    def clock(self):
        return self.t


class FillBlockTest(unittest.TestCase):
    def test_horizon_reached_by_the_last_allowed_sample_counts(self):
        f = Fake([0, 100, 200, 1000])    # the 3rd (= cap) sample reaches h = 1000
        r = fill_block(f.sample, f.coverage, f.on_ready, 1000, 2000, 3, f.clock)
        self.assertTrue(r["reached"])
        self.assertEqual(f.ready_calls, [3])
        self.assertEqual(r["tokens"], 3)

    def test_ready_happens_at_the_horizon_before_speculation(self):
        f = Fake([0, 1000, 1500, 2000])
        r = fill_block(f.sample, f.coverage, f.on_ready, 1000, 2000, 10, f.clock)
        self.assertEqual(f.ready_calls, [1])                  # commit + schedule right after the first sample
        self.assertEqual((r["ready_s"], r["block_s"], r["tokens"]), (1.0, 3.0, 3))

    def test_already_covered_block_is_ready_at_once(self):
        f = Fake([1200, 2500])           # past h already, one more sample for the speculative target
        r = fill_block(f.sample, f.coverage, f.on_ready, 1000, 2000, 10, f.clock)
        self.assertEqual((f.ready_calls, r["ready_s"], r["tokens"]), ([0], 0.0, 1))

    def test_horizon_not_reached_still_commits_once_and_reports_it(self):
        f = Fake([0, 100, 200, 300])
        r = fill_block(f.sample, f.coverage, f.on_ready, 1000, 2000, 3, f.clock)
        self.assertFalse(r["reached"])
        self.assertEqual(f.ready_calls, [3])


if __name__ == "__main__":
    unittest.main()
