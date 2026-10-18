"""Failure gates of the #1760 adapter run: stop before the optimizer step and exit non-zero (no model)."""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import cond_adapter_train as ct  # noqa: E402


class GateTest(unittest.TestCase):
    def test_step_guard_stops_on_nonfinite_and_wall_limit(self):
        self.assertIsNone(ct.step_guard(2.5, 10.0))
        self.assertIn("nonfinite", ct.step_guard(math.nan, 10.0))
        self.assertIn("nonfinite", ct.step_guard(math.inf, 10.0))
        self.assertIn("wall limit", ct.step_guard(2.5, ct.WALL_LIMIT_S + 1))

    def test_stop_saves_report_and_exits_nonzero(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "r", "report.json")
            with self.assertRaises(SystemExit) as cm:
                ct.stop({"checks": {"step0_arms_identical": False}}, path, "step 0 arms differ")
            self.assertEqual(cm.exception.code, 1)
            saved = json.load(open(path))
            self.assertEqual(saved["stopped_at"], "step 0 arms differ")
            self.assertNotIn("eval_final", saved)                       # nothing ran after the stop


if __name__ == "__main__":
    unittest.main()
