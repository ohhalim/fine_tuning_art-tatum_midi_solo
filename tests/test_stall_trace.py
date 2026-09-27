from __future__ import annotations

import gc
import time
import unittest
from types import SimpleNamespace

from inference.realtime.stall_trace import StallTracer, classify_late_events, summarize_trace

MS = 1_000_000


def rec(target_ms, started_ms, bar=0):
    return SimpleNamespace(bar_index=bar, target_ns=target_ms * MS,
                           dispatch_started_ns=started_ms * MS)


class ClassifyTest(unittest.TestCase):
    def test_priority_and_threshold(self) -> None:
        records = [rec(0, 5),          # 5 ms late: below threshold, ignored
                   rec(100, 130),      # overlaps GC >= 1 ms
                   rec(200, 230),      # overlaps external gap only
                   rec(300, 330),      # overlaps in-process gap only
                   rec(400, 430),      # nothing
                   rec(500, 530)]      # GC too short to count, in-process gap -> process
        out = classify_late_events(
            records, threshold_ns=10 * MS,
            gc_intervals=[(110 * MS, 125 * MS, 2), (505 * MS, 505 * MS + 100, 0)],
            inproc_gaps=[(115 * MS, 128 * MS), (310 * MS, 325 * MS), (510 * MS, 520 * MS)],
            external_gaps=[(205 * MS, 225 * MS)],
            producer_intervals=[(90 * MS, 150 * MS)])
        self.assertEqual([o["category"] for o in out],
                         ["gc", "system", "process", "unexplained", "process"])
        self.assertEqual([o["producer_busy"] for o in out], [True, False, False, False, False])
        tracer = StallTracer(started_ns=0, stopped_ns=1000 * MS,
                             external_gaps=[(205 * MS, 225 * MS)])
        summary = summarize_trace(tracer, out)
        self.assertAlmostEqual(summary["coverage"]["external_gaps"], 0.02)
        self.assertEqual(summary["late_by_category"],
                         {"gc": 1, "system": 1, "process": 2, "unexplained": 1})


class TracerSmokeTest(unittest.TestCase):
    def test_records_gc_and_cleans_up(self) -> None:
        tracer = StallTracer().start()
        try:
            gc.collect()
            time.sleep(0.05)
        finally:
            tracer.stop()
        self.assertGreaterEqual(len(tracer.gc_intervals), 1)
        self.assertNotIn(tracer._on_gc, gc.callbacks)
        self.assertIsNotNone(tracer._proc.returncode)


if __name__ == "__main__":
    unittest.main()
