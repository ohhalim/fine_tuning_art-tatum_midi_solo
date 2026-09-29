"""The input buffer can hand back everything it holds (docs/experiments/SESSION_CLOCK_PROBE.md)."""
from __future__ import annotations

import unittest

from mido import Message

from inference.realtime.continuous import MidiInputSnapshotBuffer


class AllEventsTest(unittest.TestCase):
    def test_all_events_ignores_the_window_and_keeps_the_cap(self) -> None:
        buf = MidiInputSnapshotBuffer(window_seconds=1.0, max_events=3)
        for i in range(5):
            buf.handle(Message("note_on", note=60 + i, velocity=90), received_ns=i * 10_000_000_000)
        self.assertEqual([e.message.note for e in buf.all_events()], [62, 63, 64])   # cap 3, oldest dropped
        self.assertEqual(len(buf.snapshot(now_ns=40_000_000_000)), 1)                # window is 1 s


if __name__ == "__main__":
    unittest.main()
