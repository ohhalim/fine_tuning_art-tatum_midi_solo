"""Late fetch + one bar of steady lead (docs/experiments/GENERATION_LEAD.md)."""
from __future__ import annotations

import threading
import time
import unittest
from threading import Event

from inference.realtime.continuous import BarBlockProducer
from inference.realtime.scheduler import (
    DEADLINE_POLICY_RECORD_AND_CONTINUE,
    MonotonicBarClock,
    OneBarMidiScheduler,
    build_deterministic_blocks,
)


class FakeTime:
    def __init__(self) -> None:
        self.now_ns = 0

    def clock_ns(self) -> int:
        return self.now_ns

    def wait_until(self, target_ns: int, stop: Event) -> None:
        if not stop.is_set():
            self.now_ns = max(self.now_ns, target_ns)


class Sink:
    def send(self, message) -> None:
        pass


class RecordingBlocks(dict):
    def __init__(self, blocks, fake):
        super().__init__(blocks)
        self.fake = fake
        self.calls = []

    def get(self, key, default=None):
        self.calls.append((key, self.fake.now_ns))
        return super().get(key, default)


def run(fetch_margin_ms, bars=4):
    fake = FakeTime()
    clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=1_000_000_000)
    blocks = RecordingBlocks(build_deterministic_blocks(clock=clock, bar_count=bars), fake)
    scheduler = OneBarMidiScheduler(sink=Sink(), clock=clock, clock_ns=fake.clock_ns,
                                    wait_until=fake.wait_until,
                                    deadline_policy=DEADLINE_POLICY_RECORD_AND_CONTINUE,
                                    fetch_margin_ms=fetch_margin_ms)
    result = scheduler.run(blocks=blocks, expected_bar_count=bars)
    return clock, blocks.calls, result


class LateFetchSchedulerTest(unittest.TestCase):
    def test_default_asks_for_the_next_bar_at_each_downbeat(self) -> None:
        clock, calls, result = run(None)
        self.assertTrue(result.run_completed)
        self.assertEqual(calls, [(0, 0)] + [(b + 1, clock.bar_start_ns(b)) for b in range(3)])

    def test_late_fetch_asks_for_each_bar_just_before_its_downbeat(self) -> None:
        clock, calls, result = run(50.0)
        self.assertTrue(result.run_completed)
        self.assertEqual(result.completed_bar_count, 4)
        self.assertEqual(calls, [(0, 0)] + [(b, clock.bar_start_ns(b) - 50_000_000) for b in (1, 2, 3)])

    def test_negative_margin_is_refused(self) -> None:
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)
        with self.assertRaises(ValueError):
            OneBarMidiScheduler(sink=Sink(), clock=clock, fetch_margin_ms=-1.0)


class SteadyLeadProducerTest(unittest.TestCase):
    def _producer(self, steady):
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)
        fallbacks = build_deterministic_blocks(clock=clock, bar_count=6)
        built = []

        def build(bar_index, events):
            built.append(bar_index)
            return fallbacks[bar_index]

        producer = BarBlockProducer(bar_count=6, fallback_blocks=fallbacks, build_block=build,
                                    clock=clock, max_lead_bars=2, steady_lead_bars=steady)
        return producer, built

    def _settle(self, built, n, timeout=2.0):
        deadline = time.monotonic() + timeout
        while len(built) < n and time.monotonic() < deadline:
            time.sleep(0.005)
        time.sleep(0.05)   # give an over-eager producer the chance to overshoot

    def test_warmup_builds_two_bars_then_one_ahead(self) -> None:
        producer, built = self._producer(steady=1)
        with producer:
            self._settle(built, 2)
            self.assertEqual(built, [0, 1])           # warm-up: lead 2 before playback
            producer.get(0)
            self._settle(built, 2)
            self.assertEqual(built, [0, 1])           # steady lead 1: bar 2 waits for bar 1's fetch
            producer.get(1)
            self._settle(built, 3)
            self.assertEqual(built, [0, 1, 2])

    def test_default_keeps_lead_two(self) -> None:
        producer, built = self._producer(steady=None)
        with producer:
            self._settle(built, 2)
            producer.get(0)
            self._settle(built, 3)
            self.assertEqual(built, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
