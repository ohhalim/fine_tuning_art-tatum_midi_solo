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

    def test_boundary_note_off_does_not_delay_the_fetch(self) -> None:
        # Astra M1: a note held to the bar line used to push the next fetch to
        # the downbeat. 120 BPM, bar 1 starts at 3.0 s -> fetch due at 2.95 s.
        from mido import Message
        from inference.realtime.scheduler import ScheduledMidiBlock, ScheduledMidiEvent
        fake = FakeTime()
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=1_000_000_000)

        def held_block(b):
            start, end = clock.bar_start_ns(b), clock.bar_start_ns(b + 1)
            return ScheduledMidiBlock(bar_index=b, target_start_ns=start, events=(
                ScheduledMidiEvent(sequence_index=0, bar_index=b, target_ns=start,
                                   message=Message("note_on", note=60, velocity=80), is_bar_start=True),
                ScheduledMidiEvent(sequence_index=1, bar_index=b, target_ns=end,
                                   message=Message("note_off", note=60, velocity=0)),
            ))

        blocks = RecordingBlocks({b: held_block(b) for b in range(3)}, fake)
        sent = []

        class OrderSink:
            def send(self, message):
                sent.append((fake.now_ns, message.type))

        scheduler = OneBarMidiScheduler(sink=OrderSink(), clock=clock, clock_ns=fake.clock_ns,
                                        wait_until=fake.wait_until,
                                        deadline_policy=DEADLINE_POLICY_RECORD_AND_CONTINUE,
                                        fetch_margin_ms=50.0)
        result = scheduler.run(blocks=blocks, expected_bar_count=3)
        self.assertTrue(result.run_completed)
        self.assertEqual(blocks.calls, [(0, 0), (1, 2_950_000_000), (2, 4_950_000_000)])
        # the boundary note-off still goes out before the next bar's note-on
        self.assertEqual([t for _, t in sent], ["note_on", "note_off"] * 3)

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


class StartBudgetTest(unittest.TestCase):
    def test_steady_bars_wait_for_their_start_time_warmup_does_not(self) -> None:
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)
        fallbacks = build_deterministic_blocks(clock=clock, bar_count=4)
        now = {"ns": 0}
        started = []

        def build(bar_index, events):
            started.append((bar_index, now["ns"]))
            return fallbacks[bar_index]

        producer = BarBlockProducer(bar_count=4, fallback_blocks=fallbacks, build_block=build,
                                    clock=clock, clock_ns=lambda: now["ns"], max_lead_bars=2,
                                    steady_lead_bars=1,
                                    start_not_before_ns=lambda b: 1_000_000_000 * b)
        with producer:
            deadline = time.monotonic() + 2
            while len(started) < 2 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual([b for b, _ in started], [0, 1])   # warm-up is not held
            producer.get(0)
            producer.get(1)
            time.sleep(0.15)
            self.assertEqual(len(started), 2)                   # bar 2 held until t = 2 s
            now["ns"] = 2_000_000_000
            deadline = time.monotonic() + 2
            while len(started) < 3 and time.monotonic() < deadline:
                time.sleep(0.005)
            self.assertEqual(started[2], (2, 2_000_000_000))

    def test_cli_needs_late_fetch_and_a_sane_fraction(self) -> None:
        import tempfile
        from scripts.run_continuous_jazz import main
        for extra in (["--fetch-margin-ms", "off", "--start-budget-bars", "0.5"],
                      ["--start-budget-bars", "1.5"]):
            with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
                main(["--output-dir", d, "--fallback-only", *extra])


class AdaptiveBudgetTest(unittest.TestCase):
    def test_budget_widens_after_slow_generations(self) -> None:
        import scripts.run_continuous_jazz as module
        captured = {}
        real = module.BarBlockProducer

        class Capture(real):
            def __init__(self, **kw):
                super().__init__(**kw)
                captured["producer"] = self
                captured["f"] = kw["start_not_before_ns"]

        from unittest import mock
        from inference.realtime.continuous import BarProductionRecord
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)   # 2 s bars
        with mock.patch.object(module, "BarBlockProducer", Capture), \
                mock.patch.object(module.OneBarMidiScheduler, "run", side_effect=RuntimeError("stop")):
            class Port:
                def reset(self): pass
                def panic(self): pass
            with self.assertRaises(RuntimeError):
                module.run_session(port=Port(), bars=4, bpm=120, chords=["C"], seed=0,
                                   generate=None, start_delay_seconds=0.0, clock=clock,
                                   fetch_margin_ms=50.0, start_budget_bars=0.5,
                                   adaptive_start_safety=1.5)
        producer, f = captured["producer"], captured["f"]
        fixed = clock.bar_start_ns(3) - 50_000_000 - 1_000_000_000
        self.assertEqual(f(3), fixed)                          # no history: fixed budget
        with producer._cv:
            producer._records[1] = BarProductionRecord(bar_index=1, source="model", used_fallback=False,
                                                       requested_ns=0, completed_ns=1_200_000_000)
        self.assertEqual(f(3), clock.bar_start_ns(3) - 50_000_000 - 1_800_000_000)   # 1.5 x 1.2 s
        with producer._cv:
            producer._records[1] = BarProductionRecord(bar_index=1, source="model", used_fallback=False,
                                                       requested_ns=0, completed_ns=5_000_000_000)
        self.assertEqual(f(3), clock.bar_start_ns(3) - 50_000_000 - 2_000_000_000)   # capped at one block


class RuntimeDefaultTest(unittest.TestCase):
    def test_start_budget_defaults_to_half_a_bar_only_with_late_fetch(self) -> None:
        import tempfile
        from unittest import mock
        import scripts.run_continuous_jazz as module
        seen = {}

        def fake_run_session(**kw):
            seen.update(kw)
            raise SystemExit(0)

        for extra, want in (([], 0.5), (["--start-budget-bars", "off"], None),
                            (["--fetch-margin-ms", "off"], None),
                            (["--start-budget-bars", "0.3"], 0.3)):
            seen.clear()
            with tempfile.TemporaryDirectory() as d, \
                    mock.patch.object(module, "run_session", fake_run_session), \
                    mock.patch("mido.open_output"), self.assertRaises(SystemExit):
                module.main(["--output-dir", d, "--fallback-only", *extra])
            self.assertEqual(seen.get("start_budget_bars"), want, extra)

    def test_late_fetch_is_the_default_with_an_off_switch(self) -> None:
        from pathlib import Path
        from scripts.run_continuous_jazz import _fetch_margin
        text = (Path(__file__).resolve().parents[1] / "scripts" / "run_continuous_jazz.py").read_text()
        self.assertIn('"--fetch-margin-ms", type=_fetch_margin, default=50.0', text)
        self.assertIsNone(_fetch_margin("off"))
        self.assertEqual(_fetch_margin("20"), 20.0)


if __name__ == "__main__":
    unittest.main()
