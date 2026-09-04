from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from threading import Event
from unittest import mock

from mido import Message

from inference.realtime.scheduler import (
    DEFAULT_SPIN_WINDOW_MS,
    MonotonicBarClock,
    OneBarMidiScheduler,
    build_deterministic_blocks,
    deterministic_events_per_bar,
    timing_histogram,
)
from scripts import run_internal_scheduler_probe as scheduler_probe


EVENTS_PER_BAR = deterministic_events_per_bar(4)


class FakeMonotonicTime:
    def __init__(self, now_ns: int = 0) -> None:
        self.now_ns = now_ns

    def clock_ns(self) -> int:
        return self.now_ns

    def wait_until(self, target_ns: int, stop: Event) -> None:
        if not stop.is_set():
            self.now_ns = max(self.now_ns, target_ns)


class RecordingSink:
    def __init__(self, fail_after: int | None = None) -> None:
        self.messages: list[Message] = []
        self.fail_after = fail_after

    def send(self, message: Message) -> None:
        if self.fail_after is not None and len(self.messages) >= self.fail_after:
            raise RuntimeError("send failed")
        self.messages.append(message.copy())


class FakeVirtualOutput:
    def __init__(self, backend: "FakeCoreMidiBackend", name: str) -> None:
        self.backend = backend
        self.name = name
        self.reset_called = False

    def __enter__(self) -> "FakeVirtualOutput":
        self.backend.source_names.add(self.name)
        return self

    def __exit__(self, *args: object) -> None:
        self.backend.source_names.remove(self.name)

    def send(self, message: Message) -> None:
        if self.backend.fail_send:
            raise RuntimeError("virtual send failed")
        callback = self.backend.callbacks.get(self.name)
        if callback is not None:
            callback(message.copy())

    def reset(self) -> None:
        self.reset_called = True


class FakeVirtualInput:
    def __init__(self, backend: "FakeCoreMidiBackend", name: str, callback: object) -> None:
        self.backend = backend
        self.name = name
        self.callback = callback

    def __enter__(self) -> "FakeVirtualInput":
        self.backend.callbacks[self.name] = self.callback
        return self

    def __exit__(self, *args: object) -> None:
        del self.backend.callbacks[self.name]


class FakeCoreMidiBackend:
    def __init__(self, *, fail_send: bool = False) -> None:
        self.source_names: set[str] = set()
        self.callbacks: dict[str, object] = {}
        self.outputs: list[FakeVirtualOutput] = []
        self.fail_send = fail_send

    def open_output(self, name: str, *, virtual: bool = False) -> FakeVirtualOutput:
        if not virtual:
            raise AssertionError("virtual output required")
        output = FakeVirtualOutput(self, name)
        self.outputs.append(output)
        return output

    def open_input(self, name: str, *, callback: object) -> FakeVirtualInput:
        return FakeVirtualInput(self, name, callback)

    def get_input_names(self) -> list[str]:
        return sorted(self.source_names)


class InternalSchedulerTest(unittest.TestCase):
    def test_monotonic_clock_maps_beats_and_bars_without_accumulated_rounding(self) -> None:
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)

        self.assertEqual(500_000_000.0, clock.beat_duration_ns)
        self.assertEqual(2_000_000_000.0, clock.bar_duration_ns)
        self.assertEqual(1_000_000_000, clock.bar_start_ns(0))
        self.assertEqual(3_000_000_000, clock.bar_start_ns(1))
        self.assertEqual(5_000_000_000, clock.bar_start_ns(2))

    def test_deterministic_blocks_have_balanced_normal_note_events(self) -> None:
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=3)

        self.assertEqual([0, 1, 2], sorted(blocks))
        self.assertTrue(
            all(len(block.events) == EVENTS_PER_BAR for block in blocks.values())
        )
        self.assertTrue(all(block.events[0].is_bar_start for block in blocks.values()))
        self.assertEqual(clock.bar_start_ns(2), blocks[2].target_start_ns)

    def test_deterministic_blocks_reach_the_spin_contention_regime(self) -> None:
        """The fixture must keep same-target-time events and sub-spin-window gaps.

        A sparse, strictly sequential fixture cannot exercise the regime where the
        busy-spin overlaps a capture callback, so losing these properties would make
        the spin window look safe without testing it.
        """
        for bpm in (90.0, 120.0, 128.0, 160.0):
            with self.subTest(bpm=bpm):
                clock = MonotonicBarClock(bpm=bpm, beats_per_bar=4, start_ns=1_000_000_000)
                blocks = build_deterministic_blocks(clock=clock, bar_count=4)
                targets = [
                    event.target_ns
                    for bar_index in sorted(blocks)
                    for event in blocks[bar_index].events
                ]
                simultaneous = sum(
                    1 for index in range(len(targets) - 1)
                    if targets[index + 1] == targets[index]
                )
                gaps_ms = [
                    (targets[index + 1] - targets[index]) / 1_000_000
                    for index in range(len(targets) - 1)
                    if targets[index + 1] != targets[index]
                ]

                self.assertGreater(simultaneous, 0)
                self.assertLess(min(gaps_ms), DEFAULT_SPIN_WINDOW_MS)
                self.assertTrue(
                    all(
                        sum(1 for event in blocks[bar_index].events if event.is_bar_start) == 1
                        for bar_index in blocks
                    )
                )

    def test_duration_rounding_never_shortens_requested_smoke(self) -> None:
        self.assertEqual(
            3,
            scheduler_probe._bar_count_for_duration(
                bpm=128.0,
                beats_per_bar=4,
                requested_duration_seconds=4.0,
            ),
        )

    def test_one_bar_scheduler_enqueues_next_block_and_completes(self) -> None:
        fake_time = FakeMonotonicTime()
        sink = RecordingSink()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=3)
        scheduler = OneBarMidiScheduler(
            sink=sink,
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=fake_time.wait_until,
        )

        result = scheduler.run(blocks=blocks, expected_bar_count=3)

        self.assertTrue(result.run_completed)
        self.assertEqual(3, result.started_bar_count)
        self.assertEqual(3, result.completed_bar_count)
        self.assertEqual(3, result.enqueued_block_count)
        self.assertEqual(0, result.queue_underrun_count)
        self.assertEqual(1, result.queue_depth_max)
        self.assertEqual(3 * EVENTS_PER_BAR, len(result.records))
        self.assertEqual(2, len(result.enqueue_lead_time_ns))
        self.assertEqual(clock.bar_start_ns(3), fake_time.now_ns)

    def test_missing_block_counts_one_bar_underrun_without_treating_rest_as_empty_queue(self) -> None:
        fake_time = FakeMonotonicTime()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=3)
        del blocks[1]
        scheduler = OneBarMidiScheduler(
            sink=RecordingSink(),
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=fake_time.wait_until,
        )

        result = scheduler.run(blocks=blocks, expected_bar_count=3)

        self.assertFalse(result.run_completed)
        self.assertEqual(1, result.queue_underrun_count)
        self.assertEqual(1, result.started_bar_count)
        self.assertEqual(1, result.completed_bar_count)
        self.assertEqual(2, result.enqueued_block_count)
        self.assertEqual(EVENTS_PER_BAR, len(result.records))
        self.assertEqual("queue_underrun", result.watchdog_trigger_reason)

    def test_send_failure_stops_run_and_records_failure(self) -> None:
        fake_time = FakeMonotonicTime()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=2)
        scheduler = OneBarMidiScheduler(
            sink=RecordingSink(fail_after=1),
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=fake_time.wait_until,
        )

        result = scheduler.run(blocks=blocks, expected_bar_count=2)

        self.assertFalse(result.run_completed)
        self.assertEqual(1, result.send_failure_count)
        self.assertEqual(1, result.started_bar_count)
        self.assertEqual(0, result.completed_bar_count)
        self.assertEqual(1, len(result.records))
        self.assertEqual("send_failure", result.watchdog_trigger_reason)

    def test_dispatch_deadline_miss_triggers_watchdog_before_late_send(self) -> None:
        fake_time = FakeMonotonicTime()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=1)

        def late_wait(target_ns: int, stop: Event) -> None:
            if not stop.is_set():
                fake_time.now_ns = target_ns + 21_000_000

        scheduler = OneBarMidiScheduler(
            sink=RecordingSink(),
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=late_wait,
            deadline_threshold_ms=20.0,
        )

        result = scheduler.run(blocks=blocks, expected_bar_count=1)

        self.assertFalse(result.run_completed)
        self.assertEqual(1, result.scheduler_dispatch_deadline_miss_count)
        self.assertEqual(1, result.started_bar_count)
        self.assertEqual(0, result.completed_bar_count)
        self.assertEqual("dispatch_deadline_miss", result.watchdog_trigger_reason)
        self.assertEqual(0, len(result.records))
        self.assertEqual(1, len(result.scheduler_dispatch_deadline_misses))
        miss = result.scheduler_dispatch_deadline_misses[0]
        self.assertEqual(21_000_000, miss.lateness_ns)
        self.assertEqual(0, miss.sequence_index)
        self.assertEqual(0, miss.bar_index)
        self.assertTrue(miss.is_bar_start)

    def test_dispatch_deadline_miss_report_round_trips_json(self) -> None:
        fake_time = FakeMonotonicTime()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=1)

        def late_wait(target_ns: int, stop: Event) -> None:
            if not stop.is_set():
                fake_time.now_ns = target_ns + 21_000_000

        scheduler = OneBarMidiScheduler(
            sink=RecordingSink(),
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=late_wait,
            deadline_threshold_ms=20.0,
        )
        run_result = scheduler.run(blocks=blocks, expected_bar_count=1)
        expected_messages = [event.message.copy() for event in blocks[0].events]
        report = scheduler_probe.build_report(
            bpm=120.0,
            beats_per_bar=4,
            requested_duration_seconds=2.0,
            scheduled_duration_seconds=2.0,
            start_delay_seconds=0.25,
            wall_clock_seconds=0.25,
            expected_messages=expected_messages,
            run_result=run_result,
            captured_messages=[],
            captured_ns=[],
            safe_reset_sent=True,
            deadline_threshold_ms=20.0,
            spin_window_ms=15.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            report_path = Path(directory) / "report.json"
            scheduler_probe.write_report(report_path, report)
            payload = json.loads(report_path.read_text(encoding="utf-8"))

        self.assertIsNone(payload["capture_deadline_miss_count"])
        self.assertFalse(payload["passed_scheduler_smoke"])
        self.assertEqual(21.0, payload["scheduler_dispatch_deadline_miss_lateness_ms"]["maximum"])
        self.assertEqual(21_000_000, payload["scheduler_dispatch_deadline_misses"][0]["lateness_ns"])
        self.assertEqual(0, payload["scheduler_dispatch_deadline_misses"][0]["sequence_index"])

    def test_histogram_uses_cumulative_latency_buckets(self) -> None:
        histogram = timing_histogram([0, 1_000_000, 3_000_000, 25_000_000, 101_000_000])

        self.assertEqual(2, histogram["le_1ms"])
        self.assertEqual(3, histogram["le_5ms"])
        self.assertEqual(4, histogram["le_50ms"])
        self.assertEqual(1, histogram["gt_100ms"])

    def test_complete_600_second_result_passes_r1_gate_and_exit_policy(self) -> None:
        fake_time = FakeMonotonicTime()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=300)
        scheduler = OneBarMidiScheduler(
            sink=RecordingSink(),
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=fake_time.wait_until,
        )
        run_result = scheduler.run(blocks=blocks, expected_bar_count=300)
        expected_messages = [
            event.message.copy()
            for bar_index in sorted(blocks)
            for event in blocks[bar_index].events
        ]
        captured_ns = [record.target_ns + 1_000_000 for record in run_result.records]

        report = scheduler_probe.build_report(
            bpm=120.0,
            beats_per_bar=4,
            requested_duration_seconds=600.0,
            scheduled_duration_seconds=600.0,
            start_delay_seconds=0.25,
            wall_clock_seconds=600.25,
            expected_messages=expected_messages,
            run_result=run_result,
            captured_messages=[record.message.copy() for record in run_result.records],
            captured_ns=captured_ns,
            safe_reset_sent=True,
            deadline_threshold_ms=20.0,
            spin_window_ms=15.0,
        )

        self.assertEqual(300, report.bar_start_capture_error_ms.sample_count)
        self.assertEqual(0, report.capture_deadline_miss_count)
        self.assertEqual(300, report.started_bar_count)
        self.assertEqual(300, report.completed_bar_count)
        self.assertTrue(report.wall_clock_soak_completed)
        self.assertTrue(report.passed_scheduler_smoke)
        self.assertTrue(report.passed_r1_internal_scheduler_gate)
        self.assertEqual(0, scheduler_probe.exit_code_for_report(report, require_r1_gate=True))

    def test_missing_capture_cannot_report_timing_or_pass_scheduler_gate(self) -> None:
        fake_time = FakeMonotonicTime()
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000_000_000)
        blocks = build_deterministic_blocks(clock=clock, bar_count=1)
        scheduler = OneBarMidiScheduler(
            sink=RecordingSink(),
            clock=clock,
            clock_ns=fake_time.clock_ns,
            wait_until=fake_time.wait_until,
        )
        run_result = scheduler.run(blocks=blocks, expected_bar_count=1)
        expected_messages = [event.message.copy() for event in blocks[0].events]

        report = scheduler_probe.build_report(
            bpm=120.0,
            beats_per_bar=4,
            requested_duration_seconds=2.0,
            scheduled_duration_seconds=2.0,
            start_delay_seconds=0.25,
            wall_clock_seconds=2.25,
            expected_messages=expected_messages,
            run_result=run_result,
            captured_messages=[record.message.copy() for record in run_result.records[:-1]],
            captured_ns=[record.target_ns + 1_000_000 for record in run_result.records[:-1]],
            safe_reset_sent=True,
            deadline_threshold_ms=20.0,
            spin_window_ms=15.0,
        )

        self.assertEqual(1, report.event_loss_count)
        self.assertEqual(0, report.sender_to_capture_timing_error_ms.sample_count)
        self.assertIsNone(report.sender_to_capture_timing_error_ms.p99)
        self.assertIsNone(report.capture_deadline_miss_count)
        self.assertFalse(report.passed_scheduler_smoke)
        self.assertFalse(report.passed_r1_internal_scheduler_gate)

    def test_fake_coremidi_runner_passes_short_smoke_and_resets(self) -> None:
        backend = FakeCoreMidiBackend()
        with (
            mock.patch.object(scheduler_probe.mido, "open_output", backend.open_output),
            mock.patch.object(scheduler_probe.mido, "open_input", backend.open_input),
            mock.patch.object(scheduler_probe.mido, "get_input_names", backend.get_input_names),
        ):
            report = scheduler_probe.run_internal_scheduler_probe(
                run_id="unit",
                bpm=6000.0,
                requested_duration_seconds=0.04,
                start_delay_seconds=0.0,
                drain_timeout_seconds=0.0,
            )

        self.assertEqual(1, report.expected_bar_count)
        self.assertEqual(EVENTS_PER_BAR, report.expected_event_count)
        self.assertEqual(EVENTS_PER_BAR, report.events_per_bar)
        self.assertEqual("dense_chord_sub_spin_v1", report.fixture_id)
        self.assertEqual(4.0, report.dense_off_gap_ms)
        self.assertEqual(EVENTS_PER_BAR, report.captured_event_count)
        self.assertTrue(report.passed_scheduler_smoke)
        self.assertFalse(report.passed_r1_internal_scheduler_gate)
        self.assertTrue(report.safe_reset_sent)
        self.assertTrue(backend.outputs[0].reset_called)

    def test_fake_coremidi_send_failure_fails_smoke_and_resets(self) -> None:
        backend = FakeCoreMidiBackend(fail_send=True)
        with (
            mock.patch.object(scheduler_probe.mido, "open_output", backend.open_output),
            mock.patch.object(scheduler_probe.mido, "open_input", backend.open_input),
            mock.patch.object(scheduler_probe.mido, "get_input_names", backend.get_input_names),
        ):
            report = scheduler_probe.run_internal_scheduler_probe(
                run_id="failure",
                bpm=6000.0,
                requested_duration_seconds=0.04,
                start_delay_seconds=0.0,
                drain_timeout_seconds=0.0,
            )

        self.assertEqual(1, report.send_failure_count)
        self.assertGreater(report.event_loss_count, 0)
        self.assertFalse(report.run_completed)
        self.assertFalse(report.passed_scheduler_smoke)
        self.assertTrue(report.safe_reset_sent)
        self.assertTrue(backend.outputs[0].reset_called)

    def test_fake_coremidi_missing_block_triggers_watchdog_and_resets(self) -> None:
        backend = FakeCoreMidiBackend()
        with (
            mock.patch.object(scheduler_probe.mido, "open_output", backend.open_output),
            mock.patch.object(scheduler_probe.mido, "open_input", backend.open_input),
            mock.patch.object(scheduler_probe.mido, "get_input_names", backend.get_input_names),
        ):
            report = scheduler_probe.run_internal_scheduler_probe(
                run_id="missing-block",
                bpm=6000.0,
                requested_duration_seconds=0.08,
                start_delay_seconds=0.0,
                drop_block_index=1,
            )

        self.assertEqual(1, report.queue_underrun_count)
        self.assertEqual("queue_underrun", report.watchdog_trigger_reason)
        self.assertFalse(report.run_completed)
        self.assertFalse(report.passed_scheduler_smoke)
        self.assertTrue(report.safe_reset_sent)
        self.assertTrue(backend.outputs[0].reset_called)


if __name__ == "__main__":
    unittest.main()
