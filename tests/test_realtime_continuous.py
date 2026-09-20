from __future__ import annotations

import time
import unittest
from threading import Event

from mido import Message

from inference.realtime.continuous import (
    SOURCE_FALLBACK_ERROR,
    SOURCE_FALLBACK_NOT_READY,
    SOURCE_MODEL,
    BarBlockProducer,
    MidiInputSnapshotBuffer,
    TimedInputMessage,
    input_events_to_notes,
    summarize_production,
)
from inference.realtime.scheduler import (
    MonotonicBarClock,
    OneBarMidiScheduler,
    ScheduledMidiBlock,
    ScheduledMidiEvent,
)

BPM = 128.0
BARS = 4


def make_clock(start_ns: int = 0) -> MonotonicBarClock:
    return MonotonicBarClock(bpm=BPM, beats_per_bar=4, start_ns=start_ns)


def make_block(clock: MonotonicBarClock, bar_index: int, *, pitch: int, adapter: str):
    start = clock.bar_start_ns(bar_index)
    end = clock.bar_start_ns(bar_index + 1)
    return ScheduledMidiBlock(
        bar_index=bar_index,
        target_start_ns=start,
        target_end_ns=end,
        events=(
            ScheduledMidiEvent(0, bar_index, start, Message("note_on", note=pitch, velocity=64),
                               is_bar_start=True),
            ScheduledMidiEvent(1, bar_index, end, Message("note_off", note=pitch, velocity=0)),
        ),
        block_id=str(bar_index),
        adapter=adapter,
        fallback_used=adapter != "model",
    )


def fallbacks(clock: MonotonicBarClock, bars: int = BARS):
    return {i: make_block(clock, i, pitch=36, adapter="fallback") for i in range(bars)}


class FakeClock:
    """Manually advanced clock so tests never depend on wall time."""

    def __init__(self) -> None:
        self.now = 0

    def __call__(self) -> int:
        return self.now

    def advance_ms(self, ms: float) -> None:
        self.now += round(ms * 1_000_000)


class InputSnapshotBufferTests(unittest.TestCase):
    def test_callback_only_timestamps_and_queues(self) -> None:
        buf = MidiInputSnapshotBuffer(window_seconds=1.0)
        buf.handle(Message("note_on", note=60, velocity=80), received_ns=1_000_000_000)

        snap = buf.snapshot(now_ns=1_500_000_000)
        self.assertEqual(1, len(snap))
        self.assertEqual(60, snap[0].message.note)
        self.assertEqual(1_000_000_000, snap[0].received_ns)
        self.assertEqual(1, buf.received_count)

    def test_events_outside_the_window_are_dropped(self) -> None:
        buf = MidiInputSnapshotBuffer(window_seconds=1.0)
        buf.handle(Message("note_on", note=60, velocity=80), received_ns=0)
        buf.handle(Message("note_on", note=62, velocity=80), received_ns=2_000_000_000)

        snap = buf.snapshot(now_ns=2_000_000_000)
        self.assertEqual([62], [e.message.note for e in snap])


class ProducerTests(unittest.TestCase):
    def test_ready_bar_is_served_from_the_model(self) -> None:
        clock = make_clock()
        built = Event()

        def build(bar_index, _events):
            built.set()
            return make_block(clock, bar_index, pitch=72, adapter="model")

        with BarBlockProducer(bar_count=BARS, fallback_blocks=fallbacks(clock),
                              build_block=build, clock=clock) as producer:
            self.assertTrue(producer.wait_for_bar(0, timeout=2.0))
            block = producer.get(0)

        self.assertTrue(built.is_set())
        self.assertEqual("model", block.adapter)
        self.assertEqual(SOURCE_MODEL, producer.record_for(0).source)
        self.assertFalse(producer.record_for(0).used_fallback)

    def test_slow_producer_falls_back_and_discards_the_late_result(self) -> None:
        """A bar that misses its slot must not be played in a later bar."""
        clock = make_clock()
        release = Event()

        def build(bar_index, _events):
            release.wait(timeout=5.0)
            return make_block(clock, bar_index, pitch=72, adapter="model")

        with BarBlockProducer(bar_count=BARS, fallback_blocks=fallbacks(clock),
                              build_block=build, clock=clock) as producer:
            # Bar 0 is still blocked inside build_block.
            block = producer.get(0)
            self.assertEqual("fallback", block.adapter)
            release.set()
            for _ in range(200):
                record = producer.record_for(0)
                if record is not None and record.completed_ns is not None:
                    break
                time.sleep(0.01)

        record = producer.record_for(0)
        self.assertEqual(SOURCE_FALLBACK_NOT_READY, record.source)
        self.assertTrue(record.discarded_late)
        # The late bar-0 block is dropped, not parked for a later bar to pick up.
        self.assertNotIn(0, producer._ready)
        self.assertEqual(1, producer.get(1).bar_index)

    def test_generation_error_falls_back_without_killing_the_producer(self) -> None:
        clock = make_clock()

        def build(bar_index, _events):
            if bar_index == 0:
                raise RuntimeError("inference exploded")
            return make_block(clock, bar_index, pitch=72, adapter="model")

        with BarBlockProducer(bar_count=BARS, fallback_blocks=fallbacks(clock),
                              build_block=build, clock=clock) as producer:
            for _ in range(200):
                if producer.record_for(0) and producer.record_for(0).completed_ns:
                    break
                time.sleep(0.01)
            self.assertEqual("fallback", producer.get(0).adapter)
            self.assertTrue(producer.wait_for_bar(1, timeout=2.0))
            self.assertEqual("model", producer.get(1).adapter)

        record = producer.record_for(0)
        self.assertEqual(SOURCE_FALLBACK_ERROR, record.source)
        self.assertIn("inference exploded", record.error)

    def test_close_cancels_a_blocked_producer(self) -> None:
        clock = make_clock()
        entered = Event()
        release = Event()

        def build(bar_index, _events):
            entered.set()
            release.wait(timeout=5.0)
            return make_block(clock, bar_index, pitch=72, adapter="model")

        producer = BarBlockProducer(bar_count=BARS, fallback_blocks=fallbacks(clock),
                                    build_block=build, clock=clock)
        producer.start()
        self.assertTrue(entered.wait(timeout=2.0))
        release.set()
        producer.close(timeout=5.0)
        self.assertIsNone(producer._thread)

    def test_lead_is_bounded(self) -> None:
        """Without consumption the producer must not run ahead indefinitely."""
        clock = make_clock()
        started: list[int] = []

        def build(bar_index, _events):
            started.append(bar_index)
            return make_block(clock, bar_index, pitch=72, adapter="model")

        with BarBlockProducer(bar_count=8, fallback_blocks=fallbacks(clock, 8),
                              build_block=build, clock=clock, max_lead_bars=1) as producer:
            time.sleep(0.2)
            # Nothing consumed yet, so watermark is -1 and lead 1 allows bar 0 only.
            self.assertEqual([0], started)
            producer.get(0)
            time.sleep(0.2)
            self.assertEqual([0, 1], started)

    def test_get_uses_injected_clock_for_timing_fields(self) -> None:
        fake = FakeClock()
        clock = make_clock()
        buf = MidiInputSnapshotBuffer(window_seconds=10.0)
        buf.handle(Message("note_on", note=60, velocity=80), received_ns=0)

        def build(bar_index, events):
            fake.advance_ms(120)
            return make_block(clock, bar_index, pitch=72, adapter="model")

        with BarBlockProducer(bar_count=BARS, fallback_blocks=fallbacks(clock),
                              build_block=build, clock=clock, input_buffer=buf,
                              clock_ns=fake) as producer:
            self.assertTrue(producer.wait_for_bar(0, timeout=2.0))

        record = producer.record_for(0)
        self.assertAlmostEqual(120.0, record.generation_ms, places=3)
        self.assertAlmostEqual(120.0, record.input_to_ready_ms, places=3)
        # Bar 0 starts at clock t=0, so the lookahead term is zero here and is
        # reported separately from the model time above.
        self.assertAlmostEqual(0.0, record.input_to_bar_start_ms, places=3)
        self.assertEqual(1, record.input_event_count)

    def test_missing_fallback_is_rejected_up_front(self) -> None:
        clock = make_clock()
        partial = fallbacks(clock)
        partial.pop(2)
        with self.assertRaisesRegex(ValueError, "fallback block missing"):
            BarBlockProducer(bar_count=BARS, fallback_blocks=partial,
                             build_block=lambda *_: None, clock=clock)


class SchedulerIntegrationTests(unittest.TestCase):
    """The producer must satisfy the mapping the scheduler already reads."""

    def _run(self, build):
        clock = make_clock(start_ns=0)
        sent: list[Message] = []

        class Sink:
            def send(self, message):
                sent.append(message)

        with BarBlockProducer(bar_count=BARS, fallback_blocks=fallbacks(clock),
                              build_block=build, clock=clock) as producer:
            producer.wait_for_bar(0, timeout=2.0)
            # Freeze the clock at the run start so no dispatch is ever late;
            # wait_until is a no-op, so the loop is driven purely by the blocks.
            result = OneBarMidiScheduler(
                sink=Sink(), clock=clock, clock_ns=lambda: 0,
                wait_until=lambda target_ns, stop: None,
            ).run(blocks=producer, expected_bar_count=BARS)
        return result, sent, producer

    def test_scheduler_consumes_the_live_producer_without_underrun(self) -> None:
        clock = make_clock()
        result, sent, producer = self._run(
            lambda i, _e: make_block(clock, i, pitch=72, adapter="model")
        )

        self.assertTrue(result.run_completed)
        self.assertEqual(BARS, result.completed_bar_count)
        self.assertEqual(0, result.queue_underrun_count)
        self.assertEqual(2 * BARS, len(sent))
        summary = summarize_production(producer.records)
        self.assertEqual(BARS, summary["bar_count"])

    def test_failing_producer_still_completes_every_bar_via_fallback(self) -> None:
        def build(_bar_index, _events):
            raise RuntimeError("always fails")

        result, sent, producer = self._run(build)

        self.assertTrue(result.run_completed)
        self.assertEqual(BARS, result.completed_bar_count)
        self.assertEqual(0, result.queue_underrun_count)
        self.assertEqual(2 * BARS, len(sent))
        summary = summarize_production(producer.records)
        self.assertEqual(0, summary["model_bar_count"])
        # wait_until is a no-op here, so the scheduler outruns the producer and
        # only some bars are attempted. Every bar still played, from fallback.
        self.assertGreaterEqual(summary["error_count"], 1)
        self.assertEqual(summary["bar_count"], summary["fallback_bar_count"])


class SummaryTests(unittest.TestCase):
    def test_empty_metric_reports_none_rather_than_zero(self) -> None:
        clock = make_clock()
        with BarBlockProducer(bar_count=1, fallback_blocks=fallbacks(clock, 1),
                              build_block=lambda *_: None, clock=clock) as producer:
            producer.get(0)
        summary = summarize_production(producer.records)

        self.assertIsNone(summary["input_to_ready_ms"])
        self.assertEqual(1, summary["fallback_bar_count"])


if __name__ == "__main__":
    unittest.main()


class InputToNotesTests(unittest.TestCase):
    def test_pairs_note_on_and_off(self) -> None:
        events = [
            TimedInputMessage(0, Message("note_on", note=60, velocity=90)),
            TimedInputMessage(500_000_000, Message("note_off", note=60, velocity=0)),
        ]
        notes = input_events_to_notes(events)

        self.assertEqual(1, len(notes))
        self.assertEqual(60, notes[0].pitch)
        self.assertEqual(90, notes[0].velocity)
        self.assertAlmostEqual(0.5, notes[0].end - notes[0].start, places=6)

    def test_held_note_is_closed_at_the_window_edge(self) -> None:
        """A key still down is the context the next bar most needs."""
        events = [TimedInputMessage(0, Message("note_on", note=64, velocity=80))]
        notes = input_events_to_notes(events, end_ns=1_000_000_000)

        self.assertEqual(1, len(notes))
        self.assertAlmostEqual(1.0, notes[0].end - notes[0].start, places=6)

    def test_quiet_input_cannot_produce_a_silent_primer_velocity(self) -> None:
        """velocity < 4 bins to 0, which decodes back to a note-off."""
        from inference.realtime.continuous import MIN_PRIMER_VELOCITY

        events = [
            TimedInputMessage(0, Message("note_on", note=60, velocity=1)),
            TimedInputMessage(200_000_000, Message("note_off", note=60, velocity=0)),
        ]
        notes = input_events_to_notes(events)

        self.assertEqual(MIN_PRIMER_VELOCITY, notes[0].velocity)
        self.assertGreaterEqual(notes[0].velocity // 4, 1)

    def test_note_on_zero_velocity_is_treated_as_note_off(self) -> None:
        events = [
            TimedInputMessage(0, Message("note_on", note=60, velocity=70)),
            TimedInputMessage(300_000_000, Message("note_on", note=60, velocity=0)),
        ]
        notes = input_events_to_notes(events)

        self.assertEqual(1, len(notes))
        self.assertAlmostEqual(0.3, notes[0].end - notes[0].start, places=6)

    def test_retrigger_closes_the_previous_note(self) -> None:
        events = [
            TimedInputMessage(0, Message("note_on", note=60, velocity=70)),
            TimedInputMessage(100_000_000, Message("note_on", note=60, velocity=90)),
            TimedInputMessage(400_000_000, Message("note_off", note=60, velocity=0)),
        ]
        notes = input_events_to_notes(events)

        self.assertEqual(2, len(notes))
        self.assertEqual([70, 90], [n.velocity for n in notes])

    def test_non_note_messages_are_ignored(self) -> None:
        events = [TimedInputMessage(0, Message("control_change", control=64, value=127))]
        self.assertEqual([], input_events_to_notes(events))
