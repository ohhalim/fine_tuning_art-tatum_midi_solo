"""Only adopted model blocks reach played metrics and the voicing pool (Astra M2)."""
from __future__ import annotations

import threading
import time
import unittest

from mido import Message

from inference.realtime.continuous import BarBlockProducer
from inference.realtime.scheduler import (MonotonicBarClock, ScheduledMidiBlock,
                                          ScheduledMidiEvent, build_deterministic_blocks)
from scripts.run_continuous_jazz import BlockMetricsRecorder, with_block_metrics


def model_block(index, pitches=(60, 64)):
    events = tuple(ScheduledMidiEvent(sequence_index=i, bar_index=index, target_ns=i,
                                      message=Message("note_on", note=p, velocity=80))
                   for i, p in enumerate(pitches))
    return ScheduledMidiBlock(bar_index=index, target_start_ns=0, events=events)


class Record:
    def __init__(self, bar_index):
        self.bar_index = bar_index


def wait_for(cond, timeout=2.0):
    deadline = time.monotonic() + timeout
    while not cond() and time.monotonic() < deadline:
        time.sleep(0.005)
    return cond()


class ProducerAdoptionTest(unittest.TestCase):
    def test_late_block_is_not_adopted_even_after_it_finishes(self) -> None:
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)
        fallbacks = build_deterministic_blocks(clock=clock, bar_count=3)
        release = threading.Event()
        built = []

        def build(bar_index, events):
            if bar_index == 0:
                release.wait(2)             # bar 0 finishes only after it was asked for
            built.append(bar_index)
            return model_block(bar_index)

        producer = BarBlockProducer(bar_count=3, fallback_blocks=fallbacks, build_block=build,
                                    clock=clock, max_lead_bars=2)
        with producer:
            self.assertIs(producer.get(0), fallbacks[0])   # not ready -> fallback
            release.set()
            self.assertTrue(wait_for(lambda: 1 in built))
            got1 = producer.get(1)
            self.assertEqual(got1.bar_index, 1)
            self.assertIsNot(got1, fallbacks[1])
            self.assertEqual(producer.adopted_blocks, frozenset({1}))
            self.assertEqual(producer.consumed_watermark, 1)
        self.assertTrue(producer.record_for(0).used_fallback)


class RecorderTest(unittest.TestCase):
    def _recorder(self, fake):
        return BlockMetricsRecorder(producer_ref=lambda: fake, chord_for_block=lambda b: "Cmaj7",
                                    adapter_for_block=lambda b: "tatum")

    def test_only_adopted_blocks_are_played_and_pooled(self) -> None:
        class Fake:
            adopted_blocks = frozenset()
            consumed_watermark = -1

            def adoption_snapshot(self):
                return self.adopted_blocks, self.consumed_watermark
        fake = Fake()
        rec = self._recorder(fake)
        rec.on_generated(model_block(0, (60, 64)), ())
        self.assertEqual(rec.played, {})                   # undecided yet
        fake.adopted_blocks, fake.consumed_watermark = frozenset({0}), 0
        rec.on_generated(model_block(1, (62,)), ())
        self.assertEqual(list(rec.played), [0])
        self.assertEqual(rec.played[0]["unique_voicings_so_far"], 1)
        fake.consumed_watermark = 1                        # bar 1 served from fallback
        rec.on_generated(model_block(2, (67,)), ())
        self.assertEqual(list(rec.played), [0])            # bar 1 discarded, not pooled
        fake.adopted_blocks, fake.consumed_watermark = frozenset({0, 2}), 2
        rec.finish([Record(0), Record(0), Record(2)])
        self.assertEqual(sorted(rec.played), [0, 2])
        self.assertEqual(rec.played[2]["unique_voicings_so_far"], 2)   # {0,4} and {7}, not {2}
        self.assertEqual([rec.generation[b]["adopted"] for b in (0, 1, 2)], [True, False, True])
        self.assertEqual(rec.played[0]["send_status"], "complete")
        self.assertEqual(rec.played[2]["send_status"], "complete")

    def test_get_between_the_two_reads_does_not_drop_an_adopted_block(self) -> None:
        # Astra's interleaving: block 0 pending; while block 1 is being recorded,
        # the scheduler's get(0) returns the model block between the two reads.
        # With separate property reads this paired adopted={} with watermark=0 and
        # dropped block 0 for good; the one-lock snapshot cannot see that pair.
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)
        fallbacks = build_deterministic_blocks(clock=clock, bar_count=3)
        producer = BarBlockProducer(bar_count=3, fallback_blocks=fallbacks, build_block=None, clock=clock)
        producer._ready[0] = model_block(0)                 # ready, not yet asked for

        class Interleaved:
            """Proxy that runs get(0) right after the first read, wherever the reader reads."""
            def __init__(self, inner):
                self.inner, self.done = inner, False

            def _once(self):
                if not self.done:
                    self.done = True
                    self.inner.get(0)

            @property
            def adopted_blocks(self):
                value = self.inner.adopted_blocks
                self._once()
                return value

            @property
            def consumed_watermark(self):
                value = self.inner.consumed_watermark
                self._once()
                return value

            def adoption_snapshot(self):
                value = self.inner.adoption_snapshot()
                self._once()
                return value

        proxy = Interleaved(producer)
        rec = self._recorder(proxy)
        rec._pending.append(0)                              # block 0 generated earlier
        rec.generation[0] = {"block": 0, "adapter": "tatum", "notes": 2, "events": 2}
        rec._voicings[0] = []
        rec.on_generated(model_block(1), ())                # get(0) lands mid-read
        self.assertEqual(producer.adopted_blocks, frozenset({0}))
        rec.finish([Record(0), Record(0)])
        self.assertIn(0, rec.played)                        # adopted block kept
        self.assertTrue(rec.generation[0]["adopted"])
        self.assertEqual(rec.played[0]["send_status"], "complete")

    def test_partial_and_unsent(self) -> None:
        class Fake:
            adopted_blocks = frozenset({0, 1})
            consumed_watermark = 1

            def adoption_snapshot(self):
                return self.adopted_blocks, self.consumed_watermark
        rec = self._recorder(Fake())
        rec.on_generated(model_block(0, (60, 64)), ())
        rec.on_generated(model_block(1, (60,)), ())
        rec.finish([Record(0)])
        self.assertEqual((rec.played[0]["send_status"], rec.played[1]["send_status"]),
                         ("partial", "none"))

    def test_astra_reproduction_through_the_producer(self) -> None:
        clock = MonotonicBarClock(bpm=120, beats_per_bar=4, start_ns=0)
        fallbacks = build_deterministic_blocks(clock=clock, bar_count=3)
        release = threading.Event()
        box = {}
        rec = BlockMetricsRecorder(producer_ref=lambda: box.get("p"), chord_for_block=lambda b: "Cmaj7",
                                   adapter_for_block=lambda b: "tatum")

        def factory(*, clock, duration):
            def build(bar_index, events):
                if bar_index == 0:
                    release.wait(2)
                return model_block(bar_index)
            return build

        build = with_block_metrics(factory, record=rec.on_generated)(clock=clock, duration=2.0)
        producer = BarBlockProducer(bar_count=3, fallback_blocks=fallbacks, build_block=build,
                                    clock=clock, max_lead_bars=2)
        box["p"] = producer
        with producer:
            producer.get(0)                                   # asked before bar 0 is ready
            release.set()
            self.assertTrue(wait_for(lambda: 1 in rec.generation))
            producer.get(1)
            self.assertTrue(wait_for(lambda: 2 in rec.generation))
        rec.finish([])
        self.assertTrue(producer.record_for(0).used_fallback)
        self.assertIn(0, rec.generation)                      # generated ...
        self.assertFalse(rec.generation[0]["adopted"])        # ... but never adopted
        self.assertNotIn(0, rec.played)
        self.assertIn(1, rec.played)


if __name__ == "__main__":
    unittest.main()
