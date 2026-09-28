"""Per-block metrics taken inside the loop (docs/experiments/LIVE_METRICS.md)."""
from __future__ import annotations

import unittest

from mido import Message

from inference.realtime.continuous import TimedInputMessage
from inference.realtime.scheduler import ScheduledMidiBlock, ScheduledMidiEvent
from scripts.run_continuous_jazz import block_metrics, with_block_metrics


def block(pitches, index=3):
    events = []
    for i, p in enumerate(pitches):
        events.append(ScheduledMidiEvent(sequence_index=2 * i, bar_index=index, target_ns=10 * i,
                                         message=Message("note_on", note=p, velocity=80)))
        events.append(ScheduledMidiEvent(sequence_index=2 * i + 1, bar_index=index, target_ns=10 * i + 5,
                                         message=Message("note_off", note=p, velocity=0)))
    return ScheduledMidiBlock(bar_index=index, target_start_ns=0, events=tuple(events))


class BlockMetricsTest(unittest.TestCase):
    def test_counts_pitch_and_chord_tones(self) -> None:
        # Cmaj7 = C E G B; D (62) is not a chord tone.
        inp = [TimedInputMessage(received_ns=1, message=Message("note_on", note=84, velocity=90)),
               TimedInputMessage(received_ns=2, message=Message("program_change", program=1))]
        m = block_metrics(block([60, 64, 62, 71]), chord="Cmaj7", adapter="tatum", input_events=inp)
        self.assertEqual((m["block"], m["notes"], m["pitch_min"], m["pitch_max"]), (3, 4, 60, 71))
        self.assertEqual(m["chord_tone_ratio"], 0.75)
        self.assertEqual((m["input_notes"], m["input_pitch_mean"]), (1, 84.0))

    def test_empty_block(self) -> None:
        m = block_metrics(block([]), chord="Dm7", adapter=None, input_events=())
        self.assertEqual((m["notes"], m["pitch_mean"], m["chord_tone_ratio"]), (0, None, None))

    def test_wrapper_records_and_returns_the_block(self) -> None:
        seen = []
        b = block([60])
        factory = with_block_metrics(lambda *, clock, duration: (lambda i, ev: b),
                                     record=lambda blk, ev: seen.append((blk.bar_index, ev)))
        build = factory(clock=None, duration=1.0)
        self.assertIs(build(3, ()), b)
        self.assertEqual(seen, [(3, ())])


if __name__ == "__main__":
    unittest.main()
