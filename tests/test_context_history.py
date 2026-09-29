"""Played history for --context-history (docs/experiments/CONTEXT_HISTORY.md, Astra review)."""
from __future__ import annotations

import tempfile
import unittest

from mido import Message

from inference.realtime.scheduler import ScheduledMidiBlock, ScheduledMidiEvent
from scripts.run_continuous_jazz import PlayedHistory, block_to_notes, main

BLOCK_S = 0.9375
BLOCK_NS = int(BLOCK_S * 1e9)


def block(index, pitch):
    start = index * BLOCK_NS
    return ScheduledMidiBlock(bar_index=index, target_start_ns=start, events=(
        ScheduledMidiEvent(sequence_index=0, bar_index=index, target_ns=start + 100_000_000,
                           message=Message("note_on", note=pitch, velocity=80)),
        ScheduledMidiEvent(sequence_index=1, bar_index=index, target_ns=start + 300_000_000,
                           message=Message("note_off", note=pitch, velocity=0)),
    ))


def pitches(tokens):
    return [t for t in tokens if 0 <= t < 128]


class PlayedHistoryTest(unittest.TestCase):
    def test_discarded_generated_block_is_replaced_by_the_fallback_that_played(self) -> None:
        generated = {0: block(0, 60), 1: block(1, 62), 2: block(2, 64)}      # 1 was generated but late
        fallbacks = {i: block(i, 40 + i) for i in range(3)}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0, 2}, watermark=2, generated=generated, fallback_for=fallbacks.__getitem__)
        self.assertEqual(pitches(hist.tokens), [60, 41, 64])
        self.assertEqual(hist.next_block, 3)

    def test_undecided_blocks_wait_and_time_stays_on_the_grid(self) -> None:
        generated = {i: block(i, 60 + i) for i in range(40)}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0}, watermark=0, generated=generated, fallback_for=lambda i: None)
        self.assertEqual(pitches(hist.tokens), [60])                          # block 1 not asked for yet
        hist.settle(adopted=set(range(40)), watermark=39, generated=generated, fallback_for=lambda i: None)
        total_ms = sum((t - 255) * 10 for t in hist.tokens if 256 <= t <= 355)
        self.assertEqual(total_ms, int(round(40 * BLOCK_S * 100)) * 10)      # 37.5 s, no 7.5 ms/block drift

    def test_block_to_notes_times_from_the_block_start(self) -> None:
        notes = block_to_notes(block(3, 67))
        self.assertEqual([(n.pitch, round(n.start, 3), round(n.end, 3)) for n in notes], [(67, 0.1, 0.3)])


class FlagTest(unittest.TestCase):
    def _exit(self, *extra):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid",
                  "--chord-primer", *extra])

    def test_history_needs_carry_tokens_and_half_bar_blocks(self) -> None:
        self._exit("--context-history")
        self._exit("--chord-blocks-per-bar", "2", "--context-carry-tokens", "32", "--context-history")

    def test_carry_needs_the_sub_block_path(self) -> None:
        self._exit("--context-carry-tokens", "32")


if __name__ == "__main__":
    unittest.main()
