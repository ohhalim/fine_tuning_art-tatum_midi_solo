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


def shifts_ms(tokens):
    return sum((t - 255) * 10 for t in tokens if 256 <= t <= 355)


def events_block(index, notes):
    """notes: (start_s, end_s, pitch) relative to the block start."""
    start = index * BLOCK_NS
    evs, seq = [], 0
    for a, b, p in notes:
        evs.append((start + int(a * 1e9), Message("note_on", note=p, velocity=80)))
        evs.append((start + int(b * 1e9), Message("note_off", note=p, velocity=0)))
    evs.sort(key=lambda x: x[0])
    return ScheduledMidiBlock(bar_index=index, target_start_ns=start, events=tuple(
        ScheduledMidiEvent(sequence_index=i, bar_index=index, target_ns=t, message=m)
        for i, (t, m) in enumerate(evs)))


class PlayedHistoryTest(unittest.TestCase):
    def test_discarded_generated_block_is_replaced_by_the_fallback_that_played(self) -> None:
        generated = {0: block(0, 60), 1: block(1, 62), 2: block(2, 64)}      # 1 was generated but late
        fallbacks = {i: block(i, 40 + i) for i in range(3)}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0, 2}, watermark=2, generated=generated, fallback_for=fallbacks.__getitem__)
        self.assertEqual(pitches(hist.tokens), [60, 41, 64])
        self.assertEqual(hist.next_block, 3)

    def test_undecided_blocks_wait(self) -> None:
        generated = {i: block(i, 60 + i) for i in range(3)}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0}, watermark=0, generated=generated, fallback_for=lambda i: None)
        self.assertEqual(pitches(hist.tokens), [60])

    def test_notes_held_to_every_block_end_do_not_stretch_time(self) -> None:
        # Astra's reproduction: note_on at 0, note_off at 0.9375 s in 40 adopted blocks.
        generated = {i: events_block(i, [(0.0, BLOCK_S, 60 + i % 12)]) for i in range(40)}
        hist = PlayedHistory(BLOCK_S, horizon_s=60.0)       # keep all 37.5 s
        hist.settle(adopted=set(range(40)), watermark=39, generated=generated, fallback_for=lambda i: None)
        self.assertEqual(shifts_ms(hist.tokens), 37500)
        self.assertEqual(hist.boundary_step, 3750)

    def test_dense_events_land_on_the_absolute_grid(self) -> None:
        from midi_processor.processor import decode_midi
        notes = [(k * 0.055, k * 0.055 + 0.04, 60 + k % 7) for k in range(17)]
        generated = {i: events_block(i, notes) for i in range(20)}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted=set(range(20)), watermark=19, generated=generated, fallback_for=lambda i: None)
        self.assertEqual(shifts_ms(hist.tokens), round(20 * 93.75) * 10)     # 18.75 s from the first note
        starts = sorted(n.start for inst in decode_midi(hist.tokens).instruments for n in inst.notes)
        want = sorted(round((i * BLOCK_S + a) * 100) / 100 for i in range(20) for a, _, _ in notes)
        self.assertEqual([round(x, 2) for x in starts], want)

    def test_empty_blocks_keep_their_time(self) -> None:
        generated = {0: block(0, 60), 1: events_block(1, []), 2: events_block(2, []), 3: block(3, 64)}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0, 1, 2, 3}, watermark=3, generated=generated, fallback_for=lambda i: None)
        # first note at 0.1 s; boundary after block 3 at 3.75 s
        self.assertEqual(shifts_ms(hist.tokens), 3750 - 100)
        self.assertEqual(pitches(hist.tokens), [60, 64])

    def test_cap_keeps_velocity_and_the_newest_end(self) -> None:
        notes = [(k * 0.05, k * 0.05 + 0.04, 60 + k % 5) for k in range(18)]
        generated = {i: events_block(i, notes) for i in range(30)}
        hist = PlayedHistory(BLOCK_S, cap=200)
        hist.settle(adopted=set(range(30)), watermark=29, generated=generated, fallback_for=lambda i: None)
        self.assertLessEqual(len(hist.tokens), 200)
        first_on = next(i for i, t in enumerate(hist.tokens) if 0 <= t < 128)
        self.assertTrue(any(356 <= t < 388 for t in hist.tokens[:first_on]))   # velocity carried
        self.assertEqual(hist.boundary_step, round(30 * 93.75))

    def test_horizon_trims_old_notes_without_breaking_alignment(self) -> None:
        generated = {i: events_block(i, [(0.0, BLOCK_S, 60)]) for i in range(40)}
        hist = PlayedHistory(BLOCK_S)                        # 30 s horizon
        hist.settle(adopted=set(range(40)), watermark=39, generated=generated, fallback_for=lambda i: None)
        self.assertEqual(shifts_ms(hist.tokens), hist.span_steps() * 10)
        self.assertLessEqual(hist.span_steps(), 3000 + round(93.75))

    def _inside(self, hist):
        return all(s0 < e <= hist.boundary_step for s0, e, _, _ in hist._notes)

    def test_last_2_5_ms_note_stays_inside_its_block(self) -> None:
        # Astra's case: note_on 0.935 s, note_off 0.9375 s (the half-bar end at 128 BPM).
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0}, watermark=0, generated={0: events_block(0, [(0.935, BLOCK_S, 60)])},
                    fallback_for=lambda i: None)
        self.assertEqual(hist.boundary_step, 94)
        self.assertEqual(hist._notes, [(93, 94, 60, 80)])
        self.assertTrue(self._inside(hist))
        self.assertEqual(shifts_ms(hist.tokens), hist.span_steps() * 10)

    def test_notes_collapsing_to_one_tick_get_a_tick_or_are_dropped(self) -> None:
        hist = PlayedHistory(BLOCK_S)
        # 4 ms note mid-block -> onset moved one tick; a zero-length note at the block
        # start cannot move earlier and is dropped.
        blk = events_block(0, [(0.100, 0.104, 62), (0.0, 0.0, 64)])
        hist.settle(adopted={0}, watermark=0, generated={0: blk}, fallback_for=lambda i: None)
        self.assertEqual(sorted(hist._notes), [(9, 10, 62, 80)])

    def test_same_pitch_across_the_boundary_keeps_off_before_on(self) -> None:
        from midi_processor.processor import decode_midi
        gen = {0: events_block(0, [(0.90, BLOCK_S, 60)]), 1: events_block(1, [(0.0, 0.2, 60)])}
        hist = PlayedHistory(BLOCK_S)
        hist.settle(adopted={0, 1}, watermark=1, generated=gen, fallback_for=lambda i: None)
        self.assertTrue(self._inside(hist))
        ons = [t for t in hist.tokens if t == 60]
        offs = [t for t in hist.tokens if t == 128 + 60]
        self.assertEqual((len(ons), len(offs)), (2, 2))
        first_off = hist.tokens.index(128 + 60)
        second_on = [i for i, t in enumerate(hist.tokens) if t == 60][1]
        self.assertLess(first_off, second_on)
        self.assertEqual(len([n for inst in decode_midi(hist.tokens).instruments for n in inst.notes]), 2)

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
