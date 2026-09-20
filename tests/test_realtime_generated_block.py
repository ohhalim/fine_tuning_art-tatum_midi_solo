from __future__ import annotations

import unittest

import pretty_midi

from inference.realtime.blocks import build_scheduled_midi_block
from inference.realtime.scheduler import MonotonicBarClock


def midi_with_notes(*notes: pretty_midi.Note) -> pretty_midi.PrettyMIDI:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    instrument = pretty_midi.Instrument(program=0, name="test_lead")
    instrument.notes.extend(notes)
    midi.instruments.append(instrument)
    return midi


class RealtimeGeneratedBlockTest(unittest.TestCase):
    def test_converts_relative_notes_to_absolute_scheduler_events(self) -> None:
        midi = midi_with_notes(
            pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5),
            pretty_midi.Note(velocity=88, pitch=62, start=0.5, end=1.0),
        )
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=1_000)

        block = build_scheduled_midi_block(
            midi=midi,
            clock=clock,
            bar_index=1,
            block_id="block-1",
            source_context_id="context-7",
            context_version=7,
            adapter="art_tatum",
            fallback_used=False,
            output_channel=2,
            sequence_start_index=20,
        )

        self.assertEqual(2_000_001_000, block.target_start_ns)
        self.assertEqual(4_000_001_000, block.target_end_ns)
        self.assertEqual("block-1", block.block_id)
        self.assertEqual("context-7", block.source_context_id)
        self.assertEqual(7, block.context_version)
        self.assertEqual("art_tatum", block.adapter)
        self.assertFalse(block.fallback_used)
        self.assertEqual([20, 21, 22, 23], [event.sequence_index for event in block.events])
        self.assertEqual(
            ["note_on", "note_off", "note_on", "note_off"],
            [event.message.type for event in block.events],
        )
        self.assertEqual([2, 2, 2, 2], [event.message.channel for event in block.events])
        self.assertTrue(block.events[0].is_bar_start)

    def test_orders_note_off_before_note_on_at_the_same_target(self) -> None:
        midi = midi_with_notes(
            pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.5),
            pretty_midi.Note(velocity=90, pitch=60, start=0.5, end=1.0),
        )
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=0)

        block = build_scheduled_midi_block(
            midi=midi,
            clock=clock,
            bar_index=0,
            block_id="block-0",
            source_context_id="context-0",
            context_version=0,
            adapter="art_tatum",
            fallback_used=True,
        )

        simultaneous = [event for event in block.events if event.target_ns == 500_000_000]
        self.assertEqual(["note_off", "note_on"], [event.message.type for event in simultaneous])

    def test_rejects_notes_outside_the_target_window(self) -> None:
        midi = midi_with_notes(
            pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=2.1),
        )
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=0)

        with self.assertRaisesRegex(ValueError, "outside the block target window"):
            build_scheduled_midi_block(
                midi=midi,
                clock=clock,
                bar_index=0,
                block_id="block-0",
                source_context_id="context-0",
                context_version=0,
                adapter="art_tatum",
                fallback_used=False,
            )

    def test_rejects_same_pitch_overlap(self) -> None:
        midi = midi_with_notes(
            pretty_midi.Note(velocity=90, pitch=60, start=0.0, end=0.75),
            pretty_midi.Note(velocity=90, pitch=60, start=0.5, end=1.0),
        )
        clock = MonotonicBarClock(bpm=120.0, beats_per_bar=4, start_ns=0)

        with self.assertRaisesRegex(ValueError, "same-pitch overlap"):
            build_scheduled_midi_block(
                midi=midi,
                clock=clock,
                bar_index=0,
                block_id="block-0",
                source_context_id="context-0",
                context_version=0,
                adapter="art_tatum",
                fallback_used=False,
            )


if __name__ == "__main__":
    unittest.main()
