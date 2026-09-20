"""Convert in-memory MIDI phrases into scheduler-ready one-bar blocks."""

from __future__ import annotations

from collections import defaultdict

import pretty_midi
from mido import Message

from .scheduler import (
    MonotonicBarClock,
    ScheduledMidiBlock,
    ScheduledMidiEvent,
)


def build_scheduled_midi_block(
    *,
    midi: pretty_midi.PrettyMIDI,
    clock: MonotonicBarClock,
    bar_index: int,
    block_id: str,
    source_context_id: str,
    context_version: int,
    adapter: str,
    fallback_used: bool,
    output_channel: int = 0,
    sequence_start_index: int = 0,
    allow_empty: bool = False,
) -> ScheduledMidiBlock:
    """Map relative note times to one absolute scheduler window without file I/O."""

    if bar_index < 0:
        raise ValueError("bar_index must not be negative")
    if not block_id:
        raise ValueError("block_id must not be empty")
    if not source_context_id:
        raise ValueError("source_context_id must not be empty")
    if context_version < 0:
        raise ValueError("context_version must not be negative")
    if not adapter:
        raise ValueError("adapter must not be empty")
    if not 0 <= output_channel <= 15:
        raise ValueError("output_channel must be between 0 and 15")
    if sequence_start_index < 0:
        raise ValueError("sequence_start_index must not be negative")

    target_start_ns = clock.bar_start_ns(bar_index)
    target_end_ns = clock.bar_start_ns(bar_index + 1)
    block_duration_seconds = (target_end_ns - target_start_ns) / 1_000_000_000.0
    notes = [
        note
        for instrument in midi.instruments
        if not instrument.is_drum
        for note in instrument.notes
    ]
    drum_note_count = sum(
        len(instrument.notes) for instrument in midi.instruments if instrument.is_drum
    )
    if drum_note_count:
        raise ValueError("drum notes are not valid in a lead MIDI block")
    if not notes and not allow_empty:
        # Off by default: an empty block is usually a generation failure. A
        # caller that can tell a whole-bar rest from a failure opts in, and
        # gets a block with no events - the scheduler simply waits out the bar.
        raise ValueError("MIDI block must contain at least one note")

    note_intervals_by_pitch: dict[int, list[tuple[float, float]]] = defaultdict(list)
    raw_events: list[tuple[int, int, int, int, str]] = []
    for note in notes:
        pitch = int(note.pitch)
        velocity = int(note.velocity)
        start = float(note.start)
        end = float(note.end)
        if not 0 <= pitch <= 127:
            raise ValueError(f"note pitch out of range: {pitch}")
        if not 1 <= velocity <= 127:
            raise ValueError(f"note velocity out of range: {velocity}")
        if start < 0.0 or end <= start:
            raise ValueError("note times must satisfy 0 <= start < end")
        if end > block_duration_seconds + 1e-9:
            raise ValueError("note ends outside the block target window")
        note_intervals_by_pitch[pitch].append((start, end))

        note_on_target_ns = target_start_ns + round(start * 1_000_000_000)
        note_off_target_ns = target_start_ns + round(end * 1_000_000_000)
        if note_off_target_ns <= note_on_target_ns:
            raise ValueError("note duration is below scheduler nanosecond resolution")
        raw_events.append((note_on_target_ns, 1, pitch, velocity, "note_on"))
        raw_events.append((note_off_target_ns, 0, pitch, 0, "note_off"))

    for pitch, intervals in note_intervals_by_pitch.items():
        previous_end = -1.0
        for start, end in sorted(intervals):
            if start < previous_end - 1e-9:
                raise ValueError(f"same-pitch overlap in MIDI block: {pitch}")
            previous_end = max(previous_end, end)

    raw_events.sort(key=lambda item: (item[0], item[1], item[2]))
    events: list[ScheduledMidiEvent] = []
    bar_start_marked = False
    for offset, raw_event in enumerate(raw_events):
        target_ns, _priority, pitch, velocity, message_type = raw_event
        is_bar_start = (
            not bar_start_marked
            and message_type == "note_on"
            and target_ns == target_start_ns
        )
        if is_bar_start:
            bar_start_marked = True
        events.append(
            ScheduledMidiEvent(
                sequence_index=sequence_start_index + offset,
                bar_index=bar_index,
                target_ns=target_ns,
                message=Message(
                    message_type,
                    channel=output_channel,
                    note=pitch,
                    velocity=velocity,
                ),
                is_bar_start=is_bar_start,
            )
        )

    return ScheduledMidiBlock(
        bar_index=bar_index,
        target_start_ns=target_start_ns,
        events=tuple(events),
        target_end_ns=target_end_ns,
        block_id=block_id,
        source_context_id=source_context_id,
        context_version=context_version,
        adapter=adapter,
        fallback_used=bool(fallback_used),
    )
