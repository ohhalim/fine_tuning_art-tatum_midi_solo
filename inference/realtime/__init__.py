"""Realtime MIDI transport primitives."""

from .blocks import build_scheduled_midi_block
from .continuous import (
    BarBlockProducer,
    BarProductionRecord,
    MidiInputSnapshotBuffer,
    TimedInputMessage,
    input_events_to_notes,
    summarize_production,
)
from .transport import (
    DirectMidiEcho,
    EchoIntegrityReport,
    MidiSendAcceptanceReport,
    RecordingMidiSink,
    TimedMidiMessage,
    build_logical_midi_fixture,
    close_mido_input,
    evaluate_event_integrity,
)
from .scheduler import (
    DEADLINE_POLICY_ABORT_ON_FIRST_MISS,
    DEADLINE_POLICY_RECORD_AND_CONTINUE,
    InternalSchedulerReport,
    MonotonicBarClock,
    OneBarMidiScheduler,
    SchedulerDispatchDeadlineMiss,
    ScheduledMidiBlock,
    ScheduledMidiEvent,
    build_deterministic_blocks,
)

__all__ = [
    "build_scheduled_midi_block",
    "BarBlockProducer",
    "BarProductionRecord",
    "MidiInputSnapshotBuffer",
    "TimedInputMessage",
    "input_events_to_notes",
    "summarize_production",
    "DirectMidiEcho",
    "EchoIntegrityReport",
    "MidiSendAcceptanceReport",
    "RecordingMidiSink",
    "TimedMidiMessage",
    "build_logical_midi_fixture",
    "close_mido_input",
    "evaluate_event_integrity",
    "DEADLINE_POLICY_ABORT_ON_FIRST_MISS",
    "DEADLINE_POLICY_RECORD_AND_CONTINUE",
    "InternalSchedulerReport",
    "MonotonicBarClock",
    "OneBarMidiScheduler",
    "SchedulerDispatchDeadlineMiss",
    "ScheduledMidiBlock",
    "ScheduledMidiEvent",
    "build_deterministic_blocks",
]
