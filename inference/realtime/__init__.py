"""Realtime MIDI transport primitives."""

from .blocks import build_scheduled_midi_block
from .continuous import (
    BarBlockProducer,
    BarProductionRecord,
    MidiInputSnapshotBuffer,
    TimedInputMessage,
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
    "summarize_production",
    "DirectMidiEcho",
    "EchoIntegrityReport",
    "MidiSendAcceptanceReport",
    "RecordingMidiSink",
    "TimedMidiMessage",
    "build_logical_midi_fixture",
    "close_mido_input",
    "evaluate_event_integrity",
    "InternalSchedulerReport",
    "MonotonicBarClock",
    "OneBarMidiScheduler",
    "SchedulerDispatchDeadlineMiss",
    "ScheduledMidiBlock",
    "ScheduledMidiEvent",
    "build_deterministic_blocks",
]
