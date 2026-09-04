"""Model-free direct MIDI echo and event-integrity measurement."""

from __future__ import annotations

import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Callable, Iterable, Protocol, Sequence

from mido import Message


REPORT_SCHEMA_VERSION = "direct_midi_echo_report_v2"
SEND_ACCEPTANCE_REPORT_SCHEMA_VERSION = "direct_midi_send_acceptance_report_v1"


class MidiSink(Protocol):
    def send(self, message: Message) -> None: ...


def close_mido_input(port: object) -> None:
    """Close a Mido RtMidi input without re-registering its callback wrapper.

    Mido 1.3.3's RtMidi ``Input._close`` assigns ``callback = None``. Its callback
    setter cancels the native callback but then installs Mido's wrapper again
    before ``close_port``. On CoreMIDI this can deadlock in ``MIDIPortDispose``
    while the new callback waits for the Python GIL.

    The RtMidi path mirrors Mido's native close/delete sequence while skipping
    that callback reinstallation. Other backends and test doubles use the public
    ``close`` method.
    """
    raw_port = getattr(port, "_rt", None)
    if raw_port is None:
        if type(port).__module__ == "mido.backends.rtmidi":
            raise RuntimeError("unsupported Mido RtMidi input internals: missing _rt")
        port.close()
        return

    raw_port.cancel_callback()
    setattr(port, "_callback", None)
    try:
        raw_port.close_port()
    finally:
        # Mark the Mido wrapper unusable before delete. If delete raises, its
        # BasePort finalizer must not enter the already-torn-down RtMidi handle.
        setattr(port, "closed", True)
        raw_port.delete()


@dataclass(frozen=True)
class TimedMidiMessage:
    logical_time_seconds: float
    message: Message


@dataclass(frozen=True)
class LatencySummary:
    sample_count: int
    p50: float | None
    p95: float | None
    p99: float | None
    maximum: float | None


@dataclass(frozen=True)
class EchoIntegrityReport:
    schema_version: str
    scope: str
    logical_duration_seconds: float
    wall_clock_seconds: float
    input_event_count: int
    output_event_count: int
    event_loss_count: int
    duplicate_output_count: int
    order_mismatch_count: int
    unmatched_note_off_count: int
    stuck_note_count: int
    crash_count: int
    minimum_input_event_count: int
    input_event_gate_passed: bool
    callback_processing_latency_ms: LatencySummary
    capture_latency_ms: LatencySummary
    safe_reset_sent: bool
    passed_event_integrity: bool
    wall_clock_soak_completed: bool
    os_midi_loopback_observed: bool
    fl_studio_audio_observed: bool
    passed_r0_transport_gate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class MidiSendAcceptanceReport:
    schema_version: str
    scope: str
    wall_clock_seconds: float
    input_event_count: int
    accepted_send_count: int
    send_failure_count: int
    minimum_input_event_count: int
    input_event_gate_passed: bool
    callback_processing_latency_ms: LatencySummary
    safe_reset_sent: bool
    output_capture_observed: bool
    passed_send_acceptance: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RecordingMidiSink:
    """In-memory output capture used by the deterministic transport probe."""

    def __init__(self) -> None:
        self.messages: list[Message] = []

    def send(self, message: Message) -> None:
        self.messages.append(message.copy())


def canonical_message(message: Message) -> tuple[int, ...]:
    return tuple(message.bytes())


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def summarize_latency(latency_ns: Sequence[int]) -> LatencySummary:
    latency_ms = [max(0, int(value)) / 1_000_000 for value in latency_ns]
    return LatencySummary(
        sample_count=len(latency_ms),
        p50=_percentile(latency_ms, 0.50),
        p95=_percentile(latency_ms, 0.95),
        p99=_percentile(latency_ms, 0.99),
        maximum=max(latency_ms) if latency_ms else None,
    )


def _note_state(messages: Iterable[Message]) -> tuple[int, int]:
    active_notes: Counter[tuple[int, int]] = Counter()
    reset_released_notes: Counter[tuple[int, int]] = Counter()
    unmatched_note_off_count = 0
    for message in messages:
        if (
            message.type == "control_change"
            and message.control in {120, 123}
        ):
            for key in [key for key in active_notes if key[0] == message.channel]:
                reset_released_notes[key] += active_notes[key]
                del active_notes[key]
            continue
        if message.type == "note_on" and message.velocity > 0:
            active_notes[(message.channel, message.note)] += 1
            continue
        if message.type not in {"note_off", "note_on"}:
            continue
        key = (message.channel, message.note)
        if active_notes[key] <= 0:
            if reset_released_notes[key] > 0:
                reset_released_notes[key] -= 1
                if reset_released_notes[key] == 0:
                    del reset_released_notes[key]
                continue
            unmatched_note_off_count += 1
            continue
        active_notes[key] -= 1
        if active_notes[key] == 0:
            del active_notes[key]
    return unmatched_note_off_count, sum(active_notes.values())


def _order_mismatch_count(
    input_keys: Sequence[tuple[int, ...]], output_keys: Sequence[tuple[int, ...]]
) -> int:
    shared_length = min(len(input_keys), len(output_keys))
    mismatches = sum(input_keys[index] != output_keys[index] for index in range(shared_length))
    return mismatches + abs(len(input_keys) - len(output_keys))


def evaluate_event_integrity(
    *,
    input_messages: Sequence[Message],
    output_messages: Sequence[Message],
    callback_latency_ns: Sequence[int],
    logical_duration_seconds: float,
    wall_clock_seconds: float,
    scope: str,
    crash_count: int = 0,
    minimum_input_event_count: int = 1,
    capture_latency_ns: Sequence[int] = (),
    safe_reset_sent: bool = False,
    wall_clock_soak_completed: bool = False,
    os_midi_loopback_observed: bool = False,
    fl_studio_audio_observed: bool = False,
) -> EchoIntegrityReport:
    input_keys = [canonical_message(message) for message in input_messages]
    output_keys = [canonical_message(message) for message in output_messages]
    input_counts = Counter(input_keys)
    output_counts = Counter(output_keys)
    event_loss_count = sum((input_counts - output_counts).values())
    duplicate_output_count = sum((output_counts - input_counts).values())
    order_mismatch_count = _order_mismatch_count(input_keys, output_keys)
    unmatched_note_off_count, stuck_note_count = _note_state(output_messages)
    input_event_gate_passed = len(input_messages) >= max(1, int(minimum_input_event_count))
    passed = all(
        value == 0
        for value in (
            event_loss_count,
            duplicate_output_count,
            order_mismatch_count,
            unmatched_note_off_count,
            stuck_note_count,
            crash_count,
        )
    ) and input_event_gate_passed
    passed_r0_transport_gate = all(
        (
            passed,
            bool(wall_clock_soak_completed),
            bool(os_midi_loopback_observed),
            bool(safe_reset_sent),
        )
    )
    return EchoIntegrityReport(
        schema_version=REPORT_SCHEMA_VERSION,
        scope=scope,
        logical_duration_seconds=float(logical_duration_seconds),
        wall_clock_seconds=float(wall_clock_seconds),
        input_event_count=len(input_messages),
        output_event_count=len(output_messages),
        event_loss_count=event_loss_count,
        duplicate_output_count=duplicate_output_count,
        order_mismatch_count=order_mismatch_count,
        unmatched_note_off_count=unmatched_note_off_count,
        stuck_note_count=stuck_note_count,
        crash_count=int(crash_count),
        minimum_input_event_count=max(1, int(minimum_input_event_count)),
        input_event_gate_passed=input_event_gate_passed,
        callback_processing_latency_ms=summarize_latency(callback_latency_ns),
        capture_latency_ms=summarize_latency(capture_latency_ns),
        safe_reset_sent=bool(safe_reset_sent),
        passed_event_integrity=passed,
        wall_clock_soak_completed=bool(wall_clock_soak_completed),
        os_midi_loopback_observed=bool(os_midi_loopback_observed),
        fl_studio_audio_observed=bool(fl_studio_audio_observed),
        passed_r0_transport_gate=passed_r0_transport_gate,
    )


class DirectMidiEcho:
    """Serialize callback delivery and forward each MIDI message once."""

    def __init__(self, sink: MidiSink, clock_ns: Callable[[], int] = time.perf_counter_ns) -> None:
        self._sink = sink
        self._clock_ns = clock_ns
        self._lock = Lock()
        self.input_messages: list[Message] = []
        self.output_messages: list[Message] = []
        self.callback_latency_ns: list[int] = []
        self.crash_count = 0

    def process(self, message: Message) -> None:
        received_ns = self._clock_ns()
        with self._lock:
            copied = message.copy()
            self.input_messages.append(copied)
            try:
                self._sink.send(copied.copy())
            except Exception:
                self.crash_count += 1
                raise
            sent_ns = self._clock_ns()
            self.output_messages.append(copied)
            self.callback_latency_ns.append(max(0, sent_ns - received_ns))

    def report(
        self,
        *,
        logical_duration_seconds: float,
        wall_clock_seconds: float,
        scope: str,
        output_messages: Sequence[Message] | None = None,
        minimum_input_event_count: int = 1,
        capture_latency_ns: Sequence[int] = (),
        safe_reset_sent: bool = False,
        wall_clock_soak_completed: bool = False,
        os_midi_loopback_observed: bool = False,
        fl_studio_audio_observed: bool = False,
    ) -> EchoIntegrityReport:
        return evaluate_event_integrity(
            input_messages=self.input_messages,
            output_messages=list(output_messages) if output_messages is not None else self.output_messages,
            callback_latency_ns=self.callback_latency_ns,
            logical_duration_seconds=logical_duration_seconds,
            wall_clock_seconds=wall_clock_seconds,
            scope=scope,
            crash_count=self.crash_count,
            minimum_input_event_count=minimum_input_event_count,
            capture_latency_ns=capture_latency_ns,
            safe_reset_sent=safe_reset_sent,
            wall_clock_soak_completed=wall_clock_soak_completed,
            os_midi_loopback_observed=os_midi_loopback_observed,
            fl_studio_audio_observed=fl_studio_audio_observed,
        )

    def send_acceptance_report(
        self,
        *,
        wall_clock_seconds: float,
        scope: str,
        minimum_input_event_count: int = 1,
        safe_reset_sent: bool = False,
    ) -> MidiSendAcceptanceReport:
        minimum_count = max(1, int(minimum_input_event_count))
        input_event_gate_passed = len(self.input_messages) >= minimum_count
        passed_send_acceptance = all(
            (
                input_event_gate_passed,
                len(self.input_messages) == len(self.output_messages),
                self.crash_count == 0,
            )
        )
        return MidiSendAcceptanceReport(
            schema_version=SEND_ACCEPTANCE_REPORT_SCHEMA_VERSION,
            scope=scope,
            wall_clock_seconds=float(wall_clock_seconds),
            input_event_count=len(self.input_messages),
            accepted_send_count=len(self.output_messages),
            send_failure_count=self.crash_count,
            minimum_input_event_count=minimum_count,
            input_event_gate_passed=input_event_gate_passed,
            callback_processing_latency_ms=summarize_latency(self.callback_latency_ns),
            safe_reset_sent=bool(safe_reset_sent),
            output_capture_observed=False,
            passed_send_acceptance=passed_send_acceptance,
        )


def build_logical_midi_fixture(
    *, logical_duration_seconds: float = 600.0, bpm: float = 120.0
) -> list[TimedMidiMessage]:
    if logical_duration_seconds <= 0:
        raise ValueError("logical_duration_seconds must be positive")
    if bpm <= 0:
        raise ValueError("bpm must be positive")

    beat_seconds = 60.0 / bpm
    note_gate_seconds = beat_seconds * 0.5
    pitch_cycle = (60, 62, 64, 65, 67, 69, 71, 72)
    events: list[TimedMidiMessage] = [
        TimedMidiMessage(0.0, Message("program_change", channel=0, program=0))
    ]
    beat_index = 0
    while True:
        note_on_time = beat_index * beat_seconds
        note_off_time = note_on_time + note_gate_seconds
        if note_off_time > logical_duration_seconds:
            break
        if beat_index % 4 == 0:
            events.append(
                TimedMidiMessage(
                    note_on_time,
                    Message("control_change", channel=0, control=1, value=(beat_index // 4) % 128),
                )
            )
        pitch = pitch_cycle[beat_index % len(pitch_cycle)]
        events.append(
            TimedMidiMessage(
                note_on_time,
                Message("note_on", channel=0, note=pitch, velocity=80 + beat_index % 24),
            )
        )
        note_off_type = "note_off" if beat_index % 2 == 0 else "note_on"
        events.append(
            TimedMidiMessage(
                note_off_time,
                Message(note_off_type, channel=0, note=pitch, velocity=0),
            )
        )
        beat_index += 1
    return events
