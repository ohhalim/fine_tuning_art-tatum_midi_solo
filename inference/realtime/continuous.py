"""Produce one bar ahead while the current bar plays.

``OneBarMidiScheduler`` only ever calls ``blocks.get(bar_index)``, so a live
view can stand in for the prebuilt mapping without touching the scheduler.
``BarBlockProducer`` is that view: a background thread fills bars ahead of
playback, and ``get`` returns whatever is ready *without blocking*, falling
back to a prebuilt block when it is not.

Two rules shape the design:

* The scheduler thread must never run the model. ``get`` only reads a dict.
* A late model result is discarded, never played in a later bar, because it
  was conditioned on input that has since gone stale.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from threading import Condition, Event, Lock, Thread
from typing import Callable, Mapping, Sequence

from mido import Message

from .scheduler import MonotonicBarClock, ScheduledMidiBlock

SOURCE_MODEL = "model"
SOURCE_FALLBACK_NOT_READY = "fallback_not_ready"
SOURCE_FALLBACK_ERROR = "fallback_error"
SOURCE_FALLBACK_NOT_STARTED = "fallback_not_started"


@dataclass(frozen=True)
class TimedInputMessage:
    """A received MIDI message with the time the callback saw it."""

    received_ns: int
    message: Message


@dataclass(frozen=True)
class BarProductionRecord:
    """What happened to one bar, from request to hand-off.

    The three latency fields are deliberately separate. ``generation_ms`` is
    model time alone; ``input_to_ready_ms`` adds the wait for input to reach
    the producer; ``input_to_bar_start_ms`` additionally includes the lookahead
    wait until the bar is actually due, which is bounded by the bar duration
    and says nothing about how fast the model is.
    """

    bar_index: int
    source: str
    used_fallback: bool
    requested_ns: int | None = None
    completed_ns: int | None = None
    generation_ms: float | None = None
    newest_input_ns: int | None = None
    input_to_ready_ms: float | None = None
    input_to_bar_start_ms: float | None = None
    input_event_count: int = 0
    discarded_late: bool = False
    error: str | None = None


class MidiInputSnapshotBuffer:
    """Timestamp and queue input; nothing else.

    Called from the MIDI callback thread, so it does no parsing, no model work
    and no allocation beyond the append. Readers take a snapshot of the recent
    window and build a primer from it on their own thread.
    """

    def __init__(self, *, window_seconds: float = 4.0, max_events: int = 512) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if max_events <= 0:
            raise ValueError("max_events must be positive")
        self._window_ns = round(window_seconds * 1_000_000_000)
        self._events: deque[TimedInputMessage] = deque(maxlen=max_events)
        self._lock = Lock()
        self._received_count = 0

    def handle(self, message: Message, *, received_ns: int | None = None) -> None:
        stamped = TimedInputMessage(
            received_ns=time.perf_counter_ns() if received_ns is None else received_ns,
            message=message,
        )
        with self._lock:
            self._events.append(stamped)
            self._received_count += 1

    def snapshot(self, *, now_ns: int | None = None) -> tuple[TimedInputMessage, ...]:
        now = time.perf_counter_ns() if now_ns is None else now_ns
        cutoff = now - self._window_ns
        with self._lock:
            return tuple(e for e in self._events if e.received_ns >= cutoff)

    @property
    def received_count(self) -> int:
        with self._lock:
            return self._received_count


class BarBlockProducer:
    """Fill bars ahead of playback on a background thread.

    Satisfies the ``blocks`` mapping that ``OneBarMidiScheduler.run`` reads.
    """

    def __init__(
        self,
        *,
        bar_count: int,
        fallback_blocks: Mapping[int, ScheduledMidiBlock],
        build_block: Callable[[int, tuple[TimedInputMessage, ...]], ScheduledMidiBlock],
        clock: MonotonicBarClock | None = None,
        input_buffer: MidiInputSnapshotBuffer | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        max_lead_bars: int = 1,
    ) -> None:
        if bar_count <= 0:
            raise ValueError("bar_count must be positive")
        if max_lead_bars < 1:
            raise ValueError("max_lead_bars must be at least 1")
        missing = [i for i in range(bar_count) if i not in fallback_blocks]
        if missing:
            raise ValueError(f"fallback block missing for bars: {missing}")
        self._bar_count = bar_count
        self._fallback = dict(fallback_blocks)
        self._build_block = build_block
        self._clock = clock
        self._input_buffer = input_buffer
        self._clock_ns = clock_ns
        self._max_lead_bars = max_lead_bars

        self._ready: dict[int, ScheduledMidiBlock] = {}
        self._records: dict[int, BarProductionRecord] = {}
        self._abandoned: set[int] = set()
        self._consumed_watermark = -1
        self._cv = Condition()
        self._stop = Event()
        self._thread: Thread | None = None

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("producer already started")
        self._thread = Thread(target=self._run, name="bar-block-producer", daemon=True)
        self._thread.start()

    def close(self, *, timeout: float = 5.0) -> None:
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=timeout)

    def __enter__(self) -> "BarBlockProducer":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- scheduler-facing --------------------------------------------------

    def get(
        self, bar_index: int, default: ScheduledMidiBlock | None = None
    ) -> ScheduledMidiBlock | None:
        """Return a block for ``bar_index`` immediately.

        Called on the scheduler thread, so it must not block and must not run
        the model. A bar that is not ready is marked abandoned: if the model
        finishes it later the result is dropped rather than played stale.
        """
        if not 0 <= bar_index < self._bar_count:
            return default
        with self._cv:
            if bar_index > self._consumed_watermark:
                self._consumed_watermark = bar_index
                self._cv.notify_all()
            block = self._ready.pop(bar_index, None)
            if block is not None:
                return block
            if bar_index not in self._records or self._records[bar_index].completed_ns is None:
                self._abandoned.add(bar_index)
                existing = self._records.get(bar_index)
                self._records[bar_index] = BarProductionRecord(
                    bar_index=bar_index,
                    source=(
                        SOURCE_FALLBACK_NOT_STARTED
                        if existing is None
                        else SOURCE_FALLBACK_NOT_READY
                    ),
                    used_fallback=True,
                    requested_ns=None if existing is None else existing.requested_ns,
                    newest_input_ns=None if existing is None else existing.newest_input_ns,
                    input_event_count=0 if existing is None else existing.input_event_count,
                    discarded_late=existing is not None,
                )
            return self._fallback[bar_index]

    # -- producer thread ---------------------------------------------------

    def _run(self) -> None:
        for bar_index in range(self._bar_count):
            with self._cv:
                while (
                    not self._stop.is_set()
                    and bar_index > self._consumed_watermark + self._max_lead_bars
                ):
                    self._cv.wait(timeout=0.05)
                if self._stop.is_set():
                    return
                if bar_index <= self._consumed_watermark:
                    # Playback already passed this bar; nothing to produce.
                    continue
            self._produce(bar_index)

    def _produce(self, bar_index: int) -> None:
        requested_ns = self._clock_ns()
        events: tuple[TimedInputMessage, ...] = ()
        if self._input_buffer is not None:
            events = self._input_buffer.snapshot(now_ns=requested_ns)
        newest_input_ns = max((e.received_ns for e in events), default=None)
        with self._cv:
            self._records[bar_index] = BarProductionRecord(
                bar_index=bar_index,
                source=SOURCE_MODEL,
                used_fallback=False,
                requested_ns=requested_ns,
                newest_input_ns=newest_input_ns,
                input_event_count=len(events),
            )
        error: str | None = None
        block: ScheduledMidiBlock | None = None
        try:
            block = self._build_block(bar_index, events)
        except Exception as exc:  # noqa: BLE001 - recorded, never raised into playback
            error = f"{type(exc).__name__}: {exc}"
        completed_ns = self._clock_ns()

        with self._cv:
            abandoned = bar_index in self._abandoned or bar_index <= self._consumed_watermark
            if block is None:
                source = SOURCE_FALLBACK_ERROR
            elif abandoned:
                source = SOURCE_FALLBACK_NOT_READY
            else:
                source = SOURCE_MODEL
                self._ready[bar_index] = block
            self._records[bar_index] = BarProductionRecord(
                bar_index=bar_index,
                source=source,
                used_fallback=source != SOURCE_MODEL,
                requested_ns=requested_ns,
                completed_ns=completed_ns,
                generation_ms=(completed_ns - requested_ns) / 1e6,
                newest_input_ns=newest_input_ns,
                input_to_ready_ms=(
                    None if newest_input_ns is None else (completed_ns - newest_input_ns) / 1e6
                ),
                input_to_bar_start_ms=self._input_to_bar_start_ms(bar_index, newest_input_ns),
                input_event_count=len(events),
                discarded_late=abandoned and block is not None,
                error=error,
            )
            self._cv.notify_all()

    def _input_to_bar_start_ms(self, bar_index: int, newest_input_ns: int | None) -> float | None:
        """Input to the moment the bar is due.

        Includes the lookahead wait, so it is bounded below by the model and
        above by the bar grid. Reported apart from ``input_to_ready_ms`` so the
        grid wait is never mistaken for model latency.
        """
        if newest_input_ns is None or self._clock is None:
            return None
        return (self._clock.bar_start_ns(bar_index) - newest_input_ns) / 1e6

    # -- inspection --------------------------------------------------------

    def wait_for_bar(self, bar_index: int, *, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        with self._cv:
            while bar_index not in self._ready:
                record = self._records.get(bar_index)
                if record is not None and record.completed_ns is not None:
                    return bar_index in self._ready
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    return False
                self._cv.wait(timeout=remaining)
            return True

    @property
    def records(self) -> tuple[BarProductionRecord, ...]:
        with self._cv:
            return tuple(self._records[i] for i in sorted(self._records))

    def record_for(self, bar_index: int) -> BarProductionRecord | None:
        with self._cv:
            return self._records.get(bar_index)


def summarize_production(
    records: Sequence[BarProductionRecord],
) -> dict[str, object]:
    """Aggregate producer records for a run report."""

    def _stats(values: Sequence[float]) -> dict[str, float] | None:
        if not values:
            return None
        ordered = sorted(values)
        return {
            "p50": ordered[len(ordered) // 2],
            "maximum": ordered[-1],
            "sample_count": len(ordered),
        }

    return {
        "bar_count": len(records),
        "model_bar_count": sum(1 for r in records if r.source == SOURCE_MODEL),
        "fallback_bar_count": sum(1 for r in records if r.used_fallback),
        "discarded_late_count": sum(1 for r in records if r.discarded_late),
        "error_count": sum(1 for r in records if r.error is not None),
        "source_counts": {
            source: sum(1 for r in records if r.source == source)
            for source in sorted({r.source for r in records})
        },
        "generation_ms": _stats([r.generation_ms for r in records if r.generation_ms is not None]),
        "input_to_ready_ms": _stats(
            [r.input_to_ready_ms for r in records if r.input_to_ready_ms is not None]
        ),
        "input_to_bar_start_ms": _stats(
            [r.input_to_bar_start_ms for r in records if r.input_to_bar_start_ms is not None]
        ),
    }


# Stage A bins velocity as ``velocity // 4``, so anything under 4 lands in bin 0
# and decodes back to MIDI velocity 0 - a note-off. A quiet player must not be
# able to push a silent velocity into the primer, so clamp on the way in.
MIN_PRIMER_VELOCITY = 4


def input_events_to_notes(
    events: Sequence[TimedInputMessage],
    *,
    end_ns: int | None = None,
    min_duration_seconds: float = 0.01,
):
    """Pair note_on/note_off from captured input into playable notes.

    Notes still held when the window closes are ended at ``end_ns`` rather than
    dropped: a key the player is holding right now is exactly the context the
    next bar should be conditioned on.
    """
    import pretty_midi

    if not events:
        return []
    origin_ns = events[0].received_ns
    closing_ns = end_ns if end_ns is not None else events[-1].received_ns
    open_notes: dict[int, tuple[float, int]] = {}
    notes = []

    def close(pitch: int, started: tuple[float, int], end_seconds: float) -> None:
        start_seconds, velocity = started
        end_seconds = max(end_seconds, start_seconds + min_duration_seconds)
        notes.append(
            pretty_midi.Note(
                velocity=max(MIN_PRIMER_VELOCITY, min(127, velocity)),
                pitch=pitch,
                start=start_seconds,
                end=end_seconds,
            )
        )

    for event in events:
        message = event.message
        if message.type not in ("note_on", "note_off"):
            continue
        seconds = (event.received_ns - origin_ns) / 1e9
        pitch = int(message.note)
        if message.type == "note_on" and message.velocity > 0:
            held = open_notes.pop(pitch, None)
            if held is not None:
                close(pitch, held, seconds)
            open_notes[pitch] = (seconds, int(message.velocity))
        else:
            held = open_notes.pop(pitch, None)
            if held is not None:
                close(pitch, held, seconds)

    tail_seconds = (closing_ns - origin_ns) / 1e9
    for pitch, held in open_notes.items():
        close(pitch, held, tail_seconds)
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes
