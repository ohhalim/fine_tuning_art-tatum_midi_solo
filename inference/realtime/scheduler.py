"""Internal monotonic clock and deterministic one-bar MIDI scheduling."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from threading import Event
from typing import Callable, Mapping, Protocol, Sequence

from mido import Message

from .transport import LatencySummary, summarize_latency


INTERNAL_SCHEDULER_REPORT_SCHEMA_VERSION = "internal_midi_scheduler_report_v5"
DETERMINISTIC_FIXTURE_ID = "dense_chord_sub_spin_v1"
DEFAULT_DEADLINE_THRESHOLD_MS = 20.0
DEFAULT_SPIN_WINDOW_MS = 15.0
DEFAULT_ENVIRONMENT_CLOCK_GAP_THRESHOLD_SECONDS = 1.0
DEADLINE_POLICY_ABORT_ON_FIRST_MISS = "abort_on_first_miss"
DEADLINE_POLICY_RECORD_AND_CONTINUE = "record_and_continue"
DEADLINE_POLICIES = frozenset(
    {DEADLINE_POLICY_ABORT_ON_FIRST_MISS, DEADLINE_POLICY_RECORD_AND_CONTINUE}
)
TIMING_HISTOGRAM_BOUNDS_MS = (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0)

# Note-off gap used on the dense beats. Kept well below DEFAULT_SPIN_WINDOW_MS so the
# fixture actually reaches the regime where the busy-spin overlaps a capture callback.
DEFAULT_DENSE_OFF_GAP_MS = 4.0

# Voices sounded on each beat of a bar, cycling every four beats. Beat 0 stays a single
# note so the bar-start sample measures clock accuracy rather than chord serialisation.
BEAT_VOICE_COUNTS = (1, 3, 1, 2)

# Beats whose note-off uses the dense gap instead of a half-beat gate.
DENSE_OFF_BEATS = frozenset({1, 2})

CHORD_SEMITONE_OFFSETS = (0, 4, 7)


def deterministic_events_per_bar(beats_per_bar: int) -> int:
    """Event count one deterministic block emits, as note-on/note-off pairs per voice."""
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be positive")
    voices = sum(
        BEAT_VOICE_COUNTS[beat_in_bar % len(BEAT_VOICE_COUNTS)]
        for beat_in_bar in range(beats_per_bar)
    )
    return voices * 2


class MidiEventSink(Protocol):
    def send(self, message: Message) -> None: ...


@dataclass(frozen=True)
class MonotonicBarClock:
    bpm: float
    beats_per_bar: int
    start_ns: int

    def __post_init__(self) -> None:
        if self.bpm <= 0:
            raise ValueError("bpm must be positive")
        if self.beats_per_bar <= 0:
            raise ValueError("beats_per_bar must be positive")
        if self.start_ns < 0:
            raise ValueError("start_ns must not be negative")

    @property
    def beat_duration_ns(self) -> float:
        return 60_000_000_000.0 / self.bpm

    @property
    def bar_duration_ns(self) -> float:
        return self.beat_duration_ns * self.beats_per_bar

    def beat_target_ns(self, absolute_beat: float) -> int:
        if absolute_beat < 0:
            raise ValueError("absolute_beat must not be negative")
        return self.start_ns + round(absolute_beat * self.beat_duration_ns)

    def bar_start_ns(self, bar_index: int) -> int:
        if bar_index < 0:
            raise ValueError("bar_index must not be negative")
        return self.beat_target_ns(bar_index * self.beats_per_bar)


@dataclass(frozen=True)
class ScheduledMidiEvent:
    sequence_index: int
    bar_index: int
    target_ns: int
    message: Message
    is_bar_start: bool = False


@dataclass(frozen=True)
class ScheduledMidiBlock:
    bar_index: int
    target_start_ns: int
    events: tuple[ScheduledMidiEvent, ...]


@dataclass(frozen=True)
class SchedulerSendRecord:
    sequence_index: int
    bar_index: int
    target_ns: int
    dispatch_started_ns: int
    send_accepted_ns: int
    message: Message
    is_bar_start: bool


@dataclass(frozen=True)
class SchedulerDispatchDeadlineMiss:
    sequence_index: int
    bar_index: int
    target_ns: int
    dispatch_started_ns: int
    lateness_ns: int
    message_type: str
    channel: int | None
    note: int | None
    is_bar_start: bool
    is_catch_up: bool
    action: str


@dataclass(frozen=True)
class SchedulerRunResult:
    deadline_policy: str
    expected_bar_count: int
    started_bar_count: int
    completed_bar_count: int
    enqueued_block_count: int
    queue_underrun_count: int
    queue_depth_max: int
    send_failure_count: int
    scheduler_dispatch_deadline_miss_count: int
    primary_dispatch_stall_count: int
    catch_up_dispatch_event_count: int
    scheduler_dispatch_deadline_misses: tuple[SchedulerDispatchDeadlineMiss, ...]
    watchdog_trigger_reason: str | None
    run_completed: bool
    enqueue_lead_time_ns: tuple[int, ...]
    dispatch_attempt_lateness_ns: tuple[int, ...]
    records: tuple[SchedulerSendRecord, ...]


@dataclass(frozen=True)
class InternalSchedulerReport:
    schema_version: str
    scope: str
    bpm: float
    beats_per_bar: int
    requested_duration_seconds: float
    scheduled_duration_seconds: float
    start_delay_seconds: float
    wall_clock_seconds: float
    realtime_elapsed_seconds: float
    monotonic_elapsed_seconds: float
    realtime_minus_monotonic_seconds: float
    scheduler_run_realtime_elapsed_seconds: float
    scheduler_run_monotonic_elapsed_seconds: float
    scheduler_run_realtime_minus_monotonic_seconds: float
    environment_clock_gap_threshold_seconds: float
    environment_valid: bool
    deadline_policy: str
    expected_bar_count: int
    started_bar_count: int
    completed_bar_count: int
    enqueued_block_count: int
    queue_depth_max: int
    queue_underrun_count: int
    # How blocks reached the scheduler. With "prebuilt_deterministic_dict" every block
    # already exists before the run, so queue_underrun_count, queue_depth_max and
    # enqueue_lead_time_ms are fixed by construction and are not real-time measurements.
    # They only become informative once a producer supplies blocks during the run.
    block_production_mode: str
    fixture_id: str
    dense_off_gap_ms: float
    events_per_bar: int
    expected_event_count: int
    sent_event_count: int
    captured_event_count: int
    event_loss_count: int
    duplicate_output_count: int
    order_mismatch_count: int
    unmatched_note_off_count: int
    stuck_note_count: int
    send_failure_count: int
    deadline_threshold_ms: float
    spin_window_ms: float
    scheduler_dispatch_deadline_miss_count: int
    scheduler_dispatch_deadline_miss_rate: float | None
    primary_dispatch_stall_count: int
    catch_up_dispatch_event_count: int
    scheduler_dispatch_deadline_miss_type_counts: dict[str, int]
    scheduler_dispatch_deadline_misses: tuple[SchedulerDispatchDeadlineMiss, ...]
    scheduler_dispatch_deadline_miss_lateness_ms: LatencySummary
    scheduler_dispatch_attempt_lateness_ms: LatencySummary
    scheduler_dispatch_attempt_lateness_p999_ms: float | None
    capture_deadline_miss_count: int | None
    scheduler_dispatch_lateness_ms: LatencySummary
    output_send_call_duration_ms: LatencySummary
    dispatch_to_capture_latency_ms: LatencySummary
    sender_to_capture_timing_error_ms: LatencySummary
    bar_start_capture_error_ms: LatencySummary
    enqueue_lead_time_ms: LatencySummary
    capture_timing_histogram: dict[str, int]
    watchdog_trigger_reason: str | None
    safe_reset_sent: bool
    output_capture_observed: bool
    run_completed: bool
    record_and_continue_completed: bool
    wall_clock_soak_completed: bool
    passed_event_integrity: bool
    passed_scheduler_smoke: bool
    passed_r1_internal_scheduler_gate: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def wait_until_ns(
    target_ns: int,
    stop: Event,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
    spin_window_ns: int = round(DEFAULT_SPIN_WINDOW_MS * 1_000_000),
) -> None:
    if spin_window_ns < 0:
        raise ValueError("spin_window_ns must not be negative")
    while not stop.is_set():
        delay_ns = target_ns - clock_ns()
        if delay_ns <= 0:
            return
        if delay_ns > spin_window_ns:
            stop.wait(timeout=(delay_ns - spin_window_ns) / 1_000_000_000)
            continue
        while not stop.is_set() and clock_ns() < target_ns:
            pass
        return


def build_deterministic_blocks(
    *,
    clock: MonotonicBarClock,
    bar_count: int,
    dense_off_gap_ms: float = DEFAULT_DENSE_OFF_GAP_MS,
) -> dict[int, ScheduledMidiBlock]:
    """Build one balanced block per bar with simultaneous voices and sub-spin gaps.

    Beat 0 is a single note carrying the bar-start marker. The remaining beats cycle
    through chord voicings and a `dense_off_gap_ms` note-off so the fixture contains
    both same-target-time events and inter-event gaps narrower than the spin window.
    """
    if bar_count <= 0:
        raise ValueError("bar_count must be positive")
    if dense_off_gap_ms <= 0:
        raise ValueError("dense_off_gap_ms must be positive")

    dense_off_gap_ns = round(dense_off_gap_ms * 1_000_000)
    pitch_cycle = (60, 62, 64, 67, 69, 67, 64, 62)
    blocks: dict[int, ScheduledMidiBlock] = {}
    sequence_index = 0
    for bar_index in range(bar_count):
        events: list[ScheduledMidiEvent] = []
        bar_beat = bar_index * clock.beats_per_bar
        for beat_in_bar in range(clock.beats_per_bar):
            root = pitch_cycle[(bar_index + beat_in_bar) % len(pitch_cycle)]
            voice_count = BEAT_VOICE_COUNTS[beat_in_bar % len(BEAT_VOICE_COUNTS)]
            pitches = tuple(
                min(127, root + CHORD_SEMITONE_OFFSETS[voice]) for voice in range(voice_count)
            )
            note_on_target = clock.beat_target_ns(bar_beat + beat_in_bar)
            if beat_in_bar % len(BEAT_VOICE_COUNTS) in DENSE_OFF_BEATS:
                note_off_target = note_on_target + dense_off_gap_ns
            else:
                note_off_target = clock.beat_target_ns(bar_beat + beat_in_bar + 0.5)
            for voice, pitch in enumerate(pitches):
                events.append(
                    ScheduledMidiEvent(
                        sequence_index=sequence_index,
                        bar_index=bar_index,
                        target_ns=note_on_target,
                        message=Message(
                            "note_on", channel=0, note=pitch, velocity=84 - voice * 4
                        ),
                        is_bar_start=beat_in_bar == 0 and voice == 0,
                    )
                )
                sequence_index += 1
            for voice, pitch in enumerate(pitches):
                events.append(
                    ScheduledMidiEvent(
                        sequence_index=sequence_index,
                        bar_index=bar_index,
                        target_ns=note_off_target,
                        message=Message("note_off", channel=0, note=pitch, velocity=0),
                    )
                )
                sequence_index += 1
        events.sort(key=lambda event: (event.target_ns, event.sequence_index))
        blocks[bar_index] = ScheduledMidiBlock(
            bar_index=bar_index,
            target_start_ns=clock.bar_start_ns(bar_index),
            events=tuple(events),
        )
    return blocks


class OneBarMidiScheduler:
    """Consume one prepared block per bar and enqueue the next block one bar ahead."""

    def __init__(
        self,
        *,
        sink: MidiEventSink,
        clock: MonotonicBarClock,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        wait_until: Callable[[int, Event], None] | None = None,
        spin_window_ms: float = DEFAULT_SPIN_WINDOW_MS,
        deadline_threshold_ms: float = DEFAULT_DEADLINE_THRESHOLD_MS,
        deadline_policy: str = DEADLINE_POLICY_ABORT_ON_FIRST_MISS,
    ) -> None:
        if spin_window_ms < 0:
            raise ValueError("spin_window_ms must not be negative")
        if deadline_threshold_ms <= 0:
            raise ValueError("deadline_threshold_ms must be positive")
        if deadline_policy not in DEADLINE_POLICIES:
            raise ValueError(f"unsupported deadline_policy: {deadline_policy!r}")
        self._sink = sink
        self._clock = clock
        self._clock_ns = clock_ns
        self._stop = Event()
        self._deadline_threshold_ns = round(deadline_threshold_ms * 1_000_000)
        self._deadline_policy = deadline_policy
        self._watchdog_trigger_reason: str | None = None
        self._wait_until = wait_until or (
            lambda target_ns, stop: wait_until_ns(
                target_ns,
                stop,
                clock_ns=self._clock_ns,
                spin_window_ns=round(spin_window_ms * 1_000_000),
            )
        )

    def stop(self, reason: str = "external_stop") -> None:
        if self._watchdog_trigger_reason is None:
            self._watchdog_trigger_reason = reason
        self._stop.set()

    def run(
        self,
        *,
        blocks: Mapping[int, ScheduledMidiBlock],
        expected_bar_count: int,
    ) -> SchedulerRunResult:
        if expected_bar_count <= 0:
            raise ValueError("expected_bar_count must be positive")

        queue: dict[int, ScheduledMidiBlock] = {}
        records: list[SchedulerSendRecord] = []
        enqueue_lead_time_ns: list[int] = []
        queue_depth_max = 0
        queue_underrun_count = 0
        enqueued_block_count = 0
        started_bar_count = 0
        completed_bar_count = 0
        send_failure_count = 0
        scheduler_dispatch_deadline_miss_count = 0
        primary_dispatch_stall_count = 0
        catch_up_dispatch_event_count = 0
        scheduler_dispatch_deadline_misses: list[SchedulerDispatchDeadlineMiss] = []
        dispatch_attempt_lateness_ns: list[int] = []
        catching_up_after_stall = False

        initial_block = blocks.get(0)
        if initial_block is not None:
            queue[0] = initial_block
            enqueued_block_count += 1
            queue_depth_max = 1

        for bar_index in range(expected_bar_count):
            target_bar_start_ns = self._clock.bar_start_ns(bar_index)
            self._wait_until(target_bar_start_ns, self._stop)
            if self._stop.is_set():
                break

            block = queue.pop(bar_index, None)
            next_bar_index = bar_index + 1
            if next_bar_index < expected_bar_count:
                next_block = blocks.get(next_bar_index)
                if next_block is not None:
                    enqueued_ns = self._clock_ns()
                    queue[next_bar_index] = next_block
                    enqueued_block_count += 1
                    enqueue_lead_time_ns.append(
                        max(0, next_block.target_start_ns - enqueued_ns)
                    )
                    queue_depth_max = max(queue_depth_max, len(queue))

            if block is None:
                queue_underrun_count += 1
                self.stop("queue_underrun")
                break

            if block.bar_index != bar_index or block.target_start_ns != target_bar_start_ns:
                raise ValueError(f"block {bar_index} does not match clock target")

            started_bar_count += 1
            block_completed = True
            for event in block.events:
                self._wait_until(event.target_ns, self._stop)
                if self._stop.is_set():
                    block_completed = False
                    break
                dispatch_started_ns = self._clock_ns()
                dispatch_lateness_ns = max(0, dispatch_started_ns - event.target_ns)
                dispatch_attempt_lateness_ns.append(dispatch_lateness_ns)
                if dispatch_lateness_ns > self._deadline_threshold_ns:
                    scheduler_dispatch_deadline_miss_count += 1
                    is_catch_up = catching_up_after_stall
                    if is_catch_up:
                        catch_up_dispatch_event_count += 1
                    else:
                        primary_dispatch_stall_count += 1
                    action = (
                        "abort_before_send"
                        if self._deadline_policy == DEADLINE_POLICY_ABORT_ON_FIRST_MISS
                        else "sent_late_for_diagnostic"
                    )
                    scheduler_dispatch_deadline_misses.append(
                        SchedulerDispatchDeadlineMiss(
                            sequence_index=event.sequence_index,
                            bar_index=event.bar_index,
                            target_ns=event.target_ns,
                            dispatch_started_ns=dispatch_started_ns,
                            lateness_ns=dispatch_lateness_ns,
                            message_type=event.message.type,
                            channel=getattr(event.message, "channel", None),
                            note=getattr(event.message, "note", None),
                            is_bar_start=event.is_bar_start,
                            is_catch_up=is_catch_up,
                            action=action,
                        )
                    )
                    if self._deadline_policy == DEADLINE_POLICY_ABORT_ON_FIRST_MISS:
                        block_completed = False
                        self.stop("dispatch_deadline_miss")
                        break
                    catching_up_after_stall = True
                else:
                    catching_up_after_stall = False
                try:
                    self._sink.send(event.message.copy())
                except Exception:
                    send_failure_count += 1
                    block_completed = False
                    self.stop("send_failure")
                    break
                records.append(
                    SchedulerSendRecord(
                        sequence_index=event.sequence_index,
                        bar_index=event.bar_index,
                        target_ns=event.target_ns,
                        dispatch_started_ns=dispatch_started_ns,
                        send_accepted_ns=self._clock_ns(),
                        message=event.message.copy(),
                        is_bar_start=event.is_bar_start,
                    )
                )
            if block_completed:
                completed_bar_count += 1
            if self._stop.is_set():
                break

        if not self._stop.is_set():
            self._wait_until(self._clock.bar_start_ns(expected_bar_count), self._stop)

        run_completed = not self._stop.is_set() and completed_bar_count == expected_bar_count
        return SchedulerRunResult(
            deadline_policy=self._deadline_policy,
            expected_bar_count=expected_bar_count,
            started_bar_count=started_bar_count,
            completed_bar_count=completed_bar_count,
            enqueued_block_count=enqueued_block_count,
            queue_underrun_count=queue_underrun_count,
            queue_depth_max=queue_depth_max,
            send_failure_count=send_failure_count,
            scheduler_dispatch_deadline_miss_count=scheduler_dispatch_deadline_miss_count,
            primary_dispatch_stall_count=primary_dispatch_stall_count,
            catch_up_dispatch_event_count=catch_up_dispatch_event_count,
            scheduler_dispatch_deadline_misses=tuple(scheduler_dispatch_deadline_misses),
            watchdog_trigger_reason=self._watchdog_trigger_reason,
            run_completed=run_completed,
            enqueue_lead_time_ns=tuple(enqueue_lead_time_ns),
            dispatch_attempt_lateness_ns=tuple(dispatch_attempt_lateness_ns),
            records=tuple(records),
        )


def timing_histogram(
    values_ns: Sequence[int],
    bounds_ms: Sequence[float] = TIMING_HISTOGRAM_BOUNDS_MS,
) -> dict[str, int]:
    histogram: dict[str, int] = {}
    for bound_ms in bounds_ms:
        bound_ns = round(bound_ms * 1_000_000)
        histogram[f"le_{bound_ms:g}ms"] = sum(value <= bound_ns for value in values_ns)
    final_bound_ns = round(bounds_ms[-1] * 1_000_000)
    histogram[f"gt_{bounds_ms[-1]:g}ms"] = sum(
        value > final_bound_ns for value in values_ns
    )
    return histogram
