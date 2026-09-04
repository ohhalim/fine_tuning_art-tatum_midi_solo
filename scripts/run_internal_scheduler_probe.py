#!/usr/bin/env python3
"""Run an internal-clock one-bar scheduler through independent CoreMIDI capture."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from threading import Event, Lock
from typing import Mapping, Sequence

import mido
from mido import Message


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.realtime.scheduler import (  # noqa: E402
    DEFAULT_DENSE_OFF_GAP_MS,
    DEFAULT_DEADLINE_THRESHOLD_MS,
    DEFAULT_SPIN_WINDOW_MS,
    DETERMINISTIC_FIXTURE_ID,
    INTERNAL_SCHEDULER_REPORT_SCHEMA_VERSION,
    InternalSchedulerReport,
    MonotonicBarClock,
    OneBarMidiScheduler,
    ScheduledMidiBlock,
    SchedulerRunResult,
    build_deterministic_blocks,
    deterministic_events_per_bar,
    timing_histogram,
)
from inference.realtime.transport import (  # noqa: E402
    canonical_message,
    evaluate_event_integrity,
    summarize_latency,
)


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "internal_midi_scheduler"


class InternalSchedulerProbeError(RuntimeError):
    pass


class PortSink:
    def __init__(self, port: object) -> None:
        self._port = port

    def send(self, message: Message) -> None:
        self._port.send(message)


def _safe_port_suffix(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", run_id).strip("-")
    return (cleaned or "scheduler")[:40]


def _wait_for_input_port(name: str, timeout_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if name in mido.get_input_names():
            return
        time.sleep(0.01)
    raise InternalSchedulerProbeError(f"virtual MIDI source did not appear: {name!r}")


def _bar_count_for_duration(
    *,
    bpm: float,
    beats_per_bar: int,
    requested_duration_seconds: float,
) -> int:
    if requested_duration_seconds <= 0:
        raise ValueError("requested_duration_seconds must be positive")
    bar_duration_seconds = beats_per_bar * 60.0 / bpm
    return max(1, math.ceil(requested_duration_seconds / bar_duration_seconds - 1e-12))


def _flatten_expected_messages(blocks: Mapping[int, ScheduledMidiBlock]) -> list[Message]:
    messages: list[Message] = []
    for bar_index in sorted(blocks):
        block = blocks[bar_index]
        messages.extend(event.message.copy() for event in block.events)
    return messages


def build_report(
    *,
    bpm: float,
    beats_per_bar: int,
    requested_duration_seconds: float,
    scheduled_duration_seconds: float,
    start_delay_seconds: float,
    wall_clock_seconds: float,
    expected_messages: Sequence[Message],
    run_result: SchedulerRunResult,
    captured_messages: Sequence[Message],
    captured_ns: Sequence[int],
    safe_reset_sent: bool,
    deadline_threshold_ms: float,
    spin_window_ms: float,
    dense_off_gap_ms: float = DEFAULT_DENSE_OFF_GAP_MS,
    block_production_mode: str = "prebuilt_deterministic_dict",
) -> InternalSchedulerReport:
    records = list(run_result.records)
    sent_messages = [record.message for record in records]
    sent_keys = [canonical_message(message) for message in sent_messages]
    captured_keys = [canonical_message(message) for message in captured_messages]
    capture_timing_error_ns: list[int] = []
    dispatch_to_capture_latency_ns: list[int] = []
    bar_start_capture_error_ns: list[int] = []
    if sent_keys == captured_keys and len(records) == len(captured_ns):
        capture_timing_error_ns = [
            max(0, int(captured_ns[index]) - record.target_ns)
            for index, record in enumerate(records)
        ]
        dispatch_to_capture_latency_ns = [
            max(0, int(captured_ns[index]) - record.dispatch_started_ns)
            for index, record in enumerate(records)
        ]
        bar_start_capture_error_ns = [
            capture_timing_error_ns[index]
            for index, record in enumerate(records)
            if record.is_bar_start
        ]

    dispatch_lateness_ns = [
        max(0, record.dispatch_started_ns - record.target_ns) for record in records
    ]
    output_send_call_duration_ns = [
        max(0, record.send_accepted_ns - record.dispatch_started_ns) for record in records
    ]
    threshold_ns = round(deadline_threshold_ms * 1_000_000)
    scheduler_dispatch_deadline_miss_count = run_result.scheduler_dispatch_deadline_miss_count
    capture_deadline_miss_count = (
        sum(latency_ns > threshold_ns for latency_ns in capture_timing_error_ns)
        if len(capture_timing_error_ns) == len(expected_messages)
        else None
    )
    dispatch_deadline_miss_lateness_ns = [
        miss.lateness_ns for miss in run_result.scheduler_dispatch_deadline_misses
    ]

    integrity = evaluate_event_integrity(
        input_messages=expected_messages,
        output_messages=captured_messages,
        callback_latency_ns=dispatch_lateness_ns,
        logical_duration_seconds=scheduled_duration_seconds,
        wall_clock_seconds=wall_clock_seconds,
        scope="internal_scheduler_coremidi_independent_capture",
        crash_count=run_result.send_failure_count,
        minimum_input_event_count=len(expected_messages),
        safe_reset_sent=safe_reset_sent,
        os_midi_loopback_observed=bool(captured_messages),
    )
    bar_start_summary = summarize_latency(bar_start_capture_error_ns)
    expected_bar_count = run_result.expected_bar_count
    bar_start_gate_passed = (
        bar_start_summary.sample_count == expected_bar_count
        and bar_start_summary.p99 is not None
        and bar_start_summary.p99 <= deadline_threshold_ms
    )
    block_gate_passed = all(
        (
            run_result.run_completed,
            run_result.started_bar_count == expected_bar_count,
            run_result.completed_bar_count == expected_bar_count,
            run_result.enqueued_block_count == expected_bar_count,
            run_result.queue_underrun_count == 0,
            run_result.send_failure_count == 0,
            run_result.scheduler_dispatch_deadline_miss_count == 0,
            run_result.watchdog_trigger_reason is None,
        )
    )
    passed_scheduler_smoke = all(
        (
            integrity.passed_event_integrity,
            block_gate_passed,
            bool(captured_messages),
            safe_reset_sent,
            scheduler_dispatch_deadline_miss_count == 0,
            capture_deadline_miss_count == 0,
            bar_start_gate_passed,
        )
    )
    wall_clock_soak_completed = all(
        (
            requested_duration_seconds >= 600.0,
            scheduled_duration_seconds >= 600.0,
            wall_clock_seconds >= scheduled_duration_seconds,
            run_result.run_completed,
        )
    )
    return InternalSchedulerReport(
        schema_version=INTERNAL_SCHEDULER_REPORT_SCHEMA_VERSION,
        scope="internal_scheduler_coremidi_independent_capture",
        bpm=float(bpm),
        beats_per_bar=int(beats_per_bar),
        requested_duration_seconds=float(requested_duration_seconds),
        scheduled_duration_seconds=float(scheduled_duration_seconds),
        start_delay_seconds=float(start_delay_seconds),
        wall_clock_seconds=float(wall_clock_seconds),
        expected_bar_count=expected_bar_count,
        started_bar_count=run_result.started_bar_count,
        completed_bar_count=run_result.completed_bar_count,
        enqueued_block_count=run_result.enqueued_block_count,
        queue_depth_max=run_result.queue_depth_max,
        queue_underrun_count=run_result.queue_underrun_count,
        block_production_mode=block_production_mode,
        fixture_id=DETERMINISTIC_FIXTURE_ID,
        dense_off_gap_ms=float(dense_off_gap_ms),
        events_per_bar=deterministic_events_per_bar(beats_per_bar),
        expected_event_count=len(expected_messages),
        sent_event_count=len(sent_messages),
        captured_event_count=len(captured_messages),
        event_loss_count=integrity.event_loss_count,
        duplicate_output_count=integrity.duplicate_output_count,
        order_mismatch_count=integrity.order_mismatch_count,
        unmatched_note_off_count=integrity.unmatched_note_off_count,
        stuck_note_count=integrity.stuck_note_count,
        send_failure_count=run_result.send_failure_count,
        deadline_threshold_ms=float(deadline_threshold_ms),
        spin_window_ms=float(spin_window_ms),
        scheduler_dispatch_deadline_miss_count=scheduler_dispatch_deadline_miss_count,
        scheduler_dispatch_deadline_misses=run_result.scheduler_dispatch_deadline_misses,
        scheduler_dispatch_deadline_miss_lateness_ms=summarize_latency(
            dispatch_deadline_miss_lateness_ns
        ),
        capture_deadline_miss_count=capture_deadline_miss_count,
        scheduler_dispatch_lateness_ms=summarize_latency(dispatch_lateness_ns),
        output_send_call_duration_ms=summarize_latency(output_send_call_duration_ns),
        dispatch_to_capture_latency_ms=summarize_latency(dispatch_to_capture_latency_ns),
        sender_to_capture_timing_error_ms=summarize_latency(capture_timing_error_ns),
        bar_start_capture_error_ms=bar_start_summary,
        enqueue_lead_time_ms=summarize_latency(run_result.enqueue_lead_time_ns),
        capture_timing_histogram=timing_histogram(capture_timing_error_ns),
        watchdog_trigger_reason=run_result.watchdog_trigger_reason,
        safe_reset_sent=bool(safe_reset_sent),
        output_capture_observed=bool(captured_messages),
        run_completed=run_result.run_completed,
        wall_clock_soak_completed=wall_clock_soak_completed,
        passed_event_integrity=integrity.passed_event_integrity,
        passed_scheduler_smoke=passed_scheduler_smoke,
        passed_r1_internal_scheduler_gate=bool(
            passed_scheduler_smoke and wall_clock_soak_completed
        ),
    )


def run_internal_scheduler_probe(
    *,
    run_id: str,
    bpm: float,
    requested_duration_seconds: float,
    beats_per_bar: int = 4,
    start_delay_seconds: float = 0.25,
    deadline_threshold_ms: float = DEFAULT_DEADLINE_THRESHOLD_MS,
    spin_window_ms: float = DEFAULT_SPIN_WINDOW_MS,
    dense_off_gap_ms: float = DEFAULT_DENSE_OFF_GAP_MS,
    drain_timeout_seconds: float = 3.0,
    drop_block_index: int | None = None,
) -> InternalSchedulerReport:
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if beats_per_bar <= 0:
        raise ValueError("beats_per_bar must be positive")
    if start_delay_seconds < 0:
        raise ValueError("start_delay_seconds must not be negative")
    if deadline_threshold_ms <= 0:
        raise ValueError("deadline_threshold_ms must be positive")
    if spin_window_ms < 0:
        raise ValueError("spin_window_ms must not be negative")
    if dense_off_gap_ms <= 0:
        raise ValueError("dense_off_gap_ms must be positive")

    expected_bar_count = _bar_count_for_duration(
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        requested_duration_seconds=requested_duration_seconds,
    )
    scheduled_duration_seconds = expected_bar_count * beats_per_bar * 60.0 / bpm
    suffix = f"{_safe_port_suffix(run_id)}-{os.getpid()}"
    output_source_name = f"JazzImprov-Scheduler-{suffix}"
    captured_messages: list[Message] = []
    captured_ns: list[int] = []
    capture_lock = Lock()
    capture_complete = Event()
    expected_event_count = expected_bar_count * deterministic_events_per_bar(beats_per_bar)

    def capture_callback(message: Message) -> None:
        with capture_lock:
            captured_messages.append(message.copy())
            captured_ns.append(time.perf_counter_ns())
            if len(captured_messages) >= expected_event_count:
                capture_complete.set()

    safe_reset_sent = False
    started = time.perf_counter()
    try:
        with mido.open_output(output_source_name, virtual=True) as output_port:
            try:
                _wait_for_input_port(output_source_name)
                clock = MonotonicBarClock(
                    bpm=bpm,
                    beats_per_bar=beats_per_bar,
                    start_ns=time.perf_counter_ns() + round(start_delay_seconds * 1_000_000_000),
                )
                expected_blocks = build_deterministic_blocks(
                    clock=clock,
                    bar_count=expected_bar_count,
                    dense_off_gap_ms=dense_off_gap_ms,
                )
                blocks = dict(expected_blocks)
                if drop_block_index is not None:
                    if drop_block_index not in blocks:
                        raise ValueError(
                            f"drop_block_index must be between 0 and {expected_bar_count - 1}"
                        )
                    del blocks[drop_block_index]
                scheduler = OneBarMidiScheduler(
                    sink=PortSink(output_port),
                    clock=clock,
                    spin_window_ms=spin_window_ms,
                    deadline_threshold_ms=deadline_threshold_ms,
                )
                with mido.open_input(output_source_name, callback=capture_callback):
                    run_result = scheduler.run(
                        blocks=blocks,
                        expected_bar_count=expected_bar_count,
                    )
                    if (
                        run_result.watchdog_trigger_reason is None
                        and not capture_complete.is_set()
                        and run_result.records
                    ):
                        capture_complete.wait(timeout=drain_timeout_seconds)
            finally:
                output_port.reset()
                safe_reset_sent = True
    except (ImportError, OSError, RuntimeError) as exc:
        raise InternalSchedulerProbeError(f"internal scheduler probe failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    with capture_lock:
        captured_messages_copy = list(captured_messages)
        captured_ns_copy = list(captured_ns)
    expected_messages = _flatten_expected_messages(expected_blocks)
    return build_report(
        bpm=bpm,
        beats_per_bar=beats_per_bar,
        requested_duration_seconds=requested_duration_seconds,
        scheduled_duration_seconds=scheduled_duration_seconds,
        start_delay_seconds=start_delay_seconds,
        wall_clock_seconds=elapsed,
        expected_messages=expected_messages,
        run_result=run_result,
        captured_messages=captured_messages_copy,
        captured_ns=captured_ns_copy,
        safe_reset_sent=safe_reset_sent,
        deadline_threshold_ms=deadline_threshold_ms,
        spin_window_ms=spin_window_ms,
        dense_off_gap_ms=dense_off_gap_ms,
    )


def write_report(path: Path, report: InternalSchedulerReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exit_code_for_report(report: InternalSchedulerReport, *, require_r1_gate: bool) -> int:
    passed = (
        report.passed_r1_internal_scheduler_gate
        if require_r1_gate
        else report.passed_scheduler_smoke
    )
    return 0 if passed else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="manual_internal_scheduler")
    parser.add_argument("--bpm", type=float, required=True)
    parser.add_argument("--duration_seconds", type=float, default=8.0)
    parser.add_argument("--beats_per_bar", type=int, default=4)
    parser.add_argument("--start_delay_seconds", type=float, default=0.25)
    parser.add_argument("--deadline_threshold_ms", type=float, default=DEFAULT_DEADLINE_THRESHOLD_MS)
    parser.add_argument("--spin_window_ms", type=float, default=DEFAULT_SPIN_WINDOW_MS)
    parser.add_argument("--dense_off_gap_ms", type=float, default=DEFAULT_DENSE_OFF_GAP_MS)
    parser.add_argument("--drain_timeout_seconds", type=float, default=3.0)
    parser.add_argument("--drop_block_index", type=int)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_internal_scheduler_probe(
            run_id=args.run_id,
            bpm=args.bpm,
            requested_duration_seconds=args.duration_seconds,
            beats_per_bar=args.beats_per_bar,
            start_delay_seconds=args.start_delay_seconds,
            deadline_threshold_ms=args.deadline_threshold_ms,
            spin_window_ms=args.spin_window_ms,
            dense_off_gap_ms=args.dense_off_gap_ms,
            drain_timeout_seconds=args.drain_timeout_seconds,
            drop_block_index=args.drop_block_index,
        )
    except (InternalSchedulerProbeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report_path = args.output_root / args.run_id / "internal_scheduler_report.json"
    write_report(report_path, report)
    print(json.dumps({"report_path": str(report_path), **report.to_dict()}, sort_keys=True))
    return exit_code_for_report(report, require_r1_gate=args.duration_seconds >= 600.0)


if __name__ == "__main__":
    raise SystemExit(main())
