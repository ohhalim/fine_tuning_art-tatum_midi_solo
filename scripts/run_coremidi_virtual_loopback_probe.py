#!/usr/bin/env python3
"""Measure sender-to-capture integrity through two CoreMIDI virtual sources."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from threading import Event, Lock
from typing import Sequence

import mido
from mido import Message


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.realtime.transport import (  # noqa: E402
    DirectMidiEcho,
    EchoIntegrityReport,
    canonical_message,
    evaluate_event_integrity,
)


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "direct_midi_echo"


class CoreMidiProbeError(RuntimeError):
    pass


class PortSink:
    def __init__(self, port: object) -> None:
        self._port = port

    def send(self, message: Message) -> None:
        self._port.send(message)


def _safe_port_suffix(run_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "-", run_id).strip("-")
    return (cleaned or "probe")[:40]


def _wait_for_input_port(name: str, timeout_seconds: float = 2.0) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if name in mido.get_input_names():
            return
        time.sleep(0.01)
    raise CoreMidiProbeError(f"virtual MIDI source did not appear as an input: {name!r}")


def _paced_message(index: int) -> Message:
    pitches = (60, 62, 64, 65, 67, 69, 71, 72)
    group_index, position = divmod(index, 4)
    pitch = pitches[(group_index // 2) % len(pitches)]
    variant = group_index % 5
    if variant == 0:
        messages = (
            Message("note_on", channel=0, note=pitch, velocity=84),
            Message("note_on", channel=0, note=min(127, pitch + 4), velocity=78),
            Message("note_off", channel=0, note=pitch, velocity=0),
            Message("note_off", channel=0, note=min(127, pitch + 4), velocity=0),
        )
    elif variant == 1:
        messages = (
            Message("note_on", channel=0, note=pitch, velocity=88),
            Message("note_off", channel=0, note=pitch, velocity=0),
            Message("note_on", channel=0, note=pitch, velocity=76),
            Message("note_off", channel=0, note=pitch, velocity=0),
        )
    elif variant == 2:
        messages = (
            Message("note_on", channel=0, note=pitch, velocity=82),
            Message("control_change", channel=0, control=123, value=0),
            Message("note_off", channel=0, note=pitch, velocity=0),
            Message("control_change", channel=0, control=1, value=group_index % 128),
        )
    elif variant == 3:
        messages = (
            Message("note_on", channel=0, note=pitch, velocity=80),
            Message("control_change", channel=0, control=120, value=0),
            Message("note_off", channel=0, note=pitch, velocity=0),
            Message("control_change", channel=0, control=11, value=96),
        )
    else:
        messages = (
            Message("program_change", channel=0, program=group_index % 128),
            Message("note_on", channel=0, note=pitch, velocity=80),
            Message("note_on", channel=0, note=pitch, velocity=0),
            Message("control_change", channel=0, control=64, value=0),
        )
    return messages[position]


def _target_offset_seconds(index: int, rate_hz: float) -> float:
    group_index, position = divmod(index, 4)
    group_start = group_index * 4.0 / rate_hz
    if group_index % 5 == 0:
        return group_start + (0.0 if position < 2 else 2.0 / rate_hz)
    return group_start + position / rate_hz


def _wait_until(target_time: float, stop: Event) -> None:
    delay = target_time - time.perf_counter()
    if delay > 0:
        stop.wait(timeout=delay)


def run_coremidi_virtual_loopback(
    *,
    run_id: str,
    duration_seconds: float,
    rate_hz: float,
    drain_timeout_seconds: float = 3.0,
) -> EchoIntegrityReport:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    if rate_hz <= 0:
        raise ValueError("rate_hz must be positive")

    event_count = int(duration_seconds * rate_hz)
    event_count -= event_count % 4
    if event_count < 4:
        raise ValueError("duration_seconds * rate_hz must produce at least four balanced events")

    suffix = f"{_safe_port_suffix(run_id)}-{os.getpid()}"
    sender_source_name = f"JazzImprov-Sender-{suffix}"
    echo_source_name = f"JazzImprov-Echo-{suffix}"
    captured: list[Message] = []
    captured_ns: list[int] = []
    callback_errors: list[Exception] = []
    capture_lock = Lock()
    capture_complete = Event()
    failure_stop = Event()

    def capture_callback(message: Message) -> None:
        with capture_lock:
            captured.append(message.copy())
            captured_ns.append(time.perf_counter_ns())
            if len(captured) >= event_count:
                capture_complete.set()

    sent: list[Message] = []
    sent_ns: list[int] = []
    safe_reset_sent = False
    started = time.perf_counter()
    try:
        with mido.open_output(sender_source_name, virtual=True) as sender_port:
            with mido.open_output(echo_source_name, virtual=True) as echo_output_port:
                try:
                    _wait_for_input_port(sender_source_name)
                    _wait_for_input_port(echo_source_name)
                    echo = DirectMidiEcho(PortSink(echo_output_port))

                    def echo_callback(message: Message) -> None:
                        try:
                            echo.process(message)
                        except Exception as exc:
                            with capture_lock:
                                callback_errors.append(exc)
                            failure_stop.set()

                    with mido.open_input(echo_source_name, callback=capture_callback):
                        with mido.open_input(sender_source_name, callback=echo_callback):
                            time.sleep(0.1)
                            send_started = time.perf_counter()
                            for index in range(event_count):
                                _wait_until(
                                    send_started + _target_offset_seconds(index, rate_hz),
                                    failure_stop,
                                )
                                if failure_stop.is_set():
                                    break
                                message = _paced_message(index)
                                sent.append(message)
                                sent_ns.append(time.perf_counter_ns())
                                sender_port.send(message)
                            _wait_until(send_started + duration_seconds, failure_stop)
                            if not failure_stop.is_set() and not capture_complete.is_set():
                                capture_complete.wait(timeout=drain_timeout_seconds)
                finally:
                    echo_output_port.reset()
                    safe_reset_sent = True
    except (ImportError, OSError, RuntimeError) as exc:
        raise CoreMidiProbeError(f"CoreMIDI virtual loopback failed: {exc}") from exc

    elapsed = time.perf_counter() - started
    with capture_lock:
        captured_copy = list(captured)
        captured_ns_copy = list(captured_ns)
        callback_error_count = len(callback_errors)

    sent_keys = [canonical_message(message) for message in sent]
    captured_keys = [canonical_message(message) for message in captured_copy]
    capture_latency_ns: list[int] = []
    if sent_keys == captured_keys and len(sent_ns) == len(captured_ns_copy):
        capture_latency_ns = [
            max(0, captured_ns_copy[index] - sent_ns[index]) for index in range(len(sent_ns))
        ]

    completed_requested_duration = len(sent) == event_count and elapsed >= duration_seconds
    return evaluate_event_integrity(
        input_messages=sent,
        output_messages=captured_copy,
        callback_latency_ns=echo.callback_latency_ns,
        capture_latency_ns=capture_latency_ns,
        logical_duration_seconds=duration_seconds,
        wall_clock_seconds=elapsed,
        scope="coremidi_virtual_loopback_independent_capture",
        crash_count=echo.crash_count + callback_error_count,
        minimum_input_event_count=event_count,
        safe_reset_sent=safe_reset_sent,
        wall_clock_soak_completed=completed_requested_duration and duration_seconds >= 600.0,
        os_midi_loopback_observed=bool(captured_copy),
        fl_studio_audio_observed=False,
    )


def write_report(path: Path, report: EchoIntegrityReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exit_code_for_report(report: EchoIntegrityReport, *, require_r0_gate: bool) -> int:
    passed = report.passed_r0_transport_gate if require_r0_gate else report.passed_event_integrity
    return 0 if passed else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="manual_coremidi_loopback")
    parser.add_argument("--duration_seconds", type=float, default=2.0)
    parser.add_argument("--rate_hz", type=float, default=50.0)
    parser.add_argument("--drain_timeout_seconds", type=float, default=3.0)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_coremidi_virtual_loopback(
            run_id=args.run_id,
            duration_seconds=args.duration_seconds,
            rate_hz=args.rate_hz,
            drain_timeout_seconds=args.drain_timeout_seconds,
        )
    except (CoreMidiProbeError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report_path = args.output_root / args.run_id / "coremidi_virtual_loopback_report.json"
    write_report(report_path, report)
    print(json.dumps({"report_path": str(report_path), **report.to_dict()}, sort_keys=True))
    return exit_code_for_report(report, require_r0_gate=args.duration_seconds >= 600.0)


if __name__ == "__main__":
    raise SystemExit(main())
