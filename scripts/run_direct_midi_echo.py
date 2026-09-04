#!/usr/bin/env python3
"""Forward a real MIDI input port to an output port without model inference."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from threading import Event
from typing import Sequence

import mido
from mido import Message


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.realtime.transport import (  # noqa: E402
    DirectMidiEcho,
    MidiSendAcceptanceReport,
    close_mido_input,
)


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "direct_midi_echo"


class MidiPortError(RuntimeError):
    pass


class MidoPortSink:
    def __init__(self, port: object) -> None:
        self._port = port

    def send(self, message: Message) -> None:
        self._port.send(message)

    def reset(self) -> None:
        self._port.reset()


def available_ports() -> dict[str, list[str]]:
    try:
        return {
            "inputs": list(mido.get_input_names()),
            "outputs": list(mido.get_output_names()),
        }
    except ModuleNotFoundError as exc:
        raise MidiPortError(
            "MIDI backend unavailable; install requirements.txt including python-rtmidi"
        ) from exc


def require_port(name: str, available: Sequence[str], kind: str) -> None:
    if name not in available:
        raise MidiPortError(f"{kind} port not found: {name!r}; available={list(available)!r}")


def write_report(path: Path, report: MidiSendAcceptanceReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_port_echo(
    *,
    input_port_name: str,
    output_port_name: str,
    duration_seconds: float,
) -> MidiSendAcceptanceReport:
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    ports = available_ports()
    require_port(input_port_name, ports["inputs"], "input")
    require_port(output_port_name, ports["outputs"], "output")
    if input_port_name == output_port_name:
        raise MidiPortError("input and output port names must differ to prevent a feedback loop")

    stopped = Event()
    callback_errors: list[Exception] = []
    started = time.perf_counter()
    safe_reset_sent = False
    with mido.open_output(output_port_name) as output_port:
        sink = MidoPortSink(output_port)
        echo = DirectMidiEcho(sink)

        def callback(message: Message) -> None:
            try:
                echo.process(message)
            except Exception as exc:
                callback_errors.append(exc)
                stopped.set()

        try:
            input_port = mido.open_input(input_port_name, callback=callback)
            try:
                stopped.wait(timeout=duration_seconds)
            finally:
                close_mido_input(input_port)
        finally:
            sink.reset()
            safe_reset_sent = True

    elapsed = time.perf_counter() - started
    if callback_errors:
        raise MidiPortError(f"MIDI callback failed: {callback_errors[0]}") from callback_errors[0]
    return echo.send_acceptance_report(
        wall_clock_seconds=elapsed,
        scope="mido_output_send_acceptance",
        safe_reset_sent=safe_reset_sent,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list_ports", action="store_true")
    parser.add_argument("--input_port")
    parser.add_argument("--output_port")
    parser.add_argument("--duration_seconds", type=float, default=600.0)
    parser.add_argument("--run_id", default="manual_port_echo")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        ports = available_ports()
        if args.list_ports:
            print(json.dumps(ports, indent=2, sort_keys=True))
            return 0
        if not args.input_port or not args.output_port:
            raise MidiPortError("--input_port and --output_port are required unless --list_ports is used")
        report = run_port_echo(
            input_port_name=args.input_port,
            output_port_name=args.output_port,
            duration_seconds=args.duration_seconds,
        )
    except (MidiPortError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2

    report_path = args.output_root / args.run_id / "direct_midi_send_acceptance_report.json"
    write_report(report_path, report)
    print(json.dumps({"report_path": str(report_path), **report.to_dict()}, sort_keys=True))
    return 0 if report.passed_send_acceptance else 1


if __name__ == "__main__":
    raise SystemExit(main())
