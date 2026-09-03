#!/usr/bin/env python3
"""Run a fast logical-duration direct MIDI echo integrity probe."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from inference.realtime.transport import (  # noqa: E402
    DirectMidiEcho,
    EchoIntegrityReport,
    RecordingMidiSink,
    build_logical_midi_fixture,
)


DEFAULT_OUTPUT_ROOT = ROOT_DIR / "outputs" / "direct_midi_echo"


def write_report(path: Path, report: EchoIntegrityReport) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_probe(*, logical_duration_seconds: float, bpm: float) -> EchoIntegrityReport:
    fixture = build_logical_midi_fixture(
        logical_duration_seconds=logical_duration_seconds,
        bpm=bpm,
    )
    sink = RecordingMidiSink()
    echo = DirectMidiEcho(sink)
    started = time.perf_counter()
    for event in fixture:
        echo.process(event.message)
    elapsed = time.perf_counter() - started
    return echo.report(
        logical_duration_seconds=logical_duration_seconds,
        wall_clock_seconds=elapsed,
        scope="in_memory_logical_fixture",
        output_messages=sink.messages,
    )


def exit_code_for_report(report: EchoIntegrityReport) -> int:
    return 0 if report.passed_event_integrity else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_id", default="manual")
    parser.add_argument("--logical_duration_seconds", type=float, default=600.0)
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_probe(
        logical_duration_seconds=args.logical_duration_seconds,
        bpm=args.bpm,
    )
    report_path = args.output_root / args.run_id / "direct_midi_echo_report.json"
    write_report(report_path, report)
    print(json.dumps({"report_path": str(report_path), **report.to_dict()}, sort_keys=True))
    return exit_code_for_report(report)


if __name__ == "__main__":
    raise SystemExit(main())
