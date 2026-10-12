#!/usr/bin/env python3
"""Independent MIDI capture for the real-clock smoke (issue #1747, docs/experiments/REALTIME_MIDI_SMOKE.md).

Waits for the sender's virtual output to appear as an input, opens it, writes a ready file, and records every message
with time.monotonic_ns() (system-wide on macOS, comparable with the other processes). Stops when the stop file
appears (written by the run script after the sender exits) or after the maximum duration.

  rt_capture.py <port name> <ready file> <stop file> <out json> <max seconds>
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time

import mido


def main() -> None:
    port_name, ready_file, stop_file, out_path, max_s = sys.argv[1:6]
    deadline = time.monotonic() + float(max_s)
    while port_name not in mido.get_input_names():
        if time.monotonic() > deadline:
            raise SystemExit(f"port {port_name!r} never appeared")
        time.sleep(0.01)
    got, lock = [], threading.Lock()

    def on_message(msg):
        t = time.monotonic_ns()
        with lock:
            got.append((t, msg.bytes()))

    port = mido.open_input(port_name, callback=on_message)
    with open(ready_file, "w") as f:
        f.write(str(time.monotonic_ns()))
    try:
        while not os.path.exists(stop_file) and time.monotonic() < deadline:
            time.sleep(0.02)
        time.sleep(0.2)                                   # let the last messages arrive
    finally:
        port.close()
    with lock:
        rows = [{"capture_ns": t, "bytes": b} for t, b in got]
    with open(out_path, "w") as f:
        json.dump({"port": port_name, "messages": rows, "stopped_by_file": os.path.exists(stop_file)}, f)


if __name__ == "__main__":
    main()
