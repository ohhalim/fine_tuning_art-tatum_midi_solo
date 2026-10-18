#!/usr/bin/env python3
"""Play a fixed left-hand note list to a virtual MIDI output port on the session clock (issue #1797).

Opens the virtual port, writes the ready file, waits for the producer's origin file (music_origin_ms t0, wall_origin_ns W),
then sends each note-on/off at W + (offset_ms + note time) ms. Logs what it sent with monotonic ns.

  rt_lh_feeder.py <port> <notes json [[pitch, on_ms, off_ms, velocity], ...]> <origin json> <ready file> <log json> [offset_ms]
"""
from __future__ import annotations

import json
import os
import sys
import time


def events(notes, offset_ms):
    """Time-ordered (ms from the music origin, kind, pitch, velocity); at the same time offs go first."""
    ev = []
    for pitch, on, off, vel in notes:
        ev.append((offset_ms + on, 1, "note_on", pitch, vel))
        ev.append((offset_ms + off, 0, "note_off", pitch, 0))
    return [(t, kind, p, v) for t, _, kind, p, v in sorted(ev)]


def main() -> None:
    import mido
    port_name, notes_path, origin_path, ready_path, log_path = sys.argv[1:6]
    offset_ms = float(sys.argv[6]) if len(sys.argv) > 6 else 937.5
    notes = json.load(open(notes_path))
    port = mido.open_output(port_name, virtual=True)
    open(ready_path, "w").close()
    log = {"port": port_name, "offset_ms": offset_ms, "sent": [], "error": None}
    try:
        t_wait = time.monotonic()
        while not os.path.exists(origin_path):
            if time.monotonic() - t_wait > 300:
                raise TimeoutError("no origin file")
            time.sleep(0.01)
        time.sleep(0.05)
        origin = json.load(open(origin_path))
        W = origin["wall_origin_ns"]
        for t_ms, kind, pitch, vel in events(notes, offset_ms):
            due = W + int(t_ms * 1e6)
            while time.monotonic_ns() < due:
                time.sleep(max(0.0, min(0.001, (due - time.monotonic_ns()) / 1e9)))
            port.send(mido.Message(kind, note=pitch, velocity=vel))
            log["sent"].append({"t_ms": t_ms, "kind": kind, "pitch": pitch, "velocity": vel, "due_ns": due, "send_ns": time.monotonic_ns()})
        time.sleep(0.5)
    except Exception as e:
        log["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        port.close()
        with open(log_path, "w") as f:
            json.dump(log, f)


if __name__ == "__main__":
    main()
