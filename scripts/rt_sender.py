#!/usr/bin/env python3
"""Real-clock MIDI sender for the smoke (issue #1747, docs/experiments/REALTIME_MIDI_SMOKE.md).

Creates a virtual MIDI output, reads the producer's JSON lines from stdin on a reader thread, and runs the clock loop:
music ms -> time.monotonic_ns() via the producer's start message, Scheduler.tick(music now), and each new midi_out entry
is sent once. Whatever ends the run (end message, EOF when the producer dies, or an error), the finally path calls
Scheduler.stop() (pending cancelled, note-off for every owned pitch), sends those offs, and closes the port.

stdin lines: {"type": "start", "music_origin_ms", "wall_origin_ns"} | {"type": "notes", "block", "notes": [[id, pitch,
velocity, on_ms, off_ms], ...]} | {"type": "rollback", "block"} | {"type": "end"}

  rt_sender.py <port name> <capture ready file> <out json>
"""
from __future__ import annotations

import json
import os
import queue
import sys
import threading
import time

import mido

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_contract import Note, Scheduler, playable  # noqa: E402

READY_TIMEOUT_S, POLL_S = 10.0, 0.001


def main() -> None:
    port_name, ready_file, out_path = sys.argv[1:4]
    log = {"port": port_name, "sent": [], "received_blocks": [], "dropped_velocity0": 0, "extended_zero_length": 0,
           "end": None, "error": None}
    q = queue.Queue()

    def reader():
        for line in sys.stdin:
            line = line.strip()
            if line:
                q.put(json.loads(line))
        q.put({"type": "eof"})

    sch = Scheduler()
    port = mido.open_output(port_name, virtual=True)
    threading.Thread(target=reader, daemon=True).start()
    origin = None                      # (music_origin_ms, wall_origin_ns)
    sent_upto = 0
    ended = False

    def music_now():
        return origin[0] + (time.monotonic_ns() - origin[1]) / 1e6

    def scheduled_ns(ms):
        return int(origin[1] + (ms - origin[0]) * 1e6)

    def flush_midi():
        nonlocal sent_upto
        while sent_upto < len(sch.midi_out):
            ms, kind, pitch, vel = sch.midi_out[sent_upto]
            msg = mido.Message("note_on", note=pitch, velocity=vel) if kind == "on" else mido.Message("note_off", note=pitch, velocity=0)
            port.send(msg)
            due = sch.midi_due[sent_upto]                  # the event's scheduled music time (ms is the tick time)
            log["sent"].append({"kind": kind, "pitch": pitch, "velocity": vel, "tick_ms": ms, "due_ms": due,
                                "scheduled_ns": scheduled_ns(due) if origin else None, "send_ns": time.monotonic_ns()})
            sent_upto += 1

    try:
        t_wait = time.monotonic() + READY_TIMEOUT_S
        while not os.path.exists(ready_file):
            if time.monotonic() > t_wait:
                raise RuntimeError("capture never became ready")
            time.sleep(0.005)
        last_activity = time.monotonic()
        while True:
            try:
                while True:
                    m = q.get_nowait()
                    last_activity = time.monotonic()
                    if m["type"] == "start":
                        origin = (m["music_origin_ms"], m["wall_origin_ns"])
                    elif m["type"] == "notes":
                        notes = [Note(*n) for n in m["notes"]]
                        kept, dropped = playable(notes)
                        log["dropped_velocity0"] += len(dropped)
                        log["extended_zero_length"] += sum(1 for n in notes if n.velocity > 0 and n.off_ms <= n.on_ms)
                        sch.schedule(kept)
                        log["received_blocks"].append({"block": m["block"], "notes": len(notes), "recv_ns": time.monotonic_ns()})
                    elif m["type"] == "end" or (m["type"] == "eof" and log["end"] is None):
                        log["end"] = m["type"]     # an eof after a normal end is the producer exiting: ignored
                        ended = True
            except queue.Empty:
                pass
            if origin is not None:
                sch.tick(music_now())
                flush_midi()
            if ended and (not sch.pending or log["end"] == "eof"):
                break
            if ended is False and time.monotonic() - last_activity > 60:
                raise RuntimeError("no input for 60 s")
            time.sleep(POLL_S)
    except Exception as e:  # recorded; the finally path still turns everything off
        log["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        now = music_now() if origin else 0.0
        log["pending_at_stop"] = len(sch.pending)
        log["sounding_at_stop"] = len(sch.pitch_owner)
        sch.stop(now)
        flush_midi()
        port.close()
        log["port_closed"] = True
        log["logical_sent"] = sch.sent
        with open(out_path, "w") as f:
            json.dump(log, f)


if __name__ == "__main__":
    main()
