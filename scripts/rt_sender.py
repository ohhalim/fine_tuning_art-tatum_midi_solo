#!/usr/bin/env python3
"""Real-clock MIDI sender for the smoke (issue #1747, docs/experiments/REALTIME_MIDI_SMOKE.md).

Creates a virtual MIDI output, reads the producer's JSON lines from stdin on a reader thread, and runs the clock loop:
music ms -> time.monotonic_ns() via the producer's start message, Scheduler.tick(music now), and each new midi_out entry
is sent once. Whatever ends the run (end message, EOF when the producer dies, a broken input line, a send error), the
cleanup turns off what it can and always closes the port and writes the record (Astra review of #1747):
- an abnormal end (EOF without end, a broken input line) is acted on before the next tick, so a due note-on is not
  sent after it; a normal end still plays out what is scheduled
- every remaining note-off is tried once on its own; a failed send is recorded, not retried in a loop
- a note-on that was not sent before the stop is cancelled and recorded, never sent (no note starts after a stop)
- the port close and the record write are each guarded, so one failure does not skip the other
- if any note-off failed, the record says the stop was not clean (silence is not claimed)

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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stream_contract import Note, Scheduler, playable  # noqa: E402

READY_TIMEOUT_S, POLL_S, IDLE_S = 10.0, 0.001, 60.0


def reader(lines, q):
    """Put each JSON line on q. A broken line (e.g. the producer killed mid-write) becomes a reader_error; an EOF is
    always delivered last, so the clock loop never waits on a dead reader."""
    try:
        for line in lines:
            line = line.strip()
            if line:
                q.put(json.loads(line))
    except Exception as e:
        q.put({"type": "reader_error", "error": f"{type(e).__name__}: {e}"[:300]})
    finally:
        q.put({"type": "eof"})


def drain(q):
    out = []
    try:
        while True:
            out.append(q.get_nowait())
    except queue.Empty:
        return out


class Sender:
    def __init__(self, port, clock_ns=time.monotonic_ns):
        self.port, self.clock_ns = port, clock_ns
        self.sch = Scheduler()
        self.origin = None                 # (music_origin_ms, wall_origin_ns)
        self.sent_upto = 0
        self.end = None                    # "end" | "eof" | "reader_error"
        self.log = {"sent": [], "send_failures": [], "cancelled_unsent_on": [], "received_blocks": [], "scheduled_notes": 0, "dropped_velocity0": 0,
                    "extended_zero_length": 0, "end": None, "error": None, "origin": None}

    def music_now(self):
        return self.origin[0] + (self.clock_ns() - self.origin[1]) / 1e6

    def scheduled_ns(self, ms):
        return int(self.origin[1] + (ms - self.origin[0]) * 1e6) if self.origin else None

    def handle(self, m):
        t = m["type"]
        if t == "start":
            self.origin = (m["music_origin_ms"], m["wall_origin_ns"])
            self.log["origin"] = list(self.origin)
        elif t == "notes":
            notes = [Note(*n) for n in m["notes"]]
            kept, dropped = playable(notes)
            self.log["dropped_velocity0"] += len(dropped)
            self.log["extended_zero_length"] += sum(1 for n in notes if n.velocity > 0 and n.off_ms <= n.on_ms)
            self.sch.schedule(kept)
            self.log["scheduled_notes"] += len(kept)
            self.log["received_blocks"].append({"block": m["block"], "notes": len(notes), "recv_ns": self.clock_ns()})
        elif t == "reader_error":
            self.log["error"] = m["error"]
            self.end = "reader_error"
        elif t == "end" or (t == "eof" and self.end is None):
            self.end = t                   # an eof after a normal end is the producer exiting: ignored

    def flush(self, best_effort=False):
        """Send midi_out entries not sent yet, once each. Normal mode raises on a failed send (the run stops).
        best_effort (cleanup after the stop) cancels unsent note-ons, records failed note-offs and goes on."""
        while self.sent_upto < len(self.sch.midi_out):
            i = self.sent_upto
            self.sent_upto += 1            # never retry the same event
            ms, kind, pitch, vel = self.sch.midi_out[i]
            due = self.sch.midi_due[i]
            msg_on = kind == "on"
            if best_effort and msg_on:
                self.log["cancelled_unsent_on"].append({"index": i, "pitch": pitch})
                continue
            try:
                self.port.send(("note_on", pitch, vel) if msg_on else ("note_off", pitch, 0))
            except Exception as e:
                self.log["send_failures"].append({"index": i, "kind": kind, "pitch": pitch, "error": f"{type(e).__name__}: {e}"[:200]})
                if not best_effort:
                    raise
                continue
            self.log["sent"].append({"index": i, "kind": kind, "pitch": pitch, "velocity": vel, "tick_ms": ms, "due_ms": due,
                                     "scheduled_ns": self.scheduled_ns(due), "send_return_ns": self.clock_ns()})

    def process(self, msgs):
        """One clock-loop iteration (the main loop calls only this): handle the drained input, then tick and send.
        An abnormal end (EOF without end, broken input) stops before any tick, so no note starts after the stop is
        known (Astra re-review of #1749); a normal end keeps playing out what is scheduled. Returns True when done."""
        for m in msgs:
            self.handle(m)
        if self.end in ("eof", "reader_error"):
            return True
        self.step()
        return self.done()

    def step(self):
        if self.origin is not None:
            self.sch.tick(self.music_now())
            self.flush()

    def done(self):
        return self.end in ("eof", "reader_error") or (self.end == "end" and not self.sch.pending)

    def cleanup(self):
        """Stop, try every remaining note-off once, close the port. Each part is guarded."""
        try:
            now = self.music_now() if self.origin else 0.0
            self.log["pending_at_stop"] = len(self.sch.pending)
            self.log["pending_on_at_stop"] = sum(1 for e in self.sch.pending if e.kind == "on")
            self.log["sounding_at_stop"] = len(self.sch.pitch_owner)
            failures_before = len(self.log["send_failures"])
            self.sch.stop(now)
            self.flush(best_effort=True)
            self.log["stop_offs_failed"] = sum(1 for f in self.log["send_failures"][failures_before:] if f["kind"] == "off")
        except Exception as e:
            self.log["cleanup_error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            try:
                self.port.close()
                self.log["port_closed"] = True
            except Exception as e:
                self.log["port_closed"] = False
                self.log["close_error"] = f"{type(e).__name__}: {e}"[:200]
        self.log["end"] = self.end
        self.log["clean_stop"] = (not self.log["send_failures"] and not self.log["cancelled_unsent_on"]
                                  and self.log.get("port_closed") is True and "cleanup_error" not in self.log)
        self.log["expected_midi"] = [[k, p, v, d] for (_, k, p, v), d in zip(self.sch.midi_out, self.sch.midi_due)]
        self.log["logical_sent"] = self.sch.sent
        return self.log


class MidoPort:
    def __init__(self, name):
        import mido
        self._mido = mido
        self._port = mido.open_output(name, virtual=True)

    def send(self, m):
        kind, pitch, vel = m
        self._port.send(self._mido.Message(kind, note=pitch, velocity=vel))

    def close(self):
        self._port.close()


def main() -> None:
    port_name, ready_file, out_path = sys.argv[1:4]
    q = queue.Queue()
    s = Sender(MidoPort(port_name))
    s.log["port"] = port_name
    threading.Thread(target=reader, args=(sys.stdin, q), daemon=True).start()
    try:
        t_wait = time.monotonic() + READY_TIMEOUT_S
        while not os.path.exists(ready_file):
            if time.monotonic() > t_wait:
                raise RuntimeError("capture never became ready")
            time.sleep(0.005)
        last = time.monotonic()
        while True:
            msgs = drain(q)
            if msgs:
                last = time.monotonic()
            if s.process(msgs):
                break
            if s.end is None and time.monotonic() - last > IDLE_S:
                raise RuntimeError(f"no input for {IDLE_S:.0f} s")
            time.sleep(POLL_S)
    except Exception as e:  # recorded; the cleanup still runs
        s.log["error"] = s.log["error"] or f"{type(e).__name__}: {e}"[:300]
    finally:
        log = s.cleanup()
        with open(out_path, "w") as f:
            json.dump(log, f)


if __name__ == "__main__":
    main()
