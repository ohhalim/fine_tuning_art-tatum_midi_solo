"""Drop our own notes when a DAW sends them back in.

FL Studio passes incoming MIDI to the selected channel as well as to the
channel whose port matches. With a MIDI Out channel selected for playing, the
model's notes go out to the DAW, into that channel and straight back to the
runtime's input. ``EchoGuard`` remembers what was sent and drops an input note
of the same kind and pitch that arrives within ``window_ms`` (#1584).
"""
from __future__ import annotations

import time
from collections import deque
from threading import Lock


def _kind(message) -> str | None:
    if message.type == "note_on" and message.velocity > 0:
        return "on"
    if message.type in ("note_off", "note_on"):
        return "off"
    return None


class EchoGuard:
    def __init__(self, window_ms: float, *, clock_ns=time.perf_counter_ns) -> None:
        self.window_ns = int(window_ms * 1_000_000)
        self._clock_ns = clock_ns
        self._sent: deque = deque(maxlen=4096)       # (sent_ns, kind, note)
        self._lock = Lock()
        self.dropped = 0

    def record(self, message) -> None:
        kind = _kind(message)
        if kind is not None:
            with self._lock:
                self._sent.append((self._clock_ns(), kind, message.note))

    def is_echo(self, message) -> bool:
        """True (and the sent note is used up) when ``message`` is one of ours coming back."""
        kind = _kind(message)
        if kind is None:
            return False
        now = self._clock_ns()
        with self._lock:
            while self._sent and now - self._sent[0][0] > self.window_ns:
                self._sent.popleft()
            for i, (_, k, note) in enumerate(self._sent):
                if k == kind and note == message.note:
                    del self._sent[i]
                    self.dropped += 1
                    return True
        return False

    def wrap(self, port):
        return _RecordingPort(port, self)


class _RecordingPort:
    """Output port that tells the guard about every message it sends."""

    def __init__(self, port, guard: EchoGuard) -> None:
        self._port = port
        self._guard = guard

    def send(self, message) -> None:
        self._guard.record(message)
        self._port.send(message)

    def __getattr__(self, name):
        return getattr(self._port, name)
