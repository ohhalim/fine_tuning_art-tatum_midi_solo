"""Our own notes coming back from a DAW are dropped (inference/realtime/echo.py, #1584)."""
from __future__ import annotations

import tempfile
import unittest

from mido import Message

from inference.realtime.echo import EchoGuard
from scripts.run_continuous_jazz import main


class Clock:
    def __init__(self) -> None:
        self.ns = 0

    def __call__(self) -> int:
        return self.ns


class Sink:
    def __init__(self) -> None:
        self.sent = []
        self.name = "sink"

    def send(self, m) -> None:
        self.sent.append(m)


class EchoGuardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.guard = EchoGuard(50, clock_ns=self.clock)
        self.sink = Sink()
        self.port = self.guard.wrap(self.sink)

    def test_a_sent_note_coming_back_soon_is_dropped_once(self) -> None:
        self.port.send(Message("note_on", note=64, velocity=90, channel=0))
        self.clock.ns = 5_000_000
        self.assertTrue(self.guard.is_echo(Message("note_on", note=64, velocity=100, channel=3)))
        self.assertFalse(self.guard.is_echo(Message("note_on", note=64, velocity=100)))   # used up
        self.assertEqual(self.guard.dropped, 1)
        self.assertEqual(len(self.sink.sent), 1)                                            # still forwarded

    def test_late_or_different_notes_pass(self) -> None:
        self.port.send(Message("note_on", note=64, velocity=90))
        self.assertFalse(self.guard.is_echo(Message("note_on", note=65, velocity=90)))
        self.clock.ns = 60_000_000
        self.assertFalse(self.guard.is_echo(Message("note_on", note=64, velocity=90)))

    def test_note_off_and_zero_velocity_note_on_are_the_same_kind(self) -> None:
        self.port.send(Message("note_off", note=60))
        self.assertFalse(self.guard.is_echo(Message("note_on", note=60, velocity=90)))     # wrong kind
        self.assertTrue(self.guard.is_echo(Message("note_on", note=60, velocity=0)))
        self.assertFalse(self.guard.is_echo(Message("control_change", control=1, value=2)))

    def test_wrapped_port_delegates_other_attributes(self) -> None:
        self.assertEqual(self.port.name, "sink")


class FlagTest(unittest.TestCase):
    def test_echo_filter_needs_an_input_port_and_a_sane_window(self) -> None:
        for extra in (["--ignore-echo-ms", "30"], ["--input-port", "x", "--ignore-echo-ms", "900"]):
            with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
                main(["--output-dir", d, "--fallback-only", *extra])


if __name__ == "__main__":
    unittest.main()
