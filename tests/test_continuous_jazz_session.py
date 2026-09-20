from __future__ import annotations

import unittest
import unittest.mock

from scripts.run_continuous_jazz import build_fallback_blocks, build_report, run_session
from inference.realtime.scheduler import MonotonicBarClock

BPM = 128
BARS = 8
CHORDS = ["Dm7", "G7", "Cmaj7", "A7"]


class Port:
    def __init__(self):
        self.sent = []
        self.resets = self.panics = 0

    def send(self, message):
        self.sent.append(message)

    def reset(self):
        self.resets += 1

    def panic(self):
        self.panics += 1


def clock():
    return MonotonicBarClock(bpm=BPM, beats_per_bar=4, start_ns=0)


class FallbackPrebuildTests(unittest.TestCase):
    def test_every_bar_has_a_prebuilt_fallback(self):
        blocks = build_fallback_blocks(clock=clock(), bars=BARS, bpm=BPM,
                                       chords=CHORDS, seed=42, duration=240.0 / BPM)

        self.assertEqual(set(range(BARS)), set(blocks))
        self.assertTrue(all(b.fallback_used for b in blocks.values()))
        self.assertTrue(all(b.events for b in blocks.values()))


class SessionTests(unittest.TestCase):
    """The session must finish and reset even when generation never succeeds."""

    def _run(self, generate, port=None):
        port = port or Port()
        # Fake clock: the run must not wait out eight real bars.
        result, producer = run_session(
            port=port, bars=BARS, bpm=BPM, chords=CHORDS, seed=42,
            generate=generate, start_delay_seconds=0.0, spin_window_ms=0.0,
            clock=clock(), clock_ns=lambda: 0, wait_until=lambda target_ns, stop: None,
        )
        return port, result, producer

    def test_failing_generation_still_plays_every_bar_and_resets(self):
        def generate(_bar_index, _events):
            raise RuntimeError("inference down")

        port, result, producer = self._run(generate)
        report = build_report(result, producer, bars=BARS, bpm=BPM)

        self.assertTrue(result.run_completed)
        self.assertEqual(BARS, result.completed_bar_count)
        self.assertEqual(0, result.queue_underrun_count)
        self.assertTrue(port.sent)
        self.assertEqual((1, 1), (port.resets, port.panics))
        self.assertEqual(0, report["production"]["model_bar_count"])
        self.assertFalse(report["realtime_coperformance_verified"])
        self.assertFalse(report["external_keyboard_verified"])
        self.assertFalse(report["daw_audio_verified"])

    def test_send_failures_are_counted_and_the_port_is_still_reset(self):
        port = Port()
        port.send = lambda _message: (_ for _ in ()).throw(RuntimeError("port died"))

        _port, result, _producer = self._run(
            lambda *_: (_ for _ in ()).throw(RuntimeError("no model")), port=port
        )

        self.assertGreater(result.send_failure_count, 0)
        self.assertEqual((1, 1), (port.resets, port.panics))

    def test_reset_runs_even_when_the_run_raises(self):
        port = Port()
        with unittest.mock.patch(
            "scripts.run_continuous_jazz.OneBarMidiScheduler"
        ) as scheduler:
            scheduler.return_value.run.side_effect = RuntimeError("scheduler died")
            with self.assertRaisesRegex(RuntimeError, "scheduler died"):
                self._run(lambda *_: (_ for _ in ()).throw(RuntimeError("no model")),
                          port=port)

        self.assertEqual((1, 1), (port.resets, port.panics))


if __name__ == "__main__":
    unittest.main()
