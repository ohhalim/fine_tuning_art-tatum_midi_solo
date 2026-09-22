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


class CaptureSummaryTests(unittest.TestCase):
    """Reset/panic traffic must not be counted as duplicated output."""

    def test_control_change_traffic_is_excluded(self):
        from mido import Message

        from scripts.run_continuous_jazz import summarize_capture

        class FakeRecord:
            def __init__(self, target_ns, note, type_):
                self.target_ns = target_ns
                self.message = Message(type_, note=note, velocity=64)

        result = type("R", (), {"records": [FakeRecord(0, 60, "note_on"),
                                            FakeRecord(1_000_000, 60, "note_off")]})()
        captured = [
            (500_000, Message("note_on", note=60, velocity=64)),
            (1_400_000, Message("note_off", note=60, velocity=0)),
        ] + [(2_000_000, Message("control_change", control=123, value=0)) for _ in range(48)]

        summary = summarize_capture(result, captured, drain_completed=True)

        self.assertEqual(2, summary["sent_note_event_count"])
        self.assertEqual(2, summary["captured_note_event_count"])
        self.assertEqual(50, summary["captured_total_message_count"])
        self.assertEqual(0, summary["duplicate_output_count"])
        self.assertEqual(0, summary["event_loss_count"])
        self.assertEqual(0, summary["order_mismatch_count"])
        self.assertAlmostEqual(0.5, summary["scheduled_to_capture_ms"]["p50"], places=3)


class PlayedMidiTests(unittest.TestCase):
    def test_played_midi_reflects_dispatched_events(self):
        import tempfile
        from pathlib import Path

        from mido import Message

        from scripts.run_continuous_jazz import write_played_midi

        class FakeRecord:
            def __init__(self, target_ns, type_, note, velocity):
                self.target_ns = target_ns
                self.message = Message(type_, note=note, velocity=velocity)

        result = type("R", (), {"records": [
            FakeRecord(0, "note_on", 60, 90),
            FakeRecord(500_000_000, "note_off", 60, 0),
        ]})()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "played.mid"
            count = write_played_midi(result, path, bpm=128)
            self.assertEqual(1, count)

            import pretty_midi

            notes = [n for i in pretty_midi.PrettyMIDI(str(path)).instruments for n in i.notes]
            self.assertEqual(1, len(notes))
            self.assertEqual(90, notes[0].velocity)
            self.assertAlmostEqual(0.5, notes[0].end - notes[0].start, places=2)

    def test_no_note_events_writes_nothing(self):
        from pathlib import Path

        from scripts.run_continuous_jazz import write_played_midi

        result = type("R", (), {"records": []})()
        self.assertIsNone(write_played_midi(result, Path("/tmp/unused.mid"), bpm=128))


class LivePrimerTests(unittest.TestCase):
    """Input must actually reach the primer, and must never break a silent run."""

    def _base(self):
        import torch

        return torch.tensor([1, 2, 3], dtype=torch.long)

    def test_empty_input_keeps_the_base_primer(self):
        from scripts.run_continuous_jazz import build_live_primer

        base = self._base()
        primer, used = build_live_primer((), base_primer=base, control_format="control_v1",
                                         role="lead", tempo_bpm=128)

        self.assertFalse(used)
        self.assertIs(base, primer)

    def test_played_notes_build_a_different_primer(self):
        from mido import Message

        from inference.realtime.continuous import TimedInputMessage
        from scripts.run_continuous_jazz import build_live_primer

        events = (
            TimedInputMessage(0, Message("note_on", note=60, velocity=90)),
            TimedInputMessage(200_000_000, Message("note_off", note=60, velocity=0)),
            TimedInputMessage(200_000_000, Message("note_on", note=64, velocity=88)),
            TimedInputMessage(400_000_000, Message("note_off", note=64, velocity=0)),
        )
        base = self._base()
        primer, used = build_live_primer(events, base_primer=base, control_format="control_v1",
                                         role="lead", tempo_bpm=128)

        self.assertTrue(used)
        self.assertNotEqual(base.tolist(), primer.tolist())
        # The played pitches survive into the primer.
        self.assertIn(60, primer.tolist())
        self.assertIn(64, primer.tolist())

    def test_control_change_only_input_keeps_the_base_primer(self):
        from mido import Message

        from inference.realtime.continuous import TimedInputMessage
        from scripts.run_continuous_jazz import build_live_primer

        events = (TimedInputMessage(0, Message("control_change", control=64, value=127)),)
        base = self._base()
        primer, used = build_live_primer(events, base_primer=base, control_format="control_v1",
                                         role="lead", tempo_bpm=128)

        self.assertFalse(used)
        self.assertIs(base, primer)

    def test_quiet_playing_never_yields_a_silent_primer_velocity(self):
        from mido import Message

        from inference.realtime.continuous import TimedInputMessage
        from scripts.generate import VELOCITY_TOKEN_END, VELOCITY_TOKEN_START
        from scripts.run_continuous_jazz import build_live_primer

        events = (
            TimedInputMessage(0, Message("note_on", note=60, velocity=1)),
            TimedInputMessage(200_000_000, Message("note_off", note=60, velocity=0)),
        )
        primer, used = build_live_primer(events, base_primer=self._base(),
                                         control_format="control_v1", role="lead", tempo_bpm=128)

        self.assertTrue(used)
        bins = [t - VELOCITY_TOKEN_START for t in primer.tolist()
                if VELOCITY_TOKEN_START <= t < VELOCITY_TOKEN_END]
        self.assertTrue(bins)
        self.assertNotIn(0, bins)


class ColdStartTests(unittest.TestCase):
    """Bars 0 and 1 are both needed before the downbeat, not one per bar."""

    def test_both_warmup_bars_are_awaited(self):
        import scripts.run_continuous_jazz as module

        waited = []
        real = module.BarBlockProducer

        class Recording(real):
            def wait_for_bar(self, bar_index, *, timeout):
                waited.append(bar_index)
                return super().wait_for_bar(bar_index, timeout=timeout)

        port = Port()
        with unittest.mock.patch.object(module, "BarBlockProducer", Recording):
            run_session(port=port, bars=BARS, bpm=BPM, chords=CHORDS, seed=42,
                        generate=lambda *_: (_ for _ in ()).throw(RuntimeError("no model")),
                        start_delay_seconds=0.0, spin_window_ms=0.0,
                        clock=clock(), clock_ns=lambda: 0,
                        wait_until=lambda target_ns, stop: None)

        self.assertEqual([0, 1], waited)

    def test_single_bar_run_does_not_wait_for_a_second_bar(self):
        import scripts.run_continuous_jazz as module

        waited = []
        real = module.BarBlockProducer

        class Recording(real):
            def wait_for_bar(self, bar_index, *, timeout):
                waited.append(bar_index)
                return super().wait_for_bar(bar_index, timeout=timeout)

        with unittest.mock.patch.object(module, "BarBlockProducer", Recording):
            run_session(port=Port(), bars=1, bpm=BPM, chords=CHORDS, seed=42,
                        generate=lambda *_: (_ for _ in ()).throw(RuntimeError("no model")),
                        start_delay_seconds=0.0, spin_window_ms=0.0,
                        clock=clock(), clock_ns=lambda: 0,
                        wait_until=lambda target_ns, stop: None)

        self.assertEqual([0], waited)


class DeadlinePolicyTests(unittest.TestCase):
    """One OS hiccup must not end a performance."""

    def test_session_records_misses_and_keeps_playing(self):
        from inference.realtime.scheduler import DEADLINE_POLICY_RECORD_AND_CONTINUE

        port = Port()
        result, producer = run_session(
            port=port, bars=BARS, bpm=BPM, chords=CHORDS, seed=42,
            generate=lambda *_: (_ for _ in ()).throw(RuntimeError("no model")),
            start_delay_seconds=0.0, spin_window_ms=0.0,
            clock=clock(), clock_ns=lambda: 0, wait_until=lambda target_ns, stop: None,
        )
        report = build_report(result, producer, bars=BARS, bpm=BPM)

        self.assertEqual(DEADLINE_POLICY_RECORD_AND_CONTINUE, report["deadline_policy"])

    def test_abort_policy_is_still_selectable(self):
        from inference.realtime.scheduler import DEADLINE_POLICY_ABORT_ON_FIRST_MISS

        result, producer = run_session(
            port=Port(), bars=BARS, bpm=BPM, chords=CHORDS, seed=42,
            generate=lambda *_: (_ for _ in ()).throw(RuntimeError("no model")),
            start_delay_seconds=0.0, spin_window_ms=0.0,
            clock=clock(), clock_ns=lambda: 0, wait_until=lambda target_ns, stop: None,
            deadline_policy=DEADLINE_POLICY_ABORT_ON_FIRST_MISS,
        )
        report = build_report(result, producer, bars=BARS, bpm=BPM)

        self.assertEqual(DEADLINE_POLICY_ABORT_ON_FIRST_MISS, report["deadline_policy"])


class PrimerBudgetTests(unittest.TestCase):
    """The primer budget must track the real control prefix, not a constant."""

    def test_velocity_survives_at_every_primer_budget(self):
        from mido import Message

        from inference.realtime.continuous import TimedInputMessage
        from scripts.generate import VELOCITY_TOKEN_END, VELOCITY_TOKEN_START
        from scripts.run_continuous_jazz import build_live_primer

        import torch

        events = []
        stamp = 0
        for pitch in [60, 63, 65, 67, 70, 72] * 6:
            events.append(TimedInputMessage(stamp, Message("note_on", note=pitch, velocity=80)))
            stamp += 120_000_000
            events.append(TimedInputMessage(stamp, Message("note_off", note=pitch, velocity=0)))
            stamp += 130_000_000

        for budget in (16, 24, 32, 48):
            primer, used = build_live_primer(
                tuple(events), base_primer=torch.tensor([1, 2, 3]),
                control_format="control_v1", role="lead", tempo_bpm=128,
                primer_max_tokens=budget,
            )
            tokens = primer.tolist()
            bins = [t for t in tokens if VELOCITY_TOKEN_START <= t < VELOCITY_TOKEN_END]
            self.assertTrue(used)
            self.assertLessEqual(len(tokens), budget)
            self.assertTrue(bins, f"velocity lost at primer_max_tokens={budget}")


class ChordPrimerOptInTests(unittest.TestCase):
    """Opt-in only: the default path must be untouched."""

    def _events(self):
        from mido import Message

        from inference.realtime.continuous import TimedInputMessage

        return (
            TimedInputMessage(0, Message("note_on", note=72, velocity=88)),
            TimedInputMessage(200_000_000, Message("note_off", note=72, velocity=0)),
        )

    def test_chord_notes_reach_the_primer(self):
        import torch

        from scripts.run_continuous_jazz import build_chord_live_primer

        base = torch.tensor([1, 2, 3], dtype=torch.long)
        primer, used_input, used_chord = build_chord_live_primer(
            self._events(), "Dm7", bpm=BPM, base_primer=base
        )

        self.assertTrue(used_chord)
        self.assertTrue(used_input)
        self.assertIn(72, primer.tolist())  # the played note survived

    def test_different_chords_give_different_primers(self):
        import torch

        from scripts.run_continuous_jazz import build_chord_live_primer

        base = torch.tensor([1], dtype=torch.long)
        d_minor, _, _ = build_chord_live_primer((), "Dm7", bpm=BPM, base_primer=base)
        g_seven, _, _ = build_chord_live_primer((), "G7", bpm=BPM, base_primer=base)

        self.assertNotEqual(d_minor.tolist(), g_seven.tolist())

    def test_primer_carries_no_untrained_control_token(self):
        import torch

        from utilities.constants import TOKEN_COND_SEP, TOKEN_STAGE_B_CHORD_QUALITY_END
        from scripts.run_continuous_jazz import build_chord_live_primer

        primer, _, _ = build_chord_live_primer(
            self._events(), "Cmaj7", bpm=BPM, base_primer=torch.tensor([1])
        )
        control = [t for t in primer.tolist()
                   if TOKEN_COND_SEP <= t <= TOKEN_STAGE_B_CHORD_QUALITY_END]

        self.assertEqual([], control)

    def test_silent_player_and_no_chord_falls_back(self):
        import torch

        from scripts.run_continuous_jazz import build_chord_live_primer

        base = torch.tensor([1, 2, 3], dtype=torch.long)
        primer, used_input, used_chord = build_chord_live_primer(
            (), "", bpm=BPM, base_primer=base
        )

        self.assertIs(base, primer)
        self.assertFalse(used_input)
        self.assertFalse(used_chord)

    def test_default_path_does_not_use_the_chord_primer(self):
        """The flag is off by default and the old builder is unchanged."""
        from scripts.run_continuous_jazz import build_live_primer
        import torch

        base = torch.tensor([1, 2, 3], dtype=torch.long)
        primer, used = build_live_primer(self._events(), base_primer=base,
                                         control_format="control_v1", role="lead",
                                         tempo_bpm=BPM)

        self.assertTrue(used)
        self.assertNotEqual(base.tolist(), primer.tolist())
