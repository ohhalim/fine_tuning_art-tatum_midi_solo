from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from mido import Message

from inference.realtime.transport import (
    REPORT_SCHEMA_VERSION,
    DirectMidiEcho,
    RecordingMidiSink,
    build_logical_midi_fixture,
    evaluate_event_integrity,
)
from scripts.run_direct_midi_echo_probe import exit_code_for_report, run_probe, write_report
from scripts import run_coremidi_virtual_loopback_probe as coremidi_probe
from scripts import run_direct_midi_echo as direct_midi_runner


class FailingSink:
    def send(self, message: Message) -> None:
        raise RuntimeError("sink failed")


class FakeCoreMidiOutput:
    def __init__(self, backend: "FakeCoreMidiBackend", name: str) -> None:
        self.backend = backend
        self.name = name
        self.reset_called = False

    def __enter__(self) -> "FakeCoreMidiOutput":
        self.backend.source_names.add(self.name)
        return self

    def __exit__(self, *args: object) -> None:
        self.backend.source_names.remove(self.name)

    def send(self, message: Message) -> None:
        if self.backend.drop_first_echo and "JazzImprov-Echo" in self.name:
            self.backend.drop_first_echo = False
            return
        callback = self.backend.callbacks.get(self.name)
        if callback is not None:
            callback(message.copy())

    def reset(self) -> None:
        self.reset_called = True


class FakeCoreMidiInput:
    def __init__(self, backend: "FakeCoreMidiBackend", name: str, callback: object) -> None:
        self.backend = backend
        self.name = name
        self.callback = callback
        self.closed = False
        self.backend.callbacks[self.name] = self.callback

    def __enter__(self) -> "FakeCoreMidiInput":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.backend.callbacks.pop(self.name, None)
        self.closed = True


class FakeCoreMidiBackend:
    def __init__(self, *, drop_first_echo: bool = False) -> None:
        self.source_names: set[str] = set()
        self.callbacks: dict[str, object] = {}
        self.opened_outputs: dict[str, FakeCoreMidiOutput] = {}
        self.drop_first_echo = drop_first_echo

    def get_input_names(self) -> list[str]:
        return sorted(self.source_names)

    def open_output(self, name: str, *, virtual: bool = False) -> FakeCoreMidiOutput:
        if not virtual:
            raise AssertionError("fake backend expects virtual output")
        output = FakeCoreMidiOutput(self, name)
        self.opened_outputs[name] = output
        return output

    def open_input(self, name: str, *, callback: object) -> FakeCoreMidiInput:
        return FakeCoreMidiInput(self, name, callback)


class FakeNamedOutput:
    def __init__(self) -> None:
        self.messages: list[Message] = []
        self.reset_called = False

    def __enter__(self) -> "FakeNamedOutput":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def send(self, message: Message) -> None:
        self.messages.append(message.copy())

    def reset(self) -> None:
        self.reset_called = True


class FakeNamedInput:
    def __init__(self, callback: object, messages: list[Message]) -> None:
        self.callback = callback
        self.messages = messages
        self.closed = False
        for message in self.messages:
            self.callback(message.copy())

    def __enter__(self) -> "FakeNamedInput":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed = True


class DirectMidiEchoTest(unittest.TestCase):
    def test_600_second_fixture_has_balanced_note_events(self) -> None:
        fixture = build_logical_midi_fixture(logical_duration_seconds=600.0, bpm=120.0)

        self.assertEqual(2701, len(fixture))
        self.assertLessEqual(fixture[-1].logical_time_seconds, 600.0)
        report = evaluate_event_integrity(
            input_messages=[event.message for event in fixture],
            output_messages=[event.message for event in fixture],
            callback_latency_ns=[],
            logical_duration_seconds=600.0,
            wall_clock_seconds=0.0,
            scope="test_fixture",
        )
        self.assertTrue(report.passed_event_integrity)
        self.assertEqual(0, report.stuck_note_count)

    def test_probe_reports_exact_in_memory_echo_without_runtime_claims(self) -> None:
        report = run_probe(logical_duration_seconds=600.0, bpm=120.0)

        self.assertEqual(REPORT_SCHEMA_VERSION, report.schema_version)
        self.assertEqual(2701, report.input_event_count)
        self.assertEqual(report.input_event_count, report.output_event_count)
        self.assertEqual(0, report.event_loss_count)
        self.assertEqual(0, report.duplicate_output_count)
        self.assertEqual(0, report.order_mismatch_count)
        self.assertEqual(0, report.stuck_note_count)
        self.assertEqual(0, report.crash_count)
        self.assertEqual(2701, report.callback_processing_latency_ms.sample_count)
        self.assertTrue(report.input_event_gate_passed)
        self.assertTrue(report.passed_event_integrity)
        self.assertFalse(report.wall_clock_soak_completed)
        self.assertFalse(report.os_midi_loopback_observed)
        self.assertFalse(report.fl_studio_audio_observed)
        self.assertFalse(report.passed_r0_transport_gate)
        self.assertEqual(0, exit_code_for_report(report))

    def test_integrity_report_detects_loss_duplicate_order_and_stuck_note(self) -> None:
        note_on = Message("note_on", channel=0, note=60, velocity=90)
        note_off = Message("note_off", channel=0, note=60, velocity=0)

        loss = evaluate_event_integrity(
            input_messages=[note_on, note_off],
            output_messages=[note_on],
            callback_latency_ns=[],
            logical_duration_seconds=1.0,
            wall_clock_seconds=0.0,
            scope="loss",
        )
        duplicate = evaluate_event_integrity(
            input_messages=[note_on, note_off],
            output_messages=[note_on, note_off, note_off],
            callback_latency_ns=[],
            logical_duration_seconds=1.0,
            wall_clock_seconds=0.0,
            scope="duplicate",
        )
        reordered = evaluate_event_integrity(
            input_messages=[note_on, note_off],
            output_messages=[note_off, note_on],
            callback_latency_ns=[],
            logical_duration_seconds=1.0,
            wall_clock_seconds=0.0,
            scope="reordered",
        )

        self.assertEqual(1, loss.event_loss_count)
        self.assertEqual(1, loss.stuck_note_count)
        self.assertEqual(1, duplicate.duplicate_output_count)
        self.assertGreater(reordered.order_mismatch_count, 0)
        self.assertFalse(loss.passed_event_integrity)
        self.assertFalse(duplicate.passed_event_integrity)
        self.assertFalse(reordered.passed_event_integrity)
        self.assertEqual(1, exit_code_for_report(loss))

    def test_empty_run_cannot_pass_integrity_or_soak_gates(self) -> None:
        report = evaluate_event_integrity(
            input_messages=[],
            output_messages=[],
            callback_latency_ns=[],
            logical_duration_seconds=600.0,
            wall_clock_seconds=600.0,
            scope="empty",
            wall_clock_soak_completed=True,
            os_midi_loopback_observed=False,
        )

        self.assertFalse(report.input_event_gate_passed)
        self.assertFalse(report.passed_event_integrity)
        self.assertTrue(report.wall_clock_soak_completed)
        self.assertFalse(report.os_midi_loopback_observed)
        self.assertFalse(report.passed_r0_transport_gate)
        self.assertEqual(1, exit_code_for_report(report))

    def test_sink_failure_is_counted_and_raised(self) -> None:
        echo = DirectMidiEcho(FailingSink())

        with self.assertRaisesRegex(RuntimeError, "sink failed"):
            echo.process(Message("note_on", note=60, velocity=80))

        self.assertEqual(1, echo.crash_count)
        self.assertEqual(1, len(echo.input_messages))
        self.assertEqual(0, len(echo.output_messages))

    def test_report_json_preserves_claim_boundaries(self) -> None:
        report = run_probe(logical_duration_seconds=4.0, bpm=120.0)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.json"
            write_report(path, report)
            payload = path.read_text(encoding="utf-8")

        self.assertIn('"scope": "in_memory_logical_fixture"', payload)
        self.assertIn('"wall_clock_soak_completed": false', payload)
        self.assertIn('"p50": null', payload)

    def test_all_notes_off_and_all_sound_off_clear_channel_note_state(self) -> None:
        for control in (120, 123):
            with self.subTest(control=control):
                messages = [
                    Message("note_on", channel=0, note=60, velocity=90),
                    Message("note_on", channel=1, note=64, velocity=90),
                    Message("control_change", channel=0, control=control, value=0),
                    Message("note_off", channel=1, note=64, velocity=0),
                ]
                report = evaluate_event_integrity(
                    input_messages=messages,
                    output_messages=messages,
                    callback_latency_ns=[],
                    logical_duration_seconds=1.0,
                    wall_clock_seconds=1.0,
                    scope="channel_mode_note_reset",
                )

                self.assertEqual(0, report.unmatched_note_off_count)
                self.assertEqual(0, report.stuck_note_count)
                self.assertTrue(report.passed_event_integrity)

    def test_late_note_off_after_channel_reset_is_not_unmatched(self) -> None:
        for control in (120, 123):
            with self.subTest(control=control):
                messages = [
                    Message("note_on", channel=0, note=60, velocity=90),
                    Message("control_change", channel=0, control=control, value=0),
                    Message("note_off", channel=0, note=60, velocity=0),
                ]
                report = evaluate_event_integrity(
                    input_messages=messages,
                    output_messages=messages,
                    callback_latency_ns=[],
                    logical_duration_seconds=1.0,
                    wall_clock_seconds=1.0,
                    scope="late_note_off_after_reset",
                )

                self.assertEqual(0, report.unmatched_note_off_count)
                self.assertEqual(0, report.stuck_note_count)
                self.assertTrue(report.passed_event_integrity)

    def test_lossy_soak_keeps_observation_separate_from_quality_gate(self) -> None:
        note_on = Message("note_on", note=60, velocity=90)
        note_off = Message("note_off", note=60, velocity=0)
        report = evaluate_event_integrity(
            input_messages=[note_on, note_off],
            output_messages=[note_on],
            callback_latency_ns=[100],
            capture_latency_ns=[],
            logical_duration_seconds=600.0,
            wall_clock_seconds=600.0,
            scope="lossy_soak",
            minimum_input_event_count=2,
            safe_reset_sent=True,
            wall_clock_soak_completed=True,
            os_midi_loopback_observed=True,
        )

        self.assertTrue(report.wall_clock_soak_completed)
        self.assertTrue(report.os_midi_loopback_observed)
        self.assertFalse(report.passed_event_integrity)
        self.assertFalse(report.passed_r0_transport_gate)
        self.assertEqual(0, report.capture_latency_ms.sample_count)
        self.assertIsNone(report.capture_latency_ms.p50)

    def test_lossless_completed_soak_passes_r0_gate_and_exit_policy(self) -> None:
        messages = [
            Message("note_on", note=60, velocity=90),
            Message("note_off", note=60, velocity=0),
        ]
        report = evaluate_event_integrity(
            input_messages=messages,
            output_messages=messages,
            callback_latency_ns=[100, 100],
            capture_latency_ns=[200, 200],
            logical_duration_seconds=600.0,
            wall_clock_seconds=600.0,
            scope="complete_soak",
            minimum_input_event_count=2,
            safe_reset_sent=True,
            wall_clock_soak_completed=True,
            os_midi_loopback_observed=True,
        )

        self.assertTrue(report.passed_event_integrity)
        self.assertTrue(report.passed_r0_transport_gate)
        self.assertEqual(0, coremidi_probe.exit_code_for_report(report, require_r0_gate=True))

        incomplete_report = evaluate_event_integrity(
            input_messages=messages,
            output_messages=messages,
            callback_latency_ns=[100, 100],
            logical_duration_seconds=2.0,
            wall_clock_seconds=2.0,
            scope="short_smoke",
            minimum_input_event_count=2,
            safe_reset_sent=True,
            wall_clock_soak_completed=False,
            os_midi_loopback_observed=True,
        )
        self.assertTrue(incomplete_report.passed_event_integrity)
        self.assertFalse(incomplete_report.passed_r0_transport_gate)
        self.assertEqual(
            1,
            coremidi_probe.exit_code_for_report(incomplete_report, require_r0_gate=True),
        )
        self.assertEqual(
            0,
            coremidi_probe.exit_code_for_report(incomplete_report, require_r0_gate=False),
        )

    def test_virtual_loopback_runner_uses_independent_capture(self) -> None:
        backend = FakeCoreMidiBackend()
        with (
            mock.patch.object(coremidi_probe.mido, "open_output", backend.open_output),
            mock.patch.object(coremidi_probe.mido, "open_input", backend.open_input),
            mock.patch.object(coremidi_probe.mido, "get_input_names", backend.get_input_names),
        ):
            report = coremidi_probe.run_coremidi_virtual_loopback(
                run_id="unit",
                duration_seconds=0.04,
                rate_hz=100.0,
                drain_timeout_seconds=0.0,
            )

        self.assertEqual(4, report.input_event_count)
        self.assertEqual(4, report.output_event_count)
        self.assertTrue(report.os_midi_loopback_observed)
        self.assertTrue(report.passed_event_integrity)
        self.assertFalse(report.wall_clock_soak_completed)
        self.assertFalse(report.passed_r0_transport_gate)
        self.assertEqual(4, report.capture_latency_ms.sample_count)

    def test_virtual_loopback_loss_reports_null_capture_latency(self) -> None:
        backend = FakeCoreMidiBackend(drop_first_echo=True)
        with (
            mock.patch.object(coremidi_probe.mido, "open_output", backend.open_output),
            mock.patch.object(coremidi_probe.mido, "open_input", backend.open_input),
            mock.patch.object(coremidi_probe.mido, "get_input_names", backend.get_input_names),
        ):
            report = coremidi_probe.run_coremidi_virtual_loopback(
                run_id="loss",
                duration_seconds=0.04,
                rate_hz=100.0,
                drain_timeout_seconds=0.0,
            )

        self.assertEqual(1, report.event_loss_count)
        self.assertTrue(report.os_midi_loopback_observed)
        self.assertFalse(report.passed_event_integrity)
        self.assertEqual(0, report.capture_latency_ms.sample_count)
        self.assertIsNone(report.capture_latency_ms.maximum)

    def test_virtual_loopback_interrupt_resets_echo_output(self) -> None:
        backend = FakeCoreMidiBackend()
        with (
            mock.patch.object(coremidi_probe.mido, "open_output", backend.open_output),
            mock.patch.object(coremidi_probe.mido, "open_input", backend.open_input),
            mock.patch.object(coremidi_probe.mido, "get_input_names", backend.get_input_names),
            mock.patch.object(coremidi_probe, "_wait_until", side_effect=KeyboardInterrupt),
            self.assertRaises(KeyboardInterrupt),
        ):
            coremidi_probe.run_coremidi_virtual_loopback(
                run_id="interrupt",
                duration_seconds=0.04,
                rate_hz=100.0,
                drain_timeout_seconds=0.0,
            )

        echo_outputs = [
            output
            for name, output in backend.opened_outputs.items()
            if "JazzImprov-Echo" in name
        ]
        self.assertEqual(1, len(echo_outputs))
        self.assertTrue(echo_outputs[0].reset_called)

    def test_named_port_runner_resets_output_and_rejects_silent_run(self) -> None:
        output = FakeNamedOutput()
        with (
            mock.patch.object(
                direct_midi_runner,
                "available_ports",
                return_value={"inputs": ["keyboard"], "outputs": ["synth"]},
            ),
            mock.patch.object(
                direct_midi_runner.mido,
                "open_output",
                return_value=output,
            ),
            mock.patch.object(
                direct_midi_runner.mido,
                "open_input",
                side_effect=lambda name, callback: FakeNamedInput(callback, []),
            ),
        ):
            report = direct_midi_runner.run_port_echo(
                input_port_name="keyboard",
                output_port_name="synth",
                duration_seconds=0.001,
            )

        self.assertTrue(output.reset_called)
        self.assertTrue(report.safe_reset_sent)
        self.assertFalse(report.input_event_gate_passed)
        self.assertFalse(report.output_capture_observed)
        self.assertFalse(report.passed_send_acceptance)

    def test_named_port_runner_forwards_each_input_once_as_send_acceptance(self) -> None:
        output = FakeNamedOutput()
        input_messages = [
            Message("note_on", note=60, velocity=90),
            Message("note_off", note=60, velocity=0),
        ]
        with (
            mock.patch.object(
                direct_midi_runner,
                "available_ports",
                return_value={"inputs": ["keyboard"], "outputs": ["synth"]},
            ),
            mock.patch.object(
                direct_midi_runner.mido,
                "open_output",
                return_value=output,
            ),
            mock.patch.object(
                direct_midi_runner.mido,
                "open_input",
                side_effect=lambda name, callback: FakeNamedInput(callback, input_messages),
            ),
        ):
            report = direct_midi_runner.run_port_echo(
                input_port_name="keyboard",
                output_port_name="synth",
                duration_seconds=0.001,
            )

        self.assertEqual(input_messages, output.messages)
        self.assertEqual(2, report.input_event_count)
        self.assertEqual(2, report.accepted_send_count)
        self.assertEqual(0, report.send_failure_count)
        self.assertTrue(report.passed_send_acceptance)
        self.assertTrue(report.safe_reset_sent)
        self.assertFalse(report.output_capture_observed)


if __name__ == "__main__":
    unittest.main()
