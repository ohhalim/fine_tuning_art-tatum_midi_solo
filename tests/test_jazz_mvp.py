import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pretty_midi
from inference.realtime.blocks import build_scheduled_midi_block
from scripts.run_jazz_mvp import fit_window, prepare_phrase, play_phrase


class Port:
    def __init__(self):
        self.resets = self.panics = 0
    def reset(self):
        self.resets += 1
    def panic(self):
        self.panics += 1


class JazzMvpTests(unittest.TestCase):
    def test_quantized_boundary_and_silence(self):
        midi = pretty_midi.PrettyMIDI()
        inst = pretty_midi.Instrument(0)
        inst.notes = [pretty_midi.Note(80, 60, 0.2, 1.880)]
        midi.instruments = [inst]
        fitted = fit_window(midi, 1.875)
        self.assertEqual(fitted.instruments[0].notes[0].end, 1.875)
        self.assertEqual(midi.instruments[0].notes[0].end, 1.880)
        inst.notes[0].end = 0.3
        self.assertEqual(fit_window(midi, 1.875).get_end_time(), 0.3)

    def test_invalid_model_fallback_and_multibar_export(self):
        phrase, midis, rows = prepare_phrase(generate=lambda _: ([], {}),
                bpm=128, bars=4, chords=['Dm7', 'G7'], seed=42)
        self.assertEqual(len(midis), 4)
        self.assertTrue(all(r['source'] == 'invalid_model_fallback' for r in rows))
        self.assertEqual(phrase.get_end_time(), 7.5)
        self.assertTrue(all(n.end <= 1.875 for m in midis for i in m.instruments for n in i.notes))

    def test_silent_model_block_is_rejected_by_token_validation(self):
        """Stage A decodes velocity as ``value * 4`` and carries it as state, so
        velocity bin 0 decodes to MIDI velocity 0 -- a note-off by the MIDI spec.
        ``validate_generated_token_block`` rejects it, so the bar falls back."""
        # velocity bin 0, note_on 60, time-shift 100 + 88 steps, note_off 60.
        tokens = [356, 60, 355, 343, 188]
        _, midis, rows = prepare_phrase(generate=lambda _: (tokens, {}),
                                        bpm=128, bars=2, chords=['Dm7'], seed=42)

        self.assertTrue(all(r['source'] == 'invalid_model_fallback' for r in rows))
        self.assertEqual(1, rows[0]['invalid_model_validation']['silent_note_count'])
        self.assertTrue(all(1 <= n.velocity <= 127
                            for m in midis for i in m.instruments for n in i.notes))

    def test_unschedulable_model_block_falls_back_instead_of_aborting(self):
        """Defense in depth: token validation and the scheduler check different
        things, so a block that passes the former can still fail the latter. That
        must fall back like any other bad bar rather than abort the whole phrase."""
        real = build_scheduled_midi_block
        calls = []

        def once(**kwargs):
            calls.append(kwargs['adapter'])
            if len(calls) == 1:
                raise ValueError('note velocity out of range: 0')
            return real(**kwargs)

        # velocity bin 16 (MIDI 64): this block passes token validation.
        tokens = [372, 60, 355, 343, 188]
        with patch('scripts.run_jazz_mvp.build_scheduled_midi_block', side_effect=once):
            _, midis, rows = prepare_phrase(generate=lambda _: (tokens, {}),
                                            bpm=128, bars=1, chords=['Dm7'], seed=42)

        self.assertEqual(['model', 'unschedulable_model_fallback'], calls)
        self.assertEqual('unschedulable_model_fallback', rows[0]['source'])
        self.assertIn('unschedulable_model_block', rows[0]['invalid_model_validation'])
        self.assertTrue(all(1 <= n.velocity <= 127
                            for m in midis for i in m.instruments for n in i.notes))

    def test_unschedulable_fallback_block_is_a_real_bug_and_aborts(self):
        """Only a *model* block may be rerouted; a failing fallback is not masked."""
        with patch('scripts.run_jazz_mvp.build_scheduled_midi_block',
                   side_effect=ValueError('boom')):
            with self.assertRaisesRegex(ValueError, 'boom'):
                prepare_phrase(generate=None, bpm=128, bars=1, chords=['Dm7'], seed=42)
        # ... and a model block that also fails on the fallback retry still raises.
        with patch('scripts.run_jazz_mvp.build_scheduled_midi_block',
                   side_effect=ValueError('boom')):
            with self.assertRaisesRegex(ValueError, 'boom'):
                prepare_phrase(generate=lambda _: ([372, 60, 355, 343, 188], {}),
                               bpm=128, bars=1, chords=['Dm7'], seed=42)

    def test_inference_error_not_hidden(self):
        def fail(_):
            raise RuntimeError('inference failed')
        with self.assertRaisesRegex(RuntimeError, 'inference failed'):
            prepare_phrase(generate=fail, bpm=128, bars=1, chords=['Dm7'], seed=42)

    def test_reset_on_scheduler_exception(self):
        _, midis, _ = prepare_phrase(generate=None, bpm=128, bars=1, chords=['Dm7'], seed=42)
        port = Port()
        with patch('scripts.run_jazz_mvp.OneBarMidiScheduler') as scheduler:
            scheduler.return_value.run.side_effect = RuntimeError('send failed')
            with self.assertRaises(RuntimeError):
                play_phrase(midis, 128, port)
        self.assertEqual((port.resets, port.panics), (1, 1))

    def test_no_downbeat_required_for_playback(self):
        _, midis, _ = prepare_phrase(generate=None, bpm=128, bars=1, chords=['Dm7'], seed=42)
        port = Port()
        with patch('scripts.run_jazz_mvp.OneBarMidiScheduler') as scheduler:
            scheduler.return_value.run.return_value = SimpleNamespace(run_completed=True,
                completed_bar_count=1, records=[], scheduler_dispatch_deadline_miss_count=0,
                send_failure_count=0)
            report = play_phrase(midis, 128, port)
        self.assertTrue(report['run_completed'])
        self.assertFalse(report['output_capture_observed'])


if __name__ == '__main__':
    unittest.main()
