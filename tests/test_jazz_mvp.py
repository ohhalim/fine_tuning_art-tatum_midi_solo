import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pretty_midi
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
