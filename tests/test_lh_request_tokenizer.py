"""The real Aria tokenizer under the export's keyword arguments (issue #1756 review Q3). Runs in the Aria venv only
(needs ariautils and mido); skipped elsewhere."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
try:
    import mido
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
except ImportError:  # not the Aria venv
    AbsTokenizer = None
from lh_request_export import TOKENIZE_KW, misaligned_notes, token_notes  # noqa: E402

TPB, TEMPO = 480, 500_000            # 120 BPM: 25 ms = 24 ticks exactly


def write(path, notes):
    """notes: (pitch, onset ms, end ms, velocity), times on a 25 ms grid."""
    ev = sorted([(on, 1, p, v) for p, on, end, v in notes] + [(end, 0, p, 0) for p, on, end, v in notes])
    mid = mido.MidiFile(ticks_per_beat=TPB)
    tr = mido.MidiTrack()
    mid.tracks.append(tr)
    tr.append(mido.MetaMessage("set_tempo", tempo=TEMPO, time=0))
    now = 0
    for ms, is_on, p, v in ev:
        tick = round(ms * TPB * 1000 / TEMPO)
        tr.append(mido.Message("note_on" if is_on else "note_off", note=p, velocity=v, time=tick - now))
        now = tick
    mid.save(path)


@unittest.skipIf(AbsTokenizer is None, "needs the Aria venv (ariautils)")
class RealTokenizerTest(unittest.TestCase):
    def tokenize(self, notes, **kw):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x.mid")
            write(path, notes)
            return AbsTokenizer().tokenize(MidiDict.from_midi(path), **kw)

    def test_origin_segments_clip_and_end(self):
        notes = [(60, 5300, 11300, 83), (64, 9975, 10075, 70), (67, 10025, 10125, 70)]   # first > 5 s, a 6 s note, 10 s boundary
        toks = self.tokenize(notes, **TOKENIZE_KW)
        target = [(p, float(on), float(end), v) for p, on, end, v in notes]
        self.assertEqual(misaligned_notes(toks, target), 0)                    # original time origin kept
        self.assertEqual(toks[2], "<T>")                                       # header, <S>, then the 5 s boundary
        self.assertEqual([on // 5000 for on, _ in token_notes(toks)], [1, 1, 2])   # each side of the 10 s boundary
        durs = [toks[i + 2][1] for i, t in enumerate(toks) if isinstance(t, tuple) and t[0] == "piano"]
        self.assertEqual(durs[0], 5000)                                        # the 6 s note is clipped to the cap
        self.assertEqual(toks[toks.index(("piano", 60, 80))], ("piano", 60, 80))   # velocity 83 -> 80
        self.assertEqual(toks[-1], "<E>")
        self.assertNotIn("<D>", toks)

    def test_the_defaults_shift_the_origin_and_add_a_dim_token(self):
        notes = [(60 + i % 24, 700 + 125 * i, 750 + 125 * i, 64) for i in range(80)]    # > 150 tokens
        target = [(p, float(on), float(end), v) for p, on, end, v in notes]
        default = self.tokenize(notes)
        self.assertGreater(misaligned_notes(default, target), 0)
        self.assertIn("<D>", default)
        fixed = self.tokenize(notes, **TOKENIZE_KW)
        self.assertEqual((misaligned_notes(fixed, target), fixed.count("<D>")), (0, 0))


if __name__ == "__main__":
    unittest.main()
