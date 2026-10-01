"""Top line only (inference/control/solo_line.py)."""
from __future__ import annotations

import tempfile
import unittest

import pretty_midi

from inference.control.solo_line import solo_line_tokens, top_notes
from scripts.generate import encode_notes_simple
from scripts.run_continuous_jazz import main
from scripts.run_resident_model_probe import stage_a_musical_duration_ms, validate_generated_token_block
from scripts.style_distance import tokens_to_notes


def note(p, s, e, v=80):
    return pretty_midi.Note(velocity=v, pitch=p, start=s, end=e)


class SoloLineTest(unittest.TestCase):
    def block(self):
        # left-hand chord under a melody, then two melody notes; block is 0.94 s
        notes = [note(48, 0.0, 0.5), note(52, 0.0, 0.5), note(55, 0.01, 0.5), note(72, 0.0, 0.3),
                 note(74, 0.3, 0.6), note(76, 0.6, 0.9)]
        toks = encode_notes_simple(sorted(notes, key=lambda n: (n.start, n.pitch)))
        return toks + [255 + 4]                                  # rest to 0.94 s

    def test_only_the_top_line_remains_and_the_length_is_kept(self) -> None:
        toks = self.block()
        solo = solo_line_tokens(toks)
        self.assertEqual([n.pitch for n in tokens_to_notes(solo)], [72, 74, 76])
        self.assertEqual(stage_a_musical_duration_ms(solo), stage_a_musical_duration_ms(toks))
        self.assertTrue(validate_generated_token_block(solo, lookahead_ms=940, allow_rest_bar=True)["valid"])

    def test_top_notes_are_monophonic(self) -> None:
        kept = top_notes([note(72, 0.0, 0.5), note(74, 0.2, 0.6), note(60, 0.21, 0.4)])
        self.assertEqual([(n.pitch, n.start, n.end) for n in kept], [(72, 0.0, 0.2), (74, 0.2, 0.6)])

    def test_rest_blocks_are_left_alone(self) -> None:
        self.assertEqual(solo_line_tokens([255 + 94]), [255 + 94])

    def test_flag_needs_the_sub_block_path(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid", "--solo-line"])


if __name__ == "__main__":
    unittest.main()
