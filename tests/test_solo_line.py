"""Top line only (inference/control/solo_line.py)."""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import pretty_midi

from inference.control import solo_line
from inference.control.solo_line import render_block, solo_line_tokens, solo_with_comp_tokens, top_notes
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

    def test_comp_adds_the_voicing_but_never_a_pitch_the_line_plays(self) -> None:
        comp = [note(48, 0.0, 0.56, 56), note(55, 0.0, 0.56, 56), note(74, 0.0, 0.56, 56)]
        out = solo_with_comp_tokens(self.block(), comp)
        pitches = sorted(n.pitch for n in tokens_to_notes(out))
        self.assertEqual(pitches, [48, 55, 72, 74, 76])                  # comp 74 dropped: the line has it
        self.assertEqual(stage_a_musical_duration_ms(out), stage_a_musical_duration_ms(self.block()))
        self.assertTrue(validate_generated_token_block(out, lookahead_ms=940, allow_rest_bar=True)["valid"])

    def test_comp_needs_solo_line(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid", "--chord-primer",
                  "--chord-blocks-per-bar", "2", "--comp"])

    def test_invalid_raw_blocks_stay_invalid_so_the_fallback_plays(self) -> None:
        # Astra's reproduction: duplicate note_on / orphan note_off became valid after the rewrite.
        stats = {}
        for raw in ([376, 60, 60, 349, 188], [376, 60, 349, 188, 188]):
            self.assertTrue(validate_generated_token_block(solo_line_tokens(raw), lookahead_ms=940,
                                                           allow_rest_bar=True)["valid"])   # the old trap
            out = render_block(raw, lookahead_ms=940, stats=stats)
            self.assertEqual(out, raw)
            self.assertFalse(validate_generated_token_block(out, lookahead_ms=940, allow_rest_bar=True)["valid"])
        self.assertEqual(stats, {"raw_invalid": 2})

    def test_valid_raw_blocks_are_rendered_and_counted(self) -> None:
        stats = {}
        out = render_block(self.block(), lookahead_ms=940, stats=stats)
        self.assertEqual([n.pitch for n in tokens_to_notes(out)], [72, 74, 76])
        self.assertEqual(stats, {"rendered": 1})

    def test_a_rewrite_that_fails_validation_plays_the_raw_block(self) -> None:
        stats = {}
        with mock.patch.object(solo_line, "solo_line_tokens", return_value=[60]):
            out = render_block(self.block(), lookahead_ms=940, stats=stats)
        self.assertEqual(out, self.block())
        self.assertEqual(stats, {"rendered_invalid": 1})

    def test_rest_blocks_are_left_alone(self) -> None:
        self.assertEqual(solo_line_tokens([255 + 94]), [255 + 94])

    def test_flag_needs_the_sub_block_path(self) -> None:
        with tempfile.TemporaryDirectory() as d, self.assertRaises(SystemExit):
            main(["--output-dir", d, "--checkpoint", "x.pt", "--conditioning-midi", "p.mid", "--solo-line"])


if __name__ == "__main__":
    unittest.main()


class ShellVoicingTest(unittest.TestCase):
    def test_root_third_seventh_without_seconds(self) -> None:
        from inference.control.solo_line import shell_voicing
        self.assertEqual(shell_voicing("Dm7"), [38, 53, 60])      # D2 F3 C4
        self.assertEqual(shell_voicing("G7"), [43, 53, 59])       # G2 F3 B3
        self.assertEqual(shell_voicing("Cmaj7"), [36, 52, 59])    # C2 E3 B3
        for ch in ("Dbmaj7", "Gb7", "Bbm7b5", "Abm7", "E7"):
            v = shell_voicing(ch)
            upper = v[1:]
            self.assertTrue(all(b - a >= 3 for a, b in zip(upper, upper[1:])), (ch, v))
