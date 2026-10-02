"""Top line only (inference/control/solo_line.py)."""
from __future__ import annotations

import tempfile
import unittest
from unittest import mock

import pretty_midi

from inference.control import solo_line
from inference.control.solo_line import PhraseBreath, render_block, solo_line_tokens, solo_with_comp_tokens, top_notes
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
        self.assertEqual(pitches, [48, 55, 72, 74, 76])                  # comp 74 dropped: overlaps the line's 74
        self.assertEqual(stage_a_musical_duration_ms(out), stage_a_musical_duration_ms(self.block()))
        self.assertTrue(validate_generated_token_block(out, lookahead_ms=940, allow_rest_bar=True)["valid"])

    def test_comp_pitch_the_line_plays_later_is_kept(self) -> None:
        # comp 76 at 0.0-0.25 does not overlap the line's 76 at 0.6-0.9
        out = solo_with_comp_tokens(self.block(), [note(76, 0.0, 0.25, 56)])
        self.assertEqual(sorted(n.pitch for n in tokens_to_notes(out)), [72, 74, 76, 76])
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


class PhraseBreathTest(unittest.TestCase):
    def run_line(self, n, ioi=0.1, max_notes=4, block=1.0):
        breath = PhraseBreath(max_notes, rest_s=0.4)
        line = [note(60 + k % 5, k * ioi, k * ioi + ioi) for k in range(n)]
        kept = []
        for b in range(int(n * ioi / block) + 1):                          # split across blocks
            part = [note(x.pitch, x.start - b * block, x.end - b * block) for x in line
                    if b * block <= x.start < (b + 1) * block]
            kept += [round(b * block + x.start, 2) for x in breath(part, b * block)]
        return kept, breath

    def test_a_rest_follows_max_notes_even_across_blocks(self) -> None:
        kept, breath = self.run_line(12, max_notes=4)
        # 4 notes (0.0-0.3) end at 0.4; starts before 0.8 drop; the block boundary at 1.0 keeps the count
        self.assertEqual(kept, [0.0, 0.1, 0.2, 0.3, 0.8, 0.9, 1.0, 1.1])
        self.assertEqual(breath.dropped, 4)

    def test_a_natural_rest_resets_the_count(self) -> None:
        breath = PhraseBreath(3, rest_s=0.4)
        notes = [note(60, 0.0, 0.1), note(62, 0.1, 0.2), note(64, 0.6, 0.7), note(65, 0.7, 0.8), note(67, 0.8, 0.9)]
        self.assertEqual(len(breath(notes, 0.0)), 5)                       # gap 0.4 s after two notes
        self.assertEqual(breath.dropped, 0)

    def test_render_with_breath_stays_valid_and_keeps_length(self) -> None:
        toks = SoloLineTest().block()
        out = render_block(toks, lookahead_ms=940, line_filter=lambda ns: PhraseBreath(1)(ns, 0.0))
        self.assertEqual([n.pitch for n in tokens_to_notes(out)], [72])
        self.assertEqual(stage_a_musical_duration_ms(out), stage_a_musical_duration_ms(toks))
        empty = solo_line_tokens(toks, line_filter=lambda ns: [])
        self.assertEqual(tokens_to_notes(empty), [])
        self.assertEqual(stage_a_musical_duration_ms(empty), stage_a_musical_duration_ms(toks))


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
