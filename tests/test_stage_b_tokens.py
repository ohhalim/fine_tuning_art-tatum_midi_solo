from __future__ import annotations

import unittest

import pretty_midi

from scripts.stage_b_tokens import (
    CHORD_QUALITIES,
    CHORD_ROOTS,
    MAX_DURATION_STEPS,
    SEQUENCE_FORMAT_STAGE_B_V1,
    STAGE_B_TOKEN_START,
    STAGE_B_VOCAB_SIZE,
    TOKEN_CHORD_QUALITY_START,
    TOKEN_CHORD_ROOT_START,
    TOKEN_NOTE_DURATION_START,
    TOKEN_NOTE_PITCH_START,
    TOKEN_POSITION_START,
    TOKEN_VELOCITY_START,
    build_stage_b_sequence,
    chord_tokens,
    decode_stage_b_notes,
    note_duration_token,
    note_pitch_token,
    parse_chord_symbol,
    position_token,
    quantize_note_duration,
    stage_b_token_name,
    velocity_bin,
)
from scripts.control_tokens import tempo_control_token
from utilities.constants import TOKEN_BAR, TOKEN_CONTROL_END, TOKEN_END, TOKEN_ROLE_LEAD


class StageBTokensTest(unittest.TestCase):
    def test_stage_b_token_range_starts_after_stage_a_control_tokens(self) -> None:
        self.assertEqual(SEQUENCE_FORMAT_STAGE_B_V1, "stage_b_v1")
        self.assertEqual(STAGE_B_TOKEN_START, TOKEN_CONTROL_END + 1)
        self.assertGreater(STAGE_B_VOCAB_SIZE, TOKEN_CONTROL_END + 1)

    def test_chord_symbol_parser_normalizes_common_jazz_chords(self) -> None:
        self.assertEqual(parse_chord_symbol("Cm7"), ("C", "min7"))
        self.assertEqual(parse_chord_symbol("F7"), ("F", "dom7"))
        self.assertEqual(parse_chord_symbol("Bbmaj7"), ("Bb", "maj7"))
        self.assertEqual(parse_chord_symbol("F#m7b5"), ("F#", "halfdim"))
        self.assertEqual(parse_chord_symbol(None), ("N", "unknown"))

    def test_chord_tokens_are_explicit_root_and_quality_tokens(self) -> None:
        root_token, quality_token = chord_tokens("Cm7")

        self.assertEqual(root_token, TOKEN_CHORD_ROOT_START + CHORD_ROOTS.index("C"))
        self.assertEqual(quality_token, TOKEN_CHORD_QUALITY_START + CHORD_QUALITIES.index("min7"))

    def test_build_stage_b_sequence_uses_position_pitch_duration_velocity_tokens(self) -> None:
        notes = [
            pretty_midi.Note(velocity=84, pitch=60, start=0.0, end=0.25),
            pretty_midi.Note(velocity=96, pitch=64, start=0.5, end=0.75),
        ]

        tokens = build_stage_b_sequence(notes, tempo_bpm=120, chords=["Cm7"], bars=1)

        self.assertEqual(tokens[0], TOKEN_ROLE_LEAD)
        self.assertEqual(tokens[1], tempo_control_token(120))
        self.assertEqual(tokens[2], TOKEN_BAR)
        self.assertIn(position_token(0), tokens)
        self.assertIn(position_token(4), tokens)
        self.assertIn(TOKEN_VELOCITY_START + velocity_bin(84), tokens)
        self.assertIn(note_pitch_token(60), tokens)
        self.assertIn(note_pitch_token(64), tokens)
        self.assertIn(note_duration_token(2), tokens)
        self.assertEqual(tokens[-1], TOKEN_END)

    def test_stage_b_roundtrip_preserves_quantized_note_shape(self) -> None:
        notes = [
            pretty_midi.Note(velocity=84, pitch=60, start=0.0, end=0.25),
            pretty_midi.Note(velocity=96, pitch=64, start=0.5, end=0.75),
        ]

        tokens = build_stage_b_sequence(notes, tempo_bpm=120, chords=["Cm7"], bars=1)
        decoded = decode_stage_b_notes(tokens, tempo_bpm=120)

        self.assertEqual([note.pitch for note in decoded], [60, 64])
        self.assertAlmostEqual(decoded[0].start, 0.0)
        self.assertAlmostEqual(decoded[0].end, 0.25)
        self.assertAlmostEqual(decoded[1].start, 0.5)
        self.assertAlmostEqual(decoded[1].end, 0.75)
        self.assertTrue(all(note.velocity > 0 for note in decoded))

    def test_duration_quantization_clamps_long_sustain(self) -> None:
        self.assertEqual(quantize_note_duration(99.0, tempo_bpm=120), MAX_DURATION_STEPS)

    def test_stage_b_token_names_are_debuggable(self) -> None:
        self.assertEqual(stage_b_token_name(TOKEN_POSITION_START), "POSITION_0")
        self.assertEqual(stage_b_token_name(TOKEN_NOTE_PITCH_START), "NOTE_PITCH_21")
        self.assertEqual(stage_b_token_name(TOKEN_NOTE_DURATION_START), "NOTE_DURATION_1")
        self.assertEqual(stage_b_token_name(TOKEN_VELOCITY_START), "VELOCITY_0")


if __name__ == "__main__":
    unittest.main()
