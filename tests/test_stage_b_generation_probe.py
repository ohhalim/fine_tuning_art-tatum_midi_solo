from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pretty_midi
import torch

from scripts.run_stage_b_generation_probe import (
    build_stage_b_primer,
    decode_tokens_to_midi,
    generate_stage_b_tokens,
)
from scripts.stage_b_tokens import (
    chord_tokens,
    note_duration_token,
    note_pitch_token,
    note_velocity_token,
    position_token,
)
from utilities.constants import TOKEN_BAR, TOKEN_END, TOKEN_ROLE_LEAD, VOCAB_SIZE


class FakeStageBModel:
    def __init__(self, returned_tokens: list[int]) -> None:
        self.returned_tokens = returned_tokens
        self.sample_vocab_size: int | None = None

    def generate(self, **kwargs):
        self.sample_vocab_size = kwargs["sample_vocab_size"]
        return torch.tensor([self.returned_tokens], dtype=torch.long)


class StageBGenerationProbeTest(unittest.TestCase):
    def test_build_stage_b_primer_contains_bar_and_first_chord(self) -> None:
        primer = build_stage_b_primer(["Cm7", "F7"], bpm=124)

        self.assertEqual(primer[0], TOKEN_ROLE_LEAD)
        self.assertEqual(primer[2], TOKEN_BAR)
        self.assertEqual(primer[3:5], chord_tokens("Cm7"))

    def test_generation_uses_full_stage_b_vocab_size(self) -> None:
        returned_tokens = build_stage_b_primer(["Cm7"], bpm=124) + [TOKEN_END]
        model = FakeStageBModel(returned_tokens)

        tokens = generate_stage_b_tokens(
            model=model,
            primer_tokens=returned_tokens[:-1],
            target_length=16,
            temperature=0.9,
            top_k=32,
            top_p=None,
        )

        self.assertEqual(tokens, returned_tokens)
        self.assertEqual(model.sample_vocab_size, VOCAB_SIZE)

    def test_decode_tokens_to_midi_writes_stage_b_notes(self) -> None:
        tokens = build_stage_b_primer(["Cm7"], bpm=120) + [
            position_token(0),
            note_velocity_token(4),
            note_pitch_token(60),
            note_duration_token(2),
            TOKEN_END,
        ]
        with tempfile.TemporaryDirectory() as tmp_dir:
            midi_path = Path(tmp_dir) / "decoded.mid"

            decode_tokens_to_midi(tokens, midi_path, bpm=120)

            midi = pretty_midi.PrettyMIDI(str(midi_path))
            notes = midi.instruments[0].notes
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0].pitch, 60)
            self.assertAlmostEqual(notes[0].start, 0.0)
            self.assertAlmostEqual(notes[0].end, 0.25)


if __name__ == "__main__":
    unittest.main()
