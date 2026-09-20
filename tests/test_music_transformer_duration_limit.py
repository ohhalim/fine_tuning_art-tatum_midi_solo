from __future__ import annotations

import unittest
import sys
from pathlib import Path
from unittest.mock import patch

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "music_transformer"))
from model.music_transformer import MusicTransformer, _apply_generated_duration_limit
from utilities.constants import RANGE_NOTE_ON, TOKEN_END, VOCAB_SIZE


class DeterministicTokenModel:
    training = False

    def __init__(self, tokens: list[int]) -> None:
        self.tokens = iter(tokens)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        logits = torch.full((1, sequence.shape[1], VOCAB_SIZE), -1000.0)
        logits[0, -1, next(self.tokens)] = 1000.0
        return logits


class MusicTransformerDurationLimitTest(unittest.TestCase):
    def test_non_time_token_does_not_change_duration(self) -> None:
        token, elapsed, reached = _apply_generated_duration_limit(60, 20, 100)

        self.assertEqual(60, token)
        self.assertEqual(20, elapsed)
        self.assertFalse(reached)

    def test_time_shift_below_remaining_duration_is_preserved(self) -> None:
        token, elapsed, reached = _apply_generated_duration_limit(265, 20, 100)

        self.assertEqual(265, token)
        self.assertEqual(30, elapsed)
        self.assertFalse(reached)

    def test_boundary_time_shift_is_cropped_to_remaining_steps(self) -> None:
        token, elapsed, reached = _apply_generated_duration_limit(355, 80, 100)

        self.assertEqual(275, token)
        self.assertEqual(100, elapsed)
        self.assertTrue(reached)

    def test_completed_duration_rejects_another_time_shift(self) -> None:
        token, elapsed, reached = _apply_generated_duration_limit(256, 100, 100)

        self.assertIsNone(token)
        self.assertEqual(100, elapsed)
        self.assertTrue(reached)

    def test_generate_stops_at_duration_and_closes_generated_notes(self) -> None:
        sampled_note_on = 61
        sampled_ten_step_shift = RANGE_NOTE_ON * 2 + 9
        model = DeterministicTokenModel([sampled_note_on, sampled_ten_step_shift])

        with patch("model.music_transformer.get_device", return_value=torch.device("cpu")):
            generated = MusicTransformer.generate(
                model,
                primer=torch.tensor([60]),
                target_seq_length=10,
                top_k=1,
                sample_vocab_size=TOKEN_END,
                grammar_mask=True,
                target_duration_steps=5,
            )

        self.assertEqual(
            [60, sampled_note_on, RANGE_NOTE_ON * 2 + 4, RANGE_NOTE_ON + sampled_note_on],
            generated[0].tolist(),
        )

    def test_generation_metadata_separates_model_steps_from_boundary_note_offs(self) -> None:
        sampled_note_on = 61
        sampled_ten_step_shift = RANGE_NOTE_ON * 2 + 9
        model = DeterministicTokenModel([sampled_note_on, sampled_ten_step_shift])

        with patch("model.music_transformer.get_device", return_value=torch.device("cpu")):
            generated, metadata = MusicTransformer.generate(
                model,
                primer=torch.tensor([60]),
                target_seq_length=3,
                top_k=1,
                sample_vocab_size=TOKEN_END,
                grammar_mask=True,
                target_duration_steps=5,
                return_metadata=True,
            )

        self.assertEqual(2, metadata["model_forward_step_count"])
        self.assertEqual(2, metadata["sampled_output_token_count"])
        self.assertEqual(1, metadata["boundary_note_off_count"])
        self.assertEqual(3, metadata["returned_generated_token_count"])
        self.assertEqual("duration_target", metadata["stop_reason"])
        self.assertTrue(metadata["duration_target_reached"])
        self.assertEqual(4, generated.shape[1])

    def test_generation_metadata_records_token_budget_underfill(self) -> None:
        sampled_note_on = 61
        # Velocity bin 16, not bin 0: bin 0 is grammar-masked because it
        # decodes to MIDI velocity 0. Any audible bin exercises the same
        # path - a token that consumes budget without advancing duration.
        sampled_velocity = RANGE_NOTE_ON * 2 + 100 + 16
        model = DeterministicTokenModel([sampled_note_on, sampled_velocity])

        with patch("model.music_transformer.get_device", return_value=torch.device("cpu")):
            _generated, metadata = MusicTransformer.generate(
                model,
                primer=torch.tensor([60]),
                target_seq_length=3,
                top_k=1,
                sample_vocab_size=TOKEN_END,
                grammar_mask=True,
                target_duration_steps=5,
                return_metadata=True,
            )

        self.assertEqual("token_budget", metadata["stop_reason"])
        self.assertEqual(2, metadata["model_forward_step_count"])
        self.assertEqual(1, metadata["boundary_note_off_count"])
        self.assertFalse(metadata["duration_target_reached"])

    def test_generation_metadata_records_end_token_without_returning_it(self) -> None:
        model = DeterministicTokenModel([TOKEN_END])

        with patch("model.music_transformer.get_device", return_value=torch.device("cpu")):
            generated, metadata = MusicTransformer.generate(
                model,
                primer=torch.tensor([60]),
                target_seq_length=3,
                top_k=1,
                sample_vocab_size=TOKEN_END + 1,
                grammar_mask=True,
                target_duration_steps=5,
                return_metadata=True,
            )

        self.assertEqual([60], generated[0].tolist())
        self.assertEqual("end_token", metadata["stop_reason"])
        self.assertEqual(1, metadata["model_forward_step_count"])
        self.assertEqual(0, metadata["sampled_output_token_count"])
        self.assertEqual(0, metadata["returned_generated_token_count"])


if __name__ == "__main__":
    unittest.main()


class GrammarMaskSilentVelocityTest(unittest.TestCase):
    """Velocity bin 0 decodes to MIDI velocity 0, which is a note-off."""

    def test_silent_velocity_token_is_masked(self) -> None:
        from model.music_transformer import SILENT_VELOCITY_TOKEN, _apply_grammar_mask

        logits = torch.zeros(1, VOCAB_SIZE)
        masked = _apply_grammar_mask(logits, set())

        self.assertEqual(float("-inf"), masked[0, SILENT_VELOCITY_TOKEN].item())
        # The neighbouring audible bin must stay available.
        self.assertEqual(0.0, masked[0, SILENT_VELOCITY_TOKEN + 1].item())

    def test_generation_never_emits_the_silent_velocity_token(self) -> None:
        from model.music_transformer import SILENT_VELOCITY_TOKEN

        model = DeterministicTokenModel([SILENT_VELOCITY_TOKEN, 61, RANGE_NOTE_ON * 2 + 9])

        with patch("model.music_transformer.get_device", return_value=torch.device("cpu")):
            generated = MusicTransformer.generate(
                model, primer=torch.tensor([60]), target_seq_length=4, top_k=1,
                sample_vocab_size=TOKEN_END, grammar_mask=True, target_duration_steps=20,
            )

        self.assertNotIn(SILENT_VELOCITY_TOKEN, generated[0].tolist())
