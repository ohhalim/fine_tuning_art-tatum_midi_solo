from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from scripts.checkpoint_utils import resize_state_dict_token_layers
from scripts.generate import (
    checkpoint_model_config,
    infer_checkpoint_max_sequence,
    resolve_full_checkpoint_path,
)


class StageACheckpointLoadingTest(unittest.TestCase):
    def test_resolves_latest_epoch_checkpoint_from_lora_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            older = root / "checkpoint_epoch1.pt"
            newer = root / "checkpoint_epoch12.pt"
            unrelated = root / "lora_weights.pt"
            older.write_bytes(b"old")
            newer.write_bytes(b"new")
            unrelated.write_bytes(b"lora")

            resolved = resolve_full_checkpoint_path(root)

        self.assertEqual(resolved, newer)

    def test_explicit_checkpoint_path_takes_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            auto_checkpoint = root / "checkpoint_epoch12.pt"
            explicit_checkpoint = root / "custom_full_checkpoint.pt"
            auto_checkpoint.write_bytes(b"auto")
            explicit_checkpoint.write_bytes(b"explicit")

            resolved = resolve_full_checkpoint_path(root, explicit_checkpoint)

        self.assertEqual(resolved, explicit_checkpoint)

    def test_missing_explicit_checkpoint_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.pt"

            with self.assertRaises(FileNotFoundError):
                resolve_full_checkpoint_path(tmp_dir, missing)

    def test_infers_model_max_sequence_from_checkpoint_state(self) -> None:
        class FakeTensor:
            shape = (512, 1, 512)

        state_dict = {"positional_encoding.pe": FakeTensor()}

        self.assertEqual(infer_checkpoint_max_sequence(state_dict, fallback=256), 512)

    def test_reads_model_config_from_training_checkpoint(self) -> None:
        checkpoint = {
            "model_config": {
                "n_layers": 2,
                "num_heads": 4,
                "d_model": 128,
                "max_sequence": 128,
            },
            "model_state_dict": {},
        }

        config = checkpoint_model_config(checkpoint)

        self.assertEqual(config["n_layers"], 2)
        self.assertEqual(config["d_model"], 128)

    def test_resizes_old_vocab_token_layers_into_current_model_shape(self) -> None:
        old_state = {
            "embedding.weight": torch.ones((390, 4)),
            "Wout.weight": torch.ones((390, 4)) * 2,
            "Wout.bias": torch.ones((390,)) * 3,
        }
        model_state = {
            "embedding.weight": torch.zeros((397, 4)),
            "Wout.weight": torch.zeros((397, 4)),
            "Wout.bias": torch.zeros((397,)),
        }

        resized, resized_keys = resize_state_dict_token_layers(old_state, model_state)

        self.assertEqual(resized_keys, ["embedding.weight", "Wout.weight", "Wout.bias"])
        self.assertEqual(tuple(resized["embedding.weight"].shape), (397, 4))
        self.assertTrue(torch.all(resized["embedding.weight"][:390] == 1))
        self.assertTrue(torch.all(resized["embedding.weight"][390:] == 0))


if __name__ == "__main__":
    unittest.main()
