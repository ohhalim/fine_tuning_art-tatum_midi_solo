from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

from scripts import train_stage_a_adapter, train_stage_a_full
from scripts.train_qlora import (
    apply_checkpoint_model_config,
    state_dict_is_lora_only,
    state_dict_uses_lora_wrappers,
    training_mode_name,
)


def training_args(**overrides):
    defaults = {
        "data_dir": "./data/roles/lead/tokenized",
        "output_dir": "./checkpoints/test",
        "checkpoint": "./checkpoints/base.pt",
        "epochs": 3,
        "batch_size": 8,
        "lr": 2e-4,
        "gradient_accumulation": 4,
        "num_workers": 0,
        "label_smoothing": 0.1,
        "seed": 42,
        "max_sequence": 512,
        "n_layers": 6,
        "num_heads": 8,
        "d_model": 512,
        "dim_feedforward": 1024,
        "rpr": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "train_full_model": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class StageATrainingModesTest(unittest.TestCase):
    def test_full_training_wrapper_enables_full_model_training(self) -> None:
        cmd = train_stage_a_full.build_train_command(training_args())

        self.assertIn("--train_full_model", cmd)
        self.assertNotIn("--checkpoint", cmd)

    def test_adapter_wrapper_requires_checkpoint_and_keeps_base_frozen(self) -> None:
        cmd = train_stage_a_adapter.build_train_command(training_args(checkpoint="./base.pt"))

        self.assertIn("--checkpoint", cmd)
        self.assertIn("./base.pt", cmd)
        self.assertNotIn("--train_full_model", cmd)

    def test_adapter_parser_rejects_missing_checkpoint(self) -> None:
        parser = train_stage_a_adapter.build_parser()

        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_checkpoint_state_detects_lora_wrapped_and_lora_only_weights(self) -> None:
        self.assertTrue(state_dict_uses_lora_wrappers({"layer.out_proj.lora_A": object()}))
        self.assertTrue(state_dict_uses_lora_wrappers({"layer.out_proj.original_layer.weight": object()}))
        self.assertTrue(state_dict_is_lora_only({"layer.out_proj.lora_A": object()}))
        self.assertFalse(
            state_dict_is_lora_only(
                {
                    "layer.out_proj.lora_A": object(),
                    "layer.out_proj.original_layer.weight": object(),
                }
            )
        )

    def test_checkpoint_model_config_overrides_training_args(self) -> None:
        args = training_args(n_layers=6, d_model=512, lora_r=16)

        apply_checkpoint_model_config(
            args,
            {
                "n_layers": 2,
                "d_model": 128,
                "lora_r": 8,
                "rpr": False,
                "lora_dropout": 0.0,
            },
        )

        self.assertEqual(args.n_layers, 2)
        self.assertEqual(args.d_model, 128)
        self.assertEqual(args.lora_r, 8)
        self.assertFalse(args.rpr)
        self.assertEqual(args.lora_dropout, 0.0)

    def test_training_mode_name_separates_full_adapter_and_random_base_lora(self) -> None:
        self.assertEqual(training_mode_name(training_args(train_full_model=True)), "full_model")
        self.assertEqual(training_mode_name(training_args(checkpoint="./base.pt")), "adapter")
        self.assertEqual(training_mode_name(training_args(checkpoint=None)), "random_base_lora")


if __name__ == "__main__":
    unittest.main()
