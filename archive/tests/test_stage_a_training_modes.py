from __future__ import annotations

import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

import torch

from scripts import train_stage_a_adapter, train_stage_a_full
from scripts.control_tokens import control_prefix_tokens
from scripts.train_qlora import (
    apply_checkpoint_model_config,
    crop_control_v1_sequence,
    state_dict_is_lora_only,
    state_dict_uses_lora_wrappers,
    training_mode_name,
)
from utilities.constants import TOKEN_COND_SEP, TOKEN_PAD


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

    def test_control_v1_crop_preserves_prompt_separator_and_target_window(self) -> None:
        prefix = control_prefix_tokens("lead", 124)
        conditioning = list(range(20, 140))
        target = list(range(200, 360))
        tokens = torch.tensor(prefix + conditioning + [TOKEN_COND_SEP] + target, dtype=torch.long)

        cropped = crop_control_v1_sequence(tokens, max_seq=96, conditioning_max_tokens=32)
        sep_index = int((cropped == TOKEN_COND_SEP).nonzero(as_tuple=False)[0].item())

        self.assertEqual(len(cropped), 96)
        self.assertEqual(cropped[:3].tolist(), prefix)
        self.assertEqual(sep_index, 35)
        self.assertEqual(cropped[3:sep_index].tolist(), conditioning[-32:])
        self.assertGreater(len(cropped[sep_index + 1 :]), 0)

    def test_control_v1_crop_pads_when_target_window_is_short(self) -> None:
        prefix = control_prefix_tokens("lead", 124)
        conditioning = list(range(20, 140))
        target = [200, 201]
        tokens = torch.tensor(prefix + conditioning + [TOKEN_COND_SEP] + target, dtype=torch.long)

        cropped = crop_control_v1_sequence(tokens, max_seq=96, conditioning_max_tokens=32)

        self.assertEqual(len(cropped), 96)
        self.assertEqual(cropped[:3].tolist(), prefix)
        self.assertIn(TOKEN_COND_SEP, cropped.tolist())
        self.assertEqual(cropped[-1].item(), TOKEN_PAD)


if __name__ == "__main__":
    unittest.main()
