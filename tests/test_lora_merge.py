"""merge_lora_for_inference must not change outputs (docs/experiments/LORA_MERGE.md)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "music_transformer"))

from utilities.device import use_cuda

use_cuda(False)

from model.music_transformer import MusicTransformer
from scripts.train_qlora import add_lora_to_model, merge_lora_for_inference

TINY = dict(n_layers=2, num_heads=2, d_model=16, dim_feedforward=32, max_sequence=48, rpr=True)


def tiny(targets, seed=0):
    torch.manual_seed(seed)
    model = MusicTransformer(**TINY)
    model, _ = add_lora_to_model(model, r=2, alpha=4, dropout=0.0, targets=targets)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "lora_B" in name:
                p.normal_(0, 0.2)
    return model.eval()


class LoraMergeTest(unittest.TestCase):
    def _check(self, targets, expect):
        model = tiny(targets)
        x = torch.randint(0, 388, (1, 30))
        with torch.no_grad():
            before = model(x)
            merged = merge_lora_for_inference(model)
            after = model(x)
            cached = model.forward_cached(x[0], {})
        self.assertEqual(merged, expect)
        self.assertLess((before - after).abs().max().item(), 1e-5)
        self.assertLess((before[0] - cached).abs().max().item(), 1e-5)
        self.assertFalse(any("lora_" in k for k in model.state_dict()))

    def test_out_proj(self) -> None:
        self._check(("out_proj",), {"linear": 2, "qkv": 0})

    def test_out_proj_qkv_ffn(self) -> None:
        self._check(("out_proj", "qkv", "ffn"), {"linear": 6, "qkv": 2})

    def test_generation_tokens_identical(self) -> None:
        primer = torch.tensor([372, 60, 260, 188, 372, 64, 262, 192])
        outs = []
        for merge in (False, True):
            model = tiny(("out_proj", "qkv"), seed=4)
            if merge:
                merge_lora_for_inference(model)
            row = []
            for seed in range(5):
                torch.manual_seed(seed)
                with torch.no_grad():
                    out = model.generate(primer=primer, target_seq_length=40, temperature=1.0,
                                         top_k=32, top_p=0.95, grammar_mask=True,
                                         target_duration_steps=150, use_kv_cache=True)
                row.append(out[0].tolist())
            outs.append(row)
        self.assertEqual(outs[0], outs[1])

    def test_plain_model_is_a_no_op(self) -> None:
        torch.manual_seed(0)
        model = MusicTransformer(**TINY).eval()
        x = torch.randint(0, 388, (1, 20))
        with torch.no_grad():
            before = model(x)
            self.assertEqual(merge_lora_for_inference(model), {"linear": 0, "qkv": 0})
            self.assertTrue(torch.equal(before, model(x)))

    def test_wrappers_from_the_top_level_module_are_merged(self) -> None:
        # scripts/generate.py imports this file as ``train_qlora``, so checkpoints
        # loaded through it carry that module's LoRALayer class (#1520).
        sys.path.insert(0, str(ROOT / "scripts"))
        import train_qlora as top_level
        self.assertIsNot(top_level.LoRALayer, sys.modules["scripts.train_qlora"].LoRALayer)
        torch.manual_seed(0)
        model = MusicTransformer(**TINY)
        model, _ = top_level.add_lora_to_model(model, r=2, alpha=4, dropout=0.0,
                                               targets=("out_proj", "qkv", "ffn"))
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.2)
        model.eval()
        x = torch.randint(0, 388, (1, 30))
        with torch.no_grad():
            before = model(x)
            self.assertEqual(merge_lora_for_inference(model), {"linear": 6, "qkv": 2})
            self.assertLess((before - model(x)).abs().max().item(), 1e-5)
        self.assertFalse(any("lora_" in k for k in model.state_dict()))


class RuntimeDefaultTest(unittest.TestCase):
    def test_runtime_merges_by_default_with_opt_out(self) -> None:
        text = (ROOT / "scripts" / "run_continuous_jazz.py").read_text()
        self.assertIn('"--merge-lora", action=argparse.BooleanOptionalAction, default=True', text)


if __name__ == "__main__":
    unittest.main()
