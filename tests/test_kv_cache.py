"""KV-cached generation must match the full-recompute path (docs/experiments/KV_CACHE.md)."""
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

TINY = dict(n_layers=2, num_heads=2, d_model=16, dim_feedforward=32, max_sequence=48, rpr=True)


def tiny(seed=0, lora=()):
    torch.manual_seed(seed)
    model = MusicTransformer(**TINY)
    if lora:
        from scripts.train_qlora import add_lora_to_model

        model, _ = add_lora_to_model(model, r=2, alpha=4, dropout=0.0, targets=lora)
        with torch.no_grad():
            for name, p in model.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.2)
    return model.eval()


class CachedForwardTest(unittest.TestCase):
    def _check(self, model, length=40):
        x = torch.randint(0, 388, (length,))
        with torch.no_grad():
            full = model(x.unsqueeze(0))[0]
            chunk = model.forward_cached(x, {})
            cache, steps = {}, []
            steps.append(model.forward_cached(x[:7], cache))
            for t in range(7, length):
                steps.append(model.forward_cached(x[t:t + 1], cache))
            step = torch.cat(steps)
        self.assertLess((full - chunk).abs().max().item(), 1e-5)
        self.assertLess((full - step).abs().max().item(), 1e-5)

    def test_matches_full_forward(self) -> None:
        self._check(tiny(0))

    def test_matches_with_lora_targets(self) -> None:
        self._check(tiny(1, lora=("out_proj", "qkv", "ffn")))

    def test_full_length(self) -> None:
        self._check(tiny(2), length=48)


class CachedGenerateTest(unittest.TestCase):
    def test_same_tokens_as_full_recompute(self) -> None:
        model = tiny(3, lora=("out_proj",))
        primer = torch.tensor([372, 60, 260, 188, 372, 64, 262, 192])
        for seed in range(6):
            outs = []
            for use in (False, True):
                torch.manual_seed(seed)
                with torch.no_grad():
                    out, meta = model.generate(primer=primer, target_seq_length=40, temperature=1.0,
                                               top_k=32, top_p=0.95, grammar_mask=True,
                                               target_duration_steps=150, return_metadata=True,
                                               use_kv_cache=use)
                outs.append((out[0].tolist(), meta["stop_reason"], meta["model_forward_step_count"]))
            self.assertEqual(outs[0], outs[1], f"seed {seed}")

    def test_non_rpr_falls_back(self) -> None:
        torch.manual_seed(0)
        model = MusicTransformer(**{**TINY, "rpr": False}).eval()
        self.assertFalse(model.supports_kv_cache())


if __name__ == "__main__":
    unittest.main()
