"""Regression for the LoRA checks used by the Mehldau Aria runner (issue #1726). CPU, tiny fake model."""
from __future__ import annotations

import os
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from aria_lora import AriaLoRA, grad_problems, input_mismatches, named_lora, unchanged  # noqa: E402


class Block(torch.nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mixed_qkv = torch.nn.Linear(d, 3 * d, bias=False)
        self.att_proj_linear = torch.nn.Linear(3 * d, d, bias=False)

    def forward(self, x):
        return x + self.att_proj_linear(self.mixed_qkv(x))


class Inner(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.tok_embeddings = torch.nn.Embedding(10, d)
        self.encode_layers = torch.nn.ModuleList(Block(d) for _ in range(n))

    def forward(self, ids):
        x = self.tok_embeddings(ids)
        for b in self.encode_layers:
            x = b(x)
        return x


class Fake(torch.nn.Module):
    def __init__(self, d=4, n=2):
        super().__init__()
        self.model = Inner(d, n)

    def forward(self, ids):
        return self.model(ids)


def frozen(seed=0):
    torch.manual_seed(seed)
    m = Fake()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class LoRAChecks(unittest.TestCase):
    def setUp(self):
        self.m = frozen()
        self.lora = AriaLoRA(self.m, rank=2, alpha=2, seed=0, device="cpu")
        self.ids = torch.tensor([[1, 2, 3]])

    def test_fresh_adapter_equals_base(self):
        self.lora.on = False
        a = self.m(self.ids)
        self.lora.on = True
        self.assertTrue(torch.equal(a, self.m(self.ids)))

    def test_grads_reach_every_tensor_after_one_step(self):
        opt = torch.optim.Adam(self.lora.parameters(), lr=0.1)
        for _ in range(2):
            opt.zero_grad(set_to_none=True)
            self.m(self.ids).sum().backward()
            self.assertEqual(grad_problems(named_lora(self.lora)), [])
            opt.step()
        self.assertTrue(all(p.grad is None for p in self.m.parameters()))

    def test_grad_problems_flags_none_and_non_finite(self):
        named = named_lora(self.lora)
        self.assertEqual(len(grad_problems(named)), len(named))          # no backward yet: all None
        self.m(self.ids).sum().backward()
        named[0][1].grad[0, 0] = float("nan")
        self.assertEqual(grad_problems(named), [(named[0][0], "non_finite")])

    def test_unchanged_detects_untouched_tensors(self):
        init = {n: p.detach().clone() for n, p in named_lora(self.lora)}
        self.assertEqual(len(unchanged(self.lora, init)), 8)   # 2 blocks x 2 targets x (A, B)
        with torch.no_grad():
            self.lora.params["0_mixed_qkv"]["B"].add_(1.0)
        self.assertNotIn("0_mixed_qkv.B", unchanged(self.lora, init))
        self.assertIn("0_mixed_qkv.A", unchanged(self.lora, init))

    def test_reload_from_file_restores_logits(self):
        with torch.no_grad():
            for p in self.lora.parameters():
                p.add_(0.3)
        before = self.m(self.ids)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.pt")
            torch.save(self.lora.state_dict(), path)
            with torch.no_grad():
                for p in self.lora.parameters():
                    p.zero_()
            self.assertFalse(torch.equal(before, self.m(self.ids)))
            self.lora.load_state_dict(torch.load(path))
        self.assertTrue(torch.equal(before, self.m(self.ids)))

    def test_input_mismatches(self):
        rec = {"data": "a", "lora": "b"}
        self.assertEqual(input_mismatches(rec, {"data": "a", "lora": "b"}), [])
        self.assertEqual(input_mismatches(rec, {"data": "a", "lora": "x"}), ["lora"])
        self.assertEqual(input_mismatches({"data": "a", "lora": None}, {"data": "a", "lora": "b"}), ["lora"])
        self.assertEqual(input_mismatches(rec, {"data": "a"}), ["lora"])


if __name__ == "__main__":
    unittest.main()
