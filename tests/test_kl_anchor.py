"""KL anchor (scripts/kl_anchor.py)."""
from __future__ import annotations

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))

from scripts.kl_anchor import kl_adapter_base, kl_positions, kl_term


class KlAnchorTest(unittest.TestCase):
    def test_zero_for_identical_and_nonnegative(self) -> None:
        torch.manual_seed(0)
        a = torch.randn(5, 50)
        self.assertLess(kl_adapter_base(a, a.clone()).abs().max().item(), 1e-6)
        b = torch.randn(5, 50)
        self.assertGreaterEqual(kl_adapter_base(a, b).min().item(), -1e-6)

    def test_finite_for_extreme_logits(self) -> None:
        a = torch.tensor([[1000.0, -1000.0, 0.0]])
        b = torch.tensor([[-1000.0, 1000.0, 0.0]])
        v = kl_adapter_base(a, b)
        self.assertTrue(torch.isfinite(v).all().item())
        self.assertGreater(v.item(), 100.0)

    def test_gradient_only_through_adapter_and_teacher_unchanged(self) -> None:
        a = torch.randn(3, 10, requires_grad=True)
        b = torch.randn(3, 10, requires_grad=True)
        before = b.detach().clone()
        loss, n = kl_term(a, b, [0, 2])
        loss.backward()
        self.assertEqual(n, 2)
        self.assertIsNotNone(a.grad)
        self.assertTrue(torch.all(a.grad[1] == 0).item())         # unselected position gets no gradient
        self.assertIsNone(b.grad)
        self.assertTrue(torch.equal(b.detach(), before))

    def test_positions_follow_shifted_mask_and_gate(self) -> None:
        # tokens: <S>, pitch, onset, dur, pitch, onset, dur ; gate after each token
        gate = [1, 0, 0, 1, 0, 0, 1]
        mask = [False, True, True, True, True, True]                # targets 1..6 (target k + 1)
        self.assertEqual(kl_positions(mask, gate[:-1]), [1, 2, 4, 5])  # positions predicting onset or dur

    def test_no_position_gives_zero(self) -> None:
        a = torch.randn(2, 4, requires_grad=True)
        loss, n = kl_term(a, a.detach(), [])
        self.assertEqual((loss.item(), n), (0.0, 0))
        loss.backward()
        self.assertTrue(torch.all(a.grad == 0).item())


if __name__ == "__main__":
    unittest.main()
