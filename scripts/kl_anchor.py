"""KL anchor to the frozen base for the WJazzD KL comparison (docs/experiments/ARIA_COND_CONTRACT_V2.md).

KL(p_adapter || p_base) over the full vocabulary from float32 log_softmax on both sides (no softmax
then log). Gradient flows through the adapter logits only; the base logits are detached. Positions:
k with loss mask True (the target is token k + 1) and gate[k] == 0 (onset or dur prediction).
With no selected position the term is zero and the count is reported.
"""
from __future__ import annotations

import torch


def kl_positions(mask, gate):
    """mask[k] is for target k + 1; gate[k] comes from tokens[0..k]. Both indexed by position k."""
    return [k for k in range(len(mask)) if mask[k] and not gate[k]]


def kl_adapter_base(logits_adapter: torch.Tensor, logits_base: torch.Tensor) -> torch.Tensor:
    """Per-position KL(p_a || p_b), shape [positions]."""
    lpa = torch.log_softmax(logits_adapter.float(), dim=-1)
    lpb = torch.log_softmax(logits_base.detach().float(), dim=-1)
    return (lpa.exp() * (lpa - lpb)).sum(-1)


def kl_term(logits_adapter, logits_base, positions):
    """Mean KL over the selected positions; zero (and n = 0) when there is none."""
    if not positions:
        return logits_adapter.sum() * 0.0, 0
    idx = torch.tensor(positions, device=logits_adapter.device)
    return kl_adapter_base(logits_adapter[idx], logits_base[idx]).mean(), len(positions)
