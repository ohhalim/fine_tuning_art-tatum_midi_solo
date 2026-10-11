"""LoRA for a frozen Aria TransformerLM, kept outside the model (same scheme as the #1724 smoke).

The low-rank path is added by forward hooks on the target Linear layers, so the base state_dict and
its hash stay untouched and an adapter is just a small dict of A/B tensors that can be swapped.
A starts kaiming-uniform from a seeded generator, B starts at zero, so a fresh adapter equals the base.
The embedding output is marked as requiring grad so Aria's reentrant gradient checkpoint can route
gradients into the blocks; the embedding weights stay frozen.
"""
from __future__ import annotations

import math

import torch

TARGETS = ("mixed_qkv", "att_proj_linear")


class AriaLoRA:
    def __init__(self, model, rank=16, alpha=16, targets=TARGETS, seed=0, device="mps"):
        self.rank, self.alpha, self.targets, self.on = rank, alpha, targets, True
        gen = torch.Generator().manual_seed(seed)
        self.params = torch.nn.ModuleDict()
        for i, block in enumerate(model.model.encode_layers):
            for name in targets:
                lin = getattr(block, name)
                a = torch.empty(rank, lin.in_features)
                bound = 1 / math.sqrt(lin.in_features)
                a.uniform_(-bound, bound, generator=gen)
                self.params[f"{i}_{name}"] = torch.nn.ParameterDict(
                    {"A": torch.nn.Parameter(a), "B": torch.nn.Parameter(torch.zeros(lin.out_features, rank))})
        self.params.to(device)
        for i, block in enumerate(model.model.encode_layers):
            for name in targets:
                getattr(block, name).register_forward_hook(self._hook(f"{i}_{name}"))
        model.model.tok_embeddings.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))

    def _hook(self, key):
        def hook(_mod, inp, out):
            if not self.on:
                return out
            pr = self.params[key]
            return out + (inp[0] @ pr["A"].t() @ pr["B"].t()) * (self.alpha / self.rank)
        return hook

    def parameters(self):
        return self.params.parameters()

    def n_params(self):
        return sum(p.numel() for p in self.params.parameters())

    def state_dict(self):
        return {k: v.detach().cpu().clone() for k, v in self.params.state_dict().items()}

    def load_state_dict(self, sd):
        self.params.load_state_dict(sd)


def grad_problems(named):
    """Names of tensors whose grad is None or not finite. `named` is [(name, param)]."""
    bad = []
    for name, p in named:
        if p.grad is None:
            bad.append((name, "none"))
        elif not bool(torch.isfinite(p.grad).all()):
            bad.append((name, "non_finite"))
    return bad


def named_lora(lora):
    return [(f"{k}.{t}", lora.params[k][t]) for k in lora.params for t in ("A", "B")]


def unchanged(lora, init_sd):
    """Names of LoRA tensors still equal to their initial value."""
    return [n for n, p in named_lora(lora) if torch.equal(p.detach().cpu(), init_sd[n])]


def input_mismatches(recorded: dict, current: dict):
    """Keys whose recorded and current hashes differ or are missing on either side."""
    return sorted(k for k in set(recorded) | set(current) if recorded.get(k) is None or recorded.get(k) != current.get(k))
