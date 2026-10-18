"""LoRA adapters of the #1724 torch format on the MLX Aria TransformerLM (issue #1834, docs/experiments/RT_ADAPTER_Q8.md).

An adapter file is a dict "{layer}_{target}.A" [r, in] / "{layer}_{target}.B" [out, r] for the targets mixed_qkv and
att_proj_linear (scripts/aria_lora.py adds out + (x @ A.T @ B.T) * alpha / rank with a forward hook). The MLX blocks use
the same layer names, so the mapping is direct. Two ways to put it on the realtime model:
- delta: wrap each target (Linear or QuantizedLinear) and add the low-rank path in the model dtype; a swap only replaces
  the small A/B arrays
- merge: W + scale * B @ A in float32, then cast and (for q8) quantize the merged weight; no runtime cost, a swap
  re-merges and re-quantizes 32 matrices
"""
from __future__ import annotations

import re

import numpy as np

TARGETS = ("mixed_qkv", "att_proj_linear")
KEY = re.compile(r"^(\d+)_(" + "|".join(TARGETS) + r")\.([AB])$")


def parse_key(k):
    """'3_mixed_qkv.A' -> (3, 'mixed_qkv', 'A'); anything else raises."""
    m = KEY.match(k)
    if not m:
        raise ValueError(f"not an adapter key: {k}")
    return int(m.group(1)), m.group(2), m.group(3)


def from_state_dict(sd):
    """{(layer, target): (A, B)} as float32 numpy, with every pair complete and one rank."""
    parts = {}
    for k, v in sd.items():
        layer, target, ab = parse_key(k)
        arr = v.detach().cpu().float().numpy() if hasattr(v, "detach") else np.asarray(v, dtype=np.float32)
        parts.setdefault((layer, target), {})[ab] = arr
    out, ranks = {}, set()
    for key, d in parts.items():
        if set(d) != {"A", "B"}:
            raise ValueError(f"incomplete pair {key}: {sorted(d)}")
        a, b = d["A"], d["B"]
        if a.ndim != 2 or b.ndim != 2 or a.shape[0] != b.shape[1]:
            raise ValueError(f"bad shapes {key}: A {a.shape} B {b.shape}")
        ranks.add(a.shape[0])
        out[key] = (a, b)
    if len(ranks) != 1:
        raise ValueError(f"mixed ranks {sorted(ranks)}")
    return out


def load_adapter(path):
    import torch
    return from_state_dict(torch.load(path, map_location="cpu", weights_only=True))


def check_against(adapter, n_layers, shapes):
    """shapes: {target: (out, in)}. Every layer x target present with matching shapes."""
    want = {(i, t) for i in range(n_layers) for t in TARGETS}
    if set(adapter) != want:
        raise ValueError(f"adapter covers {len(adapter)} of {len(want)} targets")
    for (i, t), (a, b) in adapter.items():
        out_d, in_d = shapes[t]
        if a.shape[1] != in_d or b.shape[0] != out_d:
            raise ValueError(f"layer {i} {t}: A {a.shape} B {b.shape} vs weight {(out_d, in_d)}")


def merged(w, a, b, scale):
    """W + scale * B @ A in float32 (W [out, in], A [r, in], B [out, r])."""
    return w.astype(np.float32) + np.float32(scale) * (b.astype(np.float32) @ a.astype(np.float32))


def make_lora_linear():
    """The delta wrapper class (built lazily so this module imports without mlx)."""
    import mlx.core as mx
    import mlx.nn as nn

    class LoRALinear(nn.Module):
        def __init__(self, base, a, b, scale, dtype):
            super().__init__()
            self.base = base
            self.lora_a = mx.array(a).astype(dtype)
            self.lora_b = mx.array(b).astype(dtype)
            self.scale = scale

        def set_adapter(self, a, b, dtype):
            self.lora_a = mx.array(a).astype(dtype)
            self.lora_b = mx.array(b).astype(dtype)

        def __call__(self, x):
            return self.base(x) + ((x @ self.lora_a.T) @ self.lora_b.T) * self.scale

    return LoRALinear


def apply_delta(model, adapter, scale, dtype):
    """Wrap every target of every block; returns the original modules for remove_delta."""
    LoRALinear = make_lora_linear()
    orig = {}
    for i, block in enumerate(model.model.encode_layers):
        for t in TARGETS:
            base = getattr(block, t)
            orig[(i, t)] = base
            a, b = adapter[(i, t)]
            setattr(block, t, LoRALinear(base, a, b, scale, dtype))
    return orig


def swap_delta(model, adapter, dtype):
    """Replace A/B of the wrapped targets in place (no base change)."""
    for i, block in enumerate(model.model.encode_layers):
        for t in TARGETS:
            a, b = adapter[(i, t)]
            getattr(block, t).set_adapter(a, b, dtype)


def restore(model, modules):
    for (i, t), m in modules.items():
        setattr(model.model.encode_layers[i], t, m)


def build_merged(base_weights, adapter, scale, dtype, bits=None, group=32):
    """{(layer, target): module} with the merged weight (float32 merge, cast to dtype, quantized when bits is set).
    base_weights: {(layer, target): float32 numpy [out, in]}."""
    import mlx.core as mx
    import mlx.nn as nn
    out = {}
    for key, w in base_weights.items():
        a, b = adapter[key]
        m = merged(w, a, b, scale)
        lin = nn.Linear(m.shape[1], m.shape[0], bias=False)
        lin.weight = mx.array(m).astype(dtype)
        out[key] = nn.QuantizedLinear.from_linear(lin, group_size=group, bits=bits) if bits else lin
    return out


def current_modules(model):
    return {(i, t): getattr(block, t) for i, block in enumerate(model.model.encode_layers) for t in TARGETS}


def base_target_weights(ckpt):
    """float32 numpy weights of the targets from the checkpoint: {(layer, target): [out, in]}."""
    import mlx.core as mx
    w = mx.load(ckpt)
    out = {}
    for k, v in w.items():
        m = re.match(r"^model\.encode_layers\.(\d+)\.(" + "|".join(TARGETS) + r")\.weight$", k)
        if m:
            out[(int(m.group(1)), m.group(2))] = np.array(v.astype(mx.float32))
    return out
