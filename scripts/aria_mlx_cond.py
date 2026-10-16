"""Per-position left-hand condition on the MLX runtime model (issue #1760, docs/experiments/COND_ADAPTER_MIN.md).

The condition of an input position is the 13-dim payload of block_request (12-PC multihot + available). A bias-free
Linear(13 -> d_model) adds it to the input of the last encoder layer, as the torch adapters did
(scripts/aria_cond_smoke.py: forward pre-hook on encode_layers[-1]). The upstream Aria code is not changed: cond_forward
repeats aria.inference.model_mlx.Transformer.__call__ with the injection added.
"""
from __future__ import annotations

COND_DIM = 13


def cond_vector(pay):
    """block_request payload (12-PC tuple, available) -> 13 floats."""
    pcs, available = pay
    return [float(x) for x in pcs] + [float(available)]


def cond_rows(tags, pays):
    """Per input position: the payload of the request in `tags` (-1 = cold) as 13 floats."""
    from block_request import COLD
    return [cond_vector(COLD if j < 0 else pays[j]) for j in tags]


def cond_forward(model, idxs, input_pos, offset, max_kv_pos, cond, weight):
    """aria.inference.model_mlx.TransformerLM forward with cond @ weight.T added to the last encoder layer's input.

    idxs: int32 [1, S]; input_pos: int32 [S]; cond: float32 [1, S, 13]; weight: float32 [d_model, 13] (Linear, no bias).
    Writes the KV cache at input_pos like the upstream forward and returns logits [1, S, vocab]."""
    m = model.model
    mask = m.causal_mask[None, None, input_pos, : max_kv_pos + 1]
    x = m.tok_embeddings(idxs)
    last = len(m.encode_layers) - 1
    for i, layer in enumerate(m.encode_layers):
        if i == last:
            x = x + cond @ weight.T
        x = layer(x, input_pos, max_kv_pos, offset, mask)
    return model.lm_head(m.out_layer_norm(x))
