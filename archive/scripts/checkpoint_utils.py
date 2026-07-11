"""Checkpoint loading helpers for Stage A vocabulary migrations."""

from __future__ import annotations

from typing import Iterable

import torch


TOKEN_LAYER_KEYS = ("embedding.weight", "Wout.weight", "Wout.bias")


def resize_token_layer_tensor(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Copy an older token layer into the current larger token layer shape."""
    if source.ndim != target.ndim:
        raise ValueError(f"Cannot resize tensor rank {source.ndim} into {target.ndim}")
    if source.shape[0] > target.shape[0]:
        raise ValueError(f"Checkpoint vocab size {source.shape[0]} is larger than model vocab size {target.shape[0]}")
    if tuple(source.shape[1:]) != tuple(target.shape[1:]):
        raise ValueError(f"Cannot resize token layer with incompatible trailing shape {source.shape} -> {target.shape}")

    resized = target.detach().clone()
    resized[: source.shape[0]] = source
    return resized


def resize_state_dict_token_layers(
    state_dict: dict[str, torch.Tensor],
    model_state_dict: dict[str, torch.Tensor],
    token_layer_keys: Iterable[str] = TOKEN_LAYER_KEYS,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    resized_state = dict(state_dict)
    resized_keys: list[str] = []

    for key in token_layer_keys:
        source = resized_state.get(key)
        target = model_state_dict.get(key)
        if source is None or target is None:
            continue
        if tuple(source.shape) == tuple(target.shape):
            continue
        resized_state[key] = resize_token_layer_tensor(source, target)
        resized_keys.append(key)

    return resized_state, resized_keys


def load_state_dict_with_token_resize(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    strict: bool = True,
) -> tuple[torch.nn.modules.module._IncompatibleKeys, list[str]]:
    resized_state, resized_keys = resize_state_dict_token_layers(state_dict, model.state_dict())
    result = model.load_state_dict(resized_state, strict=strict)
    return result, resized_keys
