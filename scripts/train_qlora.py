"""
Music Transformer QLoRA Fine-tuning Script
재즈 피아노 MIDI 데이터셋으로 Music Transformer를 QLoRA 파인튜닝

Usage:
    python scripts/train_qlora.py --data_dir ./data/jazz_processed --epochs 3

Reference:
    - gwinndr/MusicTransformer-Pytorch
    - ICLR 2019: Music Transformer (Huang et al.)
"""

import os
import sys
import argparse
import math
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add music_transformer to path. Walk up from this file until we find the
# repo root that actually contains music_transformer/ (robust to the script
# living in scripts/ or archive/scripts/).
def _find_repo_root(start: Path) -> Path:
    for cand in [start, *start.parents]:
        if (cand / "music_transformer").is_dir():
            return cand
    return start.parent.parent  # fallback to original behaviour

SCRIPT_DIR = _find_repo_root(Path(__file__).resolve())
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss
from utilities.constants import TOKEN_COND_SEP, TOKEN_PAD, VOCAB_SIZE

# checkpoint_utils lives next to this script; make sure its dir is importable.
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from checkpoint_utils import load_state_dict_with_token_resize
except ModuleNotFoundError:
    from scripts.checkpoint_utils import load_state_dict_with_token_resize


# =============================================================================
# LoRA Implementation for Music Transformer
# =============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for linear projections.
    
    This wrapper provides weight/bias properties for compatibility with
    rpr.py's multi_head_attention_forward_rpr which accesses .weight directly.
    """
    
    def __init__(self, original_layer: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.original_layer = original_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        
        # Freeze original weights
        self.original_layer.weight.requires_grad = False
        if self.original_layer.bias is not None:
            self.original_layer.bias.requires_grad = False
    
    @property
    def weight(self):
        """Return the effective weight with LoRA applied.
        
        This allows rpr.py to use .weight directly while still applying LoRA.
        """
        # Compute LoRA delta: B @ A gives [out_features, in_features]
        lora_weight = self.lora_B @ self.lora_A
        return self.original_layer.weight + lora_weight * self.scaling
    
    @property
    def bias(self):
        """Return the original bias"""
        return self.original_layer.bias
    
    @property
    def in_features(self):
        return self.original_layer.in_features
    
    @property
    def out_features(self):
        return self.original_layer.out_features
    
    def forward(self, x):
        # Original forward
        result = self.original_layer(x)
        # LoRA forward
        lora_out = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        return result + lora_out * self.scaling


class LoRALinearWeight(nn.Module):
    """LoRA for in_proj_weight style (combined Q, K, V projection)"""
    
    def __init__(self, embed_dim: int, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        
        # LoRA for Q, K, V separately
        self.lora_A_q = nn.Parameter(torch.zeros(r, embed_dim))
        self.lora_B_q = nn.Parameter(torch.zeros(embed_dim, r))
        self.lora_A_k = nn.Parameter(torch.zeros(r, embed_dim))
        self.lora_B_k = nn.Parameter(torch.zeros(embed_dim, r))
        self.lora_A_v = nn.Parameter(torch.zeros(r, embed_dim))
        self.lora_B_v = nn.Parameter(torch.zeros(embed_dim, r))
        
        self.lora_dropout = nn.Dropout(dropout)
        
        # Initialize
        for param in [self.lora_A_q, self.lora_A_k, self.lora_A_v]:
            nn.init.kaiming_uniform_(param, a=math.sqrt(5))
        for param in [self.lora_B_q, self.lora_B_k, self.lora_B_v]:
            nn.init.zeros_(param)
    
    def forward(self, x, original_weight, original_bias=None):
        """Apply LoRA to QKV projection"""
        # Original projection
        result = torch.nn.functional.linear(x, original_weight, original_bias)
        
        # Split into Q, K, V
        embed_dim = original_weight.shape[0] // 3
        
        # LoRA additions
        x_drop = self.lora_dropout(x)
        lora_q = x_drop @ self.lora_A_q.T @ self.lora_B_q.T * self.scaling
        lora_k = x_drop @ self.lora_A_k.T @ self.lora_B_k.T * self.scaling
        lora_v = x_drop @ self.lora_A_v.T @ self.lora_B_v.T * self.scaling
        
        # Add LoRA to result
        result_q = result[..., :embed_dim] + lora_q
        result_k = result[..., embed_dim:2*embed_dim] + lora_k
        result_v = result[..., 2*embed_dim:] + lora_v
        
        return torch.cat([result_q, result_k, result_v], dim=-1)


LORA_TARGETS = ("out_proj", "qkv", "ffn")
DEFAULT_LORA_TARGETS = ("out_proj",)
_QKV_LORA_CLASSES: dict[type, type] = {}


def _qkv_lora_class(base_cls: type) -> type:
    """Subclass whose ``in_proj_weight`` adds a q/k/v LoRA delta.

    The attention forward reads ``self.in_proj_weight``; a class-level property
    wins over the registered Parameter, which stays in ``_parameters`` so the
    checkpoint key and the frozen base weight are unchanged.
    """
    if base_cls not in _QKV_LORA_CLASSES:
        def in_proj_weight(self):
            base = self._parameters["in_proj_weight"]
            delta = torch.cat([self.lora_B_q @ self.lora_A_q,
                               self.lora_B_k @ self.lora_A_k,
                               self.lora_B_v @ self.lora_A_v], dim=0)
            return base + delta * self.lora_qkv_scaling

        _QKV_LORA_CLASSES[base_cls] = type(f"{base_cls.__name__}QKVLoRA", (base_cls,),
                                           {"in_proj_weight": property(in_proj_weight)})
    return _QKV_LORA_CLASSES[base_cls]


def enable_qkv_lora(attn: nn.Module, r: int, alpha: int) -> list[nn.Parameter]:
    if "lora_A_q" in attn._parameters:
        return []
    embed_dim = attn._parameters["in_proj_weight"].shape[1]
    params = []
    for name in ("q", "k", "v"):
        a = nn.Parameter(torch.zeros(r, embed_dim))
        nn.init.kaiming_uniform_(a, a=math.sqrt(5))
        b = nn.Parameter(torch.zeros(embed_dim, r))
        setattr(attn, f"lora_A_{name}", a)
        setattr(attn, f"lora_B_{name}", b)
        params += [a, b]
    attn.lora_qkv_scaling = alpha / r
    attn.__class__ = _qkv_lora_class(type(attn))
    return params


def _encoder_layers(model: nn.Module):
    transformer = model.transformer
    encoder = getattr(transformer, "encoder", None) or getattr(transformer, "custom_encoder", None)
    return [] if encoder is None else list(encoder.layers)


@torch.no_grad()
def merge_lora_for_inference(model: nn.Module) -> dict:
    """Fold every LoRA delta into its base weight, in place, for inference only.

    out_proj / FFN ``LoRALayer`` wrappers become plain ``nn.Linear`` layers holding
    W + scale * B @ A; QKV LoRA is added into the ``in_proj_weight`` Parameter and
    the q/k/v LoRA parameters and property subclass are removed. The forward pass
    then does no per-call delta arithmetic. Not reversible; do not train afterwards.

    Wrappers are recognised by shape, not by ``isinstance``: this file is imported
    both as ``train_qlora`` (scripts/generate.py) and ``scripts.train_qlora``, which
    gives two distinct ``LoRALayer`` classes (#1520). Any ``lora_`` tensor left
    afterwards is an error.
    """
    merged = {"linear": 0, "qkv": 0}

    def is_lora_linear(module: nn.Module) -> bool:
        return (isinstance(getattr(module, "original_layer", None), nn.Linear)
                and isinstance(getattr(module, "lora_A", None), torch.Tensor)
                and isinstance(getattr(module, "lora_B", None), torch.Tensor)
                and hasattr(module, "scaling"))

    def fold(layer: "LoRALayer") -> nn.Linear:
        base = layer.original_layer
        out = nn.Linear(base.in_features, base.out_features, bias=base.bias is not None,
                        device=base.weight.device, dtype=base.weight.dtype)
        out.weight.copy_(base.weight + (layer.lora_B @ layer.lora_A) * layer.scaling)
        if base.bias is not None:
            out.bias.copy_(base.bias)
        out.requires_grad_(False)
        return out

    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if is_lora_linear(child):
                setattr(module, name, fold(child))
                merged["linear"] += 1
    for layer in _encoder_layers(model):
        attn = getattr(layer, "self_attn", None)
        if attn is None or "lora_A_q" not in attn._parameters:
            continue
        delta = torch.cat([attn.lora_B_q @ attn.lora_A_q, attn.lora_B_k @ attn.lora_A_k,
                           attn.lora_B_v @ attn.lora_A_v], dim=0) * attn.lora_qkv_scaling
        attn._parameters["in_proj_weight"].add_(delta)
        for n in ("q", "k", "v"):
            del attn._parameters[f"lora_A_{n}"]
            del attn._parameters[f"lora_B_{n}"]
        attn.__class__ = attn.__class__.__mro__[1]    # drop the property subclass
        merged["qkv"] += 1
    left = [k for k in model.state_dict() if "lora_" in k]
    if left:
        raise RuntimeError(f"LoRA tensors left after merging: {len(left)}, e.g. {left[:3]}")
    return merged


def add_lora_targets(model: nn.Module, targets, r: int = 16, alpha: int = 32,
                     dropout: float = 0.05) -> nn.ModuleList:
    """Add LoRA for ``qkv`` and/or ``ffn`` to an already frozen model (idempotent).

    Kept separate from ``add_lora_to_model`` so a base checkpoint saved with only
    out_proj LoRA can be loaded first and the extra targets attached after.
    """
    unknown = set(targets) - set(LORA_TARGETS)
    if unknown:
        raise ValueError(f"unknown LoRA targets: {sorted(unknown)}")
    added = nn.ModuleList()
    for layer in _encoder_layers(model):
        if "qkv" in targets and hasattr(layer, "self_attn"):
            enable_qkv_lora(layer.self_attn, r, alpha)
        if "ffn" in targets:
            for name in ("linear1", "linear2"):
                current = getattr(layer, name, None)
                if isinstance(current, nn.Linear):
                    wrapped = LoRALayer(current, r=r, alpha=alpha, dropout=dropout)
                    setattr(layer, name, wrapped)
                    added.append(wrapped)
    return added


def add_lora_to_model(model: MusicTransformer, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                      targets=DEFAULT_LORA_TARGETS):
    """Add LoRA layers to Music Transformer attention layers.

    ``targets`` defaults to out_proj only, the layout of every checkpoint before
    2026-09-27; ``qkv`` and ``ffn`` are opt-in (docs/experiments/TATUM_LORA_TARGETS.md).
    """
    targets = tuple(targets)
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    # Add LoRA to each encoder layer's output projection
    lora_modules = nn.ModuleList()
    
    if hasattr(model.transformer, 'encoder'):
        encoder = model.transformer.encoder
    else:
        encoder = model.transformer.custom_encoder if hasattr(model.transformer, 'custom_encoder') else None
    
    if encoder is None:
        print("Warning: Could not find encoder in model")
        return model, lora_modules
    
    for i, layer in enumerate(encoder.layers):
        if hasattr(layer, 'self_attn'):
            attn = layer.self_attn
            
            # Add LoRA to out_proj
            if "out_proj" in targets and hasattr(attn, 'out_proj'):
                original_out_proj = attn.out_proj
                lora_out = LoRALayer(original_out_proj, r=r, alpha=alpha, dropout=dropout)
                attn.out_proj = lora_out
                lora_modules.append(lora_out)
                print(f"  Added LoRA to layer {i} out_proj")

    extra = [t for t in targets if t != "out_proj"]
    if extra:
        lora_modules.extend(add_lora_targets(model, extra, r=r, alpha=alpha, dropout=dropout))
        print(f"  Added LoRA targets: {extra}")

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, lora_modules


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def unfreeze_base_model(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def model_config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_layers": int(args.n_layers),
        "num_heads": int(args.num_heads),
        "d_model": int(args.d_model),
        "dim_feedforward": int(args.dim_feedforward),
        "max_sequence": int(args.max_sequence),
        "rpr": bool(args.rpr),
        "lora_r": int(args.lora_r),
        "lora_alpha": int(args.lora_alpha),
        "lora_dropout": float(args.lora_dropout),
        "lora_targets": list(args.lora_targets),
    }


def checkpoint_payload_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
        raise ValueError(f"Unsupported model_state_dict type: {type(state_dict)}")
    if isinstance(checkpoint, dict):
        return checkpoint
    raise ValueError(f"Unsupported checkpoint payload type: {type(checkpoint)}")


def checkpoint_model_config(checkpoint: object) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("model_config"), dict):
        return dict(checkpoint["model_config"])
    return {}


def state_dict_uses_lora_wrappers(state_dict: dict[str, torch.Tensor]) -> bool:
    return any("lora_" in key.lower() or ".original_layer." in key for key in state_dict)


def lora_targets_in_state_dict(state_dict: dict[str, torch.Tensor]) -> list[str]:
    """Which LoRA targets a saved state dict already carries."""
    found = []
    if any(".out_proj.lora_" in k for k in state_dict):
        found.append("out_proj")
    if any(".self_attn.lora_A_q" in k for k in state_dict):
        found.append("qkv")
    if any(".linear1.original_layer." in k or ".linear1.lora_" in k for k in state_dict):
        found.append("ffn")
    return found


def build_lora_model_from_state(model_config: dict[str, Any], state_dict: dict[str, torch.Tensor],
                                extra_targets=()) -> tuple[nn.Module, list[str]]:
    """Rebuild the layout a checkpoint was saved with, load it strictly, then attach extras.

    A plain base (no LoRA wrappers) is loaded before any LoRA is added; a
    LoRA-wrapped checkpoint is loaded into its own saved targets. Only targets
    it does not already carry are attached afterwards, starting from zero delta.
    """
    model = MusicTransformer(
        n_layers=int(model_config["n_layers"]), num_heads=int(model_config["num_heads"]),
        d_model=int(model_config["d_model"]), dim_feedforward=int(model_config["dim_feedforward"]),
        max_sequence=int(model_config["max_sequence"]), rpr=bool(model_config["rpr"]))
    r, alpha = int(model_config["lora_r"]), int(model_config["lora_alpha"])
    dropout = float(model_config.get("lora_dropout", 0.05))
    requested = list(dict.fromkeys(extra_targets))
    if state_dict_uses_lora_wrappers(state_dict):
        saved = lora_targets_in_state_dict(state_dict) or list(DEFAULT_LORA_TARGETS)
        model, _ = add_lora_to_model(model, r=r, alpha=alpha, dropout=dropout, targets=saved)
        load_state_dict_with_token_resize(model, state_dict, strict=True)
        remaining = [t for t in requested if t not in saved]
        if remaining:
            add_lora_targets(model, remaining, r=r, alpha=alpha, dropout=dropout)
        targets = list(dict.fromkeys([*saved, *remaining]))
    else:
        load_state_dict_with_token_resize(model, state_dict, strict=True)
        targets = requested or list(DEFAULT_LORA_TARGETS)
        model, _ = add_lora_to_model(model, r=r, alpha=alpha, dropout=dropout, targets=targets)
    return model, targets


def load_lora_snapshot(model: nn.Module, snapshot: dict[str, torch.Tensor]) -> list[str]:
    """Load a LoRA-only snapshot, failing closed on any key mismatch.

    ``load_state_dict(strict=False)`` alone would silently drop snapshot keys
    the model has no slot for (e.g. QKV tensors into an out_proj-only model)
    and leave model LoRA tensors the snapshot does not cover untouched.
    """
    if not snapshot or not all("lora_" in k for k in snapshot):
        raise ValueError("snapshot must contain only lora_ tensors")
    model_keys = {k for k in model.state_dict() if "lora_" in k}
    snap_keys = set(snapshot)
    missing, extra = sorted(model_keys - snap_keys), sorted(snap_keys - model_keys)
    if missing or extra:
        raise ValueError(f"LoRA snapshot/model mismatch: missing {len(missing)} {missing[:3]}, "
                         f"unexpected {len(extra)} {extra[:3]}")
    model.load_state_dict(snapshot, strict=False)
    return lora_targets_in_state_dict(snapshot)


def state_dict_is_lora_only(state_dict: dict[str, torch.Tensor]) -> bool:
    return bool(state_dict) and all("lora_" in key.lower() for key in state_dict)


def apply_checkpoint_model_config(args: argparse.Namespace, model_config: dict[str, Any]) -> None:
    for name in [
        "n_layers",
        "num_heads",
        "d_model",
        "dim_feedforward",
        "max_sequence",
        "lora_r",
        "lora_alpha",
    ]:
        if name in model_config:
            setattr(args, name, int(model_config[name]))
    if "rpr" in model_config:
        args.rpr = bool(model_config["rpr"])
    if "lora_dropout" in model_config:
        args.lora_dropout = float(model_config["lora_dropout"])
    if "lora_targets" in model_config:
        args.lora_targets = list(model_config["lora_targets"])


def training_mode_name(args: argparse.Namespace) -> str:
    if args.train_full_model:
        return "full_model"
    if args.checkpoint:
        return "adapter"
    return "random_base_lora"


# =============================================================================
# Dataset
# =============================================================================

CONTROL_PREFIX_LEN = 3
CONTROL_CONDITIONING_MAX_TOKENS = 64


def crop_control_v1_sequence(
    tokens: torch.Tensor,
    max_seq: int,
    conditioning_max_tokens: int = CONTROL_CONDITIONING_MAX_TOKENS,
) -> torch.Tensor:
    sep_positions = (tokens == TOKEN_COND_SEP).nonzero(as_tuple=False).flatten()
    if len(sep_positions) == 0:
        start = random.randint(0, len(tokens) - max_seq)
        return tokens[start : start + max_seq]

    sep_index = int(sep_positions[0].item())
    prefix_len = min(CONTROL_PREFIX_LEN, sep_index)
    prefix = tokens[:prefix_len]
    conditioning = tokens[prefix_len:sep_index]
    target = tokens[sep_index + 1 :]
    if len(target) == 0:
        return tokens[:max_seq]

    max_conditioning = max(0, min(int(conditioning_max_tokens), max_seq - len(prefix) - 2))
    conditioning_tail = conditioning[-max_conditioning:] if max_conditioning > 0 else conditioning[:0]
    target_budget = max_seq - len(prefix) - len(conditioning_tail) - 1
    if target_budget <= 0:
        return tokens[:max_seq]

    if len(target) > target_budget:
        target_start = random.randint(0, len(target) - target_budget)
        target = target[target_start : target_start + target_budget]

    cropped = torch.cat(
        [
            prefix,
            conditioning_tail,
            tokens.new_tensor([TOKEN_COND_SEP]),
            target,
        ]
    )
    if len(cropped) < max_seq:
        padding = torch.full((max_seq - len(cropped),), TOKEN_PAD, dtype=torch.long)
        cropped = torch.cat([cropped, padding])
    return cropped[:max_seq]


class MidiDataset(Dataset):
    """Dataset for preprocessed MIDI token sequences"""
    
    def __init__(self, data_dir: str, max_seq: int = 2048, split: str = "train"):
        self.max_seq = max_seq
        self.data_dir = Path(data_dir) / split
        
        # Load all .npy files
        self.files = list(self.data_dir.glob("*.npy"))
        if not self.files:
            raise ValueError(f"No .npy files found in {self.data_dir}")
        
        print(f"Loaded {len(self.files)} files for {split}")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        import numpy as np
        
        tokens = np.load(self.files[idx])
        tokens = torch.from_numpy(tokens).long()
        
        # Truncate or pad
        if len(tokens) > self.max_seq:
            tokens = crop_control_v1_sequence(tokens, self.max_seq)
        elif len(tokens) < self.max_seq:
            padding = torch.full((self.max_seq - len(tokens),), TOKEN_PAD, dtype=torch.long)
            tokens = torch.cat([tokens, padding])
        
        # Input and target (shifted by 1)
        x = tokens[:-1]
        y = tokens[1:]
        
        return x, y


# =============================================================================
# Training
# =============================================================================

def train_epoch(
    model,
    dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epoch,
    gradient_accumulation: int = 1,
):
    model.train()
    total_loss = 0

    gradient_accumulation = max(1, int(gradient_accumulation))
    optimizer.zero_grad(set_to_none=True)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Forward
        output = model(x)

        # Compute loss
        raw_loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
        loss = raw_loss / gradient_accumulation

        # Backward
        loss.backward()
        should_step = (
            (batch_idx + 1) % gradient_accumulation == 0
            or (batch_idx + 1) == len(dataloader)
        )
        if should_step:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.item()
        current_lr = optimizer.param_groups[0]["lr"]
        pbar.set_postfix({"loss": f"{raw_loss.item():.4f}", "lr": f"{current_lr:.2e}"})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device, crop_seed: int | None = None):
    """Mean loss over the loader.

    MidiDataset crops long songs with ``random``; with ``crop_seed`` the crops
    are the same every call, and the caller's random state is restored so the
    training crop stream is untouched.
    """
    model.eval()
    total_loss = 0
    saved_state = random.getstate() if crop_seed is not None else None
    if crop_seed is not None:
        random.seed(crop_seed)
    try:
        with torch.no_grad():
            for x, y in tqdm(dataloader, desc="Evaluating"):
                x, y = x.to(device), y.to(device)
                output = model(x)
                loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
                total_loss += loss.item()
    finally:
        if saved_state is not None:
            random.setstate(saved_state)

    return total_loss / len(dataloader)


def optimizer_updates_per_epoch(num_batches: int, gradient_accumulation: int) -> int:
    """train_epoch steps once per ``gradient_accumulation`` batches plus a trailing partial."""
    return math.ceil(num_batches / max(1, int(gradient_accumulation)))


def scheduler_total_steps(num_batches: int, gradient_accumulation: int, epochs: int,
                          mode: str = "optimizer_updates") -> int:
    """Cosine length. ``legacy_batches`` reproduces runs before 2026-09-27, which
    counted batches although the scheduler only steps on optimizer updates."""
    if mode == "legacy_batches":
        return num_batches * epochs
    if mode == "optimizer_updates":
        return optimizer_updates_per_epoch(num_batches, gradient_accumulation) * epochs
    raise ValueError(f"unknown scheduler step mode: {mode}")


def main():
    parser = argparse.ArgumentParser(description="Train Music Transformer with LoRA")
    
    # Data
    parser.add_argument("--data_dir", type=str, default="./data/jazz_processed",
                        help="Directory containing preprocessed MIDI data")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained checkpoint")
    
    # Model
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--dim_feedforward", type=int, default=1024)
    parser.add_argument("--max_sequence", type=int, default=512)
    parser.add_argument("--rpr", action="store_true", default=True,
                        help="Use Relative Position Representation")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_targets", nargs="+", default=list(DEFAULT_LORA_TARGETS),
                        choices=list(LORA_TARGETS),
                        help="out_proj (default, all existing checkpoints), qkv, ffn")
    parser.add_argument(
        "--train_full_model",
        action="store_true",
        help=(
            "Keep LoRA modules in the checkpoint format, but unfreeze the base "
            "Music Transformer too. Intended for tiny overfit smoke tests from "
            "a random base."
        ),
    )
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)  # Optimized for 24GB VRAM
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scheduler_steps",
        choices=["optimizer_updates", "legacy_batches"],
        default="optimizer_updates",
        help=(
            "Cosine T_max unit. legacy_batches reproduces D0-D4 and the first "
            "Mehldau run, whose lr barely decayed because T_max counted batches."
        ),
    )
    parser.add_argument(
        "--val_crop_seed",
        type=int,
        default=0,
        help="Seed for validation crops so val loss is comparable across epochs. -1 = random (legacy).",
    )
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints/jazz_lora")

    # Device: auto -> cuda, else mps (Apple Silicon), else cpu
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device. 'auto' prefers CUDA, then MPS, then CPU.",
    )

    args = parser.parse_args()
    set_seed(args.seed)

    # Device selection. Must stay consistent with the model's internal
    # get_device() (music_transformer.py builds the causal mask via get_device()),
    # so we drive both through utilities.device.
    from utilities.device import get_device, use_cuda, mps_device
    if args.device == "cpu":
        use_cuda(False)
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but CUDA is not available.")
        use_cuda(True)
    elif args.device == "mps":
        if mps_device() is None:
            raise SystemExit(
                "--device mps requested but MPS is not available. "
                "Need Apple Silicon + macOS 14+ and an MPS-enabled torch build. "
                "Run scripts/mps_smoke_test.py to diagnose."
            )
        use_cuda(True)  # get_device() falls through CUDA(None) -> MPS
    else:  # auto
        use_cuda(True)
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint_state_dict = None
    checkpoint_uses_lora_wrappers = False
    if args.checkpoint:
        print(f"Loading checkpoint metadata: {args.checkpoint}")
        checkpoint_payload = torch.load(args.checkpoint, map_location=device)
        checkpoint_state_dict = checkpoint_payload_state_dict(checkpoint_payload)
        if state_dict_is_lora_only(checkpoint_state_dict):
            raise ValueError(
                "Adapter/full training requires a full checkpoint or base model checkpoint; "
                f"got LoRA-only weights: {args.checkpoint}"
            )
        requested_targets = list(args.lora_targets)
        apply_checkpoint_model_config(args, checkpoint_model_config(checkpoint_payload))
        # Keep what the checkpoint already has and add what was asked for.
        args.lora_targets = list(dict.fromkeys(
            [*lora_targets_in_state_dict(checkpoint_state_dict), *args.lora_targets,
             *requested_targets]))
        checkpoint_uses_lora_wrappers = state_dict_uses_lora_wrappers(checkpoint_state_dict)
    
    # Initialize model
    print("\n=== Initializing Music Transformer ===")
    model = MusicTransformer(
        n_layers=args.n_layers,
        num_heads=args.num_heads,
        d_model=args.d_model,
        dim_feedforward=args.dim_feedforward,
        max_sequence=args.max_sequence,
        rpr=args.rpr
    )
    
    if checkpoint_state_dict is not None and not checkpoint_uses_lora_wrappers:
        print(f"Loading base checkpoint before LoRA: {args.checkpoint}")
        _, resized_keys = load_state_dict_with_token_resize(model, checkpoint_state_dict, strict=True)
        if resized_keys:
            print(f"Resized checkpoint token layers for current vocab: {', '.join(resized_keys)}")
    
    # Add LoRA
    print("\n=== Adding LoRA Layers ===")
    # A LoRA-wrapped checkpoint is loaded into the layout it was saved with;
    # targets it lacks are attached afterwards, starting from zero delta.
    initial_targets = (lora_targets_in_state_dict(checkpoint_state_dict) or list(DEFAULT_LORA_TARGETS)
                       if checkpoint_state_dict is not None and checkpoint_uses_lora_wrappers
                       else list(args.lora_targets))
    model, lora_modules = add_lora_to_model(
        model,
        r=args.lora_r,
        alpha=args.lora_alpha,
        dropout=args.lora_dropout,
        targets=initial_targets,
    )
    if checkpoint_state_dict is not None and checkpoint_uses_lora_wrappers:
        print(f"Loading LoRA-wrapped full checkpoint after LoRA: {args.checkpoint}")
        _, resized_keys = load_state_dict_with_token_resize(model, checkpoint_state_dict, strict=True)
        if resized_keys:
            print(f"Resized checkpoint token layers for current vocab: {', '.join(resized_keys)}")
        remaining = [t for t in args.lora_targets if t not in initial_targets]
        if remaining:
            lora_modules.extend(add_lora_targets(model, remaining, r=args.lora_r,
                                                 alpha=args.lora_alpha, dropout=args.lora_dropout))
            print(f"Added LoRA targets after load: {remaining}")
    if args.train_full_model:
        print("\n=== Tiny-overfit mode: unfreezing base model parameters ===")
        unfreeze_base_model(model)
        trainable_params, total_params = count_trainable_parameters(model)
        print(f"Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    model = model.to(device)
    
    # Dataset
    print("\n=== Loading Dataset ===")
    train_dataset = MidiDataset(args.data_dir, max_seq=args.max_sequence, split="train")
    val_dataset = MidiDataset(args.data_dir, max_seq=args.max_sequence, split="val")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers),
        pin_memory=device.type == "cuda",
    )
    
    # Optimizer & Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    
    updates_per_epoch = optimizer_updates_per_epoch(len(train_loader), args.gradient_accumulation)
    total_steps = scheduler_total_steps(
        len(train_loader), args.gradient_accumulation, args.epochs, args.scheduler_steps
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    print(
        f"Optimizer updates: {updates_per_epoch}/epoch, {updates_per_epoch * args.epochs} total "
        f"({len(train_loader)} batches, accumulation {args.gradient_accumulation}); "
        f"cosine T_max={total_steps} ({args.scheduler_steps})"
    )
    
    # Loss
    loss_fn = SmoothCrossEntropyLoss(
        args.label_smoothing,
        VOCAB_SIZE,
        ignore_index=TOKEN_PAD,
    )
    
    # Training loop
    print("\n=== Starting Training ===")
    best_val_loss = float("inf")
    model_config = model_config_from_args(args)
    training_config = {
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "gradient_accumulation": int(args.gradient_accumulation),
        "label_smoothing": float(args.label_smoothing),
        "seed": int(args.seed),
        "train_full_model": bool(args.train_full_model),
        "training_mode": training_mode_name(args),
        "checkpoint": args.checkpoint,
        "scheduler_steps": args.scheduler_steps,
        "scheduler_t_max": int(total_steps),
        "optimizer_updates_per_epoch": int(updates_per_epoch),
        "val_crop_seed": None if args.val_crop_seed < 0 else int(args.val_crop_seed),
    }
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            loss_fn,
            device,
            epoch,
            gradient_accumulation=args.gradient_accumulation,
        )
        val_loss = evaluate(
            model, val_loader, loss_fn, device,
            crop_seed=None if args.val_crop_seed < 0 else args.val_crop_seed,
        )
        
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            
            # Save only LoRA weights
            lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora" in k.lower()}
            torch.save(lora_state_dict, os.path.join(args.output_dir, "lora_weights.pt"))
            print(f"  Saved best LoRA weights (val_loss={val_loss:.4f})")
        
        # Save checkpoint
        torch.save({
            "epoch": epoch,
            "model_config": model_config,
            "training_config": training_config,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "optimizer_updates": int(updates_per_epoch * epoch),
        }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"LoRA weights saved to: {args.output_dir}/lora_weights.pt")


if __name__ == "__main__":
    main()
