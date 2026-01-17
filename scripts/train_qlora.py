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

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add music_transformer to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from model.music_transformer import MusicTransformer
from model.loss import SmoothCrossEntropyLoss
from utilities.constants import TOKEN_PAD


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
        """Return the effective weight (original + LoRA delta)"""
        # For compatibility: return original weight
        # The actual LoRA computation happens in forward
        return self.original_layer.weight
    
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


def add_lora_to_model(model: MusicTransformer, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    """Add LoRA layers to Music Transformer attention layers"""
    
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
            if hasattr(attn, 'out_proj'):
                original_out_proj = attn.out_proj
                lora_out = LoRALayer(original_out_proj, r=r, alpha=alpha, dropout=dropout)
                attn.out_proj = lora_out
                lora_modules.append(lora_out)
                print(f"  Added LoRA to layer {i} out_proj")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTrainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    
    return model, lora_modules


# =============================================================================
# Dataset
# =============================================================================

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
            start = random.randint(0, len(tokens) - self.max_seq)
            tokens = tokens[start:start + self.max_seq]
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

def train_epoch(model, dataloader, optimizer, scheduler, loss_fn, device, epoch):
    model.train()
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        output = model(x)
        
        # Compute loss
        loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
        
        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, loss_fn, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = loss_fn(output.view(-1, output.size(-1)), y.view(-1))
            total_loss += loss.item()
    
    return total_loss / len(dataloader)


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
    parser.add_argument("--max_sequence", type=int, default=2048)
    parser.add_argument("--rpr", action="store_true", default=True,
                        help="Use Relative Position Representation")
    
    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    
    # Output
    parser.add_argument("--output_dir", type=str, default="./checkpoints/jazz_lora")
    
    args = parser.parse_args()
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
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
    
    # Load pretrained weights if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        state_dict = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state_dict)
    
    # Add LoRA
    print("\n=== Adding LoRA Layers ===")
    model, lora_modules = add_lora_to_model(
        model, 
        r=args.lora_r, 
        alpha=args.lora_alpha, 
        dropout=args.lora_dropout
    )
    model = model.to(device)
    
    # Dataset
    print("\n=== Loading Dataset ===")
    train_dataset = MidiDataset(args.data_dir, max_seq=args.max_sequence, split="train")
    val_dataset = MidiDataset(args.data_dir, max_seq=args.max_sequence, split="val")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Optimizer & Scheduler
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.01)
    
    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=1e-6)
    
    # Loss
    loss_fn = SmoothCrossEntropyLoss(0.1, TOKEN_PAD, VOCAB_SIZE=388)
    
    # Training loop
    print("\n=== Starting Training ===")
    best_val_loss = float("inf")
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device, epoch)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        
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
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
        }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pt"))
    
    print(f"\n=== Training Complete ===")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"LoRA weights saved to: {args.output_dir}/lora_weights.pt")


if __name__ == "__main__":
    main()
