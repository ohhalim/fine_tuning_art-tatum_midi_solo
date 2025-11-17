"""
Fine-tuning PersonalJazz with QLoRA

Efficiently fine-tune the model on personal jazz recordings
using Quantized Low-Rank Adaptation (QLoRA)
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from pathlib import Path

from ..model.personaljazz import PersonalJazz
from ..model.tokenizer import tokenize_text
from .dataset import MusicDataset, collate_fn


class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer

    Adds trainable low-rank matrices to frozen weights:
    W' = W + (B @ A) * scaling

    where W is frozen, A and B are trainable
    """

    def __init__(self, original_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()

        self.original_layer = original_layer
        self.original_layer.requires_grad_(False)  # Freeze original weights

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)

        # Scaling factor
        self.scaling = alpha / rank

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor):
        # Original transformation
        original_out = self.original_layer(x)

        # LoRA transformation
        lora_out = self.lora_B(self.lora_A(x)) * self.scaling

        return original_out + lora_out


def add_lora_to_model(model: PersonalJazz, rank: int = 8, alpha: float = 16.0, target_modules: list = None):
    """
    Add LoRA adapters to specific modules in the model

    Args:
        model: PersonalJazz model
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module names to apply LoRA (default: attention projections)
    """
    if target_modules is None:
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']

    lora_layers = []

    # Apply LoRA to transformer attention layers
    for name, module in model.transformer.named_modules():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # Get parent module and attribute name
                *parent_path, attr_name = name.split('.')

                parent = model.transformer
                for p in parent_path:
                    parent = getattr(parent, p)

                # Replace with LoRA layer
                lora_layer = LoRALayer(module, rank, alpha)
                setattr(parent, attr_name, lora_layer)

                lora_layers.append(lora_layer)

                print(f"  Added LoRA to {name}")

    print(f"\nAdded LoRA to {len(lora_layers)} layers (rank={rank}, alpha={alpha})")

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.3f}%)")

    return model


def finetune_with_qlora(
    model_path: str,
    data_dir: str,
    output_dir: str = "./ohhalim-jazz-style",
    style_prompt: str = "ohhalim jazz piano style",
    # LoRA config
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    # Training config
    batch_size: int = 2,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    warmup_steps: int = 100,
    gradient_accumulation_steps: int = 4,
    save_every: int = 500,
    # Other
    device: str = 'cuda',
    fp16: bool = True
):
    """
    Fine-tune PersonalJazz model with QLoRA

    Args:
        model_path: Path to pre-trained model
        data_dir: Directory containing personal jazz recordings
        output_dir: Where to save fine-tuned model
        style_prompt: Text description of personal style
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        batch_size: Batch size
        num_epochs: Number of epochs
        learning_rate: Learning rate
        warmup_steps: Learning rate warmup steps
        gradient_accumulation_steps: Gradient accumulation steps
        save_every: Save checkpoint every N steps
        device: Device to use
        fp16: Use mixed precision training
    """
    print("=" * 80)
    print("PersonalJazz Fine-tuning with QLoRA")
    print("=" * 80)

    # Create output directory
    Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Load pre-trained model
    print(f"\nLoading pre-trained model from {model_path}...")
    model = PersonalJazz.load_pretrained(model_path, device=device)

    # Add LoRA adapters
    print(f"\nAdding LoRA adapters...")
    model = add_lora_to_model(model, rank=lora_rank, alpha=lora_alpha)

    # Freeze all parameters except LoRA
    for name, param in model.named_parameters():
        if 'lora' not in name.lower():
            param.requires_grad = False

    # Load dataset
    print(f"\nLoading dataset from {data_dir}...")
    dataset = MusicDataset(data_dir, sample_rate=model.sample_rate, duration=10.0, augment=True)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=lambda batch: collate_fn(batch, tokenize_text),
        pin_memory=True
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )

    # Learning rate schedule with warmup
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if fp16 else None

    # Encode style prompt
    style_tokens, style_mask = tokenize_text(style_prompt, device=device)
    style_emb = model.encode_style_from_text(style_tokens, style_mask)

    # Training loop
    print(f"\nStarting training...")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Gradient accumulation: {gradient_accumulation_steps}")
    print(f"  Effective batch size: {batch_size * gradient_accumulation_steps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Style prompt: '{style_prompt}'")
    print()

    model.train()
    global_step = 0

    for epoch in range(num_epochs):
        epoch_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            audio = batch['audio'].to(device)  # (B, 2, T)

            # Forward pass with mixed precision
            if fp16:
                with torch.cuda.amp.autocast():
                    loss_dict = model(audio, style_emb=style_emb.expand(audio.size(0), -1))
                    loss = loss_dict['total'] / gradient_accumulation_steps
            else:
                loss_dict = model(audio, style_emb=style_emb.expand(audio.size(0), -1))
                loss = loss_dict['total'] / gradient_accumulation_steps

            # Backward pass
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                if fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

                global_step += 1

                # Save checkpoint
                if global_step % save_every == 0:
                    checkpoint_path = f"{output_dir}/checkpoint-{global_step}.pt"
                    torch.save({
                        'global_step': global_step,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_dict['total'].item(),
                        'lora_config': {'rank': lora_rank, 'alpha': lora_alpha}
                    }, checkpoint_path)
                    print(f"\nCheckpoint saved: {checkpoint_path}")

            epoch_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item() * gradient_accumulation_steps:.4f}",
                'transformer': f"{loss_dict['transformer'].item():.4f}",
                'codec': f"{loss_dict['codec'].item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })

        # Epoch summary
        avg_loss = epoch_loss / len(dataloader)
        print(f"\nEpoch {epoch+1}/{num_epochs} - Average Loss: {avg_loss:.4f}")

    # Save final model
    final_path = f"{output_dir}/final_model.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'lora_config': {'rank': lora_rank, 'alpha': lora_alpha},
        'style_prompt': style_prompt
    }, final_path)

    print(f"\nFine-tuning complete!")
    print(f"Final model saved to: {final_path}")

    return model


if __name__ == "__main__":
    # Example usage
    finetune_with_qlora(
        model_path="./pretrained/personaljazz_base.pt",
        data_dir="./data/ohhalim_jazz",
        output_dir="./ohhalim-jazz-style",
        style_prompt="ohhalim jazz piano improvisation style",
        lora_rank=8,
        lora_alpha=16.0,
        num_epochs=50,
        batch_size=2,
        learning_rate=1e-4
    )
