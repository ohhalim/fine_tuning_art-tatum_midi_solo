"""
Training script for TatumFlow
Implements multi-objective loss with diffusion training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse
from tqdm import tqdm
import json

from .model import TatumFlow, create_tatumflow_model
from .tokenizer import TatumFlowTokenizer
from .dataset import create_dataloaders


class TatumFlowLoss(nn.Module):
    """
    Multi-objective loss for TatumFlow:
    1. Reconstruction loss (cross-entropy)
    2. Diffusion denoising loss (MSE in latent space)
    3. Style VAE loss (KL divergence)
    4. Music theory disentanglement loss
    """

    def __init__(
        self,
        vocab_size: int,
        lambda_recon: float = 1.0,
        lambda_diffusion: float = 0.5,
        lambda_kl: float = 0.1,
        lambda_theory: float = 0.2
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.lambda_recon = lambda_recon
        self.lambda_diffusion = lambda_diffusion
        self.lambda_kl = lambda_kl
        self.lambda_theory = lambda_theory

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        latent: torch.Tensor,
        latent_target: Optional[torch.Tensor],
        style_mu: Optional[torch.Tensor],
        style_logvar: Optional[torch.Tensor],
        theory_components: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all loss components

        Args:
            logits: (B, L, V) model predictions
            targets: (B, L) target token IDs
            latent: (B, L, D) latent representation
            latent_target: (B, L, D) target latent (clean)
            style_mu: (B, style_dim) VAE mu
            style_logvar: (B, style_dim) VAE logvar
            theory_components: dict of theory representations

        Returns:
            Dictionary of losses
        """
        B, L, V = logits.shape

        # 1. Reconstruction loss (cross-entropy)
        recon_loss = F.cross_entropy(
            logits.reshape(-1, V),
            targets.reshape(-1),
            ignore_index=0  # Assuming 0 is PAD token
        )

        # 2. Diffusion loss (if latent_target provided)
        diffusion_loss = torch.tensor(0.0, device=logits.device)
        if latent_target is not None:
            diffusion_loss = F.mse_loss(latent, latent_target)

        # 3. Style VAE KL loss
        kl_loss = torch.tensor(0.0, device=logits.device)
        if style_mu is not None and style_logvar is not None:
            kl_loss = -0.5 * torch.sum(
                1 + style_logvar - style_mu.pow(2) - style_logvar.exp(),
                dim=-1
            ).mean()

        # 4. Music theory disentanglement loss
        # Encourage orthogonality between components
        theory_loss = torch.tensor(0.0, device=logits.device)
        if theory_components:
            components_list = list(theory_components.values())
            if len(components_list) > 1:
                # Compute pairwise cosine similarity and penalize high correlation
                for i in range(len(components_list)):
                    for j in range(i + 1, len(components_list)):
                        comp_i = F.normalize(components_list[i], dim=-1)
                        comp_j = F.normalize(components_list[j], dim=-1)
                        similarity = (comp_i * comp_j).sum(dim=-1).abs().mean()
                        theory_loss += similarity

        # Total loss
        total_loss = (
            self.lambda_recon * recon_loss +
            self.lambda_diffusion * diffusion_loss +
            self.lambda_kl * kl_loss +
            self.lambda_theory * theory_loss
        )

        return {
            'total': total_loss,
            'recon': recon_loss,
            'diffusion': diffusion_loss,
            'kl': kl_loss,
            'theory': theory_loss
        }


class TatumFlowTrainer:
    """Trainer for TatumFlow model"""

    def __init__(
        self,
        model: TatumFlow,
        train_loader,
        val_loader,
        tokenizer: TatumFlowTokenizer,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        num_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        log_dir: str = 'logs',
        diffusion_prob: float = 0.5
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.device = device
        self.num_epochs = num_epochs
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.diffusion_prob = diffusion_prob

        # Optimizer and scheduler
        self.optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.95)
        )

        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs * len(train_loader),
            eta_min=learning_rate * 0.1
        )

        # Loss function
        self.criterion = TatumFlowLoss(vocab_size=tokenizer.vocab_size)

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Logging
        self.writer = SummaryWriter(log_dir)
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_losses = {
            'total': 0.0,
            'recon': 0.0,
            'diffusion': 0.0,
            'kl': 0.0,
            'theory': 0.0
        }

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Create targets (shift input by 1)
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            # Randomly apply diffusion during training
            use_diffusion = np.random.rand() < self.diffusion_prob
            timestep = None
            latent_target = None

            if use_diffusion:
                # Sample random timestep
                batch_size = input_ids.shape[0]
                timestep = torch.randint(
                    0,
                    self.model.diffusion.num_steps,
                    (batch_size,),
                    device=self.device
                )

                # Get clean latent as target
                with torch.no_grad():
                    clean_outputs = self.model(input_ids, timestep=None, mask=attention_mask)
                    latent_target = clean_outputs['latent'].detach()

            # Forward pass
            outputs = self.model(input_ids, timestep=timestep, mask=attention_mask)

            # Compute loss
            losses = self.criterion(
                logits=outputs['logits'],
                targets=targets,
                latent=outputs['latent'],
                latent_target=latent_target,
                style_mu=outputs.get('style_mu'),
                style_logvar=outputs.get('style_logvar'),
                theory_components=outputs.get('theory_components', {})
            )

            loss = losses['total'] / self.gradient_accumulation_steps
            loss.backward()

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()

            # Gradient accumulation
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm
                )

                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1

                # Log to tensorboard
                if self.global_step % 100 == 0:
                    for key, value in losses.items():
                        self.writer.add_scalar(f'train/{key}_loss', value, self.global_step)
                    self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total'].item():.4f}",
                'recon': f"{losses['recon'].item():.4f}"
            })

        # Average losses
        num_batches = len(self.train_loader)
        for key in total_losses:
            total_losses[key] /= num_batches

        return total_losses

    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate model"""
        self.model.eval()
        total_losses = {
            'total': 0.0,
            'recon': 0.0,
            'diffusion': 0.0,
            'kl': 0.0,
            'theory': 0.0
        }

        pbar = tqdm(self.val_loader, desc=f"Validation {epoch}")

        for batch in pbar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)

            # Create targets
            targets = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()
            attention_mask = attention_mask[:, :-1].contiguous()

            # Forward pass (no diffusion during validation)
            outputs = self.model(input_ids, timestep=None, mask=attention_mask)

            # Compute loss
            losses = self.criterion(
                logits=outputs['logits'],
                targets=targets,
                latent=outputs['latent'],
                latent_target=None,
                style_mu=outputs.get('style_mu'),
                style_logvar=outputs.get('style_logvar'),
                theory_components=outputs.get('theory_components', {})
            )

            # Accumulate losses
            for key in total_losses:
                total_losses[key] += losses[key].item()

        # Average losses
        num_batches = len(self.val_loader)
        for key in total_losses:
            total_losses[key] /= num_batches

        # Log to tensorboard
        for key, value in total_losses.items():
            self.writer.add_scalar(f'val/{key}_loss', value, epoch)

        return total_losses

    def save_checkpoint(self, epoch: int, val_loss: float, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'global_step': self.global_step
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pt'
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best.pt'
            torch.save(checkpoint, best_path)
            print(f"  Saved best model with val_loss: {val_loss:.4f}")

        # Save periodic checkpoints
        if epoch % 10 == 0:
            epoch_path = self.checkpoint_dir / f'epoch_{epoch}.pt'
            torch.save(checkpoint, epoch_path)

    def train(self):
        """Full training loop"""
        print("Starting TatumFlow training...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")

            # Train
            train_losses = self.train_epoch(epoch)
            print(f"Train - Total: {train_losses['total']:.4f}, Recon: {train_losses['recon']:.4f}")

            # Validate
            val_losses = self.validate(epoch)
            print(f"Val   - Total: {val_losses['total']:.4f}, Recon: {val_losses['recon']:.4f}")

            # Save checkpoint
            is_best = val_losses['total'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_losses['total']

            self.save_checkpoint(epoch, val_losses['total'], is_best)

        print("\nTraining completed!")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train TatumFlow model')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing MIDI files')
    parser.add_argument('--cache_dir', type=str, default='cache', help='Cache directory for tokenized data')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()

    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = TatumFlowTokenizer()

    # Find MIDI files
    print(f"Scanning {args.data_dir} for MIDI files...")
    data_path = Path(args.data_dir)
    all_midis = list(data_path.rglob("*.mid")) + list(data_path.rglob("*.midi"))
    all_midis = [str(p) for p in all_midis]

    print(f"Found {len(all_midis)} MIDI files")

    # Split train/val (90/10)
    split_idx = int(len(all_midis) * 0.9)
    train_paths = all_midis[:split_idx]
    val_paths = all_midis[split_idx:]

    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        num_workers=args.num_workers,
        cache_dir=args.cache_dir
    )

    # Create model
    print(f"Creating {args.model_size} model...")
    model = create_tatumflow_model(
        model_size=args.model_size,
        vocab_size=tokenizer.vocab_size
    )

    # Create trainer
    trainer = TatumFlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=args.device,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        checkpoint_dir=args.checkpoint_dir
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
