"""
Stage 0: VQ-VAE Pretraining on MAESTRO

This script pretrains the VQ-VAE on classical piano data
to learn good latent representations of piano rolls
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.vqvae import VQVAE


def create_dummy_dataloader(data_dir: str, batch_size: int = 16):
    """
    Create dummy dataloader for testing

    TODO: Replace with actual MIDI dataloader
    """
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size

        def __len__(self):
            return self.size

        def __getitem__(self, idx):
            # Dummy piano roll: [2, 128, 256]
            # 2 channels: onset + sustain
            # 128: MIDI pitches
            # 256: time steps (~8 bars at 16th note resolution)
            return torch.randn(2, 128, 256)

    dataset = DummyDataset(size=1000)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def train_vqvae(
    model: VQVAE,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 50,
    lr: float = 1e-4,
    device: str = "cuda",
    save_dir: str = "./checkpoints/vqvae"
):
    """
    Train VQ-VAE

    Args:
        model: VQ-VAE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        epochs: Number of epochs
        lr: Learning rate
        device: Device to train on
        save_dir: Directory to save checkpoints
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.1
    )

    os.makedirs(save_dir, exist_ok=True)

    best_val_loss = float('inf')

    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = {
            'total': 0,
            'recon': 0,
            'vq': 0,
            'perplexity': 0
        }

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [Train]")
        for batch_idx, piano_roll in enumerate(pbar):
            piano_roll = piano_roll.to(device)

            # Forward pass
            recon, vq_loss, perplexity = model(piano_roll)

            # Reconstruction loss
            recon_loss = nn.functional.mse_loss(recon, piano_roll)

            # Total loss
            loss = recon_loss + vq_loss

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Logging
            train_losses['total'] += loss.item()
            train_losses['recon'] += recon_loss.item()
            train_losses['vq'] += vq_loss.item()
            train_losses['perplexity'] += perplexity.item()

            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{recon_loss.item():.4f}",
                'vq': f"{vq_loss.item():.4f}",
                'ppl': f"{perplexity.item():.1f}"
            })

        # Average losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)

        # Validation
        model.eval()
        val_losses = {
            'total': 0,
            'recon': 0,
            'vq': 0,
            'perplexity': 0
        }

        with torch.no_grad():
            for piano_roll in tqdm(val_loader, desc=f"Epoch {epoch + 1}/{epochs} [Val]"):
                piano_roll = piano_roll.to(device)

                recon, vq_loss, perplexity = model(piano_roll)
                recon_loss = nn.functional.mse_loss(recon, piano_roll)
                loss = recon_loss + vq_loss

                val_losses['total'] += loss.item()
                val_losses['recon'] += recon_loss.item()
                val_losses['vq'] += vq_loss.item()
                val_losses['perplexity'] += perplexity.item()

        for key in val_losses:
            val_losses[key] /= len(val_loader)

        # Print epoch summary
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Train - Loss: {train_losses['total']:.4f}, Recon: {train_losses['recon']:.4f}, "
              f"VQ: {train_losses['vq']:.4f}, Perplexity: {train_losses['perplexity']:.1f}")
        print(f"Val   - Loss: {val_losses['total']:.4f}, Recon: {val_losses['recon']:.4f}, "
              f"VQ: {val_losses['vq']:.4f}, Perplexity: {val_losses['perplexity']:.1f}")

        # Save best model
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': {
                    'in_channels': model.encoder.conv_in.in_channels,
                    'hidden_dim': 256,
                    'num_embeddings': model.vq.num_embeddings,
                    'latent_dim': model.vq.embedding_dim
                }
            }, os.path.join(save_dir, 'best.pt'))
            print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_losses['total']
            }, os.path.join(save_dir, f'epoch_{epoch + 1}.pt'))

        scheduler.step()

    print(f"\n✅ VQ-VAE training complete! Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train VQ-VAE")
    parser.add_argument("--data_dir", default="./data/maestro", help="Data directory")
    parser.add_argument("--save_dir", default="./checkpoints/vqvae", help="Save directory")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--test", action="store_true", help="Test mode with dummy data")

    args = parser.parse_args()

    print("=" * 50)
    print("VQ-VAE Pretraining")
    print("=" * 50)
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print("=" * 50)

    # Create model
    model = VQVAE(
        in_channels=2,
        hidden_dim=256,
        num_embeddings=512,
        latent_dim=64,
        commitment_cost=0.25
    )

    # Create dataloaders
    if args.test:
        print("\n⚠️  Test mode: Using dummy data")
        train_loader = create_dummy_dataloader(args.data_dir, args.batch_size)
        val_loader = create_dummy_dataloader(args.data_dir, args.batch_size // 2)
    else:
        # TODO: Implement real dataloader
        print("\n⚠️  Real dataloader not implemented yet, using dummy data")
        train_loader = create_dummy_dataloader(args.data_dir, args.batch_size)
        val_loader = create_dummy_dataloader(args.data_dir, args.batch_size // 2)

    # Train
    train_vqvae(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        device=args.device,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()
