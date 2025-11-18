#!/usr/bin/env python3
"""
Quick start training script for TatumFlow
Uses configuration from config.yaml
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
from tatumflow import (
    create_tatumflow_model,
    TatumFlowTokenizer,
    create_dataloaders,
    TatumFlowTrainer,
    load_config,
    set_seed,
    print_model_summary,
    create_directories
)


def main():
    # Load configuration
    config_path = Path(__file__).parent.parent / 'config.yaml'
    config = load_config(str(config_path))

    # Set seed
    set_seed(42)

    # Create directories
    create_directories(config)

    print("="*80)
    print("TatumFlow Training")
    print("="*80)

    # Initialize tokenizer
    print("\n[1/5] Initializing tokenizer...")
    tokenizer = TatumFlowTokenizer()
    print(f"  Vocabulary size: {tokenizer.vocab_size}")

    # Find MIDI files
    print("\n[2/5] Loading dataset...")
    data_dir = Path(config['data']['data_dir'])

    if not data_dir.exists():
        print(f"  Warning: Data directory {data_dir} not found!")
        print("  Please update config.yaml with correct data_dir")
        print("  Example MIDI files should be in: data/midi/")
        return

    all_midis = list(data_dir.rglob("*.mid")) + list(data_dir.rglob("*.midi"))
    all_midis = [str(p) for p in all_midis]

    if len(all_midis) == 0:
        print(f"  No MIDI files found in {data_dir}")
        print("  Please add MIDI files to the data directory")
        return

    print(f"  Found {len(all_midis)} MIDI files")

    # Split train/val
    split_idx = int(len(all_midis) * config['data']['train_split'])
    train_paths = all_midis[:split_idx]
    val_paths = all_midis[split_idx:]
    print(f"  Train: {len(train_paths)}, Val: {len(val_paths)}")

    # Create dataloaders
    print("\n[3/5] Creating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_paths=train_paths,
        val_paths=val_paths,
        tokenizer=tokenizer,
        batch_size=config['training']['batch_size'],
        max_seq_len=config['data']['max_seq_len'],
        num_workers=config['data']['num_workers'],
        cache_dir=config['data']['cache_dir']
    )
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Create model
    print(f"\n[4/5] Creating {config['model']['size']} model...")
    model = create_tatumflow_model(
        model_size=config['model']['size'],
        vocab_size=tokenizer.vocab_size
    )

    print_model_summary(model)

    # Create trainer
    print("\n[5/5] Starting training...")
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    trainer = TatumFlowTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        device=device,
        learning_rate=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay'],
        num_epochs=config['training']['num_epochs'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        max_grad_norm=config['training']['max_grad_norm'],
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir'],
        diffusion_prob=config['training']['diffusion_prob']
    )

    # Train
    trainer.train()

    print("\n" + "="*80)
    print("Training completed successfully!")
    print("="*80)
    print(f"\nCheckpoints saved to: {config['paths']['checkpoint_dir']}")
    print(f"Logs saved to: {config['paths']['log_dir']}")
    print("\nTo generate music, run:")
    print("  python scripts/generate_music.py --checkpoint checkpoints/best.pt --mode continuation --prompt input.mid --output output.mid")


if __name__ == "__main__":
    main()
