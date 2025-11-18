"""
Utility functions for TatumFlow
"""

import torch
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any
import random


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str):
    """Save configuration to YAML file"""
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(device: str = 'auto') -> torch.device:
    """Get torch device"""
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"


def create_directories(config: Dict[str, Any]):
    """Create necessary directories from config"""
    dirs_to_create = [
        config['data'].get('cache_dir', 'cache'),
        config['paths'].get('checkpoint_dir', 'checkpoints'),
        config['paths'].get('log_dir', 'logs'),
        config['paths'].get('output_dir', 'outputs')
    ]

    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """Get current learning rate from optimizer"""
    for param_group in optimizer.param_groups:
        return param_group['lr']
    return 0.0


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_summary(model: torch.nn.Module):
    """Print model architecture summary"""
    print("\n" + "="*80)
    print("TatumFlow Model Summary")
    print("="*80)

    # Count parameters by component
    components = {
        'Token Embedding': model.token_embedding,
        'Style Encoder': model.style_encoder,
        'Style Decoder': model.style_decoder,
        'Theory Encoder': model.theory_encoder,
        'Latent Encoder': model.latent_encoder,
        'Latent Decoder': model.latent_decoder,
        'Diffusion': model.diffusion,
        'Transformer Blocks': model.blocks,
        'Output Head': model.head
    }

    total_params = 0
    for name, component in components.items():
        params = sum(p.numel() for p in component.parameters())
        total_params += params
        print(f"{name:.<40} {params:>15,} parameters")

    print("-"*80)
    print(f"{'Total':.<40} {total_params:>15,} parameters")
    print("="*80 + "\n")


def midi_to_pianoroll(midi_path: str, fs: int = 100) -> np.ndarray:
    """
    Convert MIDI to piano roll representation

    Args:
        midi_path: Path to MIDI file
        fs: Sampling frequency (frames per second)

    Returns:
        Piano roll array (time, pitch)
    """
    import mido
    from mido import MidiFile

    midi = MidiFile(midi_path)

    # Get total length in ticks
    max_tick = 0
    for track in midi.tracks:
        current_tick = 0
        for msg in track:
            current_tick += msg.time
            max_tick = max(max_tick, current_tick)

    # Convert to time
    ticks_per_beat = midi.ticks_per_beat
    tempo = 500000  # Default 120 BPM
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo = msg.tempo
                break

    tick_duration = tempo / ticks_per_beat / 1000000  # seconds
    total_time = max_tick * tick_duration

    # Create piano roll
    num_frames = int(total_time * fs)
    piano_roll = np.zeros((num_frames, 128))

    # Fill piano roll
    for track in midi.tracks:
        current_tick = 0
        active_notes = {}

        for msg in track:
            current_tick += msg.time
            current_time = current_tick * tick_duration
            frame = int(current_time * fs)

            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = frame
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    start_frame = active_notes[msg.note]
                    end_frame = min(frame, num_frames)
                    piano_roll[start_frame:end_frame, msg.note] = msg.velocity / 127.0
                    del active_notes[msg.note]

    return piano_roll


def visualize_piano_roll(piano_roll: np.ndarray, save_path: str):
    """
    Visualize piano roll and save as image

    Args:
        piano_roll: Piano roll array (time, pitch)
        save_path: Path to save image
    """
    import matplotlib.pyplot as plt

    plt.figure(figsize=(15, 8))
    plt.imshow(
        piano_roll.T,
        aspect='auto',
        origin='lower',
        cmap='Blues',
        interpolation='nearest'
    )
    plt.xlabel('Time (frames)')
    plt.ylabel('MIDI Pitch')
    plt.title('Piano Roll Visualization')
    plt.colorbar(label='Velocity')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()

    print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    # Test utilities
    print("Testing TatumFlow utilities...")

    # Test config loading
    config = {
        'model': {'size': 'base'},
        'training': {'batch_size': 4},
        'paths': {'checkpoint_dir': 'checkpoints'}
    }

    print("Config example:", config)
    print("Utility module ready!")
