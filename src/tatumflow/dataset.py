"""
Dataset loaders for TatumFlow training
Supports PiJAMA, Maestro, and custom MIDI datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import json
from .tokenizer import TatumFlowTokenizer


class MIDIDataset(Dataset):
    """
    Dataset for MIDI files tokenized for TatumFlow

    Supports:
    - Caching tokenized sequences
    - Data augmentation (pitch shift, tempo)
    - Chunk-based training
    """

    def __init__(
        self,
        midi_paths: List[str],
        tokenizer: TatumFlowTokenizer,
        max_seq_len: int = 2048,
        cache_dir: Optional[str] = None,
        augment: bool = False
    ):
        self.midi_paths = midi_paths
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.augment = augment

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Preprocess all files
        self.samples = []
        self._preprocess()

    def _preprocess(self):
        """Tokenize and cache all MIDI files"""
        print(f"Preprocessing {len(self.midi_paths)} MIDI files...")

        for idx, midi_path in enumerate(self.midi_paths):
            if idx % 100 == 0:
                print(f"  Processing {idx}/{len(self.midi_paths)}")

            # Check cache
            cache_path = None
            if self.cache_dir:
                cache_path = self.cache_dir / f"{Path(midi_path).stem}.npy"
                if cache_path.exists():
                    tokens = np.load(cache_path)
                    self._add_chunks(tokens, midi_path)
                    continue

            # Tokenize
            try:
                tokens = self.tokenizer.encode_midi(midi_path)
                tokens = np.array(tokens, dtype=np.int32)

                # Save to cache
                if cache_path:
                    np.save(cache_path, tokens)

                self._add_chunks(tokens, midi_path)

            except Exception as e:
                print(f"  Error processing {midi_path}: {e}")
                continue

        print(f"Loaded {len(self.samples)} sequences")

    def _add_chunks(self, tokens: np.ndarray, source: str):
        """Split long sequence into chunks"""
        # Split into max_seq_len chunks with overlap
        overlap = self.max_seq_len // 4

        if len(tokens) <= self.max_seq_len:
            self.samples.append({
                'tokens': tokens,
                'source': source
            })
        else:
            start = 0
            while start < len(tokens):
                end = min(start + self.max_seq_len, len(tokens))
                chunk = tokens[start:end]

                self.samples.append({
                    'tokens': chunk,
                    'source': source
                })

                start += self.max_seq_len - overlap

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        tokens = sample['tokens'].copy()

        # Data augmentation
        if self.augment and np.random.rand() < 0.3:
            tokens = self._augment_tokens(tokens)

        # Pad to max_seq_len
        if len(tokens) < self.max_seq_len:
            padding = np.full(
                self.max_seq_len - len(tokens),
                self.tokenizer.pad_id,
                dtype=np.int32
            )
            tokens = np.concatenate([tokens, padding])

        # Create attention mask
        mask = (tokens != self.tokenizer.pad_id).astype(np.int32)

        return {
            'input_ids': torch.from_numpy(tokens).long(),
            'attention_mask': torch.from_numpy(mask).long(),
            'source': sample['source']
        }

    def _augment_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """
        Data augmentation by pitch shifting
        Modifies NOTE_ON and NOTE_OFF tokens
        """
        # Random pitch shift: -6 to +6 semitones
        shift = np.random.randint(-6, 7)
        if shift == 0:
            return tokens

        augmented = tokens.copy()

        for i, token_id in enumerate(tokens):
            token_str = self.tokenizer.id_to_token.get(token_id, "")

            if token_str.startswith("NOTE_ON_") or token_str.startswith("NOTE_OFF_"):
                parts = token_str.split("_")
                pitch = int(parts[2])
                new_pitch = pitch + shift

                # Check bounds
                if self.tokenizer.config.min_pitch <= new_pitch <= self.tokenizer.config.max_pitch:
                    prefix = "_".join(parts[:2])
                    new_token = f"{prefix}_{new_pitch}"
                    if new_token in self.tokenizer.token_to_id:
                        augmented[i] = self.tokenizer.token_to_id[new_token]

        return augmented


class ArtTatumDataset(MIDIDataset):
    """
    Specialized dataset for Art Tatum style learning
    Filters PiJAMA for Art Tatum performances
    """

    def __init__(
        self,
        pijama_dir: str,
        tokenizer: TatumFlowTokenizer,
        artist_filter: str = "art tatum",
        **kwargs
    ):
        # Find Art Tatum MIDI files
        pijama_path = Path(pijama_dir)
        all_midis = list(pijama_path.rglob("*.mid")) + list(pijama_path.rglob("*.midi"))

        # Filter by artist name (assuming metadata or filename contains artist)
        filtered_midis = []
        for midi_path in all_midis:
            # Check if artist name in filename or parent directory
            path_str = str(midi_path).lower()
            if artist_filter.lower() in path_str:
                filtered_midis.append(str(midi_path))

        print(f"Found {len(filtered_midis)} Art Tatum MIDI files")

        super().__init__(
            midi_paths=filtered_midis,
            tokenizer=tokenizer,
            **kwargs
        )


def create_dataloaders(
    train_paths: List[str],
    val_paths: List[str],
    tokenizer: TatumFlowTokenizer,
    batch_size: int = 4,
    max_seq_len: int = 2048,
    num_workers: int = 4,
    cache_dir: Optional[str] = None
) -> tuple:
    """
    Create train and validation dataloaders

    Args:
        train_paths: List of training MIDI file paths
        val_paths: List of validation MIDI file paths
        tokenizer: TatumFlowTokenizer instance
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        num_workers: Number of data loading workers
        cache_dir: Directory for caching tokenized sequences

    Returns:
        (train_loader, val_loader)
    """
    train_dataset = MIDIDataset(
        midi_paths=train_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        cache_dir=cache_dir,
        augment=True
    )

    val_dataset = MIDIDataset(
        midi_paths=val_paths,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        cache_dir=cache_dir,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    return train_loader, val_loader


if __name__ == "__main__":
    # Test dataset
    from .tokenizer import TatumFlowTokenizer

    tokenizer = TatumFlowTokenizer()

    # Create dummy dataset
    print("Testing dataset loader...")
    # dataset = MIDIDataset(
    #     midi_paths=["path/to/midi.mid"],
    #     tokenizer=tokenizer,
    #     max_seq_len=512
    # )
    print("Dataset module ready")
