"""
Dataset for PersonalJazz training

Loads audio files and optional text descriptions
"""

import torch
from torch.utils.data import Dataset
import torchaudio
from pathlib import Path
from typing import Optional, List, Tuple
import json


class MusicDataset(Dataset):
    """
    Music dataset for PersonalJazz training

    Expected directory structure:
        data_dir/
            audio/
                track001.wav
                track002.wav
                ...
            metadata.json  (optional: text descriptions)

    metadata.json format:
        {
            "track001.wav": "slow jazz piano ballad in the style of Bill Evans",
            "track002.wav": "fast bebop improvisation",
            ...
        }
    """

    def __init__(
        self,
        data_dir: str,
        sample_rate: int = 48000,
        duration: float = 10.0,  # Load 10-second chunks
        augment: bool = False
    ):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.duration = duration
        self.augment = augment

        # Find all audio files
        audio_dir = self.data_dir / "audio"
        self.audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac']:
            self.audio_files.extend(list(audio_dir.glob(ext)))

        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")

        print(f"Found {len(self.audio_files)} audio files")

        # Load metadata if available
        metadata_path = self.data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
            print(f"Loaded metadata for {len(self.metadata)} files")
        else:
            self.metadata = {}
            print("No metadata found, using filename as description")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]

        # Load audio
        audio, sr = torchaudio.load(audio_path)

        # Resample if necessary
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            audio = resampler(audio)

        # Convert to stereo if mono
        if audio.shape[0] == 1:
            audio = audio.repeat(2, 1)

        # Extract chunk
        chunk_samples = int(self.duration * self.sample_rate)

        if audio.shape[1] >= chunk_samples:
            # Random crop
            if self.augment:
                start = torch.randint(0, audio.shape[1] - chunk_samples + 1, (1,)).item()
            else:
                start = 0
            audio = audio[:, start:start + chunk_samples]
        else:
            # Pad if too short
            pad_amount = chunk_samples - audio.shape[1]
            audio = torch.nn.functional.pad(audio, (0, pad_amount))

        # Normalize
        audio = audio / (audio.abs().max() + 1e-8)

        # Get text description
        filename = audio_path.name
        text = self.metadata.get(filename, f"jazz piano improvisation")

        return {
            'audio': audio,  # (2, T)
            'text': text,
            'filename': filename
        }


def collate_fn(batch: List[dict], tokenize_fn=None):
    """
    Collate function for DataLoader

    Args:
        batch: List of dataset items
        tokenize_fn: Function to tokenize text

    Returns:
        Batched tensors
    """
    audios = torch.stack([item['audio'] for item in batch])  # (B, 2, T)

    texts = [item['text'] for item in batch]

    # Tokenize texts if tokenizer provided
    if tokenize_fn is not None:
        # Tokenize all texts
        max_len = 128
        all_token_ids = []
        all_masks = []

        for text in texts:
            token_ids, mask = tokenize_fn(text, max_length=max_len)
            all_token_ids.append(token_ids)
            all_masks.append(mask)

        text_tokens = torch.cat(all_token_ids, dim=0)  # (B, max_len)
        attention_masks = torch.cat(all_masks, dim=0)  # (B, max_len)
    else:
        text_tokens = None
        attention_masks = None

    return {
        'audio': audios,
        'text': texts,
        'text_tokens': text_tokens,
        'attention_masks': attention_masks
    }
