"""
Brad Mehldau Data Preprocessing Pipeline

Efficient pipeline for preparing Brad Mehldau MIDI for Moonbeam LoRA fine-tuning

Input: 15-20 Brad Mehldau MIDI files
Output: 180-240 augmented training examples (5D format)

This is 10x more efficient than SCG approach which requires 100s of songs
"""

import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import json
from dataclasses import dataclass, asdict
from tqdm import tqdm

from .midi_5d_representation import (
    MIDI5DConverter,
    Note5D,
    ChordExtractor,
    DataAugmenter5D
)


@dataclass
class BradMehldauSample:
    """Single training sample for Moonbeam"""
    song_id: str
    notes_5d: List[Note5D]
    chords: List[Tuple[float, str]]
    duration_seconds: float
    original_file: str
    augmentation_type: str  # 'original', 'transpose_+2', 'tempo_0.9', etc.

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'song_id': self.song_id,
            'notes_5d': [[n.onset_time, n.duration, n.octave, n.pitch_class, n.velocity] for n in self.notes_5d],
            'chords': self.chords,
            'duration_seconds': self.duration_seconds,
            'original_file': self.original_file,
            'augmentation_type': self.augmentation_type
        }


class BradMehldauDataset:
    """
    Brad Mehldau dataset for Moonbeam fine-tuning

    Features:
    - 5D MIDI representation
    - Chord progression extraction
    - Data augmentation (12x multiplier)
    - Efficient storage format
    """

    def __init__(
        self,
        data_dir: str = "./data/brad_mehldau",
        output_dir: str = "./moonbeam_data/brad_processed",
        min_notes: int = 100,  # Minimum notes per sample
        max_duration: float = 60.0,  # Maximum duration (seconds)
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.min_notes = min_notes
        self.max_duration = max_duration

        self.converter = MIDI5DConverter(quantize=True)
        self.chord_extractor = ChordExtractor()
        self.augmenter = DataAugmenter5D()

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_brad_midi_files(self) -> List[Path]:
        """
        Collect Brad Mehldau MIDI files

        Sources:
        1. PiJAMA dataset (8.9 hours of Brad Mehldau)
        2. Manually collected/transcribed files
        3. YouTube → audio-to-MIDI conversions
        """
        midi_files = []

        # Find all MIDI files
        for ext in ['*.mid', '*.midi']:
            midi_files.extend(self.data_dir.glob(ext))
            midi_files.extend(self.data_dir.glob(f'**/{ext}'))

        # Filter for quality
        filtered_files = []
        for midi_file in midi_files:
            try:
                notes = self.converter.midi_to_5d(str(midi_file))
                if len(notes) >= self.min_notes:
                    filtered_files.append(midi_file)
            except Exception as e:
                print(f"⚠️  Skipping {midi_file.name}: {e}")

        return filtered_files

    def process_single_file(
        self,
        midi_file: Path,
        apply_augmentation: bool = True
    ) -> List[BradMehldauSample]:
        """
        Process single MIDI file into training samples

        Args:
            midi_file: Path to MIDI file
            apply_augmentation: Whether to generate augmented versions

        Returns:
            List of samples (1 original + N augmented)
        """
        samples = []

        # Convert to 5D
        notes_5d = self.converter.midi_to_5d(str(midi_file))

        # Extract chords
        chords = self.chord_extractor.extract_chords(notes_5d)

        # Calculate duration
        if notes_5d:
            duration = max(n.onset_time + n.duration for n in notes_5d)
        else:
            return []

        # Create original sample
        original_sample = BradMehldauSample(
            song_id=midi_file.stem,
            notes_5d=notes_5d,
            chords=chords,
            duration_seconds=duration,
            original_file=str(midi_file),
            augmentation_type='original'
        )
        samples.append(original_sample)

        # Augmentation
        if apply_augmentation:
            # Transpose to 12 keys
            for semitones in range(-6, 6):
                if semitones == 0:
                    continue  # Skip original key

                transposed_notes = self.augmenter.transpose(notes_5d, semitones)
                transposed_chords = self._transpose_chords(chords, semitones)

                aug_sample = BradMehldauSample(
                    song_id=f"{midi_file.stem}_transpose{semitones:+d}",
                    notes_5d=transposed_notes,
                    chords=transposed_chords,
                    duration_seconds=duration,
                    original_file=str(midi_file),
                    augmentation_type=f'transpose_{semitones:+d}'
                )
                samples.append(aug_sample)

            # Tempo variations
            for tempo_factor in [0.9, 1.1]:
                stretched_notes = self.augmenter.time_stretch(notes_5d, tempo_factor)
                stretched_chords = self._stretch_chords(chords, tempo_factor)

                aug_sample = BradMehldauSample(
                    song_id=f"{midi_file.stem}_tempo{tempo_factor:.1f}",
                    notes_5d=stretched_notes,
                    chords=stretched_chords,
                    duration_seconds=duration * tempo_factor,
                    original_file=str(midi_file),
                    augmentation_type=f'tempo_{tempo_factor:.1f}'
                )
                samples.append(aug_sample)

        return samples

    def _transpose_chords(
        self,
        chords: List[Tuple[float, str]],
        semitones: int
    ) -> List[Tuple[float, str]]:
        """Transpose chord names by semitones"""
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        transposed_chords = []
        for time, chord_name in chords:
            if chord_name == 'N':
                transposed_chords.append((time, 'N'))
                continue

            # Parse chord (e.g., "Cmaj7" → root='C', type='maj7')
            root = chord_name[0]
            if len(chord_name) > 1 and chord_name[1] == '#':
                root = chord_name[:2]
                chord_type = chord_name[2:]
            else:
                chord_type = chord_name[1:]

            # Transpose root
            root_idx = note_names.index(root)
            new_root_idx = (root_idx + semitones) % 12
            new_root = note_names[new_root_idx]

            transposed_chords.append((time, f"{new_root}{chord_type}"))

        return transposed_chords

    def _stretch_chords(
        self,
        chords: List[Tuple[float, str]],
        factor: float
    ) -> List[Tuple[float, str]]:
        """Stretch chord times by factor"""
        return [(time * factor, chord) for time, chord in chords]

    def process_all_files(self) -> List[BradMehldauSample]:
        """
        Process all Brad Mehldau MIDI files

        Returns:
            List of all training samples
        """
        print("🎹 Processing Brad Mehldau MIDI files...")

        # Collect files
        midi_files = self.collect_brad_midi_files()
        print(f"   Found {len(midi_files)} Brad Mehldau MIDI files")

        # Process each file
        all_samples = []
        for midi_file in tqdm(midi_files, desc="Processing"):
            samples = self.process_single_file(midi_file, apply_augmentation=True)
            all_samples.extend(samples)

        print(f"   Generated {len(all_samples)} training samples")
        print(f"   Augmentation factor: {len(all_samples) / len(midi_files):.1f}x")

        return all_samples

    def save_dataset(self, samples: List[BradMehldauSample]):
        """
        Save processed dataset

        Format: JSON for metadata, NPZ for efficient 5D arrays
        """
        print("\n💾 Saving dataset...")

        # Save metadata
        metadata = {
            'num_samples': len(samples),
            'samples': [s.to_dict() for s in samples]
        }

        metadata_path = self.output_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"   Saved metadata: {metadata_path}")

        # Save 5D arrays (more efficient)
        for i, sample in enumerate(tqdm(samples, desc="Saving arrays")):
            # Convert to numpy arrays
            notes_array = np.array([
                [n.onset_time, n.duration, n.octave, n.pitch_class, n.velocity]
                for n in sample.notes_5d
            ], dtype=np.float32)

            # Save
            np.savez_compressed(
                self.output_dir / f'sample_{i:04d}.npz',
                notes=notes_array,
                song_id=sample.song_id
            )

        print(f"   Saved {len(samples)} samples to {self.output_dir}")

    def load_dataset(self) -> List[BradMehldauSample]:
        """Load processed dataset"""
        metadata_path = self.output_dir / 'metadata.json'

        if not metadata_path.exists():
            raise FileNotFoundError(f"Dataset not found: {metadata_path}")

        with open(metadata_path) as f:
            metadata = json.load(f)

        print(f"📂 Loaded {metadata['num_samples']} samples")
        return metadata

    def get_statistics(self, samples: List[BradMehldauSample]) -> Dict:
        """Compute dataset statistics"""
        total_notes = sum(len(s.notes_5d) for s in samples)
        total_duration = sum(s.duration_seconds for s in samples)
        num_original = sum(1 for s in samples if s.augmentation_type == 'original')

        # Chord distribution
        all_chords = []
        for sample in samples:
            all_chords.extend([c[1] for c in sample.chords])

        unique_chords = len(set(all_chords))

        return {
            'num_samples': len(samples),
            'num_original_songs': num_original,
            'augmentation_factor': len(samples) / max(1, num_original),
            'total_notes': total_notes,
            'avg_notes_per_sample': total_notes / len(samples),
            'total_duration_hours': total_duration / 3600,
            'unique_chords': unique_chords,
        }


def create_training_splits(
    samples: List[BradMehldauSample],
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Tuple[List[BradMehldauSample], List[BradMehldauSample], List[BradMehldauSample]]:
    """
    Split dataset into train/val/test

    Important: Split by SONG, not by sample (to avoid data leakage)
    """
    # Group by original file
    songs_dict = {}
    for sample in samples:
        original = sample.original_file
        if original not in songs_dict:
            songs_dict[original] = []
        songs_dict[original].append(sample)

    # Split songs
    songs = list(songs_dict.keys())
    np.random.shuffle(songs)

    num_val = int(len(songs) * val_ratio)
    num_test = int(len(songs) * test_ratio)

    val_songs = songs[:num_val]
    test_songs = songs[num_val:num_val + num_test]
    train_songs = songs[num_val + num_test:]

    # Collect samples
    train_samples = []
    val_samples = []
    test_samples = []

    for song in train_songs:
        train_samples.extend(songs_dict[song])

    for song in val_songs:
        val_samples.extend(songs_dict[song])

    for song in test_songs:
        test_samples.extend(songs_dict[song])

    return train_samples, val_samples, test_samples


if __name__ == "__main__":
    print("=" * 60)
    print("Brad Mehldau Data Processing Pipeline")
    print("=" * 60)

    # Create dataset
    dataset = BradMehldauDataset(
        data_dir="./data/brad_mehldau",
        output_dir="./moonbeam_data/brad_processed"
    )

    # Process all files
    samples = dataset.process_all_files()

    # Statistics
    stats = dataset.get_statistics(samples)
    print("\n📊 Dataset Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        else:
            print(f"   {key}: {value}")

    # Train/val/test split
    train, val, test = create_training_splits(samples, val_ratio=0.1, test_ratio=0.1)
    print(f"\n📈 Data Split:")
    print(f"   Train: {len(train)} samples")
    print(f"   Val:   {len(val)} samples")
    print(f"   Test:  {len(test)} samples")

    # Save
    dataset.save_dataset(samples)

    print("\n✅ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Upload to Runpod: scp -r moonbeam_data/ runpod:/workspace/")
    print(f"2. Start LoRA fine-tuning: python train_moonbeam_lora.py")
    print(f"3. Expected training time: 4-6 hours on RTX 4090")
    print(f"4. Expected cost: $3-4")
