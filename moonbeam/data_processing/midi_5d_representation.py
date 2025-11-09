"""
5D MIDI Representation for Moonbeam

Moonbeam uses a novel 5-dimensional representation:
1. onset_time: When the note starts (continuous)
2. duration: How long the note lasts (continuous)
3. octave: Which octave (0-10)
4. pitch_class: Which note in the octave (0-11, C=0, C#=1, ..., B=11)
5. velocity: How hard the note is played (0-127)

This is more efficient than traditional piano roll and captures musical structure better.
"""

import numpy as np
import pretty_midi
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Note5D:
    """5D note representation"""
    onset_time: float
    duration: float
    octave: int
    pitch_class: int
    velocity: int

    def to_midi_pitch(self) -> int:
        """Convert octave + pitch_class to MIDI pitch (0-127)"""
        return self.octave * 12 + self.pitch_class

    @classmethod
    def from_midi_note(cls, note: pretty_midi.Note) -> 'Note5D':
        """Convert pretty_midi.Note to 5D representation"""
        pitch = note.pitch
        octave = pitch // 12
        pitch_class = pitch % 12

        return cls(
            onset_time=note.start,
            duration=note.end - note.start,
            octave=octave,
            pitch_class=pitch_class,
            velocity=note.velocity
        )

    def to_pretty_midi_note(self) -> pretty_midi.Note:
        """Convert back to pretty_midi.Note"""
        return pretty_midi.Note(
            velocity=self.velocity,
            pitch=self.to_midi_pitch(),
            start=self.onset_time,
            end=self.onset_time + self.duration
        )


class MIDI5DConverter:
    """
    Convert between MIDI files and 5D representation

    This is the core data format for Moonbeam fine-tuning
    """

    def __init__(
        self,
        time_resolution: float = 0.0625,  # 16th note at 120 BPM
        quantize: bool = True
    ):
        """
        Args:
            time_resolution: Time quantization (0.0625 = 16th note)
            quantize: Whether to quantize onset times and durations
        """
        self.time_resolution = time_resolution
        self.quantize = quantize

    def midi_to_5d(self, midi_path: str) -> List[Note5D]:
        """
        Convert MIDI file to 5D representation

        Args:
            midi_path: Path to MIDI file

        Returns:
            List of Note5D objects, sorted by onset time
        """
        midi = pretty_midi.PrettyMIDI(midi_path)

        # Combine all instruments (or filter for piano only)
        all_notes = []
        for instrument in midi.instruments:
            if instrument.is_drum:
                continue  # Skip drums for piano solo
            all_notes.extend(instrument.notes)

        # Convert to 5D
        notes_5d = [Note5D.from_midi_note(note) for note in all_notes]

        # Quantize if requested
        if self.quantize:
            notes_5d = self._quantize_notes(notes_5d)

        # Sort by onset time
        notes_5d.sort(key=lambda n: n.onset_time)

        return notes_5d

    def notes_5d_to_midi(
        self,
        notes_5d: List[Note5D],
        output_path: Optional[str] = None,
        tempo: int = 120
    ) -> pretty_midi.PrettyMIDI:
        """
        Convert 5D representation back to MIDI

        Args:
            notes_5d: List of Note5D objects
            output_path: Optional path to save MIDI file
            tempo: Tempo in BPM

        Returns:
            pretty_midi.PrettyMIDI object
        """
        midi = pretty_midi.PrettyMIDI(initial_tempo=tempo)
        piano = pretty_midi.Instrument(program=0)  # Acoustic Grand Piano

        # Convert all notes
        for note_5d in notes_5d:
            piano.notes.append(note_5d.to_pretty_midi_note())

        midi.instruments.append(piano)

        if output_path:
            midi.write(output_path)

        return midi

    def _quantize_notes(self, notes: List[Note5D]) -> List[Note5D]:
        """Quantize onset times and durations to grid"""
        quantized = []

        for note in notes:
            # Quantize onset and duration
            onset_quantized = round(note.onset_time / self.time_resolution) * self.time_resolution
            duration_quantized = max(
                self.time_resolution,  # Minimum duration = 16th note
                round(note.duration / self.time_resolution) * self.time_resolution
            )

            quantized.append(Note5D(
                onset_time=onset_quantized,
                duration=duration_quantized,
                octave=note.octave,
                pitch_class=note.pitch_class,
                velocity=note.velocity
            ))

        return quantized

    def tokenize_5d(self, notes_5d: List[Note5D]) -> np.ndarray:
        """
        Convert 5D notes to tokens for Moonbeam

        Returns:
            tokens: [num_notes, 5] array
        """
        tokens = np.array([
            [note.onset_time, note.duration, note.octave, note.pitch_class, note.velocity]
            for note in notes_5d
        ], dtype=np.float32)

        return tokens

    def detokenize_5d(self, tokens: np.ndarray) -> List[Note5D]:
        """Convert tokens back to Note5D objects"""
        notes_5d = []

        for token in tokens:
            onset_time, duration, octave, pitch_class, velocity = token
            notes_5d.append(Note5D(
                onset_time=float(onset_time),
                duration=float(duration),
                octave=int(octave),
                pitch_class=int(pitch_class),
                velocity=int(velocity)
            ))

        return notes_5d


class ChordExtractor:
    """
    Extract chord progression from MIDI

    For conditional generation (chord → melody)
    """

    def __init__(self, chord_window: float = 2.0):
        """
        Args:
            chord_window: Time window for chord detection (seconds)
        """
        self.chord_window = chord_window
        self.chord_templates = self._init_chord_templates()

    def _init_chord_templates(self) -> Dict[str, List[int]]:
        """Chord templates (intervals from root)"""
        return {
            'maj': [0, 4, 7],
            'min': [0, 3, 7],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            '7': [0, 4, 7, 10],
            'dim': [0, 3, 6],
            'dim7': [0, 3, 6, 9],
            'aug': [0, 4, 8],
            'sus4': [0, 5, 7],
            'm7b5': [0, 3, 6, 10],  # half-diminished
            'maj9': [0, 4, 7, 11, 14],
            'min9': [0, 3, 7, 10, 14],
        }

    def extract_chords(self, notes_5d: List[Note5D]) -> List[Tuple[float, str]]:
        """
        Extract chord progression from 5D notes

        Returns:
            List of (time, chord_name) tuples
        """
        if not notes_5d:
            return []

        chords = []

        # Analyze chords in time windows
        max_time = max(note.onset_time + note.duration for note in notes_5d)
        current_time = 0

        while current_time < max_time:
            # Get notes in current window
            window_notes = [
                note for note in notes_5d
                if current_time <= note.onset_time < current_time + self.chord_window
            ]

            if window_notes:
                # Detect chord
                chord_name = self._detect_chord(window_notes)
                chords.append((current_time, chord_name))

            current_time += self.chord_window

        return chords

    def _detect_chord(self, notes: List[Note5D]) -> str:
        """Detect chord from notes in a time window"""
        if not notes:
            return "N"  # No chord

        # Get unique pitch classes
        pitch_classes = list(set(note.pitch_class for note in notes))

        if len(pitch_classes) < 3:
            return "N"  # Too few notes

        # Try to match chord templates
        for root in range(12):
            # Normalize pitch classes relative to root
            normalized = sorted([(pc - root) % 12 for pc in pitch_classes])

            # Match against templates
            for chord_type, template in self.chord_templates.items():
                if all(interval in normalized for interval in template):
                    # Convert root to note name
                    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                    root_name = note_names[root]
                    return f"{root_name}{chord_type}"

        return "N"  # No match


class DataAugmenter5D:
    """
    Data augmentation for 5D MIDI

    - Transpose (12 keys)
    - Tempo variation
    - Velocity variation
    """

    def transpose(self, notes_5d: List[Note5D], semitones: int) -> List[Note5D]:
        """
        Transpose notes by semitones

        Args:
            notes_5d: Original notes
            semitones: Number of semitones to transpose (-12 to +12)

        Returns:
            Transposed notes
        """
        transposed = []

        for note in notes_5d:
            new_pitch = note.to_midi_pitch() + semitones

            # Clip to valid MIDI range (0-127)
            new_pitch = max(0, min(127, new_pitch))

            new_octave = new_pitch // 12
            new_pitch_class = new_pitch % 12

            transposed.append(Note5D(
                onset_time=note.onset_time,
                duration=note.duration,
                octave=new_octave,
                pitch_class=new_pitch_class,
                velocity=note.velocity
            ))

        return transposed

    def time_stretch(self, notes_5d: List[Note5D], factor: float) -> List[Note5D]:
        """
        Stretch time by factor

        Args:
            factor: Time stretch factor (0.8 = faster, 1.2 = slower)
        """
        stretched = []

        for note in notes_5d:
            stretched.append(Note5D(
                onset_time=note.onset_time * factor,
                duration=note.duration * factor,
                octave=note.octave,
                pitch_class=note.pitch_class,
                velocity=note.velocity
            ))

        return stretched

    def velocity_variation(self, notes_5d: List[Note5D], variation: float = 0.1) -> List[Note5D]:
        """
        Add random velocity variation

        Args:
            variation: Variation amount (0.1 = ±10%)
        """
        varied = []

        for note in notes_5d:
            vel_delta = int(note.velocity * variation * np.random.randn())
            new_velocity = max(20, min(127, note.velocity + vel_delta))

            varied.append(Note5D(
                onset_time=note.onset_time,
                duration=note.duration,
                octave=note.octave,
                pitch_class=note.pitch_class,
                velocity=new_velocity
            ))

        return varied

    def augment_dataset(
        self,
        notes_5d: List[Note5D],
        num_augmentations: int = 12
    ) -> List[List[Note5D]]:
        """
        Generate augmented versions

        Args:
            notes_5d: Original notes
            num_augmentations: Number of augmented versions (default 12 for all keys)

        Returns:
            List of augmented note sequences
        """
        augmented = []

        # Transpose to all 12 keys
        for semitones in range(-6, 6):
            transposed = self.transpose(notes_5d, semitones)
            augmented.append(transposed)

        # Add tempo variations
        if num_augmentations > 12:
            for factor in [0.9, 1.1]:
                stretched = self.time_stretch(notes_5d, factor)
                augmented.append(stretched)

        return augmented[:num_augmentations]


if __name__ == "__main__":
    print("Testing 5D MIDI Representation...")

    # Test converter
    converter = MIDI5DConverter()

    # Create sample notes
    sample_notes = [
        Note5D(0.0, 0.5, 4, 0, 80),  # C4, quarter note
        Note5D(0.5, 0.5, 4, 4, 75),  # E4
        Note5D(1.0, 0.5, 4, 7, 70),  # G4
        Note5D(1.5, 0.5, 4, 11, 85), # B4
    ]

    print(f"Sample notes (5D): {len(sample_notes)}")

    # Convert to MIDI
    midi = converter.notes_5d_to_midi(sample_notes, tempo=120)
    print(f"Converted to MIDI: {len(midi.instruments[0].notes)} notes")

    # Test chord extraction
    chord_extractor = ChordExtractor()
    chords = chord_extractor.extract_chords(sample_notes)
    print(f"Detected chords: {chords}")

    # Test augmentation
    augmenter = DataAugmenter5D()
    transposed = augmenter.transpose(sample_notes, 2)  # Up 2 semitones
    print(f"Transposed notes: {len(transposed)}")

    augmented = augmenter.augment_dataset(sample_notes, num_augmentations=12)
    print(f"Augmented dataset: {len(augmented)} versions")

    print("\n✅ 5D representation tests passed!")
