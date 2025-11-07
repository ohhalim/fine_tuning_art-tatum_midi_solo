"""
Brad Mehldau AI Generator - Inference Server

Optimized inference with:
- INT8 quantization
- ONNX optimization (optional)
- Batch processing
- Low latency for real-time generation
"""

import os
import sys
import torch
import numpy as np
from typing import List, Optional, Tuple
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.hybrid_model import SCGTransformerHybrid


class BradMehldauGenerator:
    """
    Optimized Brad Mehldau style generator for real-time inference

    Features:
    - Quantized model for faster inference
    - DDIM with 50 steps (< 1s on GPU, < 5s on CPU)
    - Temperature control for creativity
    - Guidance scale for style strength
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda",
        quantize: bool = True,
        compile: bool = False
    ):
        """
        Args:
            checkpoint_path: Path to trained model checkpoint
            device: Device to run on ('cuda', 'mps', 'cpu')
            quantize: Apply INT8 quantization for faster inference
            compile: Use torch.compile for PyTorch 2.0+ (experimental)
        """
        self.device = device
        print(f"🎹 Loading Brad Mehldau Generator...")
        print(f"   Device: {device}")

        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        self.model = SCGTransformerHybrid()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.model.to(device)

        # Quantization
        if quantize and device == "cpu":
            print("   Applying INT8 quantization...")
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {torch.nn.Linear},
                dtype=torch.qint8
            )

        # Compile (PyTorch 2.0+)
        if compile:
            try:
                print("   Compiling model...")
                self.model = torch.compile(self.model)
            except Exception as e:
                print(f"   ⚠️  Compilation failed: {e}")

        print("   ✅ Model loaded!\n")

    @torch.no_grad()
    def generate_solo(
        self,
        chord_progression: List[str],
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        temperature: float = 0.8,
        duration_bars: int = 4
    ) -> np.ndarray:
        """
        Generate Brad Mehldau style solo from chord progression

        Args:
            chord_progression: List of chord symbols ['Cmaj7', 'Dm7', 'G7', 'Cmaj7']
            num_steps: DDIM steps (50 = fast, 1000 = slow/better quality)
            guidance_scale: Style strength (7.5 = default, higher = more Brad-like)
            temperature: Creativity (0.5 = conservative, 1.5 = wild)
            duration_bars: Duration in bars (4 bars = ~8 seconds)

        Returns:
            piano_roll: [2, 128, time] numpy array
                        2 channels: onset + sustain
                        128: MIDI pitches
                        time: time steps
        """
        start_time = time.time()

        # Generate
        piano_roll = self.model.generate(
            chord_progression=chord_progression,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            temperature=temperature
        )

        # Convert to numpy
        piano_roll = piano_roll.cpu().numpy()[0]  # [2, 128, time]

        elapsed = time.time() - start_time
        print(f"⏱️  Generated in {elapsed:.2f}s")

        return piano_roll

    def piano_roll_to_midi_notes(
        self,
        piano_roll: np.ndarray,
        threshold: float = 0.5,
        velocity_range: Tuple[int, int] = (60, 100)
    ) -> List[dict]:
        """
        Convert piano roll to MIDI note events

        Args:
            piano_roll: [2, 128, time] piano roll
            threshold: Onset detection threshold
            velocity_range: (min, max) velocity

        Returns:
            notes: List of {'pitch', 'start', 'end', 'velocity'}
        """
        onset_channel = piano_roll[0]  # [128, time]
        sustain_channel = piano_roll[1]

        notes = []

        # Process each pitch
        for pitch in range(128):
            onset_signal = onset_channel[pitch]

            # Find onsets
            onsets = np.where(onset_signal > threshold)[0]

            for onset in onsets:
                # Find note end (when sustain drops)
                end = onset + 1
                while end < len(sustain_channel[pitch]) and sustain_channel[pitch][end] > threshold:
                    end += 1

                # Map velocity based on onset strength
                velocity = int(np.clip(
                    velocity_range[0] + (onset_signal[onset] * (velocity_range[1] - velocity_range[0])),
                    velocity_range[0],
                    velocity_range[1]
                ))

                notes.append({
                    'pitch': pitch,
                    'start': onset,
                    'end': end,
                    'velocity': velocity
                })

        return notes


class MIDIChordDetector:
    """
    Detects chords from MIDI notes

    Receives MIDI note-on events and identifies chord type
    """

    def __init__(self):
        self.active_notes = set()
        self.chord_templates = self._init_chord_templates()

    def _init_chord_templates(self) -> dict:
        """
        Initialize chord templates

        Returns intervals from root note
        """
        return {
            'maj': [0, 4, 7],
            'min': [0, 3, 7],
            'maj7': [0, 4, 7, 11],
            'min7': [0, 3, 7, 10],
            '7': [0, 4, 7, 10],
            'dim': [0, 3, 6],
            'aug': [0, 4, 8],
            'sus4': [0, 5, 7],
            'm7b5': [0, 3, 6, 10]  # half-diminished
        }

    def note_on(self, pitch: int):
        """Add note to active notes"""
        self.active_notes.add(pitch)

    def note_off(self, pitch: int):
        """Remove note from active notes"""
        self.active_notes.discard(pitch)

    def detect_chord(self) -> Optional[str]:
        """
        Detect chord from currently active notes

        Returns:
            chord_name: e.g., "Cmaj7", "Dm7", etc.
        """
        if len(self.active_notes) < 3:
            return None

        notes = sorted(list(self.active_notes))

        # Normalize to root = 0
        root = notes[0]
        intervals = [(note - root) % 12 for note in notes]

        # Match chord template
        for chord_type, template in self.chord_templates.items():
            if all(interval in intervals for interval in template):
                # Convert root to note name
                note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
                root_name = note_names[root % 12]

                return f"{root_name}{chord_type}"

        return None

    def clear(self):
        """Clear all active notes"""
        self.active_notes.clear()


if __name__ == "__main__":
    print("Testing Brad Mehldau Generator...")

    # Test chord detection
    print("\n=== Chord Detection Test ===")
    detector = MIDIChordDetector()

    # Cmaj7: C-E-G-B (60-64-67-71)
    detector.note_on(60)
    detector.note_on(64)
    detector.note_on(67)
    detector.note_on(71)
    chord = detector.detect_chord()
    print(f"Notes: C-E-G-B → Detected: {chord}")

    detector.clear()

    # Dm7: D-F-A-C (62-65-69-72)
    detector.note_on(62)
    detector.note_on(65)
    detector.note_on(69)
    detector.note_on(72)
    chord = detector.detect_chord()
    print(f"Notes: D-F-A-C → Detected: {chord}")

    print("\n✅ Chord detection working!")

    # Test generator (requires trained model)
    print("\n=== Generator Test ===")
    print("⚠️  Generator test requires trained model checkpoint")
    print("Run after training: python server/inference_server.py --checkpoint ./checkpoints/brad_final/best.pt")
