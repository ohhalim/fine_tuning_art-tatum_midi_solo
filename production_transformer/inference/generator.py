"""
Music Generator

Production inference wrapper for Music Transformer
"""

import torch
from typing import List, Optional
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import MusicTransformerForGeneration
from data import EventTokenizer


class MusicGenerator:
    """
    High-level generator for Brad Mehldau-style piano solos

    Example:
        >>> generator = MusicGenerator("experiments/brad_mehldau_v1/final_model")
        >>> midi = generator.generate(
        ...     prompt="Cmaj7 Am7 Dm7 G7",
        ...     max_length=512,
        ...     temperature=0.9
        ... )
        >>> generator.save_midi(midi, "output/solo.mid")
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None
    ):
        """
        Initialize generator

        Args:
            checkpoint_path: Path to trained model
            device: Device to run on (auto-detected if None)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading model from {checkpoint_path}...")
        self.model = MusicTransformerForGeneration.from_pretrained(checkpoint_path)
        self.model.to(self.device)
        self.model.eval()

        self.tokenizer = EventTokenizer()

        print(f"✅ Model loaded on {self.device}")

    def parse_chords(self, chord_string: str) -> Optional[List[int]]:
        """
        Parse chord string to chord IDs

        Args:
            chord_string: Space-separated chords (e.g., "Cmaj7 Am7 Dm7 G7")

        Returns:
            List of chord IDs or None
        """
        if not chord_string or chord_string.strip() == "":
            return None

        chords = chord_string.strip().split()
        chord_ids = []

        for chord in chords:
            if chord in self.tokenizer.vocab.chord_to_id:
                chord_ids.append(self.tokenizer.vocab.chord_to_id[chord])
            else:
                print(f"Warning: Unknown chord '{chord}', using 'N' (no chord)")
                chord_ids.append(self.tokenizer.vocab.chord_to_id['N'])

        return chord_ids if chord_ids else None

    @torch.no_grad()
    def generate(
        self,
        prompt: Optional[str] = None,
        start_tokens: Optional[torch.Tensor] = None,
        max_length: int = 512,
        temperature: float = 0.9,
        top_p: float = 0.95,
        top_k: Optional[int] = None,
        num_return_sequences: int = 1
    ) -> List[torch.Tensor]:
        """
        Generate music sequences

        Args:
            prompt: Chord progression (e.g., "Cmaj7 Am7 Dm7 G7")
            start_tokens: Alternative to prompt (raw tokens)
            max_length: Maximum sequence length
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling (optional)
            num_return_sequences: Number of sequences to generate

        Returns:
            List of generated token sequences
        """
        # Parse chords if provided
        chord_ids = None
        if prompt:
            chord_ids = self.parse_chords(prompt)
            if chord_ids:
                chord_ids = torch.tensor([chord_ids], device=self.device)

        # Create start tokens if not provided
        if start_tokens is None:
            start_tokens = torch.tensor(
                [[self.tokenizer.bos_token_id]],
                device=self.device
            )
        else:
            start_tokens = start_tokens.to(self.device)

        # Expand for multiple sequences
        if num_return_sequences > 1:
            start_tokens = start_tokens.repeat(num_return_sequences, 1)
            if chord_ids is not None:
                chord_ids = chord_ids.repeat(num_return_sequences, 1)

        # Generate
        generated = self.model.generate(
            input_ids=start_tokens,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            chord_ids=chord_ids,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )

        return [generated[i] for i in range(num_return_sequences)]

    def tokens_to_midi(self, tokens: torch.Tensor, output_path: Optional[str] = None):
        """
        Convert tokens to MIDI file

        Args:
            tokens: Generated token sequence
            output_path: Optional path to save MIDI

        Returns:
            pretty_midi.PrettyMIDI object
        """
        # Convert to events
        token_list = tokens.cpu().tolist()
        events = self.tokenizer.decode(token_list)

        # Convert to MIDI
        midi = self.tokenizer.events_to_midi(events, output_path=output_path)

        return midi

    def generate_and_save(
        self,
        prompt: Optional[str] = None,
        output_path: str = "output.mid",
        **generate_kwargs
    ):
        """
        Generate music and save to MIDI file

        Args:
            prompt: Chord progression
            output_path: Output MIDI path
            **generate_kwargs: Arguments for generate()

        Returns:
            Path to saved MIDI file
        """
        # Generate
        generated = self.generate(prompt=prompt, num_return_sequences=1, **generate_kwargs)

        # Save to MIDI
        self.tokens_to_midi(generated[0], output_path)

        print(f"✅ MIDI saved to {output_path}")
        return output_path


if __name__ == "__main__":
    """Test generator"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--prompt", type=str, default="Cmaj7 Am7 Dm7 G7")
    parser.add_argument("--output", type=str, default="output.mid")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--num_samples", type=int, default=1)

    args = parser.parse_args()

    # Create generator
    generator = MusicGenerator(args.checkpoint)

    # Generate
    print(f"\nGenerating with prompt: {args.prompt}")
    print(f"Temperature: {args.temperature}")
    print(f"Max length: {args.max_length}\n")

    for i in range(args.num_samples):
        output_path = args.output if args.num_samples == 1 else f"{args.output[:-4]}_{i+1}.mid"
        generator.generate_and_save(
            prompt=args.prompt,
            output_path=output_path,
            max_length=args.max_length,
            temperature=args.temperature
        )
