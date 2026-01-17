"""
Music Transformer MIDI 생성 스크립트
학습된 LoRA 가중치로 재즈 스타일 MIDI 생성

Usage:
    python scripts/generate.py --lora_path ./checkpoints/jazz_lora --output ./samples/
"""

import os
import sys
import argparse
from pathlib import Path
import torch

# Add music_transformer to path
SCRIPT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from model.music_transformer import MusicTransformer
from utilities.constants import TOKEN_END
from utilities.device import get_device
from midi_processor.processor import decode_midi

# Import LoRA from train script
from train_qlora import add_lora_to_model


def load_model_with_lora(lora_path: str, 
                         n_layers: int = 6,
                         num_heads: int = 8, 
                         d_model: int = 512,
                         dim_feedforward: int = 1024,
                         max_sequence: int = 2048,
                         rpr: bool = True,
                         lora_r: int = 16,
                         lora_alpha: int = 32):
    """Load Music Transformer with LoRA weights"""
    
    device = get_device()
    
    # Initialize model
    model = MusicTransformer(
        n_layers=n_layers,
        num_heads=num_heads,
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        max_sequence=max_sequence,
        rpr=rpr
    )
    
    # Add LoRA layers
    model, _ = add_lora_to_model(model, r=lora_r, alpha=lora_alpha)
    
    # Load LoRA weights
    lora_weights_path = Path(lora_path) / "lora_weights.pt"
    if lora_weights_path.exists():
        lora_state = torch.load(lora_weights_path, map_location=device)
        model.load_state_dict(lora_state, strict=False)
        print(f"Loaded LoRA weights from {lora_weights_path}")
    else:
        print(f"Warning: LoRA weights not found at {lora_weights_path}")
    
    model = model.to(device)
    model.eval()
    
    return model


def generate_midi(model, 
                  primer_tokens=None,
                  target_length: int = 1024,
                  temperature: float = 1.0,
                  output_path: str = None):
    """Generate MIDI from model"""
    
    device = get_device()
    
    # Use default primer if not provided
    if primer_tokens is None:
        # Random start token
        primer_tokens = torch.randint(0, 100, (1,), dtype=torch.long)
    
    primer_tokens = primer_tokens.to(device)
    
    print(f"Generating {target_length} tokens...")
    
    with torch.no_grad():
        generated = model.generate(
            primer=primer_tokens,
            target_seq_length=target_length,
            beam=0,
            beam_chance=1.0
        )
    
    generated = generated.cpu().numpy().flatten()
    
    # Decode to MIDI
    if output_path:
        print(f"Saving to {output_path}...")
        decode_midi(generated.tolist(), output_path)
        print(f"MIDI saved to {output_path}")
    
    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate MIDI with LoRA-finetuned Music Transformer")
    
    # Model
    parser.add_argument("--lora_path", type=str, default="./checkpoints/jazz_lora",
                        help="Path to LoRA checkpoint directory")
    parser.add_argument("--base_checkpoint", type=str, default=None,
                        help="Path to base model checkpoint (optional)")
    
    # Generation
    parser.add_argument("--length", type=int, default=1024,
                        help="Target sequence length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to generate")
    
    # Output
    parser.add_argument("--output", type=str, default="./samples",
                        help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Load model
    print("\n=== Loading Model with LoRA ===")
    model = load_model_with_lora(args.lora_path)
    
    # Generate samples
    print(f"\n=== Generating {args.num_samples} samples ===")
    
    for i in range(args.num_samples):
        output_path = os.path.join(args.output, f"jazz_sample_{i+1}.mid")
        
        print(f"\n--- Sample {i+1}/{args.num_samples} ---")
        generate_midi(
            model,
            target_length=args.length,
            temperature=args.temperature,
            output_path=output_path
        )
    
    print(f"\n=== Done ===")
    print(f"Samples saved to: {args.output}/")


if __name__ == "__main__":
    main()
