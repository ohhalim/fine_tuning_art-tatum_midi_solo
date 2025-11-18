#!/usr/bin/env python3
"""
Quick start generation script for TatumFlow
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import argparse
from tatumflow import TatumFlowGenerator, load_model_from_checkpoint


def main():
    parser = argparse.ArgumentParser(
        description='Generate music with TatumFlow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate continuation of a MIDI file
  python generate_music.py --checkpoint checkpoints/best.pt \\
      --mode continuation --prompt input.mid --output output.mid

  # Generate Art Tatum-style improvisation
  python generate_music.py --checkpoint checkpoints/best.pt \\
      --mode improvise --prompt input.mid --output improvisation.mid \\
      --num_variations 5 --creativity 0.7

  # Transfer style between pieces
  python generate_music.py --checkpoint checkpoints/best.pt \\
      --mode style_transfer --prompt classical.mid \\
      --target_style jazz.mid --output jazz_style.mid
        """
    )

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['continuation', 'style_transfer', 'improvise'],
                       help='Generation mode')
    parser.add_argument('--prompt', type=str, required=True,
                       help='Input MIDI file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output MIDI file path')

    # Optional arguments
    parser.add_argument('--target_style', type=str,
                       help='Target style MIDI (for style_transfer mode)')
    parser.add_argument('--num_tokens', type=int, default=512,
                       help='Number of tokens to generate (continuation mode)')
    parser.add_argument('--num_variations', type=int, default=5,
                       help='Number of variations (improvise mode)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Sampling temperature (higher = more random)')
    parser.add_argument('--top_k', type=int, default=50,
                       help='Top-k sampling parameter')
    parser.add_argument('--top_p', type=float, default=0.95,
                       help='Nucleus sampling parameter')
    parser.add_argument('--creativity', type=float, default=0.5,
                       help='Creativity level 0-1 (improvise mode)')
    parser.add_argument('--model_size', type=str, default='base',
                       choices=['small', 'base', 'large'])
    parser.add_argument('--device', type=str,
                       default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Validate inputs
    if args.mode == 'style_transfer' and not args.target_style:
        parser.error("--target_style is required for style_transfer mode")

    if not Path(args.prompt).exists():
        parser.error(f"Prompt file not found: {args.prompt}")

    if args.mode == 'style_transfer' and not Path(args.target_style).exists():
        parser.error(f"Target style file not found: {args.target_style}")

    print("="*80)
    print("TatumFlow Music Generation")
    print("="*80)

    # Load model
    print(f"\n[1/3] Loading model from {args.checkpoint}...")
    model, tokenizer = load_model_from_checkpoint(
        args.checkpoint,
        model_size=args.model_size,
        device=args.device
    )

    # Create generator
    print(f"\n[2/3] Initializing generator...")
    generator = TatumFlowGenerator(model, tokenizer, args.device)
    print(f"  Device: {args.device}")
    print(f"  Mode: {args.mode}")

    # Generate
    print(f"\n[3/3] Generating music...")

    if args.mode == 'continuation':
        print(f"  Generating {args.num_tokens} tokens continuation of {args.prompt}")
        print(f"  Temperature: {args.temperature}, Top-k: {args.top_k}, Top-p: {args.top_p}")

        generated = generator.generate_continuation(
            prompt_midi=args.prompt,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        generator.tokens_to_midi(generated, args.output)

    elif args.mode == 'style_transfer':
        print(f"  Transferring style from {args.target_style}")
        print(f"  to {args.prompt}")

        generated = generator.style_transfer(
            source_midi=args.prompt,
            target_style_midi=args.target_style,
            temperature=args.temperature
        )
        generator.tokens_to_midi(generated, args.output)

    elif args.mode == 'improvise':
        print(f"  Generating {args.num_variations} improvisations")
        print(f"  Creativity: {args.creativity}")

        variations = generator.improvise(
            base_midi=args.prompt,
            num_variations=args.num_variations,
            creativity=args.creativity
        )

        # Save all variations
        output_path = Path(args.output)
        for i, variation in enumerate(variations):
            if args.num_variations > 1:
                output_file = output_path.parent / f"{output_path.stem}_var{i+1}{output_path.suffix}"
            else:
                output_file = output_path

            generator.tokens_to_midi(variation, str(output_file))

    print("\n" + "="*80)
    print("Generation completed successfully!")
    print("="*80)

    if args.mode == 'improvise' and args.num_variations > 1:
        print(f"\nGenerated {args.num_variations} variations:")
        for i in range(args.num_variations):
            output_path = Path(args.output)
            var_file = output_path.parent / f"{output_path.stem}_var{i+1}{output_path.suffix}"
            print(f"  {var_file}")
    else:
        print(f"\nOutput saved to: {args.output}")


if __name__ == "__main__":
    main()
