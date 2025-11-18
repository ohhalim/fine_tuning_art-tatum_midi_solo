"""
Generation and improvisation script for TatumFlow
Supports multiple generation modes:
- Unconditional generation
- Prompt continuation
- Style transfer improvisation
- Infilling
"""

import torch
import argparse
from pathlib import Path
import numpy as np
from typing import Optional

from .model import TatumFlow, create_tatumflow_model
from .tokenizer import TatumFlowTokenizer


class TatumFlowGenerator:
    """Generator for TatumFlow model"""

    def __init__(
        self,
        model: TatumFlow,
        tokenizer: TatumFlowTokenizer,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate_continuation(
        self,
        prompt_midi: Optional[str] = None,
        prompt_tokens: Optional[torch.Tensor] = None,
        num_tokens: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        style: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Generate continuation of a prompt

        Args:
            prompt_midi: Path to prompt MIDI file
            prompt_tokens: Or provide tokens directly
            num_tokens: Number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling
            style: Optional style vector

        Returns:
            Generated token sequence
        """
        # Get prompt tokens
        if prompt_tokens is None:
            if prompt_midi is None:
                # Start from scratch
                prompt_tokens = torch.tensor(
                    [[self.tokenizer.sos_id]],
                    dtype=torch.long,
                    device=self.device
                )
            else:
                # Encode MIDI
                tokens = self.tokenizer.encode_midi(prompt_midi)
                prompt_tokens = torch.tensor([tokens], dtype=torch.long, device=self.device)
        else:
            if prompt_tokens.dim() == 1:
                prompt_tokens = prompt_tokens.unsqueeze(0)
            prompt_tokens = prompt_tokens.to(self.device)

        # Generate
        generated = self.model.generate(
            prompt_tokens=prompt_tokens,
            max_length=num_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            style=style
        )

        return generated

    @torch.no_grad()
    def style_transfer(
        self,
        source_midi: str,
        target_style_midi: str,
        num_iterations: int = 3,
        denoise_strength: float = 0.7,
        temperature: float = 1.0
    ) -> torch.Tensor:
        """
        Transfer style from target to source using latent diffusion

        Args:
            source_midi: Source MIDI file (content)
            target_style_midi: Target style MIDI file
            num_iterations: Number of diffusion iterations
            denoise_strength: How much to denoise (0-1)
            temperature: Sampling temperature

        Returns:
            Generated tokens with transferred style
        """
        # Encode source
        source_tokens = torch.tensor(
            [self.tokenizer.encode_midi(source_midi)],
            dtype=torch.long,
            device=self.device
        )

        # Encode target style
        target_tokens = torch.tensor(
            [self.tokenizer.encode_midi(target_style_midi)],
            dtype=torch.long,
            device=self.device
        )

        # Extract style from target
        target_style, _, _ = self.model.encode_style(target_tokens)

        # Get source latent
        source_outputs = self.model(source_tokens, timestep=None)
        source_latent = source_outputs['latent']

        # Iterative denoising with target style
        current_latent = source_latent
        max_timestep = int(self.model.diffusion.num_steps * denoise_strength)

        for iteration in range(num_iterations):
            # Add noise
            t = torch.tensor([max_timestep - iteration * (max_timestep // num_iterations)], device=self.device)
            noise = torch.randn_like(current_latent)
            noisy_latent = self.model.diffusion.q_sample(current_latent, t, noise)

            # Denoise with target style
            # This is a simplified version - full implementation would use DDIM sampling
            with torch.no_grad():
                # Create dummy input for diffusion forward pass
                outputs = self.model(
                    source_tokens,
                    style=target_style,
                    timestep=t
                )
                current_latent = outputs['latent']

        # Decode to tokens
        x = self.model.latent_decoder(current_latent)

        # Add style
        style_emb = self.model.style_decoder(target_style).unsqueeze(1)
        x = x + style_emb

        # Transformer blocks
        time_emb = torch.zeros(1, self.model.hidden_dim, device=self.device)
        for block in self.model.blocks:
            x = block(x, time_emb, None)

        # Output
        x = self.model.norm_out(x)
        logits = self.model.head(x)

        # Sample tokens
        generated_tokens = []
        for i in range(logits.shape[1]):
            token_logits = logits[0, i, :] / temperature
            probs = torch.softmax(token_logits, dim=-1)
            token = torch.multinomial(probs, 1)
            generated_tokens.append(token.item())

        return torch.tensor([generated_tokens], dtype=torch.long)

    @torch.no_grad()
    def improvise(
        self,
        base_midi: str,
        num_variations: int = 5,
        creativity: float = 0.5,
        preserve_structure: bool = True
    ) -> list:
        """
        Generate improvisations of a base MIDI file

        Args:
            base_midi: Base MIDI file to improvise on
            num_variations: Number of variations to generate
            creativity: How creative (0=conservative, 1=very creative)
            preserve_structure: Whether to preserve musical structure

        Returns:
            List of generated token sequences
        """
        # Encode base MIDI
        base_tokens = torch.tensor(
            [self.tokenizer.encode_midi(base_midi)],
            dtype=torch.long,
            device=self.device
        )

        # Get base style
        base_style, style_mu, style_logvar = self.model.encode_style(base_tokens)

        variations = []

        for i in range(num_variations):
            # Sample from style distribution with creativity control
            if creativity > 0:
                # Add noise to style
                style_std = torch.exp(0.5 * style_logvar) * creativity
                style_noise = torch.randn_like(base_style) * style_std
                variation_style = base_style + style_noise
            else:
                variation_style = base_style

            # Generate with modified style
            if preserve_structure:
                # Use base tokens as strong prompt
                generated = self.model.generate(
                    prompt_tokens=base_tokens[:, :64],  # Use first 64 tokens as anchor
                    max_length=base_tokens.shape[1] - 64,
                    temperature=0.9 + creativity * 0.5,
                    top_k=40,
                    top_p=0.95,
                    style=variation_style
                )
            else:
                # Generate from scratch with style
                start_tokens = torch.tensor(
                    [[self.tokenizer.sos_id]],
                    dtype=torch.long,
                    device=self.device
                )
                generated = self.model.generate(
                    prompt_tokens=start_tokens,
                    max_length=512,
                    temperature=1.0 + creativity * 0.5,
                    top_k=50,
                    top_p=0.9,
                    style=variation_style
                )

            variations.append(generated)

        return variations

    def tokens_to_midi(self, tokens: torch.Tensor, output_path: str):
        """
        Convert tokens to MIDI and save

        Args:
            tokens: Token tensor
            output_path: Output MIDI file path
        """
        if tokens.dim() == 2:
            tokens = tokens[0]  # Take first batch

        token_list = tokens.cpu().numpy().tolist()

        # Decode to MIDI
        midi = self.tokenizer.decode_tokens(token_list)

        # Save
        midi.save(output_path)
        print(f"Saved MIDI to {output_path}")


def load_model_from_checkpoint(
    checkpoint_path: str,
    model_size: str = 'base',
    device: str = 'cuda'
) -> tuple:
    """
    Load TatumFlow model from checkpoint

    Returns:
        (model, tokenizer)
    """
    # Create tokenizer
    tokenizer = TatumFlowTokenizer()

    # Create model
    model = create_tatumflow_model(
        model_size=model_size,
        vocab_size=tokenizer.vocab_size
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Val loss: {checkpoint.get('val_loss', 'N/A'):.4f}")

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description='Generate with TatumFlow')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--mode', type=str, required=True,
                       choices=['continuation', 'style_transfer', 'improvise'],
                       help='Generation mode')
    parser.add_argument('--prompt', type=str, help='Prompt MIDI file')
    parser.add_argument('--target_style', type=str, help='Target style MIDI (for style transfer)')
    parser.add_argument('--output', type=str, required=True, help='Output MIDI file')
    parser.add_argument('--num_tokens', type=int, default=512, help='Number of tokens to generate')
    parser.add_argument('--num_variations', type=int, default=5, help='Number of variations (improvise mode)')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=50, help='Top-k sampling')
    parser.add_argument('--top_p', type=float, default=0.95, help='Nucleus sampling')
    parser.add_argument('--creativity', type=float, default=0.5, help='Creativity (0-1)')
    parser.add_argument('--model_size', type=str, default='base', choices=['small', 'base', 'large'])
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    # Load model
    print("Loading model...")
    model, tokenizer = load_model_from_checkpoint(
        args.checkpoint,
        model_size=args.model_size,
        device=args.device
    )

    # Create generator
    generator = TatumFlowGenerator(model, tokenizer, args.device)

    print(f"\nGeneration mode: {args.mode}")

    # Generate based on mode
    if args.mode == 'continuation':
        print(f"Generating continuation of {args.prompt}...")
        generated = generator.generate_continuation(
            prompt_midi=args.prompt,
            num_tokens=args.num_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p
        )
        generator.tokens_to_midi(generated, args.output)

    elif args.mode == 'style_transfer':
        if not args.target_style:
            raise ValueError("--target_style required for style_transfer mode")
        print(f"Transferring style from {args.target_style} to {args.prompt}...")
        generated = generator.style_transfer(
            source_midi=args.prompt,
            target_style_midi=args.target_style,
            temperature=args.temperature
        )
        generator.tokens_to_midi(generated, args.output)

    elif args.mode == 'improvise':
        print(f"Generating {args.num_variations} improvisations of {args.prompt}...")
        variations = generator.improvise(
            base_midi=args.prompt,
            num_variations=args.num_variations,
            creativity=args.creativity
        )

        # Save all variations
        output_path = Path(args.output)
        for i, variation in enumerate(variations):
            output_file = output_path.parent / f"{output_path.stem}_var{i+1}{output_path.suffix}"
            generator.tokens_to_midi(variation, str(output_file))

    print("\nGeneration complete!")


if __name__ == "__main__":
    main()
