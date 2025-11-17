"""
Generate jazz audio with PersonalJazz model

Simple inference script for generating personalized jazz improvisations
"""

import torch
import torchaudio
from pathlib import Path

from ..model.personaljazz import PersonalJazz


def generate_jazz(
    model_path: str,
    style_prompt: str = "ohhalim jazz piano style",
    output_path: str = "./generated_jazz.wav",
    duration: float = 16.0,
    temperature: float = 0.95,
    top_p: float = 0.9,
    device: str = 'cuda'
):
    """
    Generate jazz improvisation

    Args:
        model_path: Path to trained/fine-tuned model
        style_prompt: Text description of desired style
        output_path: Where to save generated audio
        duration: Duration in seconds
        temperature: Sampling temperature (higher = more random)
        top_p: Nucleus sampling parameter
        device: Device to use

    Returns:
        audio: Generated audio tensor
    """
    print("=" * 80)
    print("PersonalJazz Generation")
    print("=" * 80)

    # Load model
    print(f"\nLoading model from {model_path}...")
    model = PersonalJazz.load_pretrained(model_path, device=device)

    # Generate
    print(f"\nGenerating {duration}s of jazz...")
    print(f"  Style: '{style_prompt}'")
    print(f"  Temperature: {temperature}")
    print(f"  Top-p: {top_p}")
    print()

    with torch.no_grad():
        audio = model.generate(
            style_prompt=style_prompt,
            duration=duration,
            temperature=temperature,
            top_p=top_p,
            device=device
        )

    # audio shape: (1, 2, T)
    audio = audio.squeeze(0).cpu()  # (2, T)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    torchaudio.save(
        str(output_path),
        audio,
        sample_rate=model.sample_rate
    )

    print(f"\n✅ Generated audio saved to: {output_path}")
    print(f"   Duration: {duration}s")
    print(f"   Sample rate: {model.sample_rate} Hz")
    print(f"   Channels: Stereo")

    return audio


if __name__ == "__main__":
    # Example: Generate jazz with fine-tuned model
    generate_jazz(
        model_path="./ohhalim-jazz-style/final_model.pt",
        style_prompt="ohhalim jazz piano modal improvisation",
        output_path="./output/ohhalim_jazz_01.wav",
        duration=16.0,
        temperature=0.95
    )
