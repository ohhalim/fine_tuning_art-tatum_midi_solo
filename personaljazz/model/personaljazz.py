"""
PersonalJazz: Real-time Personalized Jazz Improvisation Model

Complete model integrating:
- StyleEncoder: Text/Audio → Style embedding
- AudioCodec: Audio ↔ Discrete tokens
- MusicTransformer: Token sequence generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import torchaudio

from .transformer import MusicTransformer
from .codec import AudioCodec
from .style_encoder import StyleEncoder


class PersonalJazz(nn.Module):
    """
    PersonalJazz: End-to-end personalized jazz generation model

    Architecture:
        1. StyleEncoder encodes text prompt → style embedding
        2. MusicTransformer generates token sequence conditioned on style
        3. AudioCodec decodes tokens → audio waveform

    Supports real-time generation with chunk-based processing
    """

    def __init__(
        self,
        # Codec config
        sample_rate: int = 48000,
        codebook_size: int = 2048,
        num_quantizers: int = 8,

        # Transformer config
        vocab_size: int = 2048,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,

        # Style encoder config
        style_vocab_size: int = 50000,
        style_dim: int = 512
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers

        # Components
        self.codec = AudioCodec(
            sample_rate=sample_rate,
            in_channels=2,
            codebook_size=codebook_size,
            num_quantizers=num_quantizers
        )

        self.style_encoder = StyleEncoder(
            vocab_size=style_vocab_size,
            d_model=style_dim
        )

        self.transformer = MusicTransformer(
            vocab_size=codebook_size,  # One token per quantizer level
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            style_dim=style_dim
        )

        print(f"\nPersonalJazz Model Summary:")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Codebook: {codebook_size}^{num_quantizers} = {codebook_size ** num_quantizers:.2e} states")
        print(f"  Transformer: {self.transformer.count_parameters() / 1e6:.1f}M params")
        print(f"  Total: {self.count_parameters() / 1e6:.1f}M params\n")

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def encode_style_from_text(self, text_tokens: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Encode text prompt to style embedding"""
        return self.style_encoder.encode_text(text_tokens, attention_mask)

    def encode_style_from_audio(self, audio: torch.Tensor):
        """Encode audio to style embedding"""
        return self.style_encoder.encode_audio(audio)

    def audio_to_tokens(self, audio: torch.Tensor):
        """Convert audio to discrete tokens"""
        tokens, loss = self.codec.encode(audio)

        # Flatten quantizer dimension: (B, T, Q) → (B, T*Q)
        # Each timestep has Q tokens (one per quantizer level)
        B, T, Q = tokens.shape
        tokens_flat = tokens.reshape(B, T * Q)

        return tokens_flat, loss

    def tokens_to_audio(self, tokens_flat: torch.Tensor):
        """Convert discrete tokens to audio"""
        # Unflatten: (B, T*Q) → (B, T, Q)
        B, TQ = tokens_flat.shape
        T = TQ // self.num_quantizers
        tokens = tokens_flat.reshape(B, T, self.num_quantizers)

        audio = self.codec.decode(tokens)
        return audio

    def forward(
        self,
        audio: torch.Tensor,
        style_emb: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ):
        """
        Training forward pass

        Args:
            audio: Input audio (B, C, T_audio)
            style_emb: Pre-computed style embedding (B, style_dim)
            text_tokens: Text tokens for style (B, T_text)
            attention_mask: Text attention mask (B, T_text)

        Returns:
            loss_dict: Dictionary of losses
        """
        # Encode style if not provided
        if style_emb is None:
            if text_tokens is not None:
                style_emb = self.encode_style_from_text(text_tokens, attention_mask)
            else:
                # Use audio itself as style (for self-reconstruction)
                style_emb = self.encode_style_from_audio(audio)

        # Encode audio to tokens
        tokens_flat, codec_loss = self.audio_to_tokens(audio)

        # Predict next token
        # Input: tokens[:-1], Target: tokens[1:]
        logits = self.transformer(tokens_flat[:, :-1], style_emb)

        # Cross-entropy loss
        transformer_loss = F.cross_entropy(
            logits.reshape(-1, self.codebook_size),
            tokens_flat[:, 1:].reshape(-1)
        )

        # Total loss
        total_loss = transformer_loss + 0.1 * codec_loss

        loss_dict = {
            'total': total_loss,
            'transformer': transformer_loss,
            'codec': codec_loss
        }

        return loss_dict

    @torch.no_grad()
    def generate(
        self,
        style_prompt: str = None,
        style_emb: torch.Tensor = None,
        duration: float = 16.0,
        temperature: float = 1.0,
        top_p: float = 0.9,
        chunk_duration: float = 2.0,
        device: str = 'cuda'
    ):
        """
        Generate audio from style prompt

        Args:
            style_prompt: Text description (e.g., "ohhalim jazz piano style")
            style_emb: Pre-computed style embedding (alternative to text)
            duration: Total duration in seconds
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            chunk_duration: Duration of each chunk (for real-time generation)
            device: Device to run on

        Returns:
            audio: Generated audio (1, C, T)
        """
        self.eval()

        # Encode style
        if style_emb is None:
            if style_prompt is not None:
                # Tokenize text (simple word-level for demo)
                # In production, use proper tokenizer (e.g., SentencePiece)
                from .tokenizer import tokenize_text
                text_tokens, attention_mask = tokenize_text(style_prompt, device=device)
                style_emb = self.encode_style_from_text(text_tokens, attention_mask)
            else:
                raise ValueError("Must provide either style_prompt or style_emb")

        # Calculate number of tokens to generate
        # Codec downsamples by 640x: 48000 Hz → 75 Hz
        # Each timestep has num_quantizers tokens
        tokens_per_sec = self.sample_rate // 640
        total_tokens = int(duration * tokens_per_sec * self.num_quantizers)

        # Generate in chunks
        chunk_tokens = int(chunk_duration * tokens_per_sec * self.num_quantizers)
        num_chunks = int(duration / chunk_duration)

        # Start tokens (random or learned start token)
        start_token = torch.randint(0, self.codebook_size, (1, 1), device=device)

        all_tokens = start_token
        kv_caches = None

        for i in range(num_chunks):
            print(f"Generating chunk {i+1}/{num_chunks}...")

            # Generate chunk
            tokens, kv_caches = self.transformer.generate(
                all_tokens,
                max_new_tokens=chunk_tokens,
                style_emb=style_emb,
                temperature=temperature,
                top_p=top_p
            )

            all_tokens = tokens

        # Convert tokens to audio
        audio = self.tokens_to_audio(all_tokens[:, 1:])  # Skip start token

        return audio

    def save_pretrained(self, path: str):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'sample_rate': self.sample_rate,
                'codebook_size': self.codebook_size,
                'num_quantizers': self.num_quantizers
            }
        }, path)
        print(f"Model saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str, device: str = 'cuda'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        print(f"Model loaded from {path}")
        return model
