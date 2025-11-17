"""
Audio Codec for PersonalJazz

Neural audio codec using Residual Vector Quantization (RVQ)
Based on SoundStream/EnCodec architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class ResidualVectorQuantizer(nn.Module):
    """
    Residual Vector Quantization (RVQ)

    Quantizes audio features into discrete tokens across multiple levels
    """

    def __init__(
        self,
        dim: int = 512,
        codebook_size: int = 2048,
        num_quantizers: int = 8,
        commitment_weight: float = 0.25
    ):
        super().__init__()

        self.dim = dim
        self.codebook_size = codebook_size
        self.num_quantizers = num_quantizers
        self.commitment_weight = commitment_weight

        # Codebooks for each quantization level
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, dim)
            for _ in range(num_quantizers)
        ])

        # Initialize codebooks
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1 / codebook_size, 1 / codebook_size)

    def forward(self, x: torch.Tensor):
        """
        Quantize input

        Args:
            x: Input features (B, T, dim)

        Returns:
            quantized: Quantized features (B, T, dim)
            indices: Codebook indices (B, T, num_quantizers)
            loss: Commitment and codebook loss
        """
        B, T, D = x.shape

        residual = x
        quantized = torch.zeros_like(x)
        indices = torch.zeros(B, T, self.num_quantizers, dtype=torch.long, device=x.device)
        total_loss = 0.0

        for i, codebook in enumerate(self.codebooks):
            # Find nearest code
            distances = (
                residual.pow(2).sum(dim=-1, keepdim=True)
                - 2 * residual @ codebook.weight.t()
                + codebook.weight.pow(2).sum(dim=1)
            )

            min_indices = distances.argmin(dim=-1)  # (B, T)
            indices[:, :, i] = min_indices

            # Get quantized values
            quantized_level = codebook(min_indices)  # (B, T, dim)
            quantized = quantized + quantized_level

            # Commitment loss
            commit_loss = F.mse_loss(residual.detach(), quantized_level)
            codebook_loss = F.mse_loss(residual, quantized_level.detach())

            total_loss = total_loss + commit_loss * self.commitment_weight + codebook_loss

            # Update residual
            residual = residual - quantized_level.detach()

        # Straight-through estimator
        quantized = x + (quantized - x).detach()

        return quantized, indices, total_loss / self.num_quantizers

    def decode(self, indices: torch.Tensor):
        """
        Decode indices to features

        Args:
            indices: Codebook indices (B, T, num_quantizers)

        Returns:
            quantized: Decoded features (B, T, dim)
        """
        B, T, Q = indices.shape
        quantized = torch.zeros(B, T, self.dim, device=indices.device)

        for i, codebook in enumerate(self.codebooks):
            quantized = quantized + codebook(indices[:, :, i])

        return quantized


class ConvBlock(nn.Module):
    """Convolutional block with residual connection"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 7, stride: int = 1):
        super().__init__()

        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=kernel_size // 2
        )
        self.norm = nn.GroupNorm(min(32, out_channels // 4), out_channels)
        self.activation = nn.SiLU()

        # Residual projection
        self.residual = nn.Conv1d(in_channels, out_channels, 1, stride=stride) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = self.residual(x)
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x + residual


class Encoder(nn.Module):
    """Audio Encoder: Raw audio → Latent features"""

    def __init__(
        self,
        in_channels: int = 2,  # Stereo
        hidden_dim: int = 512,
        latent_dim: int = 512,
        num_layers: int = 5
    ):
        super().__init__()

        # Initial conv
        self.input_conv = nn.Conv1d(in_channels, hidden_dim, kernel_size=7, padding=3)

        # Downsampling blocks (reduce by 2^num_layers)
        layers = []
        current_dim = hidden_dim
        for i in range(num_layers):
            out_dim = min(hidden_dim * (2 ** (i + 1)), 1024)
            layers.append(ConvBlock(current_dim, out_dim, kernel_size=7, stride=2))
            current_dim = out_dim

        self.encoder_blocks = nn.Sequential(*layers)

        # Output projection
        self.output_proj = nn.Conv1d(current_dim, latent_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor):
        """
        Encode audio

        Args:
            x: Raw audio (B, C, T) where C=2 for stereo

        Returns:
            latent: Latent features (B, latent_dim, T // downsampling_factor)
        """
        x = self.input_conv(x)
        x = self.encoder_blocks(x)
        x = self.output_proj(x)
        return x


class Decoder(nn.Module):
    """Audio Decoder: Latent features → Raw audio"""

    def __init__(
        self,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        out_channels: int = 2,  # Stereo
        num_layers: int = 5
    ):
        super().__init__()

        # Input projection
        self.input_proj = nn.Conv1d(latent_dim, hidden_dim * (2 ** (num_layers - 1)), kernel_size=3, padding=1)

        # Upsampling blocks
        layers = []
        current_dim = hidden_dim * (2 ** (num_layers - 1))
        for i in range(num_layers - 1, -1, -1):
            out_dim = hidden_dim * (2 ** i) if i > 0 else hidden_dim

            layers.append(nn.ConvTranspose1d(current_dim, out_dim, kernel_size=8, stride=2, padding=3))
            layers.append(nn.GroupNorm(min(32, out_dim // 4), out_dim))
            layers.append(nn.SiLU())

            current_dim = out_dim

        self.decoder_blocks = nn.Sequential(*layers)

        # Output conv
        self.output_conv = nn.Conv1d(hidden_dim, out_channels, kernel_size=7, padding=3)
        self.output_activation = nn.Tanh()

    def forward(self, x: torch.Tensor):
        """
        Decode latent features

        Args:
            x: Latent features (B, latent_dim, T)

        Returns:
            audio: Reconstructed audio (B, C, T * upsampling_factor)
        """
        x = self.input_proj(x)
        x = self.decoder_blocks(x)
        x = self.output_conv(x)
        x = self.output_activation(x)
        return x


class AudioCodec(nn.Module):
    """
    Complete Neural Audio Codec

    Compresses stereo audio (48kHz) into discrete tokens
    Compression: 48000 Hz → ~75 Hz (640x compression)
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        in_channels: int = 2,
        latent_dim: int = 512,
        codebook_size: int = 2048,
        num_quantizers: int = 8,
        downsample_factor: int = 640  # 48000 / 75 ≈ 640
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.in_channels = in_channels
        self.downsample_factor = downsample_factor

        # Calculate num_layers from downsample_factor
        # downsample_factor = 2^num_layers
        self.num_layers = int(np.log2(downsample_factor))

        # Encoder, Decoder, Quantizer
        self.encoder = Encoder(in_channels, 512, latent_dim, self.num_layers)
        self.decoder = Decoder(latent_dim, 512, in_channels, self.num_layers)
        self.quantizer = ResidualVectorQuantizer(latent_dim, codebook_size, num_quantizers)

        print(f"AudioCodec: {sample_rate}Hz → {sample_rate // downsample_factor}Hz tokens ({downsample_factor}x compression)")
        print(f"   Codebook: {codebook_size} codes × {num_quantizers} levels = {codebook_size ** num_quantizers:.2e} states")

    def encode(self, audio: torch.Tensor):
        """
        Encode audio to discrete tokens

        Args:
            audio: Raw audio (B, C, T) where T is in samples

        Returns:
            tokens: Discrete token indices (B, T // downsample_factor, num_quantizers)
            loss: Quantization loss
        """
        # Encode to latent
        latent = self.encoder(audio)  # (B, latent_dim, T')

        # Transpose for quantizer
        latent = latent.transpose(1, 2)  # (B, T', latent_dim)

        # Quantize
        quantized, tokens, loss = self.quantizer(latent)

        return tokens, loss

    def decode(self, tokens: torch.Tensor):
        """
        Decode tokens to audio

        Args:
            tokens: Token indices (B, T', num_quantizers)

        Returns:
            audio: Reconstructed audio (B, C, T)
        """
        # Decode tokens to latent
        quantized = self.quantizer.decode(tokens)  # (B, T', latent_dim)

        # Transpose for decoder
        quantized = quantized.transpose(1, 2)  # (B, latent_dim, T')

        # Decode to audio
        audio = self.decoder(quantized)  # (B, C, T)

        return audio

    def forward(self, audio: torch.Tensor):
        """
        Full encode-decode cycle

        Args:
            audio: Raw audio (B, C, T)

        Returns:
            reconstructed: Reconstructed audio (B, C, T)
            tokens: Discrete tokens (B, T', num_quantizers)
            loss: Quantization loss
        """
        tokens, loss = self.encode(audio)
        reconstructed = self.decode(tokens)

        return reconstructed, tokens, loss
