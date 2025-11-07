"""
DiT (Diffusion Transformer)

Transformer-based diffusion model for latent space

Reference: Scalable Diffusion Models with Transformers
"""

import torch
import torch.nn as nn
import math
from typing import Optional
from einops import rearrange


class TimestepEmbedding(nn.Module):
    """
    Sinusoidal timestep embedding (like in Transformers)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch] - Timesteps
        Returns:
            emb: [batch, dim] - Timestep embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class PatchEmbed(nn.Module):
    """
    Convert latent to patches (like ViT)

    Args:
        latent_size: (height, width) of latent
        patch_size: Size of each patch
        in_channels: Latent channels
        embed_dim: Embedding dimension
    """

    def __init__(self, latent_size=(32, 64), patch_size=2, in_channels=64, embed_dim=384):
        super().__init__()

        self.latent_size = latent_size
        self.patch_size = patch_size
        self.num_patches = (latent_size[0] // patch_size) * (latent_size[1] // patch_size)

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, channels, height, width]
        Returns:
            patches: [batch, num_patches, embed_dim]
        """
        x = self.proj(x)  # [batch, embed_dim, H', W']
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class DiTBlock(nn.Module):
    """
    Transformer block with adaptive layer norm (adaLN)

    Conditioning is applied via adaptive normalization
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()

        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_size),
            nn.Dropout(dropout)
        )

        # Adaptive layer norm (conditioning)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, num_patches, hidden_size] - Input
            c: [batch, hidden_size] - Conditioning (timestep + style)
        Returns:
            out: [batch, num_patches, hidden_size]
        """
        # AdaLN modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=-1)

        # Multi-head self-attention with adaLN
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + gate_msa.unsqueeze(1) * attn_out

        # MLP with adaLN
        x_norm = self.norm2(x)
        x_norm = x_norm * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(x_norm)

        return x


class DiT(nn.Module):
    """
    Diffusion Transformer (DiT)

    Args:
        latent_size: (height, width) of VQ-VAE latent
        patch_size: Patch size for patchification
        in_channels: Latent channels (VQ-VAE output)
        hidden_size: Transformer hidden dimension
        depth: Number of transformer blocks
        num_heads: Number of attention heads
        mlp_ratio: MLP hidden dim ratio
    """

    def __init__(
        self,
        latent_size=(32, 64),
        patch_size=2,
        in_channels=64,
        hidden_size=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()

        self.latent_size = latent_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        # Patch embedding
        self.patch_embed = PatchEmbed(latent_size, patch_size, in_channels, hidden_size)
        num_patches = self.patch_embed.num_patches

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size))

        # Timestep embedding
        self.time_embed = nn.Sequential(
            TimestepEmbedding(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])

        # Output layers
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_size ** 2 * in_channels, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Positional embedding
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Linear layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        self.apply(_basic_init)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert patches back to latent

        Args:
            x: [batch, num_patches, patch_size^2 * channels]
        Returns:
            latent: [batch, channels, height, width]
        """
        p = self.patch_size
        h = self.latent_size[0] // p
        w = self.latent_size[1] // p

        x = rearrange(
            x,
            'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
            h=h, w=w, p1=p, p2=p
        )
        return x

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            x: [batch, channels, height, width] - Noisy latent
            t: [batch] - Timesteps
            context: [batch, hidden_size] - Style conditioning (optional)

        Returns:
            pred: [batch, channels, height, width] - Predicted noise
        """
        # Patch embedding
        x = self.patch_embed(x)  # [batch, num_patches, hidden_size]
        x = x + self.pos_embed

        # Timestep embedding
        t_emb = self.time_embed(t)  # [batch, hidden_size]

        # Add style conditioning if provided
        if context is not None:
            c = t_emb + context  # Combine timestep and style
        else:
            c = t_emb

        # Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # Output
        x = self.final_norm(x)
        x = self.final_linear(x)
        x = self.unpatchify(x)

        return x


if __name__ == "__main__":
    print("Testing DiT...")

    # Create model
    model = DiT(
        latent_size=(32, 64),
        patch_size=2,
        in_channels=64,
        hidden_size=384,
        depth=12,
        num_heads=6
    )

    # Create dummy input
    batch_size = 4
    noisy_latent = torch.randn(batch_size, 64, 32, 64)
    timesteps = torch.randint(0, 1000, (batch_size,))
    style_context = torch.randn(batch_size, 384)

    # Forward pass
    pred_noise = model(noisy_latent, timesteps, style_context)

    print(f"Input shape: {noisy_latent.shape}")
    print(f"Timesteps shape: {timesteps.shape}")
    print(f"Style context shape: {style_context.shape}")
    print(f"Output shape: {pred_noise.shape}")

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {num_params:,}")

    print("\n✅ DiT test passed!")
