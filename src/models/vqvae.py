"""
VQ-VAE for MIDI Piano Roll Encoding

Piano roll → Latent code → Piano roll

Reference: SCG (Rule-Guided Music Generation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class VectorQuantizer(nn.Module):
    """
    Vector Quantization layer

    Args:
        num_embeddings: Size of the codebook
        embedding_dim: Dimension of each embedding vector
        commitment_cost: Weight for commitment loss
    """

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64, commitment_cost: float = 0.25):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            z: [batch, channels, height, width] - Encoder output

        Returns:
            quantized: [batch, channels, height, width] - Quantized latent
            loss: Scalar - VQ loss (commitment + codebook)
            perplexity: Scalar - Codebook usage metric
        """
        # Reshape to [batch * height * width, channels]
        z_flattened = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z_flattened.view(-1, self.embedding_dim)

        # Calculate distances to codebook vectors
        distances = (
            torch.sum(z_flattened ** 2, dim=1, keepdim=True)
            + torch.sum(self.embeddings.weight ** 2, dim=1)
            - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
        )

        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize
        quantized = torch.matmul(encodings, self.embeddings.weight)
        quantized = quantized.view(z.shape[0], z.shape[2], z.shape[3], self.embedding_dim)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = z + (quantized - z).detach()

        # Perplexity (codebook usage)
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class ResidualBlock(nn.Module):
    """Residual block for VQ-VAE"""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)

        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        x = self.conv1(x)
        x = self.norm1(x)
        x = F.relu(x, inplace=True)

        x = self.conv2(x)
        x = self.norm2(x)

        return F.relu(x + residual, inplace=True)


class Encoder(nn.Module):
    """VQ-VAE Encoder: Piano roll → Latent"""

    def __init__(self, in_channels: int = 2, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, hidden_dim // 4, kernel_size=3, padding=1)

        self.down1 = nn.Sequential(
            ResidualBlock(hidden_dim // 4, hidden_dim // 2),
            nn.Conv2d(hidden_dim // 2, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        )

        self.down2 = nn.Sequential(
            ResidualBlock(hidden_dim // 2, hidden_dim),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=4, stride=2, padding=1)
        )

        self.residual = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )

        self.conv_out = nn.Conv2d(hidden_dim, latent_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, 2, 128, time] - Piano roll (onset + sustain)
        Returns:
            z: [batch, latent_dim, 32, time//4]
        """
        x = self.conv_in(x)
        x = self.down1(x)
        x = self.down2(x)
        x = self.residual(x)
        z = self.conv_out(x)
        return z


class Decoder(nn.Module):
    """VQ-VAE Decoder: Latent → Piano roll"""

    def __init__(self, latent_dim: int = 64, hidden_dim: int = 256, out_channels: int = 2):
        super().__init__()

        self.conv_in = nn.Conv2d(latent_dim, hidden_dim, kernel_size=1)

        self.residual = nn.Sequential(
            ResidualBlock(hidden_dim, hidden_dim),
            ResidualBlock(hidden_dim, hidden_dim)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1),
            ResidualBlock(hidden_dim // 2, hidden_dim // 2)
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1),
            ResidualBlock(hidden_dim // 4, hidden_dim // 4)
        )

        self.conv_out = nn.Conv2d(hidden_dim // 4, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: [batch, latent_dim, 32, time//4]
        Returns:
            x: [batch, 2, 128, time] - Reconstructed piano roll
        """
        x = self.conv_in(z)
        x = self.residual(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.conv_out(x)
        return x


class VQVAE(nn.Module):
    """
    Complete VQ-VAE model for piano roll encoding

    Args:
        in_channels: Input channels (2 for onset + sustain)
        hidden_dim: Hidden dimension
        num_embeddings: Codebook size
        latent_dim: Latent embedding dimension
        commitment_cost: VQ commitment loss weight
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_dim: int = 256,
        num_embeddings: int = 512,
        latent_dim: int = 64,
        commitment_cost: float = 0.25
    ):
        super().__init__()

        self.encoder = Encoder(in_channels, hidden_dim, latent_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, in_channels)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch, 2, 128, time] - Piano roll

        Returns:
            recon: [batch, 2, 128, time] - Reconstructed piano roll
            vq_loss: Scalar - VQ loss
            perplexity: Scalar - Codebook usage
        """
        z = self.encoder(x)
        z_q, vq_loss, perplexity = self.vq(z)
        recon = self.decoder(z_q)

        return recon, vq_loss, perplexity

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode piano roll to latent"""
        z = self.encoder(x)
        z_q, _, _ = self.vq(z)
        return z_q

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to piano roll"""
        return self.decoder(z)


if __name__ == "__main__":
    print("Testing VQ-VAE...")

    # Create model
    model = VQVAE(
        in_channels=2,
        hidden_dim=256,
        num_embeddings=512,
        latent_dim=64
    )

    # Create dummy piano roll
    batch_size = 4
    piano_roll = torch.randn(batch_size, 2, 128, 256)  # 256 time steps

    # Forward pass
    recon, vq_loss, perplexity = model(piano_roll)

    print(f"Input shape: {piano_roll.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Perplexity: {perplexity.item():.2f}")

    # Test encode/decode
    latent = model.encode(piano_roll)
    decoded = model.decode(latent)
    print(f"\nLatent shape: {latent.shape}")
    print(f"Decoded shape: {decoded.shape}")

    print("\n✅ VQ-VAE test passed!")
