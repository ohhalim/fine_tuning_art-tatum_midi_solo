"""
SCG + Transformer Hybrid Model

Combines:
- VQ-VAE for latent encoding
- DiT for diffusion generation
- Style Encoder Transformer for Brad Mehldau conditioning
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
import math

from .vqvae import VQVAE
from .dit import DiT
from .style_encoder import BradMehldauStyleEncoder, StyleAdapter, ChordAwareAttention


class GaussianDiffusion(nn.Module):
    """
    Gaussian Diffusion Process (DDPM/DDIM)

    Implements forward diffusion (adding noise) and reverse diffusion (denoising)
    """

    def __init__(self, timesteps: int = 1000, beta_schedule: str = "linear"):
        super().__init__()

        self.timesteps = timesteps

        # Define beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(0.0001, 0.02, timesteps)
        elif beta_schedule == "cosine":
            s = 0.008
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward diffusion: Add noise to x_start at timestep t

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """
        Predict x_0 from x_t and predicted noise
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1)

        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise


class SCGTransformerHybrid(nn.Module):
    """
    Main Hybrid Model: SCG + Transformer

    Architecture:
        Input: Chord progression + (optional) melody
        ↓
        [StyleEncoder] → style_embedding
        ↓
        [DiT + Diffusion] with style conditioning
        ↓
        [VQ-VAE Decoder] → MIDI piano roll
    """

    def __init__(
        self,
        # VQ-VAE config
        vqvae_config: dict = None,
        # DiT config
        dit_config: dict = None,
        # Style Encoder config
        style_encoder_config: dict = None,
        # Diffusion config
        diffusion_timesteps: int = 1000,
        beta_schedule: str = "cosine"
    ):
        super().__init__()

        # Default configs
        if vqvae_config is None:
            vqvae_config = {
                "in_channels": 2,
                "hidden_dim": 256,
                "num_embeddings": 512,
                "latent_dim": 64
            }

        if dit_config is None:
            dit_config = {
                "latent_size": (32, 64),
                "patch_size": 2,
                "in_channels": 64,
                "hidden_size": 384,
                "depth": 12,
                "num_heads": 6
            }

        if style_encoder_config is None:
            style_encoder_config = {
                "vocab_size": 2000,
                "hidden_size": 768,
                "num_layers": 8,
                "num_heads": 12,
                "style_dim": 256
            }

        # Initialize models
        self.vqvae = VQVAE(**vqvae_config)
        self.dit = DiT(**dit_config)
        self.style_encoder = BradMehldauStyleEncoder(**style_encoder_config)

        # Style adapter (256 → 384)
        self.style_adapter = StyleAdapter(
            style_dim=style_encoder_config["style_dim"],
            dit_hidden_dim=dit_config["hidden_size"]
        )

        # Diffusion
        self.diffusion = GaussianDiffusion(
            timesteps=diffusion_timesteps,
            beta_schedule=beta_schedule
        )

        self.timesteps = diffusion_timesteps

    def forward(
        self,
        piano_roll: torch.Tensor,
        chord_tokens: torch.Tensor,
        chord_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training forward pass

        Args:
            piano_roll: [batch, 2, 128, time] - Target piano roll
            chord_tokens: [batch, seq_len] - Chord progression tokens
            chord_mask: [batch, seq_len] - Attention mask

        Returns:
            loss: Total loss
            vq_loss: VQ-VAE loss
            diffusion_loss: Diffusion loss
        """
        batch_size = piano_roll.shape[0]

        # 1. Encode piano roll to latent
        z_0 = self.vqvae.encode(piano_roll)  # [batch, 64, 32, time//4]

        # 2. Extract Brad Mehldau style
        style_emb, chord_features = self.style_encoder(chord_tokens, chord_mask)

        # 3. Adapt style for DiT conditioning
        style_cond = self.style_adapter(style_emb)  # [batch, 384]

        # 4. Sample random timestep
        t = torch.randint(0, self.timesteps, (batch_size,), device=piano_roll.device)

        # 5. Add noise (forward diffusion)
        noise = torch.randn_like(z_0)
        z_t = self.diffusion.q_sample(z_0, t, noise)

        # 6. Predict noise with DiT
        predicted_noise = self.dit(z_t, t, context=style_cond)

        # 7. Diffusion loss
        diffusion_loss = torch.nn.functional.mse_loss(predicted_noise, noise)

        # 8. VQ-VAE reconstruction loss (for end-to-end training)
        recon, vq_loss, _ = self.vqvae(piano_roll)
        recon_loss = torch.nn.functional.mse_loss(recon, piano_roll)

        # Total loss
        total_loss = diffusion_loss + 0.1 * vq_loss + 0.1 * recon_loss

        return total_loss, vq_loss, diffusion_loss

    @torch.no_grad()
    def generate(
        self,
        chord_progression: List[str],
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        temperature: float = 1.0,
        eta: float = 0.0
    ) -> torch.Tensor:
        """
        Generate Brad Mehldau style MIDI from chord progression

        Args:
            chord_progression: List of chord names ["Cmaj7", "Dm7", "G7", ...]
            num_steps: Number of DDIM steps (50 = fast, 1000 = slow but better)
            guidance_scale: Classifier-free guidance strength (7.5 = default)
            temperature: Sampling temperature (0.8-1.2)
            eta: DDIM eta parameter (0 = deterministic, 1 = stochastic)

        Returns:
            piano_roll: [1, 2, 128, time] - Generated MIDI
        """
        device = next(self.parameters()).device

        # Tokenize chords (placeholder - implement proper tokenizer)
        chord_tokens = self._tokenize_chords(chord_progression).to(device)

        # Extract style
        style_emb, _ = self.style_encoder(chord_tokens)
        style_cond = self.style_adapter(style_emb)

        # DDIM sampling
        z_t = torch.randn(1, 64, 32, 64, device=device) * temperature

        # Time steps for DDIM
        timesteps = torch.linspace(self.timesteps - 1, 0, num_steps, dtype=torch.long, device=device)

        for i, t in enumerate(timesteps):
            t_batch = t.unsqueeze(0)

            # Predict noise
            predicted_noise = self.dit(z_t, t_batch, context=style_cond)

            # DDIM update
            z_t = self._ddim_step(z_t, predicted_noise, t, timesteps, i, eta)

        # Decode to piano roll
        piano_roll = self.vqvae.decode(z_t)

        return piano_roll

    def _ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        timesteps: torch.Tensor,
        step: int,
        eta: float
    ) -> torch.Tensor:
        """
        Single DDIM denoising step
        """
        alpha_t = self.diffusion.alphas_cumprod[t]

        if step < len(timesteps) - 1:
            alpha_t_prev = self.diffusion.alphas_cumprod[timesteps[step + 1]]
        else:
            alpha_t_prev = torch.tensor(1.0, device=x_t.device)

        # Predict x_0
        x_0_pred = self.diffusion.predict_start_from_noise(x_t, t.unsqueeze(0), noise_pred)

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1 - alpha_t_prev - eta ** 2 * (1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise_pred

        # Random noise
        noise = torch.randn_like(x_t) if eta > 0 else 0

        # DDIM update
        x_t_prev = torch.sqrt(alpha_t_prev) * x_0_pred + dir_xt + eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev)) * noise

        return x_t_prev

    def _tokenize_chords(self, chord_progression: List[str]) -> torch.Tensor:
        """
        Tokenize chord progression (placeholder)

        TODO: Implement proper chord tokenizer
        """
        # Simple placeholder - return random tokens
        return torch.randint(0, 2000, (1, len(chord_progression)))


if __name__ == "__main__":
    print("Testing SCG + Transformer Hybrid Model...")

    # Create model
    model = SCGTransformerHybrid()

    # Training test
    print("\n=== Training Test ===")
    batch_size = 2
    piano_roll = torch.randn(batch_size, 2, 128, 256)
    chord_tokens = torch.randint(0, 2000, (batch_size, 8))

    total_loss, vq_loss, diffusion_loss = model(piano_roll, chord_tokens)

    print(f"Total loss: {total_loss.item():.4f}")
    print(f"VQ loss: {vq_loss.item():.4f}")
    print(f"Diffusion loss: {diffusion_loss.item():.4f}")

    # Generation test
    print("\n=== Generation Test ===")
    chord_progression = ["Cmaj7", "Dm7", "G7", "Cmaj7"]
    generated = model.generate(chord_progression, num_steps=10)  # Fast test

    print(f"Generated piano roll shape: {generated.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== Model Info ===")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\n✅ Hybrid model test passed!")
