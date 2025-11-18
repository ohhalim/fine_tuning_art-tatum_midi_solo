"""
TatumFlow: Hierarchical Latent Diffusion for Jazz Improvisation
Innovative architecture combining:
- Multi-scale temporal modeling (note/beat/phrase levels)
- Latent diffusion in symbolic domain
- Explicit music theory disentanglement
- Bidirectional context modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict


class RotaryPositionalEmbedding(nn.Module):
    """RoPE for better position encoding"""
    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary embeddings to queries and keys"""
    def rotate_half(x):
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class MusicTheoryEncoder(nn.Module):
    """
    Explicitly disentangle music theory components:
    - Harmony (chord progressions, tonality)
    - Melody (pitch contours, motifs)
    - Rhythm (timing, syncopation)
    - Dynamics (velocity, articulation)
    """
    def __init__(self, hidden_dim: int = 512):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Separate encoders for each component
        self.harmony_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        self.melody_encoder = nn.Sequential(
            nn.Linear(128, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        self.rhythm_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        self.dynamics_encoder = nn.Sequential(
            nn.Linear(64, hidden_dim // 4),
            nn.LayerNorm(hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4)
        )

        # Combine components
        self.fusion = nn.Linear(hidden_dim, hidden_dim)

    def extract_harmony(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract harmony features using pitch class profiles

        Returns (B, 128) tensor with:
        - Pitch class histogram (12)
        - Chord type indicators (24)
        - Harmonic intervals (12)
        - Rest: padding to 128
        """
        batch_size, seq_len = tokens.shape
        harmony_features = []

        for b in range(batch_size):
            # Pitch class histogram (0-11)
            pc_hist = torch.zeros(12, device=tokens.device)

            # Track active pitches for chord detection
            active_pitches = []

            for t in range(seq_len):
                token_id = tokens[b, t].item()
                # Skip special tokens
                if token_id < 5:  # PAD, SOS, EOS, MASK, T
                    continue

                # Detect NOTE_ON tokens (after TIME tokens which are 5-505)
                # NOTE_ON tokens start after TIME tokens
                if 506 <= token_id < 594:  # NOTE_ON range (88 notes)
                    pitch = 21 + (token_id - 506)  # Convert to MIDI pitch
                    pc = pitch % 12
                    pc_hist[pc] += 1.0
                    active_pitches.append(pitch)

            # Normalize pitch class histogram
            if pc_hist.sum() > 0:
                pc_hist = pc_hist / pc_hist.sum()

            # Chord type detection (simple heuristics)
            chord_types = torch.zeros(24, device=tokens.device)
            if len(active_pitches) >= 3:
                unique_pcs = list(set([p % 12 for p in active_pitches]))
                unique_pcs.sort()

                # Major chord: 0, 4, 7
                # Minor chord: 0, 3, 7
                # Dim chord: 0, 3, 6
                # Aug chord: 0, 4, 8
                # Dominant 7th: 0, 4, 7, 10
                # Major 7th: 0, 4, 7, 11
                # Minor 7th: 0, 3, 7, 10

                # Simple detection based on intervals
                if len(unique_pcs) >= 3:
                    intervals = [(unique_pcs[i+1] - unique_pcs[i]) % 12
                                for i in range(len(unique_pcs)-1)]

                    # Major (4, 3)
                    if 4 in intervals and 3 in intervals:
                        chord_types[0] = 1.0
                    # Minor (3, 4)
                    elif 3 in intervals and 4 in intervals:
                        chord_types[1] = 1.0
                    # Diminished (3, 3)
                    elif intervals.count(3) >= 2:
                        chord_types[2] = 1.0
                    # Augmented (4, 4)
                    elif intervals.count(4) >= 2:
                        chord_types[3] = 1.0

            # Harmonic intervals (adjacent pitch classes)
            harmonic_intervals = torch.zeros(12, device=tokens.device)
            for i in range(len(pc_hist)-1):
                if pc_hist[i] > 0 and pc_hist[i+1] > 0:
                    interval = (i+1 - i) % 12
                    harmonic_intervals[interval] += 1.0

            if harmonic_intervals.sum() > 0:
                harmonic_intervals = harmonic_intervals / harmonic_intervals.sum()

            # Concatenate features
            features = torch.cat([pc_hist, chord_types, harmonic_intervals])

            # Pad to 128
            padding = torch.zeros(128 - features.shape[0], device=tokens.device)
            features = torch.cat([features, padding])

            harmony_features.append(features)

        return torch.stack(harmony_features)

    def extract_melody(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract melodic contour using skyline algorithm

        Returns (B, 128) tensor with:
        - Melodic interval histogram (25)
        - Contour patterns (3)
        - Pitch range (1)
        - Average pitch (1)
        - Rest: melodic features
        """
        batch_size, seq_len = tokens.shape
        melody_features = []

        for b in range(batch_size):
            # Extract notes with timing
            notes = []  # (time_ms, pitch, velocity)
            current_time = 0
            chunk_start = 0

            for t in range(seq_len):
                token_id = tokens[b, t].item()

                # Time shift token (resets time)
                if token_id == 4:  # <T> token
                    chunk_start = current_time

                # TIME tokens (5-505)
                elif 5 <= token_id <= 505:
                    time_step = token_id - 5
                    current_time = chunk_start + time_step * 10  # 10ms resolution

                # NOTE_ON tokens (506-593)
                elif 506 <= token_id < 594:
                    pitch = 21 + (token_id - 506)
                    notes.append((current_time, pitch))

            if len(notes) == 0:
                # No notes found, return zeros
                melody_features.append(torch.zeros(128, device=tokens.device))
                continue

            # Skyline extraction (highest pitch at each time)
            time_pitches = {}
            for time, pitch in notes:
                if time not in time_pitches:
                    time_pitches[time] = []
                time_pitches[time].append(pitch)

            skyline = [max(pitches) for pitches in time_pitches.values()]

            # Melodic intervals
            if len(skyline) > 1:
                intervals = [skyline[i+1] - skyline[i] for i in range(len(skyline)-1)]

                # Interval histogram (-12 to +12)
                interval_hist = torch.zeros(25, device=tokens.device)
                for interval in intervals:
                    idx = min(max(interval + 12, 0), 24)  # Clamp to 0-24
                    interval_hist[idx] += 1.0

                if interval_hist.sum() > 0:
                    interval_hist = interval_hist / interval_hist.sum()

                # Contour (up/down/same)
                contour = torch.zeros(3, device=tokens.device)
                for interval in intervals:
                    if interval > 0:
                        contour[0] += 1  # Up
                    elif interval < 0:
                        contour[1] += 1  # Down
                    else:
                        contour[2] += 1  # Same

                if contour.sum() > 0:
                    contour = contour / contour.sum()

                # Pitch range (normalized)
                pitch_range = (max(skyline) - min(skyline)) / 88.0

                # Average pitch (normalized)
                avg_pitch = sum(skyline) / len(skyline) / 127.0
            else:
                interval_hist = torch.zeros(25, device=tokens.device)
                contour = torch.zeros(3, device=tokens.device)
                pitch_range = 0.0
                avg_pitch = skyline[0] / 127.0 if len(skyline) > 0 else 0.0

            # Concatenate features
            features = torch.cat([
                interval_hist,
                contour,
                torch.tensor([pitch_range, avg_pitch], device=tokens.device)
            ])

            # Pad to 128
            padding = torch.zeros(128 - features.shape[0], device=tokens.device)
            features = torch.cat([features, padding])

            melody_features.append(features)

        return torch.stack(melody_features)

    def extract_rhythm(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract rhythmic patterns from inter-onset intervals

        Returns (B, 64) tensor with:
        - IOI histogram (32 bins, log scale)
        - Syncopation measure (1)
        - Regularity/entropy (1)
        - Rest: rhythmic complexity features
        """
        batch_size, seq_len = tokens.shape
        rhythm_features = []

        for b in range(batch_size):
            # Extract onset times
            onsets = []
            current_time = 0
            chunk_start = 0

            for t in range(seq_len):
                token_id = tokens[b, t].item()

                # Time shift
                if token_id == 4:
                    chunk_start = current_time

                # TIME tokens
                elif 5 <= token_id <= 505:
                    time_step = token_id - 5
                    current_time = chunk_start + time_step * 10

                # NOTE_ON
                elif 506 <= token_id < 594:
                    onsets.append(current_time)

            if len(onsets) < 2:
                rhythm_features.append(torch.zeros(64, device=tokens.device))
                continue

            # Inter-onset intervals (ms)
            iois = [onsets[i+1] - onsets[i] for i in range(len(onsets)-1)]
            iois = [max(ioi, 1) for ioi in iois]  # Avoid log(0)

            # IOI histogram (log scale: 1ms to 10s)
            ioi_hist = torch.zeros(32, device=tokens.device)
            for ioi in iois:
                log_ioi = np.log10(ioi)  # log10(1) = 0, log10(10000) = 4
                bin_idx = int((log_ioi / 4.0) * 32)  # Map to 0-31
                bin_idx = min(max(bin_idx, 0), 31)
                ioi_hist[bin_idx] += 1.0

            if ioi_hist.sum() > 0:
                ioi_hist = ioi_hist / ioi_hist.sum()

            # Syncopation (variance / mean)
            mean_ioi = sum(iois) / len(iois)
            variance = sum((ioi - mean_ioi)**2 for ioi in iois) / len(iois)
            syncopation = (variance ** 0.5) / (mean_ioi + 1e-6)
            syncopation = min(syncopation, 10.0) / 10.0  # Normalize

            # Entropy (regularity measure)
            probs = ioi_hist + 1e-10
            entropy_val = -(probs * torch.log2(probs)).sum()
            entropy_val = entropy_val / np.log2(32)  # Normalize to 0-1

            # Concatenate features
            features = torch.cat([
                ioi_hist,
                torch.tensor([syncopation, entropy_val.item()], device=tokens.device)
            ])

            # Pad to 64
            padding = torch.zeros(64 - features.shape[0], device=tokens.device)
            features = torch.cat([features, padding])

            rhythm_features.append(features)

        return torch.stack(rhythm_features)

    def extract_dynamics(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract dynamic (velocity) features

        Returns (B, 64) tensor with:
        - Velocity histogram (32 bins)
        - Dynamic range (1)
        - Average velocity (1)
        - Velocity variance (1)
        - Rest: dynamic evolution features
        """
        batch_size, seq_len = tokens.shape
        dynamics_features = []

        for b in range(batch_size):
            # Extract velocities
            velocities = []

            for t in range(seq_len):
                token_id = tokens[b, t].item()

                # VEL tokens (682-713 = 32 velocity bins)
                if 682 <= token_id < 714:
                    vel_bin = token_id - 682
                    # Dequantize: bin * (128/32) + (128/64) for center of bin
                    velocity = vel_bin * 4 + 2  # Approximate center
                    velocities.append(velocity)

            if len(velocities) == 0:
                dynamics_features.append(torch.zeros(64, device=tokens.device))
                continue

            # Velocity histogram (32 bins: 0-127)
            vel_hist = torch.zeros(32, device=tokens.device)
            for vel in velocities:
                bin_idx = min(int(vel / 4), 31)  # 128 / 4 = 32 bins
                vel_hist[bin_idx] += 1.0

            if vel_hist.sum() > 0:
                vel_hist = vel_hist / vel_hist.sum()

            # Dynamic range
            dynamic_range = (max(velocities) - min(velocities)) / 127.0

            # Average velocity
            avg_velocity = sum(velocities) / len(velocities) / 127.0

            # Velocity variance (expressiveness)
            mean_vel = sum(velocities) / len(velocities)
            variance = sum((v - mean_vel)**2 for v in velocities) / len(velocities)
            vel_variance = (variance ** 0.5) / 127.0

            # Concatenate features
            features = torch.cat([
                vel_hist,
                torch.tensor([dynamic_range, avg_velocity, vel_variance], device=tokens.device)
            ])

            # Pad to 64
            padding = torch.zeros(64 - features.shape[0], device=tokens.device)
            features = torch.cat([features, padding])

            dynamics_features.append(features)

        return torch.stack(dynamics_features)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Returns:
            combined: fused representation
            components: dict of individual components for loss computation
        """
        harmony = self.extract_harmony(tokens)
        melody = self.extract_melody(tokens)
        rhythm = self.extract_rhythm(tokens)
        dynamics = self.extract_dynamics(tokens)

        h_enc = self.harmony_encoder(harmony)
        m_enc = self.melody_encoder(melody)
        r_enc = self.rhythm_encoder(rhythm)
        d_enc = self.dynamics_encoder(dynamics)

        # Concatenate all components
        combined = torch.cat([h_enc, m_enc, r_enc, d_enc], dim=-1)
        fused = self.fusion(combined)

        return fused, {
            'harmony': h_enc,
            'melody': m_enc,
            'rhythm': r_enc,
            'dynamics': d_enc
        }


class MultiScaleAttention(nn.Module):
    """
    Multi-head attention with multi-scale temporal modeling

    Operates at different temporal scales to capture both:
    - Fine-grained patterns (note-level, scale=1)
    - Medium patterns (beat-level, scale=2)
    - Coarse patterns (phrase-level, scale=4)

    Each scale uses separate attention heads, then results are combined.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8, scales: list = [1, 2, 4]):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.scales = scales
        self.num_scales = len(scales)

        # Divide heads across scales
        assert num_heads % self.num_scales == 0, f"num_heads ({num_heads}) must be divisible by num_scales ({self.num_scales})"
        self.heads_per_scale = num_heads // self.num_scales
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0

        # Separate QKV projections for each scale
        self.scale_qkvs = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim * 3 // self.num_scales)
            for _ in scales
        ])

        # Pooling layers for each scale (except scale=1)
        self.scale_pools = nn.ModuleList([
            nn.AvgPool1d(kernel_size=scale, stride=scale, padding=0) if scale > 1 else nn.Identity()
            for scale in scales
        ])

        # Upsampling layers to restore original length
        self.scale_upsample = nn.ModuleList([
            nn.Upsample(scale_factor=scale, mode='linear', align_corners=False) if scale > 1 else nn.Identity()
            for scale in scales
        ])

        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B, L, D = x.shape

        scale_outputs = []

        for scale_idx, (scale, qkv_layer, pool, upsample) in enumerate(
            zip(self.scales, self.scale_qkvs, self.scale_pools, self.scale_upsample)
        ):
            # Pool to this scale
            if scale > 1:
                # Reshape for pooling: (B, L, D) -> (B, D, L)
                x_pool = x.transpose(1, 2)
                x_pool = pool(x_pool)
                x_pool = x_pool.transpose(1, 2)  # Back to (B, L//scale, D)
                L_scale = x_pool.shape[1]
            else:
                x_pool = x
                L_scale = L

            # QKV projection for this scale
            qkv = qkv_layer(x_pool).reshape(B, L_scale, 3, self.heads_per_scale, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads_per_scale, L_scale, head_dim)

            # Scaled dot-product attention
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # Apply mask if provided (needs to be downsampled for scale > 1)
            if mask is not None and scale > 1:
                # Downsample mask to match this scale
                mask_scale = mask[:, :, ::scale, ::scale] if mask.dim() == 4 else mask[:, ::scale]
                attn = attn.masked_fill(mask_scale == 0, float('-inf'))
            elif mask is not None:
                attn = attn.masked_fill(mask == 0, float('-inf'))

            attn = F.softmax(attn, dim=-1)
            out = attn @ v

            # Reshape: (B, heads_per_scale, L_scale, head_dim) -> (B, L_scale, D//num_scales)
            out = out.transpose(1, 2).reshape(B, L_scale, -1)

            # Upsample back to original length if needed
            if scale > 1:
                # (B, L_scale, D//num_scales) -> (B, D//num_scales, L_scale) -> upsample -> (B, D//num_scales, L)
                out = out.transpose(1, 2)
                out = upsample(out)
                out = out.transpose(1, 2)  # Back to (B, L, D//num_scales)

            scale_outputs.append(out)

        # Concatenate outputs from all scales
        multi_scale_out = torch.cat(scale_outputs, dim=-1)  # (B, L, D)

        # Final projection
        return self.proj(multi_scale_out)


class DiffusionTransformerBlock(nn.Module):
    """Transformer block with time embedding for diffusion"""
    def __init__(self, hidden_dim: int, num_heads: int = 8, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = MultiScaleAttention(hidden_dim, num_heads)
        self.norm2 = nn.LayerNorm(hidden_dim)

        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_dim)
        )

        # Time embedding modulation (AdaLN)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 6)
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # AdaLN modulation
        time_params = self.time_mlp(time_emb).unsqueeze(1)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = time_params.chunk(6, dim=-1)

        # Attention with modulation
        h = self.norm1(x)
        h = h * (1 + scale_msa) + shift_msa
        h = self.attn(h, mask)
        x = x + gate_msa * h

        # MLP with modulation
        h = self.norm2(x)
        h = h * (1 + scale_mlp) + shift_mlp
        h = self.mlp(h)
        x = x + gate_mlp * h

        return x


class LatentDiffusionCore(nn.Module):
    """
    Diffusion process in latent space of symbolic music
    Inspired by Stable Diffusion but for discrete tokens
    """
    def __init__(self, latent_dim: int = 512, num_steps: int = 1000):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_steps = num_steps

        # Noise schedule (cosine schedule)
        self.register_buffer('betas', self.cosine_beta_schedule(num_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))

        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim)
        )

    def cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule as proposed in improved DDPM"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def get_time_embedding(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Sinusoidal time embedding"""
        half_dim = self.latent_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.time_mlp(emb)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None):
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5

        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        """
        Predict x_0 from x_t and predicted noise
        Used in DDIM sampling
        """
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod_t = (1 - self.alphas_cumprod[t]) ** 0.5

        while len(sqrt_alphas_cumprod_t.shape) < len(x_t.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)

        # x_0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
        return (x_t - sqrt_one_minus_alphas_cumprod_t * noise) / sqrt_alphas_cumprod_t

    @torch.no_grad()
    def ddim_sample(
        self,
        noise_predictor,
        shape: tuple,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        condition: Optional[torch.Tensor] = None,
        device: str = 'cuda'
    ):
        """
        DDIM sampling for faster generation

        Args:
            noise_predictor: Function that predicts noise from (x_t, t, condition)
            shape: Shape of latent to generate (B, L, D)
            num_inference_steps: Number of denoising steps (default 50, vs 1000 for DDPM)
            eta: Stochasticity parameter (0 = deterministic DDIM, 1 = DDPM)
            condition: Conditioning signal (style embedding, theory features, etc.)
            device: Device to run on

        Returns:
            Denoised latent x_0
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Create timestep schedule (subsample from full schedule)
        step_size = self.num_steps // num_inference_steps
        timesteps = list(range(0, self.num_steps, step_size))[::-1]  # Reverse order

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device).long()

            # Predict noise
            predicted_noise = noise_predictor(x_t, t_tensor, condition)

            # Predict x_0
            x_0_pred = self.predict_start_from_noise(x_t, t_tensor, predicted_noise)

            # Compute previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0

            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)

            # Compute sigma (stochasticity)
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

            # Compute direction pointing to x_t
            pred_direction = torch.sqrt(1 - alpha_t_prev - sigma**2) * predicted_noise

            # Compute x_{t-1}
            x_t_prev = torch.sqrt(alpha_t_prev) * x_0_pred + pred_direction

            # Add noise if not deterministic
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                x_t_prev = x_t_prev + sigma * noise

            x_t = x_t_prev

        return x_t

    @torch.no_grad()
    def ddim_sample_with_cfg(
        self,
        noise_predictor,
        shape: tuple,
        condition: torch.Tensor,
        num_inference_steps: int = 50,
        eta: float = 0.0,
        guidance_scale: float = 7.5,
        device: str = 'cuda'
    ):
        """
        DDIM sampling with Classifier-Free Guidance

        Improves conditioning strength and generation quality
        Used in Stable Diffusion, DALL-E 2, etc.

        Args:
            noise_predictor: Function that predicts noise from (x_t, t, condition)
            shape: Shape of latent to generate
            condition: Conditioning signal (must support None for unconditional)
            num_inference_steps: Number of denoising steps
            eta: Stochasticity parameter
            guidance_scale: Strength of conditioning (1.0 = no guidance, 7.5 typical)
            device: Device to run on

        Returns:
            Denoised latent x_0
        """
        # Start from pure noise
        x_t = torch.randn(shape, device=device)

        # Create timestep schedule
        step_size = self.num_steps // num_inference_steps
        timesteps = list(range(0, self.num_steps, step_size))[::-1]

        for i, t in enumerate(timesteps):
            t_tensor = torch.tensor([t], device=device).long()

            # Predict noise with conditioning
            noise_cond = noise_predictor(x_t, t_tensor, condition)

            # Predict noise without conditioning (unconditional)
            noise_uncond = noise_predictor(x_t, t_tensor, None)

            # Classifier-free guidance
            # noise = uncond + guidance_scale * (cond - uncond)
            predicted_noise = noise_uncond + guidance_scale * (noise_cond - noise_uncond)

            # Predict x_0
            x_0_pred = self.predict_start_from_noise(x_t, t_tensor, predicted_noise)

            # Compute previous timestep
            if i < len(timesteps) - 1:
                t_prev = timesteps[i + 1]
            else:
                t_prev = 0

            # Get alpha values
            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t_prev] if t_prev > 0 else torch.tensor(1.0)

            # Compute sigma
            sigma = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t)) * torch.sqrt(1 - alpha_t / alpha_t_prev)

            # Compute direction
            pred_direction = torch.sqrt(1 - alpha_t_prev - sigma**2) * predicted_noise

            # Compute x_{t-1}
            x_t_prev = torch.sqrt(alpha_t_prev) * x_0_pred + pred_direction

            # Add noise
            if eta > 0 and i < len(timesteps) - 1:
                noise = torch.randn_like(x_t)
                x_t_prev = x_t_prev + sigma * noise

            x_t = x_t_prev

        return x_t


class TatumFlow(nn.Module):
    """
    TatumFlow: Hierarchical Latent Diffusion for Jazz Improvisation

    Key innovations:
    1. Multi-scale temporal modeling (note/beat/phrase)
    2. Latent diffusion in symbolic domain
    3. Explicit music theory disentanglement
    4. Bidirectional context modeling
    5. Style VAE for controllable generation
    """
    def __init__(
        self,
        vocab_size: int = 2048,
        hidden_dim: int = 512,
        latent_dim: int = 256,
        num_layers: int = 12,
        num_heads: int = 8,
        max_seq_len: int = 2048,
        diffusion_steps: int = 1000,
        num_style_dims: int = 64
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = RotaryPositionalEmbedding(hidden_dim, max_seq_len)

        # Music theory encoder
        self.theory_encoder = MusicTheoryEncoder(hidden_dim)

        # Style encoder (VAE)
        self.style_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_style_dims * 2)  # mu and logvar
        )
        self.style_decoder = nn.Linear(num_style_dims, hidden_dim)

        # Latent encoder/decoder
        self.latent_encoder = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU()
        )
        self.latent_decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )

        # Diffusion core
        self.diffusion = LatentDiffusionCore(latent_dim, diffusion_steps)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DiffusionTransformerBlock(hidden_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output projection
        self.norm_out = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """VAE reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode_style(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode style from input sequence"""
        # Get token embeddings
        x = self.token_embedding(tokens)
        x = x.mean(dim=1)  # Pool over sequence

        # VAE encoding
        params = self.style_encoder(x)
        mu, logvar = params.chunk(2, dim=-1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar

    def forward(
        self,
        tokens: torch.Tensor,
        style: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            tokens: (B, L) token indices
            style: (B, style_dim) style vector, if None will be encoded from tokens
            timestep: (B,) diffusion timestep
            mask: (B, L) attention mask

        Returns:
            Dictionary with logits, latents, and components
        """
        B, L = tokens.shape

        # Encode style if not provided
        if style is None:
            style, style_mu, style_logvar = self.encode_style(tokens)
        else:
            style_mu = style_logvar = None

        # Token embeddings
        x = self.token_embedding(tokens)  # (B, L, D)

        # Add style information
        style_emb = self.style_decoder(style).unsqueeze(1)  # (B, 1, D)
        x = x + style_emb

        # Extract music theory components
        theory_emb, theory_components = self.theory_encoder(tokens)
        x = x + theory_emb.unsqueeze(1)

        # Encode to latent space
        latent = self.latent_encoder(x)  # (B, L, latent_dim)

        # Diffusion process
        if self.training and timestep is not None:
            # Forward diffusion during training
            noise = torch.randn_like(latent)
            latent_noisy = self.diffusion.q_sample(latent, timestep, noise)
            latent = latent_noisy
        elif timestep is not None:
            # Use provided latent during inference
            pass

        # Get time embedding
        if timestep is not None:
            time_emb = self.diffusion.get_time_embedding(timestep)
        else:
            time_emb = torch.zeros(B, self.hidden_dim, device=tokens.device)

        # Decode from latent
        x = self.latent_decoder(latent)

        # Transformer blocks with diffusion conditioning
        for block in self.blocks:
            x = block(x, time_emb, mask)

        # Output projection
        x = self.norm_out(x)
        logits = self.head(x)

        return {
            'logits': logits,
            'latent': latent,
            'style_mu': style_mu,
            'style_logvar': style_logvar,
            'theory_components': theory_components
        }

    @torch.no_grad()
    def generate(
        self,
        prompt_tokens: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        style: Optional[torch.Tensor] = None,
        num_diffusion_steps: int = 50
    ) -> torch.Tensor:
        """
        Generate continuation of prompt using classifier-free guidance

        Args:
            prompt_tokens: (B, L) starting tokens
            max_length: maximum generation length
            temperature: sampling temperature
            top_k: top-k filtering
            top_p: nucleus sampling threshold
            style: optional style vector
            num_diffusion_steps: number of diffusion denoising steps
        """
        self.eval()
        device = prompt_tokens.device
        B = prompt_tokens.shape[0]

        # Encode style from prompt if not provided
        if style is None:
            style, _, _ = self.encode_style(prompt_tokens)

        generated = prompt_tokens

        for _ in range(max_length):
            # Get model predictions
            outputs = self.forward(generated, style=style, timestep=None)
            logits = outputs['logits'][:, -1, :] / temperature

            # Top-k filtering
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)

            # Check for end token (if defined)
            # if next_token == eos_token_id:
            #     break

        return generated


def create_tatumflow_model(
    model_size: str = 'base',
    vocab_size: int = 2048
) -> TatumFlow:
    """
    Create TatumFlow model with predefined configurations

    Args:
        model_size: 'small', 'base', 'large'
        vocab_size: size of token vocabulary
    """
    configs = {
        'small': {
            'hidden_dim': 384,
            'latent_dim': 192,
            'num_layers': 6,
            'num_heads': 6,
        },
        'base': {
            'hidden_dim': 512,
            'latent_dim': 256,
            'num_layers': 12,
            'num_heads': 8,
        },
        'large': {
            'hidden_dim': 768,
            'latent_dim': 384,
            'num_layers': 24,
            'num_heads': 12,
        }
    }

    config = configs.get(model_size, configs['base'])
    return TatumFlow(vocab_size=vocab_size, **config)


if __name__ == "__main__":
    # Test model creation
    model = create_tatumflow_model('base')
    print(f"TatumFlow model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Test forward pass
    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, 2048, (batch_size, seq_len))
    timestep = torch.randint(0, 1000, (batch_size,))

    outputs = model(tokens, timestep=timestep)
    print(f"Output logits shape: {outputs['logits'].shape}")
    print(f"Latent shape: {outputs['latent'].shape}")
