"""
Flow Matching for Music Generation

Conditional Flow Matching (CFM) - 최신 생성 모델
Diffusion보다 10배 빠르고, 더 안정적인 학습

References:
- Lipman et al., "Flow Matching for Generative Modeling" (ICLR 2023)
- Tong et al., "Improving and Generalizing Flow-Based Generative Models" (ICML 2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching

    핵심 아이디어:
    - x_0: 노이즈 (source)
    - x_1: 실제 데이터 (target)
    - x_t = (1-t) * x_0 + t * x_1  (linear interpolation)
    - v_t = x_1 - x_0  (velocity field)

    Model은 v_t를 예측하도록 학습
    """

    def __init__(
        self,
        model: nn.Module,
        sigma_min: float = 1e-4
    ):
        super().__init__()
        self.model = model
        self.sigma_min = sigma_min

    def get_target(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor
    ) -> torch.Tensor:
        """
        Target velocity field 계산

        Args:
            x_0: Source (noise)
            x_1: Target (data)

        Returns:
            v: Velocity field (x_1 - x_0)
        """
        return x_1 - x_0

    def interpolate(
        self,
        x_0: torch.Tensor,
        x_1: torch.Tensor,
        t: torch.Tensor
    ) -> torch.Tensor:
        """
        Linear interpolation between x_0 and x_1

        x_t = (1-t) * x_0 + t * x_1

        Args:
            x_0: Source [B, ...]
            x_1: Target [B, ...]
            t: Time [B, 1, ...]

        Returns:
            x_t: Interpolated [B, ...]
        """
        return (1 - t) * x_0 + t * x_1

    def forward(
        self,
        x_1: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Training forward pass

        Args:
            x_1: Target data [B, T, D]
            condition: Conditioning (e.g., style embedding) [B, C]

        Returns:
            loss: Flow matching loss
        """
        B, T, D = x_1.shape

        # Sample time uniformly
        t = torch.rand(B, 1, 1, device=x_1.device)

        # Sample source (noise)
        x_0 = torch.randn_like(x_1) * self.sigma_min

        # Interpolate
        x_t = self.interpolate(x_0, x_1, t)

        # Target velocity
        v_target = self.get_target(x_0, x_1)

        # Predict velocity
        t_broadcast = t.expand(B, T, 1)  # [B, T, 1]
        v_pred = self.model(x_t, t_broadcast, condition)

        # Loss: MSE between predicted and target velocity
        loss = F.mse_loss(v_pred, v_target)

        return loss

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        condition: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        device: str = 'cuda',
        method: str = 'euler'
    ) -> torch.Tensor:
        """
        Sample from the flow model (ODE solver)

        Args:
            shape: Output shape (B, T, D)
            condition: Conditioning [B, C]
            num_steps: Number of integration steps
            device: Device
            method: Integration method ('euler' or 'rk4')

        Returns:
            x_1: Generated samples [B, T, D]
        """
        B, T, D = shape

        # Start from noise
        x = torch.randn(shape, device=device) * self.sigma_min

        # Time steps
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full((B, T, 1), i / num_steps, device=device)

            if method == 'euler':
                # Euler integration
                v = self.model(x, t, condition)
                x = x + v * dt

            elif method == 'rk4':
                # Runge-Kutta 4th order (more accurate, slower)
                k1 = self.model(x, t, condition)

                k2 = self.model(
                    x + 0.5 * dt * k1,
                    t + 0.5 * dt,
                    condition
                )

                k3 = self.model(
                    x + 0.5 * dt * k2,
                    t + 0.5 * dt,
                    condition
                )

                k4 = self.model(
                    x + dt * k3,
                    t + dt,
                    condition
                )

                x = x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

            else:
                raise ValueError(f"Unknown method: {method}")

        return x


class FlowMatchingVelocityModel(nn.Module):
    """
    Velocity field model for Flow Matching

    Simple MLP-based velocity predictor for testing
    (In production, use Transformer)
    """

    def __init__(
        self,
        input_dim: int = 128,
        hidden_dim: int = 512,
        condition_dim: int = 512,
        num_layers: int = 4
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.condition_dim = condition_dim

        # Time embedding (sinusoidal)
        self.time_dim = 128
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Condition embedding
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Main network
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, input_dim))

        self.net = nn.Sequential(*layers)

    def time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal time embedding

        Args:
            t: Time [B, T, 1]

        Returns:
            emb: Time embedding [B, T, time_dim]
        """
        half_dim = self.time_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t * emb.unsqueeze(0).unsqueeze(0)  # [B, T, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, T, time_dim]
        return emb

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Predict velocity field

        Args:
            x: Input [B, T, D]
            t: Time [B, T, 1]
            condition: Conditioning [B, C]

        Returns:
            v: Velocity [B, T, D]
        """
        B, T, D = x.shape

        # Time embedding
        t_emb = self.time_embedding(t)  # [B, T, time_dim]
        t_emb = self.time_mlp(t_emb)  # [B, T, hidden_dim]

        # Condition embedding
        if condition is not None:
            c_emb = self.condition_mlp(condition)  # [B, hidden_dim]
            c_emb = c_emb.unsqueeze(1).expand(B, T, -1)  # [B, T, hidden_dim]
        else:
            c_emb = 0

        # Add embeddings to input
        h = self.net[0](x) + t_emb + c_emb  # [B, T, hidden_dim]

        # Pass through network
        for layer in self.net[1:]:
            h = layer(h)

        return h


# ============================================================================
# Test & Demo
# ============================================================================

def test_flow_matching():
    """Flow Matching 테스트"""
    print("=" * 80)
    print("Flow Matching 테스트")
    print("=" * 80)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")

    # Create model
    print("\n[1] 모델 생성")
    velocity_model = FlowMatchingVelocityModel(
        input_dim=128,
        hidden_dim=512,
        condition_dim=512
    ).to(device)

    flow = FlowMatching(velocity_model)

    params = sum(p.numel() for p in velocity_model.parameters())
    print(f"   Parameters: {params:,}")

    # Test training
    print("\n[2] Training 테스트")
    batch_size = 4
    seq_len = 100
    dim = 128

    x_1 = torch.randn(batch_size, seq_len, dim, device=device)
    condition = torch.randn(batch_size, 512, device=device)

    loss = flow(x_1, condition)
    print(f"   Loss: {loss.item():.4f}")

    # Test sampling
    print("\n[3] Sampling 테스트")
    samples = flow.sample(
        shape=(2, 50, 128),
        condition=condition[:2],
        num_steps=10,
        device=device,
        method='euler'
    )
    print(f"   Generated shape: {samples.shape}")
    print(f"   Generated range: [{samples.min():.3f}, {samples.max():.3f}]")

    # Compare Euler vs RK4
    print("\n[4] Euler vs RK4 비교")
    import time

    start = time.time()
    samples_euler = flow.sample((1, 100, 128), num_steps=50, method='euler', device=device)
    time_euler = time.time() - start

    start = time.time()
    samples_rk4 = flow.sample((1, 100, 128), num_steps=50, method='rk4', device=device)
    time_rk4 = time.time() - start

    print(f"   Euler: {time_euler:.3f}s")
    print(f"   RK4: {time_rk4:.3f}s")
    print(f"   Speedup: {time_rk4 / time_euler:.2f}x")

    print("\n" + "=" * 80)
    print("✅ Flow Matching 테스트 완료!")
    print("=" * 80)


if __name__ == "__main__":
    test_flow_matching()
