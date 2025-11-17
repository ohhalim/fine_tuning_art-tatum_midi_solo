"""
Music Transformer for PersonalJazz

Autoregressive transformer for generating music token sequences.
Based on GPT-2 architecture with modifications for music generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better long-context modeling"""

    def __init__(self, dim: int, max_seq_len: int = 8192):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Precompute for efficiency
        t = torch.arange(max_seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor, seq_len: int):
        return (
            self.cos_cached[:, :, :seq_len, ...],
            self.sin_cached[:, :, :seq_len, ...]
        )


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Apply rotary position embedding to query and key"""
    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot


def rotate_half(x: torch.Tensor):
    """Rotate half the hidden dims of the input"""
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with KV-cache for fast generation"""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        # Q, K, V projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ):
        B, T, C = x.shape

        # Project to Q, K, V
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply rotary position embedding
        if rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Use KV-cache if available
        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.o_proj(out)

        new_cache = (k, v) if use_cache else None

        return out, new_cache


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GELU activation"""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """Single transformer block with pre-norm"""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple] = None,
        use_cache: bool = False,
        rope_cos_sin: Optional[Tuple] = None
    ):
        # Pre-norm attention
        residual = x
        x = self.ln1(x)
        attn_out, new_cache = self.attn(x, mask, kv_cache, use_cache, rope_cos_sin)
        x = residual + self.dropout(attn_out)

        # Pre-norm feedforward
        residual = x
        x = self.ln2(x)
        ff_out = self.ff(x)
        x = residual + self.dropout(ff_out)

        return x, new_cache


class MusicTransformer(nn.Module):
    """
    Autoregressive Transformer for music token generation

    Args:
        vocab_size: Size of token vocabulary (e.g., 2048 for codebook)
        d_model: Hidden dimension (default: 1024)
        num_layers: Number of transformer blocks (default: 24)
        num_heads: Number of attention heads (default: 16)
        d_ff: Feed-forward dimension (default: 4096)
        dropout: Dropout rate (default: 0.1)
        max_seq_len: Maximum sequence length (default: 8192)
        style_dim: Dimension of style conditioning (default: 512)
    """

    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 1024,
        num_layers: int = 24,
        num_heads: int = 16,
        d_ff: int = 4096,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        style_dim: int = 512
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Rotary position embedding
        self.rope = RotaryPositionalEmbedding(d_model // num_heads, max_seq_len)

        # Style conditioning
        self.style_proj = nn.Linear(style_dim, d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # Output
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)

        # Initialize weights
        self.apply(self._init_weights)

        print(f"MusicTransformer initialized: {self.count_parameters() / 1e6:.1f}M parameters")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        tokens: torch.Tensor,
        style_emb: Optional[torch.Tensor] = None,
        kv_caches: Optional[list] = None,
        use_cache: bool = False
    ):
        """
        Forward pass

        Args:
            tokens: Token IDs (B, T)
            style_emb: Style embedding (B, style_dim)
            kv_caches: List of KV caches for each layer
            use_cache: Whether to return KV caches

        Returns:
            logits: (B, T, vocab_size)
            new_caches: List of new KV caches (if use_cache=True)
        """
        B, T = tokens.shape

        # Token embeddings
        x = self.token_emb(tokens)  # (B, T, d_model)

        # Add style conditioning
        if style_emb is not None:
            style = self.style_proj(style_emb).unsqueeze(1)  # (B, 1, d_model)
            x = x + style

        x = self.dropout(x)

        # Rotary position embeddings
        cos, sin = self.rope(x, T)
        rope_cos_sin = (cos, sin)

        # Causal mask
        mask = torch.tril(torch.ones(T, T, device=tokens.device)).view(1, 1, T, T)

        # Transformer blocks
        new_caches = []
        for i, block in enumerate(self.blocks):
            kv_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, mask, kv_cache, use_cache, rope_cos_sin)
            if use_cache:
                new_caches.append(new_cache)

        # Output
        x = self.ln_f(x)
        logits = self.head(x)

        if use_cache:
            return logits, new_caches
        return logits

    @torch.no_grad()
    def generate(
        self,
        start_tokens: torch.Tensor,
        max_new_tokens: int,
        style_emb: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = 0.9
    ):
        """
        Autoregressive generation with sampling

        Args:
            start_tokens: Initial tokens (B, T)
            max_new_tokens: Number of tokens to generate
            style_emb: Style embedding
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Nucleus sampling

        Returns:
            generated: Generated tokens (B, T + max_new_tokens)
        """
        tokens = start_tokens
        kv_caches = None

        for _ in range(max_new_tokens):
            # Forward pass (only new token if using cache)
            if kv_caches is not None:
                logits, kv_caches = self(tokens[:, -1:], style_emb, kv_caches, use_cache=True)
            else:
                logits, kv_caches = self(tokens, style_emb, None, use_cache=True)

            # Get logits for last token
            logits = logits[:, -1, :] / temperature

            # Top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append to sequence
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
