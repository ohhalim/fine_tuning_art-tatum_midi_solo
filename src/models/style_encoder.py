"""
Brad Mehldau Style Encoder Transformer

이 모듈은 코드 진행과 멜로디를 입력받아 Brad Mehldau의 스타일 특징을 추출합니다.
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig
from typing import Optional, Tuple


class BradMehldauStyleEncoder(nn.Module):
    """
    Brad Mehldau 스타일 특징 추출 Transformer

    Architecture:
        - BERT-like Transformer (8 layers, 12 heads)
        - Style projection layer
        - Chord & melody conditioning

    Args:
        vocab_size (int): MIDI token vocabulary size
        hidden_size (int): Transformer hidden dimension
        num_layers (int): Number of transformer layers
        num_heads (int): Number of attention heads
        style_dim (int): Output style embedding dimension
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        hidden_size: int = 768,
        num_layers: int = 8,
        num_heads: int = 12,
        style_dim: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()

        # BERT-like Transformer configuration
        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=hidden_size * 4,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
            max_position_embeddings=512
        )

        self.transformer = BertModel(config)

        # Style projection head
        self.style_proj = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, style_dim)
        )

        # Chord-specific projection (optional, for better chord understanding)
        self.chord_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU()
        )

        self.hidden_size = hidden_size
        self.style_dim = style_dim

    def forward(
        self,
        chord_tokens: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        melody_tokens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            chord_tokens: [batch_size, seq_len] - Tokenized chord progression
            attention_mask: [batch_size, seq_len] - Attention mask
            melody_tokens: [batch_size, seq_len] - Optional melody tokens for training

        Returns:
            style_emb: [batch_size, style_dim] - Brad Mehldau style embedding
            chord_features: [batch_size, seq_len, 256] - Chord-level features
        """
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = (chord_tokens != 0).long()

        # Transformer encoding
        outputs = self.transformer(
            input_ids=chord_tokens,
            attention_mask=attention_mask,
            return_dict=True
        )

        # Extract features
        sequence_output = outputs.last_hidden_state  # [batch, seq_len, hidden]
        pooled_output = outputs.pooler_output  # [batch, hidden]

        # Style embedding (global feature)
        style_emb = self.style_proj(pooled_output)

        # Chord features (local features for cross-attention)
        chord_features = self.chord_proj(sequence_output)

        return style_emb, chord_features

    def freeze_layers(self, num_layers: int):
        """
        Freeze first N transformer layers for fine-tuning

        Args:
            num_layers: Number of layers to freeze from the bottom
        """
        for i in range(num_layers):
            for param in self.transformer.encoder.layer[i].parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True


class ChordAwareAttention(nn.Module):
    """
    Cross-attention mechanism for chord-aware generation

    DiT features attend to chord features extracted by StyleEncoder
    """

    def __init__(self, dim: int = 384, num_heads: int = 6, dropout: float = 0.1):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        dit_features: torch.Tensor,
        chord_features: torch.Tensor,
        chord_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Cross-attention: DiT features attend to chord features

        Args:
            dit_features: [batch, n_patches, dim] - DiT intermediate features
            chord_features: [batch, seq_len, dim] - Chord features from StyleEncoder
            chord_mask: [batch, seq_len] - Attention mask for chords

        Returns:
            attended_features: [batch, n_patches, dim]
        """
        # Cross-attention
        attended, _ = self.attention(
            query=dit_features,
            key=chord_features,
            value=chord_features,
            key_padding_mask=chord_mask if chord_mask is not None else None
        )

        # Residual connection & normalization
        output = self.norm(dit_features + self.dropout(attended))

        return output


class StyleAdapter(nn.Module):
    """
    Adapter to convert style embedding to DiT conditioning

    Transforms style_dim → dit_hidden_dim
    """

    def __init__(self, style_dim: int = 256, dit_hidden_dim: int = 384, dropout: float = 0.1):
        super().__init__()

        self.adapter = nn.Sequential(
            nn.Linear(style_dim, dit_hidden_dim),
            nn.LayerNorm(dit_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dit_hidden_dim, dit_hidden_dim)
        )

    def forward(self, style_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            style_emb: [batch, style_dim]
        Returns:
            adapted: [batch, dit_hidden_dim]
        """
        return self.adapter(style_emb)


if __name__ == "__main__":
    # Test the model
    print("Testing BradMehldauStyleEncoder...")

    # Create model
    model = BradMehldauStyleEncoder(
        vocab_size=2000,
        hidden_size=768,
        num_layers=8,
        num_heads=12,
        style_dim=256
    )

    # Create dummy input
    batch_size = 4
    seq_len = 32
    chord_tokens = torch.randint(0, 2000, (batch_size, seq_len))

    # Forward pass
    style_emb, chord_features = model(chord_tokens)

    print(f"Input shape: {chord_tokens.shape}")
    print(f"Style embedding shape: {style_emb.shape}")
    print(f"Chord features shape: {chord_features.shape}")

    # Test adapter
    print("\nTesting StyleAdapter...")
    adapter = StyleAdapter(style_dim=256, dit_hidden_dim=384)
    adapted = adapter(style_emb)
    print(f"Adapted style shape: {adapted.shape}")

    # Test chord-aware attention
    print("\nTesting ChordAwareAttention...")
    attention = ChordAwareAttention(dim=256, num_heads=8)
    dit_features = torch.randn(batch_size, 64, 256)  # 64 patches
    attended = attention(dit_features, chord_features)
    print(f"Attended features shape: {attended.shape}")

    print("\n✅ All tests passed!")
