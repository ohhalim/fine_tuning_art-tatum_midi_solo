"""
Style Encoder for PersonalJazz

Encodes text prompts and audio into a shared embedding space
Based on CLIP/CoCa architecture for contrastive learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TextEncoder(nn.Module):
    """
    Text encoder for style prompts

    Uses a simple transformer encoder to encode text descriptions
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_seq_len: int = 128
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.ln = nn.LayerNorm(d_model)

        # Output projection to style space
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Encode text

        Args:
            token_ids: Token IDs (B, T)
            attention_mask: Attention mask (B, T)

        Returns:
            text_emb: Text embedding (B, d_model)
        """
        B, T = token_ids.shape

        # Token + position embeddings
        positions = torch.arange(T, device=token_ids.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(token_ids) + self.pos_emb(positions)

        # Transformer encoding
        # Convert attention_mask to transformer format
        if attention_mask is not None:
            # attention_mask: 1 for valid, 0 for padding
            # transformer expects: False for valid, True for padding
            mask = (attention_mask == 0)
        else:
            mask = None

        x = self.transformer(x, src_key_padding_mask=mask)

        # Pool: use [CLS] token (first token) or mean pooling
        # Here we use mean pooling over valid tokens
        if attention_mask is not None:
            mask_expanded = attention_mask.unsqueeze(-1).float()
            x = (x * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            x = x.mean(dim=1)

        x = self.ln(x)
        x = self.proj(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        return x


class AudioEncoder(nn.Module):
    """
    Audio encoder for music

    Encodes audio into the same embedding space as text
    """

    def __init__(
        self,
        in_channels: int = 2,
        d_model: int = 512,
        num_layers: int = 6
    ):
        super().__init__()

        # Convolutional frontend
        self.conv1 = nn.Conv1d(in_channels, 128, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv1d(256, d_model, kernel_size=3, stride=2, padding=1)

        self.norm = nn.LayerNorm(d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, audio: torch.Tensor):
        """
        Encode audio

        Args:
            audio: Raw audio (B, C, T)

        Returns:
            audio_emb: Audio embedding (B, d_model)
        """
        # Convolutional feature extraction
        x = F.gelu(self.conv1(audio))
        x = F.gelu(self.conv2(x))
        x = F.gelu(self.conv3(x))

        # Transpose for transformer
        x = x.transpose(1, 2)  # (B, T', d_model)

        # Layer norm
        x = self.norm(x)

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)  # (B, d_model)

        x = self.proj(x)

        # L2 normalize
        x = F.normalize(x, dim=-1)

        return x


class StyleEncoder(nn.Module):
    """
    Joint text-audio style encoder

    Learns a shared embedding space for text descriptions and audio
    via contrastive learning (CLIP-style)
    """

    def __init__(
        self,
        vocab_size: int = 50000,
        d_model: int = 512,
        num_text_layers: int = 6,
        num_audio_layers: int = 6
    ):
        super().__init__()

        self.text_encoder = TextEncoder(vocab_size, d_model, num_text_layers)
        self.audio_encoder = AudioEncoder(2, d_model, num_audio_layers)

        # Learnable temperature for contrastive loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        print(f"StyleEncoder: {d_model}d embedding space")

    def encode_text(self, token_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """Encode text prompt"""
        return self.text_encoder(token_ids, attention_mask)

    def encode_audio(self, audio: torch.Tensor):
        """Encode audio"""
        return self.audio_encoder(audio)

    def forward(self, text_tokens: torch.Tensor, audio: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        """
        Forward pass for contrastive learning

        Args:
            text_tokens: Text token IDs (B, T_text)
            audio: Audio waveforms (B, C, T_audio)
            attention_mask: Text attention mask (B, T_text)

        Returns:
            text_emb: Text embeddings (B, d_model)
            audio_emb: Audio embeddings (B, d_model)
            logit_scale: Temperature parameter
        """
        text_emb = self.encode_text(text_tokens, attention_mask)
        audio_emb = self.encode_audio(audio)

        return text_emb, audio_emb, self.logit_scale.exp()


def contrastive_loss(text_emb: torch.Tensor, audio_emb: torch.Tensor, logit_scale: torch.Tensor):
    """
    Contrastive loss (CLIP-style)

    Maximize similarity between matching text-audio pairs,
    minimize similarity between non-matching pairs

    Args:
        text_emb: Text embeddings (B, D)
        audio_emb: Audio embeddings (B, D)
        logit_scale: Temperature scaling

    Returns:
        loss: Contrastive loss
    """
    B = text_emb.shape[0]

    # Compute similarity matrix
    # (B, D) @ (D, B) = (B, B)
    logits = logit_scale * text_emb @ audio_emb.t()

    # Labels: diagonal elements are positive pairs
    labels = torch.arange(B, device=text_emb.device)

    # Symmetric cross-entropy loss
    loss_text = F.cross_entropy(logits, labels)
    loss_audio = F.cross_entropy(logits.t(), labels)

    loss = (loss_text + loss_audio) / 2

    return loss


# For compatibility
import numpy as np
