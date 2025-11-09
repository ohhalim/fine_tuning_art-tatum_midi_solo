"""
Production Music Transformer Models

HuggingFace-compatible implementations for music generation
"""

from .music_transformer import (
    MusicTransformerConfig,
    MusicTransformerModel,
    MusicTransformerForGeneration
)
from .qlora import apply_qlora_to_model, QLoRAConfig

__all__ = [
    'MusicTransformerConfig',
    'MusicTransformerModel',
    'MusicTransformerForGeneration',
    'apply_qlora_to_model',
    'QLoRAConfig'
]
