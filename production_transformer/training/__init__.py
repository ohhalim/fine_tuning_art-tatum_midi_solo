"""
Training module for Music Transformer

Industry-standard training with:
- HuggingFace Trainer
- Weights & Biases integration
- Best practices for production
"""

from .train import main as train_main

__all__ = ['train_main']
