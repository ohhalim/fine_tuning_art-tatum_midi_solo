"""
PersonalJazz: Real-time Personalized Jazz Improvisation Model

A state-of-the-art model for generating personalized jazz improvisations
in real-time, suitable for live DJ performance.
"""

from .personaljazz import PersonalJazz
from .transformer import MusicTransformer
from .codec import AudioCodec
from .style_encoder import StyleEncoder

__version__ = "1.0.0"
__all__ = ["PersonalJazz", "MusicTransformer", "AudioCodec", "StyleEncoder"]
