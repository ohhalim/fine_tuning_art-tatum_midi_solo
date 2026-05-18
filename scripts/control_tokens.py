"""Stage A control-token helpers.

The base MIDI processor only knows note/time/velocity tokens plus END/PAD.
Stage A control_v1 keeps those event tokens intact and prepends a small,
explicit prompt before the conditioning MIDI events.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer"))
sys.path.insert(0, str(SCRIPT_DIR / "music_transformer" / "third_party"))

from utilities.constants import (
    CONTROL_TOKEN_NAMES,
    TOKEN_BAR,
    TOKEN_COND_SEP,
    TOKEN_END,
    TOKEN_ROLE_LEAD,
    TOKEN_TEMPO_DANCE,
    TOKEN_TEMPO_FAST,
    TOKEN_TEMPO_MEDIUM,
    TOKEN_TEMPO_SLOW,
)


SEQUENCE_FORMAT_CONTROL_V1 = "control_v1"
SEQUENCE_FORMAT_LEGACY_SEP = "legacy_sep"
SEQUENCE_FORMAT_CHOICES = (SEQUENCE_FORMAT_CONTROL_V1, SEQUENCE_FORMAT_LEGACY_SEP)


def tempo_control_token(tempo_bpm: float | int | None) -> int:
    tempo = float(tempo_bpm or 120.0)
    if tempo < 90.0:
        return TOKEN_TEMPO_SLOW
    if tempo < 120.0:
        return TOKEN_TEMPO_MEDIUM
    if tempo < 150.0:
        return TOKEN_TEMPO_DANCE
    return TOKEN_TEMPO_FAST


def role_control_token(role: str | None) -> int:
    # Stage A currently trains only the lead/solo role. Keep unknown roles
    # mapped to ROLE_LEAD until the dataset has multiple labeled roles.
    return TOKEN_ROLE_LEAD


def control_prefix_tokens(role: str | None = "lead", tempo_bpm: float | int | None = 120.0) -> list[int]:
    return [role_control_token(role), tempo_control_token(tempo_bpm), TOKEN_BAR]


def token_names(tokens: Sequence[int]) -> list[str]:
    return [CONTROL_TOKEN_NAMES.get(int(token), str(int(token))) for token in tokens]


def build_control_sequence(
    conditioning_tokens: Sequence[int],
    target_tokens: Sequence[int],
    role: str | None = "lead",
    tempo_bpm: float | int | None = 120.0,
) -> list[int]:
    return (
        control_prefix_tokens(role=role, tempo_bpm=tempo_bpm)
        + [int(token) for token in conditioning_tokens]
        + [TOKEN_COND_SEP]
        + [int(token) for token in target_tokens]
        + [TOKEN_END]
    )


def build_control_primer(
    conditioning_tokens: Sequence[int],
    role: str | None = "lead",
    tempo_bpm: float | int | None = 120.0,
    append_sep_token: bool = True,
    primer_max_tokens: int = 64,
) -> list[int]:
    prefix = control_prefix_tokens(role=role, tempo_bpm=tempo_bpm)
    suffix = [TOKEN_COND_SEP] if append_sep_token else []
    available_conditioning = max(0, int(primer_max_tokens) - len(prefix) - len(suffix))
    body = [int(token) for token in conditioning_tokens]
    if primer_max_tokens > 0:
        body = body[-available_conditioning:] if available_conditioning > 0 else []
    return prefix + body + suffix


def build_legacy_sequence(conditioning_tokens: Sequence[int], target_tokens: Sequence[int]) -> list[int]:
    return [int(token) for token in conditioning_tokens] + [TOKEN_END] + [int(token) for token in target_tokens] + [TOKEN_END]


def build_legacy_primer(
    conditioning_tokens: Sequence[int],
    append_sep_token: bool = True,
    primer_max_tokens: int = 64,
) -> list[int]:
    tokens = [int(token) for token in conditioning_tokens]
    if append_sep_token:
        tokens = tokens + [TOKEN_END]
    if primer_max_tokens > 0:
        tokens = tokens[-int(primer_max_tokens) :]
    return tokens
