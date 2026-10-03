"""Soft penalty on the current chord's avoid notes while sampling (docs/experiments/HARMONY_BIAS.md).

The runtime model barely conditions on the chord guide (same output for different
chords in 10-37% of blocks, #1636), and the user still heard "a lot of dissonance"
after candidate selection and carry (2026-10-03). This subtracts ``strength`` from
the logits of note_on tokens whose pitch class is an avoid note of the block's
chord (standard chord-scale practice), so they become rarer, not impossible:
chromatic passing and approach notes stay reachable.

Avoid pitch classes, relative to the root:
  maj7  b9 b3 11 b13 b7     (Ionian / Lydian: #11 allowed)
  7     11 7                (Mixolydian + altered tensions b9 #9 #11 b13 allowed)
  m7    b9 3 #11 b13 7      (Dorian)
  m7b5  b9 3 5 13 7         (Locrian #2)
  dim   b9 3 5 b7           (whole-half)
"""
from __future__ import annotations

NOTE_ON_START, NOTE_ON_END = 0, 127
AVOID = {(0, 4, 7, 11): {1, 3, 5, 8, 10}, (0, 4, 7, 10): {5, 11}, (0, 3, 7, 10): {1, 4, 6, 8, 11},
         (0, 3, 6, 10): {1, 4, 7, 9, 11}, (0, 3, 6, 9): {1, 4, 7, 10}}


def avoid_pitch_classes(chord: str) -> set[int]:
    from inference.app.fallback import parse_chord

    root, iv = parse_chord(chord)
    return {(root + k) % 12 for k in AVOID.get(tuple(iv), set())}


class HarmonyBias:
    """logits_processor for generate_once: ``(logits, sequence) -> logits``."""

    def __init__(self, chord: str, strength: float = 2.0, repeat: float = 0.0) -> None:
        import torch

        self.chord, self.strength, self.repeat = chord, strength, repeat
        pcs = avoid_pitch_classes(chord)
        self.bias = None
        self._mask = torch.tensor([float(p % 12 in pcs) for p in range(NOTE_ON_END + 1)])
        self.steps = 0

    def __call__(self, logits, sequence=None):
        self.steps += 1
        if self.bias is None or self.bias.device != logits.device or self.bias.dtype != logits.dtype:
            self.bias = (-self.strength * self._mask).to(device=logits.device, dtype=logits.dtype)
        out = logits.clone()
        out[..., NOTE_ON_START:NOTE_ON_END + 1] = out[..., NOTE_ON_START:NOTE_ON_END + 1] + self.bias
        if self.repeat and sequence is not None and len(sequence):
            # Penalise striking the most recent note_on pitch again (#1658): with the avoid penalty the
            # model fell back on repeating it (same-note share .109 -> .175; real bebop .055).
            ons = (sequence <= NOTE_ON_END).nonzero()
            if len(ons):
                last = int(sequence[int(ons[-1])])
                out[..., last] = out[..., last] - self.repeat
        return out
