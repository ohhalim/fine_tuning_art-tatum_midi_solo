"""Synthetic Cmaj7/Cm7 contrast pairs for the 32-step condition test (docs/experiments/ARIA_COND_32STEP.md).

Fixed generator, no real song. Every family has one prefix built only from C and G (tones Cmaj7
and Cm7 share), so the prefix tokens of a P (Cmaj7) and Q (Cm7) pair are identical and only the
chord condition differs. The first note after the prefix is the first one that tells the chords
apart: the third (E / Eb) or the seventh (B / Bb). The chord is fixed for the whole sequence, so
the chord-change boundary limit of the timing contract does not enter. Pure Python.
"""
from __future__ import annotations

VEL = 80
DIFF = {"3rd": {"P": 4, "Q": 3}, "7th": {"P": 11, "Q": 10}}
# family: prefix pitches (C/G only), inter-onset seconds (one value or one per note), first
# discriminating tone, octave of that tone
FAMILIES = {
    "F0": ([60, 67, 72, 67], 0.25, "3rd", 4),
    "F1": ([48, 55, 60, 67, 72], 0.25, "7th", 4),
    "F2": ([72, 67, 60], 0.5, "3rd", 5),
    "F3": ([60, 60, 67, 67], [0.25, 0.25, 0.5, 0.25], "7th", 4),
    "F4": ([55, 60, 67], 0.375, "3rd", 4),
    "F5": ([67, 72, 79, 72, 67], 0.2, "7th", 4),
    "F6": ([48, 60, 72], 0.5, "3rd", 5),
    "F7": ([60, 55, 48, 55, 60, 67], 0.25, "7th", 4),
    "E8": ([67, 60, 67, 72], [0.5, 0.25, 0.25, 0.5], "3rd", 4),
    "E9": ([72, 79, 72, 67, 60], 0.3, "7th", 4),
    "E10": ([55, 48, 55, 60], 0.4, "3rd", 5),
    "E11": ([60, 72, 60, 67], [0.25, 0.5, 0.25, 0.25], "7th", 4),
}
TRAIN = ["F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"]
EVAL = ["E8", "E9", "E10", "E11"]
CHORD = {"P": ("C", "maj7"), "Q": ("C", "m7")}


def example(family: str, quality: str):
    """Notes (pitch, onset s, end s, velocity) and the two candidate pitches of the first discriminating note."""
    pitches, ioi, kind, octave = FAMILIES[family]
    iois = ioi if isinstance(ioi, list) else [ioi] * len(pitches)
    notes, t = [], 0.0
    for p, d in zip(pitches, iois):
        notes.append((p, round(t, 3), round(t + 0.8 * d, 3), VEL))
        t += d
    base = 12 * (octave + 1)
    target = base + DIFF[kind][quality]
    other_kind = "7th" if kind == "3rd" else "3rd"
    for p in (target, 67, 12 * 5 + DIFF[other_kind][quality], 72):      # chord tones after the target
        notes.append((p, round(t, 3), round(t + 0.2, 3), VEL))
        t += 0.25
    return notes, {"P": base + DIFF[kind]["P"], "Q": base + DIFF[kind]["Q"]}


def signature(family: str):
    """Transposition-free pattern signature: intervals and inter-onset times of the prefix."""
    pitches, ioi, kind, _ = FAMILIES[family]
    iois = ioi if isinstance(ioi, list) else [ioi] * len(pitches)
    return tuple(b - a for a, b in zip(pitches, pitches[1:])), tuple(iois), kind


def first_discriminating_index(toks, cand) -> int:
    """Index of the first piano token whose pitch is one of the two candidates."""
    return next(i for i, x in enumerate(toks) if isinstance(x, tuple) and x[0] == "piano" and x[1] in cand.values())


def target_mask(toks, first_idx: int, arm: str) -> list[bool]:
    """Loss mask over shifted targets: mask[k] is for predicting toks[k + 1].
    A: every target. B: targets from the first discriminating note on (its pitch, onset, duration,
    later notes and <E>); prefix targets dropped. C: as B without the <E> target. Inputs are never
    changed; only which targets count."""
    out = []
    for k in range(len(toks) - 1):
        j = k + 1
        if arm == "A":
            out.append(True)
        elif arm == "B":
            out.append(j >= first_idx)
        elif arm == "C":
            out.append(j >= first_idx and toks[j] != "<E>")
        else:
            raise ValueError(arm)
    return out
