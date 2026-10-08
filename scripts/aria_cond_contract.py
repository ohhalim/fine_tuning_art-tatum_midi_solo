"""Chord condition timing for Aria token sequences (docs/experiments/ARIA_COND_SMOKE.md).

The chord plan is an external schedule known in advance. Input position i (the hidden state that
predicts token i + 1) gets the chord sounding at the prefix time of tokens[0..i]: the latest onset
already in the input, or the latest 5 s boundary, whichever is later. Nothing after position i is
read, so a note's own onset never conditions the tokens before it. Pure Python; no Aria import.

| input token at i  | what it reveals             | prefix time for position i       |
|-------------------|-----------------------------|----------------------------------|
| prefix, <S>       | nothing                     | 0                                |
| (piano, p, v)     | the next note's pitch, vel  | unchanged                        |
| (onset, x)        | this note's onset           | 5000 * (<T> so far) + x          |
| (dur, d)          | this note's duration        | unchanged (this note's onset)    |
| <T>               | a 5 s boundary has passed   | max(previous, 5000 * <T> so far) |
| <E>, others       | nothing about time          | unchanged                        |
"""
from __future__ import annotations

SEGMENT_MS = 5000


def prefix_times(tokens) -> list[int]:
    """Prefix time in ms for every input position, from tokens[0..i] only."""
    out, n_t, now = [], 0, 0
    for tok in tokens:
        if tok == "<T>":
            n_t += 1
            now = max(now, SEGMENT_MS * n_t)
        elif isinstance(tok, tuple) and tok[0] == "onset":
            now = SEGMENT_MS * n_t + tok[1]
        out.append(now)
    return out


def chord_at(plan, t_ms: int):
    """plan: [(onset_ms, end_ms, pitch classes)]. Outside the plan there is no chord."""
    for on, end, pcs in plan:
        if on <= t_ms < end:
            return pcs
    return None


def chroma_per_position(tokens, plan) -> list[list[float]]:
    rows = []
    for t in prefix_times(tokens):
        pcs = chord_at(plan, t)
        rows.append([1.0 if pcs is not None and k in pcs else 0.0 for k in range(12)])
    return rows
