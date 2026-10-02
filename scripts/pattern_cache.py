"""Pattern-cache logit bias for sampling (docs/experiments/PATTERN_CACHE_DECODING.md).

Real players re-use top-line interval patterns from the last few seconds; the
model's rollouts almost never do (#1597, #1600, #1602). At each sampling step
this looks at the last three top-line notes of the sequence so far; wherever
their two intervals started an interval 3-gram inside the preceding 8 s, the
note_on of the pitch that would complete that 3-gram gets ``bias`` added to
its logit. Probabilities only: the model still chooses, and rhythm is free.
"""
from __future__ import annotations

import math

from scripts.repeat_weights import top_line_tokens


def candidate_pitches(tokens, horizon_s: float = 8.0) -> set[int]:
    line = top_line_tokens(tokens)
    if len(line) < 3:
        return set()
    (t1, p1, _), (t2, p2, _), (t3, p3, _) = line[-3:]
    head = (p2 - p1, p3 - p2)
    if head == (0, 0):
        return set()
    out = set()
    for w in range(len(line) - 3):                         # windows wholly before the last three notes
        if t1 - line[w][0] > horizon_s:
            continue
        if w + 3 >= len(line) - 3:            # must not share a note with the current window
            break
        a, b, c, d = (line[w + k][1] for k in range(4))
        if (b - a, c - b) == head:
            nxt = p3 + (d - c)
            if 0 <= nxt < 128:
                out.add(nxt)
    return out


class PatternCacheBias:
    """Callable for ``generate(logits_processor=...)``; counts how often it fired."""

    def __init__(self, bias: float = math.log(3.0), horizon_s: float = 8.0) -> None:
        self.bias = bias
        self.horizon_s = horizon_s
        self.steps = 0
        self.fired = 0

    def __call__(self, token_logits, sequence):
        self.steps += 1
        cands = candidate_pitches([int(t) for t in sequence], self.horizon_s)
        if not cands:
            return token_logits
        self.fired += 1
        out = token_logits.clone()
        for p in cands:
            out[..., p] = out[..., p] + self.bias
        return out
