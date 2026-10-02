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
    """Callable for ``generate(logits_processor=...)``; counts how often it fired.

    Incremental: new tokens extend the top line, and every completed 4-note
    window is indexed by its first two intervals, so a step costs O(new tokens
    + matches) instead of re-reading the whole sequence. ``candidates`` gives the
    same set as ``candidate_pitches`` on the same prefix (tested)."""

    def __init__(self, bias: float = math.log(3.0), horizon_s: float = 8.0, window_s: float = 0.05) -> None:
        self.bias = bias
        self.horizon_s = horizon_s
        self.window_s = window_s
        self.steps = 0
        self.fired = 0
        self._n = 0                 # tokens consumed
        self._time = 0              # 10 ms steps
        self._done = []             # completed clusters: (onset s, pitch)
        self._cur = None            # current cluster: [onset s, pitch]
        self._index = {}            # (i1, i2) -> [(window onset, w, i3)]

    def _complete(self, cluster) -> None:
        self._done.append((cluster[0], cluster[1]))
        if len(self._done) >= 4:
            w = len(self._done) - 4
            a, b, c, d = (self._done[w + k][1] for k in range(4))
            self._index.setdefault((b - a, c - b), []).append((self._done[w][0], w, d - c))

    def _advance(self, tokens) -> None:
        for t in tokens[self._n:]:
            t = int(t)
            if 256 <= t <= 355:
                self._time += t - 255
            elif 0 <= t < 128:
                s = self._time / 100
                if self._cur is None or s - self._cur[0] > self.window_s:
                    if self._cur is not None:
                        self._complete(self._cur)
                    self._cur = [s, t]
                elif t > self._cur[1]:
                    self._cur[1] = t
        self._n = len(tokens)

    def candidates(self, tokens) -> set[int]:
        self._advance(tokens)
        line = self._done + ([tuple(self._cur)] if self._cur is not None else [])
        n = len(line)
        if n < 3:
            return set()
        (t1, p1), (_, p2), (_, p3) = line[-3:]
        head = (p2 - p1, p3 - p2)
        if head == (0, 0):
            return set()
        out = set()
        for start, w, i3 in self._index.get(head, ()):
            if w + 3 < n - 3 and t1 - start <= self.horizon_s and 0 <= p3 + i3 < 128:
                out.add(p3 + i3)
        return out

    def __call__(self, token_logits, sequence):
        self.steps += 1
        cands = self.candidates(sequence.tolist() if hasattr(sequence, "tolist") else list(sequence))
        if not cands:
            return token_logits
        self.fired += 1
        out = token_logits.clone()
        for p in cands:
            out[..., p] = out[..., p] + self.bias
        return out
