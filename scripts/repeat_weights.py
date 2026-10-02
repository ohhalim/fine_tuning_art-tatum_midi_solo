"""Which training tokens re-complete a recent top-line pattern (docs/experiments/REPEAT_WEIGHT_PILOT.md).

A note_on token is marked when it is the fourth note of a top-line window
(highest note of each 50 ms onset cluster) whose interval 3-gram already
occurred in an earlier, non-overlapping window that started within the
preceding 8 s: the same rule as ``coherence_metrics.motif_reuse``. Same-note
patterns (0, 0, 0) are not marked.
"""
from __future__ import annotations

TS_START, TS_END = 256, 355


def top_line_tokens(tokens, window_s: float = 0.05):
    """(cluster onset s, pitch, token index of that note_on) per onset cluster."""
    line, first, best, steps = [], None, None, 0
    for i, t in enumerate(tokens):
        t = int(t)
        if TS_START <= t <= TS_END:
            steps += t - TS_START + 1
        elif 0 <= t < 128:
            s = steps / 100
            if first is None or s - first > window_s:
                if best is not None:
                    line.append((first, best[0], best[1]))
                first, best = s, (t, i)
            elif t > best[0]:
                best = (t, i)
    if best is not None:
        line.append((first, best[0], best[1]))
    return line


def repeat_note_on_mask(tokens, horizon_s: float = 8.0) -> list[bool]:
    from collections import Counter, deque

    mask = [False] * len(tokens)
    line = top_line_tokens(tokens)
    grams = [(line[w][0], tuple(line[w + k + 1][1] - line[w + k][1] for k in range(3)), line[w + 3][2])
             for w in range(len(line) - 3)]
    pending, window, counts = deque(), deque(), Counter()
    for w, (t, g, idx) in enumerate(grams):
        while pending and pending[0][0] <= w - 4:          # earlier windows not overlapping window w
            _, tj, gj = pending.popleft()
            window.append((tj, gj))
            counts[gj] += 1
        while window and t - window[0][0] > horizon_s:
            _, old = window.popleft()
            counts[old] -= 1
        if counts[g] > 0 and any(g):
            mask[idx] = True
        pending.append((w, t, g))
    return mask


def batch_weights(x, y, weight: float, pad: int):
    """Per-target weights for a batch: ``weight`` where y re-completes a pattern, else 1 (0 at PAD)."""
    import torch

    w = torch.ones_like(y, dtype=torch.float32)
    for b in range(y.shape[0]):
        seq = [int(x[b, 0])] + [int(t) for t in y[b]]
        m = repeat_note_on_mask(seq)[1:]
        w[b] = torch.tensor([weight if mk else 1.0 for mk in m], dtype=torch.float32)
    return w.masked_fill(y == pad, 0.0)


def weighted_smooth_ce(logits, target, weights, label_smoothing: float, vocab_size: int):
    """Label-smoothed CE per token (as SmoothCrossEntropyLoss), weighted mean; weight 0 drops a token."""
    import torch
    import torch.nn.functional as F

    q = F.one_hot(target.clamp(min=0).long(), vocab_size).float()
    q = (1.0 - label_smoothing) * q + label_smoothing / vocab_size
    ce = -(q * F.log_softmax(logits, dim=-1)).sum(-1)
    return (ce * weights).sum() / weights.sum().clamp(min=1e-8)
