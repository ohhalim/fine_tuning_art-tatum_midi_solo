"""Pick one of N generated candidates for a block (docs/experiments/CANDIDATE_SELECT.md, #1637).

The same fixed ranker as the offline check (scripts/candidate_select_eval.py):
among valid candidates whose solo top line (>= G3) has at least ``MIN_NOTES``
notes, the highest fit - clash against the chord's pitch classes, duration
weighted; ties go to the lowest index; with no qualifying candidate, candidate 0.
Candidates are all model output: nothing is edited or filtered.
"""
from __future__ import annotations

MIN_NOTES = 3


def solo_line(tokens, block_s: float):
    from inference.control.harmony_contract import SOLO_SPLIT
    from inference.control.solo_line import top_notes
    from scripts.style_distance import tokens_to_notes

    line = top_notes(tokens_to_notes([int(t) for t in tokens]))
    return [(n.pitch, n.start, min(n.end, block_s)) for n in line if n.pitch >= SOLO_SPLIT and n.start < block_s]


def score(solo, pcs) -> float | None:
    if len(solo) < MIN_NOTES:
        return None
    dur = [max(e - s, 0.01) for _, s, e in solo]
    tot = sum(dur)
    fit = sum(d for (p, _, _), d in zip(solo, dur) if p % 12 in pcs) / tot
    clash = sum(d for (p, _, _), d in zip(solo, dur)
                if min(min((p - q) % 12, (q - p) % 12) for q in pcs) == 1) / tot
    return fit - clash


def pick(candidates, pcs, *, block_s: float, valid) -> tuple[int, list]:
    """(index, scores) for token lists ``candidates``; ``valid(tokens) -> bool``."""
    best, best_i, scores = None, 0, []
    for i, toks in enumerate(candidates):
        s = score(solo_line(toks, block_s), pcs) if valid(toks) else None
        scores.append(s)
        if s is not None and (best is None or s > best):
            best, best_i = s, i
    return best_i, scores
