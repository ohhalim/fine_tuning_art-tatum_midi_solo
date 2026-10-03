"""Chord symbols estimated from the accompaniment proxy of a real window (#1637 draft).

Real bebop transcriptions carry no chord labels, so the runtime's chord-symbol
metrics had no same-source reference. Template matching over the five qualities
the runtime knows: the bass pitch class is preferred as root; a window counts as
*confident* only when one candidate wins outright and the third and seventh are
both present (so the quality is actually observed, not guessed).
"""
from __future__ import annotations

QUALITIES = {"maj7": (0, 4, 7, 11), "7": (0, 4, 7, 10), "m7": (0, 3, 7, 10), "m7b5": (0, 3, 6, 10),
             "dim": (0, 3, 6, 9)}
NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MISS, EXTRA, BASS_BONUS = 1.0, 1.0, 1.5


def candidates(pcs, bass: int | None):
    out = []
    for root in range(12):
        for q, iv in QUALITIES.items():
            t = {(root + i) % 12 for i in iv}
            score = len(pcs & t) - EXTRA * len(pcs - t) - MISS * 0.25 * len(t - pcs)
            if bass is not None and bass == root:
                score += BASS_BONUS
            out.append((score, root, q))
    return sorted(out, reverse=True)


def estimate(pcs, bass: int | None) -> dict:
    """Best (root, quality), its pitch classes, and whether it is confident."""
    pcs = set(pcs)
    ranked = candidates(pcs, bass)
    best, second = ranked[0], ranked[1]
    _, root, q = best
    iv = QUALITIES[q]
    third, seventh = (root + iv[1]) % 12, (root + iv[3]) % 12
    confident = best[0] > second[0] and third in pcs and seventh in pcs
    return {"root": root, "quality": q, "symbol": NAMES[root] + q, "pcs": {(root + i) % 12 for i in iv},
            "margin": best[0] - second[0], "confident": confident}
