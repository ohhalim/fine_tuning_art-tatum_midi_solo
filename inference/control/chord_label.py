"""Chord candidates for one vertical slice of accompaniment (docs/experiments/LABEL_AUDIT.md).

``chord_estimate`` (#1637 draft) scores one best template and prefers the bass as
root. The audit needs the full candidate set instead, without assuming bass = root:
a candidate is a (root, quality) whose guide tones (3rd and 7th, or 6th) are all
sounding and whose chord tones plus allowed tensions cover every sounding pitch
class. The root may be absent (rootless voicings). D-F-A-C stays ambiguous (Dm7 and
F6), as do rootless dominants and their tritone substitutes.

Triads are candidates too, and a slice needs at least three pitch classes: the first
audit run read major-third dyads (left-hand tenths) as minor-major sevenths and plain
major triads as rootless minor sevenths, because nothing else could explain them.
"""
from __future__ import annotations

NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

# quality: (chord tones, required tones, allowed tensions), intervals from the root
QUALITIES = {
    "maj7": ({0, 4, 7, 11}, {4, 11}, {2, 6, 9}),
    "7": ({0, 4, 7, 10}, {4, 10}, {1, 2, 3, 6, 8, 9}),
    "m7": ({0, 3, 7, 10}, {3, 10}, {2, 5, 9}),
    "m7b5": ({0, 3, 6, 10}, {3, 6, 10}, {2, 5, 8}),
    "dim7": ({0, 3, 6, 9}, {3, 6, 9}, {2, 5, 8, 11}),
    "6": ({0, 4, 7, 9}, {4, 9}, {2}),
    "m6": ({0, 3, 7, 9}, {3, 9}, {2, 5}),
    "mMaj7": ({0, 3, 7, 11}, {3, 11}, {2, 5}),
    "maj": ({0, 4, 7}, {4, 7}, {2}),
    "min": ({0, 3, 7}, {3, 7}, {2}),
}
MIN_PCS = 3


def candidates(pcs) -> list[tuple[int, str]]:
    """Every (root pc, quality) consistent with the sounding pitch classes."""
    pcs = {p % 12 for p in pcs}
    out = []
    if len(pcs) < MIN_PCS:
        return out
    for root in range(12):
        rel = {(p - root) % 12 for p in pcs}
        for q, (tones, required, tensions) in QUALITIES.items():
            if required <= rel and rel <= tones | tensions:
                out.append((root, q))
    return out


def classify(pcs, bass: int | None) -> tuple[str, tuple[int, str] | None, list[tuple[int, str]]]:
    """(class, label, candidates): clear / bass_resolved / ambiguous / unknown."""
    cands = candidates(pcs)
    if not cands:
        return "unknown", None, cands
    if len(cands) == 1:
        return "clear", cands[0], cands
    if bass is not None:
        at_bass = [c for c in cands if c[0] == bass % 12]
        if len({c[0] for c in at_bass}) == 1 and len(at_bass) == 1:
            return "bass_resolved", at_bass[0], cands
    return "ambiguous", None, cands


def symbol(label) -> str:
    return "-" if label is None else NAMES[label[0]] + label[1]
