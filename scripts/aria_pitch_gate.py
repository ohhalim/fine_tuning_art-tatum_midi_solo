"""Pitch-position gate for the condition adapter (docs/experiments/WJAZZD_PILOT.md, gate pilot).

gate[i] = 1 when, after reading tokens[0..i] only, the next token may be a pitch token, i.e. the
sequence is between notes: right after <S>, after a note's dur, after <T> or after <D>. It is 0
inside the header, right after a pitch token (the onset is next), right after an onset (the dur is
next), after <E>, and after any other token. The gate never looks at the token it predicts. It is
not an output mask: every token keeps its probability, and conditions injected at earlier positions
still reach later positions through attention.
"""
from __future__ import annotations


def _kind(t):
    return t[0] if isinstance(t, tuple) else t


def gate_step(state: str, tok) -> tuple[str, int]:
    """state in {header, note, onset, dur, end}; returns the new state and the gate after tok."""
    k = _kind(tok)
    if state == "header":
        state = "note" if k == "<S>" else "header"
    elif state == "end":
        pass
    elif k == "<E>":
        state = "end"
    elif state == "note":
        if k == "piano":
            state = "onset"
        elif k in ("<T>", "<D>"):
            state = "note"
        else:
            state = "other"
    elif state == "onset":
        state = "dur" if k == "onset" else "other"
    elif state == "dur":
        state = "note" if k == "dur" else "other"
    if state == "other":                      # unexpected token: resync only at a later dur or <T>/<D>
        state = "note" if k in ("dur", "<T>", "<D>") else "other"
    return state, int(state == "note")


def gates(tokens) -> list[int]:
    state, out = "header", []
    for t in tokens:
        state, g = gate_step(state, t)
        out.append(g)
    return out
