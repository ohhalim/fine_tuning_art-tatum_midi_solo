"""Keep only the top line of a generated block (#1589).

The model learned whole piano parts, so a block mixes left-hand chords with the
melody; in the FL Studio test the user heard "not soloing, chords go in too".
This keeps the highest note of each 50 ms onset cluster (the same top line the
coherence metrics use), makes it monophonic and re-encodes it, padding with
rest so the block still fills exactly the same time. Notes are not changed,
only removed: no claim that the result sounds like a solo.
"""
from __future__ import annotations

TIME_SHIFT_START = 256
TIME_SHIFT_END = 355


def _steps(tokens) -> int:
    return sum(int(t) - TIME_SHIFT_START + 1 for t in tokens if TIME_SHIFT_START <= int(t) <= TIME_SHIFT_END)


def top_notes(notes, window_s: float = 0.05):
    """Highest note of each onset cluster, each ended by the next kept onset."""
    import pretty_midi

    kept, first, best = [], None, None
    for n in sorted(notes, key=lambda n: (n.start, -n.pitch)):
        if first is None or n.start - first > window_s:
            if best is not None:
                kept.append(best)
            first, best = n.start, n
        elif n.pitch > best.pitch:
            best = n
    if best is not None:
        kept.append(best)
    out = []
    for i, n in enumerate(kept):
        end = min(n.end, kept[i + 1].start) if i + 1 < len(kept) else n.end
        out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=max(end, n.start + 0.01)))
    return out


def solo_line_tokens(tokens) -> list[int]:
    """Re-encode ``tokens`` as their top line, same total length; unchanged if that is impossible."""
    from scripts.generate import encode_notes_simple
    from scripts.style_distance import tokens_to_notes

    tokens = [int(t) for t in tokens]
    notes = tokens_to_notes(tokens)
    if not notes:
        return tokens
    out = encode_notes_simple(top_notes(notes))
    remaining = _steps(tokens) - _steps(out)
    if remaining < 0:
        return tokens
    while remaining > 0:
        step = min(remaining, TIME_SHIFT_END - TIME_SHIFT_START + 1)
        out.append(TIME_SHIFT_START + step - 1)
        remaining -= step
    return out
