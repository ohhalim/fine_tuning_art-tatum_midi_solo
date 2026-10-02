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


def solo_with_comp_tokens(tokens, comp_notes) -> list[int]:
    """Top line plus ``comp_notes`` (e.g. the chord guide voicing), same total length.

    A comp pitch that the top line also plays in this block is left out, so the
    two never sound the same key twice."""
    import pretty_midi

    from scripts.generate import encode_notes_simple
    from scripts.style_distance import tokens_to_notes

    tokens = [int(t) for t in tokens]
    total = _steps(tokens)
    line = top_notes(tokens_to_notes(tokens))
    used = {n.pitch for n in line}
    limit = total / 100
    comp = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=min(n.end, limit))
            for n in comp_notes if n.pitch not in used and n.start < limit]
    notes = sorted(line + comp, key=lambda n: (n.start, n.pitch))
    if not notes:
        return tokens
    out = encode_notes_simple(notes)
    remaining = total - _steps(out)
    if remaining < 0:
        return solo_line_tokens(tokens)
    while remaining > 0:
        step = min(remaining, TIME_SHIFT_END - TIME_SHIFT_START + 1)
        out.append(TIME_SHIFT_START + step - 1)
        remaining -= step
    return out


def render_block(tokens, *, lookahead_ms: float, comp_notes=None, stats: dict | None = None) -> list[int]:
    """What to schedule for a generated block under --solo-line / --comp.

    Astra review: the raw model block is validated first. An invalid raw block
    is returned unchanged so the block builder rejects it and the usual fallback
    plays; the rewrite must not turn a model error into a valid block. The
    rewritten block is validated again; if it fails, the valid raw block plays.
    ``stats`` counts raw_invalid / rendered / rendered_invalid separately, per
    generation attempt: not per adopted block, not per block actually sent."""
    from scripts.run_resident_model_probe import validate_generated_token_block

    def valid(t) -> bool:
        return bool(validate_generated_token_block(t, lookahead_ms=lookahead_ms, allow_rest_bar=True)["valid"])

    stats = stats if stats is not None else {}
    raw = [int(t) for t in tokens]
    if not valid(raw):
        stats["raw_invalid"] = stats.get("raw_invalid", 0) + 1
        return raw
    rendered = solo_with_comp_tokens(raw, comp_notes) if comp_notes else solo_line_tokens(raw)
    if not valid(rendered):
        stats["rendered_invalid"] = stats.get("rendered_invalid", 0) + 1
        return raw
    stats["rendered"] = stats.get("rendered", 0) + 1
    return rendered


def shell_voicing(chord: str) -> list[int]:
    """Root low (C2-G2 / Ab1-B1), third and seventh in E3-Eb4: no seconds between them.

    The chord-guide voicing used before had seconds in it (e.g. Dbmaj7 C-Db) and
    rang for 60% of the block; the user heard the comping as messy (2026-10-02)."""
    from inference.app.fallback import parse_chord

    root, iv = parse_chord(chord)
    r = 36 + root if root <= 7 else 24 + root
    third = 52 + ((root + iv[1]) - 52) % 12
    seventh = 52 + ((root + iv[3]) - 52) % 12
    return sorted({r, third, seventh})
