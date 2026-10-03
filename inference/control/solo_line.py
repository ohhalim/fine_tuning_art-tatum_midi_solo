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


class PhraseBreath:
    """Opt-in rest after long phrases in the top line, kept across blocks (#1626).

    Real bebop right hands rest about every 16 notes; the runtime line ran up to
    ~100 notes without a 0.3 s gap (blues, #1620). Once ``max_notes`` notes have
    played since the last gap of ``REST_S``, notes starting within ``rest_s`` of
    the last kept note's end are dropped. Notes are only removed, never moved.
    State follows generated blocks in order (not adoption), a known limitation."""

    REST_S = 0.3

    def __init__(self, max_notes: int, rest_s: float = 0.4) -> None:
        self.max_notes, self.rest_s = max_notes, rest_s
        self.count, self.last_end = 0, None
        self.dropped = 0

    def __call__(self, notes, block_start: float):
        kept = []
        for n in notes:
            start, end = block_start + n.start, block_start + n.end
            if self.count >= self.max_notes:                        # breathing: wait rest_s
                if start < self.last_end + self.rest_s:
                    self.dropped += 1
                    continue
                self.count = 0
            elif self.last_end is not None and start - self.last_end >= self.REST_S:
                self.count = 0                                       # a natural rest
            kept.append(n)
            self.count += 1
            self.last_end = end
        return kept


def _padded(notes, total: int):
    """``notes`` encoded and padded with rest to ``total`` steps; None if they do not fit."""
    from scripts.generate import encode_notes_simple

    out = encode_notes_simple(notes) if notes else []
    remaining = total - _steps(out)
    if remaining < 0:
        return None
    while remaining > 0:
        step = min(remaining, TIME_SHIFT_END - TIME_SHIFT_START + 1)
        out.append(TIME_SHIFT_START + step - 1)
        remaining -= step
    return out


def solo_line_tokens(tokens, line_filter=None) -> list[int]:
    """Re-encode ``tokens`` as their top line, same total length; unchanged if that is impossible.

    ``line_filter(notes)`` may remove top-line notes first (e.g. a bound PhraseBreath)."""
    from scripts.style_distance import tokens_to_notes

    tokens = [int(t) for t in tokens]
    notes = tokens_to_notes(tokens)
    if not notes:
        return tokens
    line = top_notes(notes)
    if line_filter is not None:
        line = line_filter(line)                                 # all filtered out: a rest block
    out = _padded(line, _steps(tokens))
    return tokens if out is None else out


def _note_row(n) -> list:
    return [int(n.pitch), round(float(n.start), 4), round(float(n.end), 4), int(n.velocity)]


def _grid(t: float) -> float:
    """The 10 ms token grid. Model output is already on it; comp hits at beat fractions
    (e.g. 1.5 beats = 0.703125 s at 128 BPM) are not, and the encoder rounds each delta
    while keeping raw times, so off-grid comp shifted the solo notes after it by up to
    ~28 ms (#1649 review)."""
    return round(round(t * 100) / 100, 2)


def solo_with_comp_tokens(tokens, comp_notes, line_filter=None, trace: dict | None = None) -> list[int]:
    """Top line plus ``comp_notes`` (e.g. the chord guide voicing), same total length.

    A comp note is left out only when the top line holds the same pitch at an
    overlapping time, so the two never sound the same key at once (#1621 review:
    dropping it for the whole block lost chord tones the line used elsewhere)."""
    import pretty_midi

    from scripts.style_distance import tokens_to_notes

    tokens = [int(t) for t in tokens]
    total = _steps(tokens)
    line = top_notes(tokens_to_notes(tokens))
    if line_filter is not None:
        # Once per block (it keeps phrase state), before the comp clash check (#1627 review H1).
        line = line_filter(line)
    limit = total / 100

    def clashes(c) -> bool:
        return any(m.pitch == c.pitch and m.start < c.end and c.start < m.end for m in line)
    comp_notes = [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=_grid(n.start),
                                   end=max(_grid(n.end), _grid(n.start) + 0.01)) for n in comp_notes]
    comp, dropped = [], {}
    for n in comp_notes:
        reason = "past_block_end" if n.start >= limit else "same_pitch_overlap" if clashes(n) else None
        if reason:
            dropped[reason] = dropped.get(reason, 0) + 1
            continue
        comp.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=min(n.end, limit)))
    if trace is not None:                            # the solo alone, for context carry without the comp
        solo_only = _padded(line, total)
        trace["solo_tokens"] = solo_only if solo_only is not None else None
    if trace is not None:                            # where each planned comp note went (#1647 review M1)
        trace.update(planned=[_note_row(n) for n in comp_notes], emitted=[_note_row(n) for n in comp],
                     dropped=dropped, clipped=sum(1 for n in comp_notes if n.start < limit < n.end))
    notes = sorted(line + comp, key=lambda n: (n.start, n.pitch))
    if not notes and line_filter is None:
        return tokens
    out = _padded(notes, total)
    if out is None:                                  # comp does not fit: the (filtered) line alone
        out = _padded(line, total)
        if trace is not None:
            trace["dropped"]["reencode_fallback"] = len(trace["emitted"])
            trace["emitted"] = []
    return tokens if out is None else out


def render_block(tokens, *, lookahead_ms: float, comp_notes=None, stats: dict | None = None,
                 line_filter=None, trace: dict | None = None) -> list[int]:
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
    trace = trace if trace is not None else {}
    raw = [int(t) for t in tokens]
    if not valid(raw):
        stats["raw_invalid"] = stats.get("raw_invalid", 0) + 1
        trace.update(outcome="raw_invalid", emitted=[])
        return raw
    rendered = (solo_with_comp_tokens(raw, comp_notes, line_filter, trace) if comp_notes
                else solo_line_tokens(raw, line_filter))
    if not valid(rendered):
        stats["rendered_invalid"] = stats.get("rendered_invalid", 0) + 1
        trace.update(outcome="rendered_invalid", emitted=[])
        return raw
    stats["rendered"] = stats.get("rendered", 0) + 1
    trace["outcome"] = "rendered"
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
