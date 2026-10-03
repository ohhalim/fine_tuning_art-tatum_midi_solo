"""Phrase plan -> pitches over the chords (docs/experiments/PHRASE_REALIZER.md).

Instead of patching each sampled note (#1656-#1662, still "a lot of dissonance"),
separate what a line does from which pitches it uses, as grammar / analysis-by-
synthesis jazz generators do (Impro-Visor abstract melodies; Frieler & Zaddach's
chord-scale mapping): an abstract phrase keeps the rhythm, rests and the signed
interval contour of a source line (a real solo or the model's), and the realizer
chooses pitches for the chords actually sounding.

Realization, one note at a time (single pass, one-note lookahead):
* anchors (on a beat, long notes, phrase ends) take the chord tone nearest the
  contour target, in the contour's direction
* other notes take the scale tone nearest the target; a weak note right before an
  anchor a step away becomes its chromatic approach (half step below/above it)
* a repeated pitch only where the source repeats (interval 0)
* range C4-C6: a target leaving it moves an octave back toward the middle (register
  displacement) instead of sticking at the edge, where clamping piled up repeats
* a phrase starts near the previous phrase's last note
"""
from __future__ import annotations

REST_S = 0.3
LOW, HIGH = 60, 84
LONG_S = 0.3


def abstract(line) -> list[dict]:
    """[(onset, dur, interval from previous note in the phrase or None, velocity)] of a top line."""
    out, prev = [], None
    for n in sorted(line, key=lambda n: n.start):
        new_phrase = prev is None or n.start - prev.end >= REST_S
        out.append({"onset": n.start, "dur": n.end - n.start, "velocity": n.velocity,
                    "interval": None if new_phrase else n.pitch - prev.pitch, "phrase_end": False})
        if out[:-1] and new_phrase:
            out[-2]["phrase_end"] = True
        prev = n
    if out:
        out[-1]["phrase_end"] = True
    return out


def _fold(target: float) -> float:
    while target > HIGH:
        target -= 12
    while target < LOW:
        target += 12
    return target


def _nearest(target: float, allowed_pcs, direction: int = 0, prev: int | None = None, avoid=()) -> int:
    cands = [p for p in range(LOW, HIGH + 1) if p % 12 in allowed_pcs]
    if avoid:                             # no semitone / minor 9th / major 7th against a sounding comp note
        clear = [p for p in cands if all((p - q) % 12 not in (1, 11) for q in avoid)]
        cands = clear or cands
    target = _fold(target)
    if direction and prev is not None:
        dirn = [p for p in cands if (p - prev) * direction > 0]
        if not dirn:                      # at the edge: continue the line an octave away, not on the same key
            dirn = [p for p in cands if p != prev]
        cands = dirn
    return min(cands, key=lambda p: (abs(p - target), p))


def realize(events, *, chord_at, bpm: float, start: int = 72, comp_at=None):
    """Pitched notes [(pitch, onset, end, velocity)] for an abstract phrase sequence.

    ``chord_at(t) -> (chord tone pcs, scale pcs)``; ``comp_at(t0, t1) -> comp pitches sounding in
    [t0, t1)`` (optional): candidates a semitone / minor 9th / major 7th from them are skipped.

    An approach note is placed only on a short weak note right before an anchor, and
    the anchor is then fixed to the goal it approaches (#1669: the first version computed
    the next note independently, so approaches did not resolve and Gb over Fmaj7 or B
    over Gm7 stayed). Chromatic approaches come from a half step below; from above the
    approach is the scale tone above the goal."""
    beat = 60.0 / bpm
    out, prev, forced = [], None, None

    def is_anchor(ev):
        return abs(ev["onset"] - round(ev["onset"] / beat) * beat) <= 0.03 or ev["dur"] >= LONG_S or ev["phrase_end"]

    for i, ev in enumerate(events):
        tones, scale = chord_at(ev["onset"])
        anchor = is_anchor(ev)
        avoid = comp_at(ev["onset"], ev["onset"] + ev["dur"]) if comp_at else ()
        if forced is not None:
            pitch, forced = forced, None
        elif ev["interval"] is None or prev is None:
            pitch = _nearest(prev if prev is not None else start, tones, avoid=avoid)
        elif ev["interval"] == 0:
            pitch = prev                                          # the source repeats here
        else:
            direction = 1 if ev["interval"] > 0 else -1
            pitch = _nearest(prev + ev["interval"], tones if anchor else scale, direction, prev, avoid)
            nxt = events[i + 1] if i + 1 < len(events) else None
            if (not anchor and ev["dur"] <= 0.2 and nxt is not None and nxt["interval"] is not None
                    and 0 < abs(nxt["interval"]) <= 2 and is_anchor(nxt)):
                ntones, nscale = chord_at(nxt["onset"])
                goal = _nearest(pitch + nxt["interval"], ntones)
                if nxt["interval"] > 0:
                    approach = goal - 1                           # chromatic from below
                else:
                    above = [p for p in range(goal + 1, goal + 3) if p % 12 in nscale]
                    approach = above[0] if above else goal + 2    # scale tone from above
                if LOW <= approach <= HIGH and approach != prev:
                    pitch, forced = approach, goal
        pitch = int(_fold(pitch))
        out.append((pitch, ev["onset"], ev["onset"] + ev["dur"], ev["velocity"]))
        prev = pitch
    return out


def clear_comp(solo, comp, chord_at):
    """Final pass on the played timeline: a solo note sounding a semitone / minor 9th / major 7th
    from an overlapping comp note moves to the nearest chord tone that clashes with nothing (#1669)."""
    out = []
    for k, (p, s, e, v) in enumerate(solo):
        sound = [q for q, cs, ce, _ in comp if min(e, ce) - max(s, cs) > 0.02]
        if any((p - q) % 12 in (1, 11) for q in sound):
            tones, _ = chord_at(s)
            cands = [c for c in range(LOW, HIGH + 1) if c % 12 in tones and all((c - q) % 12 not in (1, 11) for q in sound)]
            if cands:
                p = min(cands, key=lambda c: (abs(c - p), c))
        out.append((p, s, e, v))
    return out
