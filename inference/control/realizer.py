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


def _nearest(target: float, allowed_pcs, direction: int = 0, prev: int | None = None) -> int:
    cands = [p for p in range(LOW, HIGH + 1) if p % 12 in allowed_pcs]
    target = _fold(target)
    if direction and prev is not None:
        dirn = [p for p in cands if (p - prev) * direction > 0]
        if not dirn:                      # at the edge: continue the line an octave away, not on the same key
            dirn = [p for p in cands if p != prev]
        cands = dirn
    return min(cands, key=lambda p: (abs(p - target), p))


def realize(events, *, chord_at, bpm: float, start: int = 72):
    """Pitched notes [(pitch, onset, end, velocity)] for an abstract phrase sequence.

    ``chord_at(t) -> (chord tone pcs, scale pcs)``."""
    beat = 60.0 / bpm
    out, prev = [], None
    for i, ev in enumerate(events):
        tones, scale = chord_at(ev["onset"])
        anchor = (abs(ev["onset"] - round(ev["onset"] / beat) * beat) <= 0.03 or ev["dur"] >= LONG_S
                  or ev["phrase_end"])
        if ev["interval"] is None or prev is None:
            target, direction = (prev if prev is not None else start), 0
        else:
            target, direction = prev + ev["interval"], (ev["interval"] > 0) - (ev["interval"] < 0)
        if direction == 0 and ev["interval"] == 0 and prev is not None:
            pitch = prev                                          # the source repeats here
        else:
            pitch = _nearest(target, tones if anchor else scale, direction, prev)
            nxt = events[i + 1] if i + 1 < len(events) else None
            if (not anchor and nxt is not None and nxt["interval"] is not None and 0 < abs(nxt["interval"]) <= 2):
                ntones, _ = chord_at(nxt["onset"])
                nxt_anchor = (abs(nxt["onset"] - round(nxt["onset"] / beat) * beat) <= 0.03
                              or nxt["dur"] >= LONG_S or nxt["phrase_end"])
                if nxt_anchor:
                    goal = _nearest(pitch + nxt["interval"], ntones)
                    approach = goal - (1 if nxt["interval"] > 0 else -1)
                    if LOW <= approach <= HIGH:
                        pitch = approach                          # chromatic approach into the anchor
        pitch = max(LOW, min(HIGH, pitch))
        out.append((pitch, ev["onset"], ev["onset"] + ev["dur"], ev["velocity"]))
        prev = pitch
    return out
