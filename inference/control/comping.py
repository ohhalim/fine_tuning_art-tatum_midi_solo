"""Comping that varies like a player's, still telling the chord (docs/experiments/COMPING.md).

The user heard the runtime comp (root-3rd-7th on beat 1 and the & of 3, same
velocity and length every half bar) as "too mechanical" (2026-10-03). This
keeps the job (make the chord audible, short, under the solo) and changes how.
Decided one half bar at a time, like the runtime's blocks (the next chord can
arrive live, so it is not known in advance):

* rhythm: one of a few hand-written half-bar figures, never the same figure
  twice in a row; pairs of halves give the familiar bar shapes (Charleston,
  anticipations, two-feel, pushes). Hand-written templates, not learned
* voicing: rootless 3rd + 7th + 5th or 9th inside E3-E4 (under the solo's G3+
  top line), choosing the one closest to the previous voicing (voice leading);
  the root low on the first hit of a new chord
* a new chord always gets a hit within its first beat; "rest" is only chosen
  when the chord did not change
* velocity: accents on off-beat hits plus a small deterministic variation

Deterministic for (chord sequence, block, seed).
"""
from __future__ import annotations

import itertools
import random

# (start beat, length in beats, accent) inside a half bar (2 beats)
FIGURES = {
    "on1": [(0.0, 0.9, 0)],
    "long": [(0.0, 1.7, 0)],
    "push": [(0.5, 0.4, 1)],
    "and2": [(1.5, 0.45, 1)],
    "on1_and2": [(0.0, 0.45, 0), (1.5, 0.4, 1)],
    "and1_and2": [(0.5, 0.35, 1), (1.5, 0.4, 1)],
    "rest": [],
}
LOW, HIGH = 52, 64             # rootless voicing window (E3-E4)
BASS_LOW = 36                  # root C2-B2
BASE_VEL, ACCENT, JITTER = 52, 7, 4


def chord_tones(chord: str) -> tuple[int, list[int]]:
    """(root pc, [3rd, 5th, 7th, 9th] pcs) for the runtime's chord vocabulary (natural 9th)."""
    from inference.app.fallback import parse_chord

    root, iv = parse_chord(chord)
    return root, [(root + iv[1]) % 12, (root + iv[2]) % 12, (root + iv[3]) % 12, (root + 2) % 12]


def voicings(chord: str) -> list[tuple[int, ...]]:
    """Rootless 3-note voicings (3rd + 7th + 5th or 9th) inside LOW-HIGH, within an octave."""
    _, (third, fifth, seventh, ninth) = chord_tones(chord)
    out = set()
    for extra in (fifth, ninth):
        cands = [[p for p in range(LOW, HIGH + 1) if p % 12 == pc] for pc in (third, seventh, extra)]
        for combo in itertools.product(*cands):
            v = tuple(sorted(combo))
            if v[-1] - v[0] <= 12 and len(set(v)) == 3:
                out.add(v)
    return sorted(out)


def closest(options, previous):
    if previous is None:
        return min(options, key=lambda v: (abs(sum(v) / len(v) - (LOW + HIGH) / 2), v))
    return min(options, key=lambda v: (sum(abs(a - b) for a, b in zip(v, previous)), v))


def comp_half(chord: str, *, block: int, bpm: float, seed: int, state: dict):
    """Notes (pitch, start s, end s, velocity) for one half-bar block, times from its start; and the figure."""
    beat = 60.0 / bpm
    rng = random.Random(seed * 7919 + block)
    changed = chord != state.get("chord")
    names = [f for f in FIGURES if f != state.get("figure") and not (changed and f == "rest")]
    figure = names[rng.randrange(len(names))]
    hits = list(FIGURES[figure])
    if changed and not any(h[0] < 1.0 for h in hits):
        hits.insert(0, (0.0, 0.45, 0))                       # state the new chord within its first beat
    state["figure"] = figure
    notes = []
    for i, (start, length, accent) in enumerate(hits):
        end = min(start + length, 2.0, hits[i + 1][0] if i + 1 < len(hits) else 2.0)
        voicing = closest(voicings(chord), state.get("voicing"))
        state["voicing"] = voicing
        vel = BASE_VEL + ACCENT * accent + rng.randint(-JITTER, JITTER)
        t0, t1 = start * beat, end * beat
        if chord != state.get("chord"):
            notes.append((BASS_LOW + chord_tones(chord)[0], t0, t1, vel))
            state["chord"] = chord
        notes += [(p, t0, t1, vel) for p in voicing]
    return notes, figure


def shell_half(chord: str, *, sub_index: int, bpm: float):
    """The previous comp (#1620): root-3rd-7th, beat 1 or the & of 3, 0.25 s, velocity 56."""
    from inference.control.solo_line import shell_voicing

    at = 0.0 if sub_index == 0 else 0.5 * 60.0 / bpm
    return [(p, at, at + 0.25, 56) for p in shell_voicing(chord)]



GUIDE_LOW, GUIDE_HIGH = 48, 59     # C3-B3: A2-Ab3 put a minor 3rd / tritone below the low interval limits (#1669)
GUIDE_FIGURES = [[(0.0, 0.6)], [(1.5, 0.5)], [(0.0, 0.6)], [(0.0, 0.6)]]   # per half bar: 1 | &2 | 1 | 1
GUIDE_VEL = 46


def guide_tones(chord: str, previous=None) -> tuple[int, ...]:
    """3rd and 7th inside C3-B3, the pair closest to the previous pair."""
    _, (third, _fifth, seventh, _ninth) = chord_tones(chord)
    cands = []
    for a in range(GUIDE_LOW, GUIDE_HIGH + 1):
        for b in range(GUIDE_LOW, GUIDE_HIGH + 1):
            if a % 12 == third and b % 12 == seventh:
                cands.append(tuple(sorted((a, b))))
    return closest(sorted(set(cands)), previous)


def guide_half(chord: str, *, block: int, bpm: float, state: dict):
    """Clean comp (#1665): root low on a chord change, 3rd + 7th in C3-B3, soft and short.

    The figure follows a fixed 2-bar cycle (beat 1, & of 2, beat 1, beat 1), so it is
    predictable rather than random; a new chord is always stated on its first beat."""
    beat = 60.0 / bpm
    changed = chord != state.get("chord")
    figure = [(0.0, 0.6)] if changed else GUIDE_FIGURES[block % len(GUIDE_FIGURES)]
    pair = guide_tones(chord, state.get("voicing"))
    state["voicing"] = pair
    notes = []
    for start, length in figure:
        t0, t1 = start * beat, (start + length) * beat
        if changed:
            notes.append((BASS_LOW + chord_tones(chord)[0], t0, t1, GUIDE_VEL + 4))
        notes += [(p, t0, t1, GUIDE_VEL) for p in pair]
    state["chord"] = chord
    return notes, "guide"


GUIDE_ALTERNATIVES = (("3", "7"), ("3", "5"), ("5", "7"), ("1", "3"), ("1", "7"))   # docs/experiments/C1_COMP_AB.md


def solo_clashes(upper, solo, t0: float, t1: float, overlap: float = 0.02):
    """[(solo pitch, comp pitch, overlap s)] where a solo note sounding over [t0, t1) lies a
    semitone (mod 12) above a comp pitch: minor 2nd, minor 9th and their compounds."""
    out = []
    for p, s, e in solo:
        ov = min(e, t1) - max(s, t0)
        if ov > overlap:
            out += [(p, q, ov) for q in upper if p > q and (p - q) % 12 == 1]
    return out


def revoice_strike(chord: str, upper, solo, t0: float, t1: float):
    """C1: the guide pair for one strike, re-chosen among the chord's own tones in C3-B3
    when it clashes with the solo (``solo_clashes``). Dominant sevenths keep 3 + 7 (their
    tritone is the function); no clash-free candidate keeps the original. Returns
    (pitches, info) with info["status"] in clear / changed / unresolved / dominant_kept."""
    from inference.app.fallback import parse_chord

    root, iv = parse_chord(chord)
    tone = {"1": root, "3": (root + iv[1]) % 12, "5": (root + iv[2]) % 12, "7": (root + iv[3]) % 12}
    place = lambda pc: GUIDE_LOW + (pc - GUIDE_LOW) % 12
    before = solo_clashes(upper, solo, t0, t1)
    info = {"before": before}
    if not before:
        return tuple(upper), {**info, "status": "clear"}
    if iv[1] == 4 and iv[3] == 10:
        return tuple(upper), {**info, "status": "dominant_kept"}
    for names in GUIDE_ALTERNATIVES:
        cand = tuple(sorted(place(tone[n]) for n in names))
        if len(set(c % 12 for c in cand)) == 2 and not solo_clashes(cand, solo, t0, t1):
            omitted = [n for n in ("3", "7") if n not in names]
            filled = {n: any(p % 12 == tone[n] and min(e, t1) - max(s, t0) > 0.02 for p, s, e in solo) for n in omitted}
            return cand, {**info, "status": "changed", "voicing": names, "omitted": omitted, "solo_fills_omitted": filled}
    return tuple(upper), {**info, "status": "unresolved"}
