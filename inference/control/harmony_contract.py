"""One harmony guide format shared by training data and the runtime (#1631).

The runtime states the chord to the model as a short low-register voicing
before each block (``chord_guide_notes_for_duration``). That voicing dropped
chord tones: it took the lowest three chord tones above C3, so G7 came out as
G-D-F-G (no third) and Dm7b5 as D-C-D-F, the same as Dm7 (no flat fifth).

This module builds the guide from a bass pitch class and a set of pitch classes,
so the same function serves both sides of the contract:

* runtime: chord symbol -> (root, chord pitch classes)
* training/evaluation: the accompaniment proxy of a real window -> (lowest
  pitch class, sounding low pitch classes)

The proxy is not a hand separation or a chord label: it is the notes below the
solo register that start in, or still sound at the start of, the window.

Serialised example: ``[guide][the window's solo line]``. The guide starts at 0
and ends at ``GUIDE_FRACTION`` of the window, as at runtime; the solo line
follows it in the token stream, its times relative to the window start.
"""
from __future__ import annotations

GUIDE_VELOCITY = 58
GUIDE_FRACTION = 0.85
SOLO_SPLIT = 55          # top-line notes at or above G3 are the solo (build_rh_dataset)
BASS_LOW, UPPER_LOW = 36, 48


def guide_pitches(bass_pc: int, pcs) -> list[int]:
    """Bass in C2-B2, every other pitch class once in C3-B3, ascending."""
    bass = BASS_LOW + bass_pc % 12
    upper = sorted(UPPER_LOW + pc % 12 for pc in set(pcs) if pc % 12 != bass_pc % 12)
    return [bass, *upper]


def guide_notes(bass_pc: int, pcs, seconds: float):
    import pretty_midi

    if seconds <= 0:
        return []
    end = max(seconds * GUIDE_FRACTION, min(seconds, 0.05))
    return [pretty_midi.Note(velocity=GUIDE_VELOCITY, pitch=p, start=0.0, end=end)
            for p in guide_pitches(bass_pc, pcs)]


def chord_pcs(chord: str) -> tuple[int, set[int]]:
    from inference.app.fallback import parse_chord

    root, intervals = parse_chord(chord)
    return root, {(root + i) % 12 for i in intervals}


def accompaniment_proxy(notes, t0: float, t1: float, *, split: int = SOLO_SPLIT):
    """(bass pc, pcs, stats) of the notes below ``split`` starting in [t0, t1) or sounding at t0.

    ``notes`` must be sorted by start. Returns bass ``None`` when nothing qualifies."""
    low = [n for n in notes if n.pitch < split and n.start < t1 and (n.start >= t0 or n.end > t0 + 0.05)]
    if not low:
        return None, set(), {"empty": True, "pcs": 0, "onset_groups": 0}
    pcs = {n.pitch % 12 for n in low}
    onsets = sorted({round(max(n.start, t0), 2) for n in low})
    groups = sum(1 for i, t in enumerate(onsets) if i == 0 or t - onsets[i - 1] > 0.05)
    bass = min(low, key=lambda n: n.pitch).pitch % 12
    return bass, pcs, {"empty": False, "pcs": len(pcs), "onset_groups": groups}


def solo_window(line, t0: float, t1: float):
    """Solo-line notes starting in [t0, t1), times relative to t0, clipped at the window end."""
    import pretty_midi

    return [pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start - t0,
                             end=min(n.end, t1) - t0)
            for n in line if t0 <= n.start < t1 and min(n.end, t1) > n.start]


def serialize(guide, solo, seconds: float) -> tuple[list[int], int]:
    """Tokens ``[guide][solo]`` and the index where the solo starts.

    The solo is encoded on its own timeline after the guide's last event; a rest
    pads the solo to ``seconds`` so the window length is part of the example."""
    from inference.control.solo_line import _padded
    from scripts.generate import encode_notes_simple

    head = encode_notes_simple(sorted(guide, key=lambda n: (n.start, n.pitch))) if guide else []
    steps = int(round(seconds * 100))
    body = _padded(sorted(solo, key=lambda n: (n.start, n.pitch)), steps)
    if body is None:
        body = encode_notes_simple(sorted(solo, key=lambda n: (n.start, n.pitch)))
    return head + body, len(head)
