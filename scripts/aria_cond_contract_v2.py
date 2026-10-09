"""Chord condition contract v2: current chord, next planned chord, time to the change
(docs/experiments/ARIA_COND_CONTRACT_V2.md). Pure Python; no Aria import.

The prefix time t of input position i is the v1 definition (aria_cond_contract.prefix_times): only
tokens[0..i] are read. The plan is an external schedule known in advance. For every position the
feature row is

    current chroma (12) | next chroma (12) | current_known | has_next | next_known | delta_norm | delta_clamped

- current: the plan segment with onset <= t < end (a change at exactly t is already current)
- next: the first segment after t whose chord differs from the current one (onset strictly > t);
  consecutive segments with the same chord are merged first
- delta_sec = s - t from the plan's tempo map; delta_norm = min(delta_sec, CAP_S) / CAP_S;
  delta_clamped = 1 when delta_sec > CAP_S. No next chord: has_next 0, delta_norm 0, clamped 0,
  so "far away" (has_next 1, clamped 1) and "none" stay apart
- unknown (outside the plan, or a segment marked unknown) has known = 0; a no-chord segment
  (N.C.) is known = 1 with an all-zero chroma. The two are never confused
- never reads a future note's onset, duration or any target token
"""
from __future__ import annotations

from aria_cond_contract import prefix_times

CAP_S = 4.0          # two bars at 128 BPM in 4/4 is 3.75 s
FEATURE_DIM = 29
NC = "N.C."


def beats_to_ms(beat: float, tempo_map) -> float:
    """tempo_map: [(beat, bpm)] sorted, first at beat 0."""
    ms, last_b, bpm = 0.0, 0.0, tempo_map[0][1]
    for b, new_bpm in tempo_map[1:]:
        if b >= beat:
            break
        ms += (b - last_b) * 60000.0 / bpm
        last_b, bpm = b, new_bpm
    return ms + (beat - last_b) * 60000.0 / bpm


def plan_from_beats(segments, tempo_map, shift_ms: float = 0.0):
    """segments: [(onset_beat, end_beat, pcs | "N.C." | None)], None = unknown. Returns ms segments
    in the token timeline (shift_ms = leading silence the tokenizer removed)."""
    out = [(beats_to_ms(a, tempo_map) - shift_ms, beats_to_ms(b, tempo_map) - shift_ms, c) for a, b, c in segments]
    return sorted(out, key=lambda s: s[0])


def _merged(plan):
    out = []
    for on, end, c in sorted(plan, key=lambda s: s[0]):
        key = frozenset(c) if isinstance(c, (set, frozenset)) else c
        if out and out[-1][2] == key and abs(out[-1][1] - on) < 1e-9:
            out[-1] = (out[-1][0], end, key)
        else:
            out.append((on, end, key))
    return out


def _chroma(c):
    if c is None or c == NC:
        return [0.0] * 12
    return [1.0 if k in c else 0.0 for k in range(12)]


def feature_at(t_ms: float, plan) -> dict:
    segs = _merged(plan)
    cur = next((s for s in segs if s[0] <= t_ms < s[1]), None)
    cur_chord = cur[2] if cur is not None else None
    later = [s for s in segs if s[0] > t_ms and s[2] != cur_chord]
    nxt = later[0] if later else None
    delta = (nxt[0] - t_ms) / 1000.0 if nxt is not None else None
    return {"t_ms": t_ms,
            "current": cur_chord, "next": nxt[2] if nxt else None,
            "current_known": int(cur is not None and cur_chord is not None),
            "has_next": int(nxt is not None),
            "next_known": int(nxt is not None and nxt[2] is not None),
            "delta_sec": delta,
            "delta_norm": 0.0 if delta is None else min(delta, CAP_S) / CAP_S,
            "delta_clamped": int(delta is not None and delta > CAP_S)}


def vector(f: dict) -> list[float]:
    return (_chroma(f["current"]) + _chroma(f["next"]) +
            [float(f["current_known"]), float(f["has_next"]), float(f["next_known"]), f["delta_norm"], float(f["delta_clamped"])])


def features(tokens, plan) -> list[dict]:
    return [feature_at(t, plan) for t in prefix_times(tokens)]
