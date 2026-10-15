"""Left-hand role proxies and causal conditions for the chord-conditioning audit (issue #1750,
docs/experiments/CAUSAL_LH_AUDIT.md). Pure Python, no model.

PiJAMA has no hand labels. The three rules below are role proxies, not hand truth: they split a two-hand
performance into "condition" notes (harmony-like, what a player's left hand would give the model) and "target" notes
(what the model would learn to play). Every note keeps a unique id; a rule also says when the note's role becomes
known (available_at), because some rules need a closed window before they can decide.

  g3        pitch < G3 (55) -> condition, else target. Known at the note's onset
  top50     50 ms onset groups (from the group's first onset): the top note is the target (excluded if below G3),
            the rest are condition. Known when the window closes (first onset + 50 ms)
  grid300   300 ms grid windows with >= 3 onsets: all but the top are condition, the top is the target (excluded if
            below G3). Other windows: notes >= G3 are targets, the rest excluded. Known at the window end

Conditions
  online_condition(d)   condition notes whose role is known by d, that started by d and were not released by d
                        (a release after d is simply "not observed yet"). This is what a real-time system can see
  offline_condition(t)  the usual dataset shortcut: condition notes sounding at the target onset t, regardless of when
                        their role became known. Used only to measure how much future information it would use
"""
from __future__ import annotations

from dataclasses import dataclass

SPLIT_PITCH, GROUP_MS, GRID_MS, GRID_MIN = 55, 50.0, 300.0, 3
LEAD_MS = 468.75          # the runtime commit lead: 1 beat at 128 BPM
INF = float("inf")


@dataclass(frozen=True)
class Note:
    nid: int
    pitch: int
    on: float
    off: float
    vel: int = 64


@dataclass(frozen=True)
class Role:
    role: str             # "cond" | "target" | "excluded"
    available_at: float
    reason: str = ""


def assign_g3(notes):
    return {n.nid: Role("cond" if n.pitch < SPLIT_PITCH else "target", n.on) for n in notes}


def _top_rest(group):
    top = max(group, key=lambda n: (n.pitch, -n.on, -n.nid))
    return top, [n for n in group if n.nid != top.nid]


def _target_or_excluded(n, avail):
    return Role("target", avail) if n.pitch >= SPLIT_PITCH else Role("excluded", avail, "top below G3")


def assign_top50(notes):
    roles, ordered, i = {}, sorted(notes, key=lambda n: (n.on, n.pitch, n.nid)), 0
    while i < len(ordered):
        g0 = ordered[i].on
        group = [n for n in ordered[i:] if n.on < g0 + GROUP_MS]
        avail = g0 + GROUP_MS
        top, rest = _top_rest(group)
        roles[top.nid] = _target_or_excluded(top, avail)
        for n in rest:
            roles[n.nid] = Role("cond", avail)
        i += len(group)
    return roles


def assign_grid300(notes):
    roles, windows = {}, {}
    for n in notes:
        windows.setdefault(int(n.on // GRID_MS), []).append(n)
    for k, group in windows.items():
        avail = (k + 1) * GRID_MS
        if len(group) >= GRID_MIN:
            top, rest = _top_rest(group)
            roles[top.nid] = _target_or_excluded(top, avail)
            for n in rest:
                roles[n.nid] = Role("cond", avail)
        else:
            for n in group:
                roles[n.nid] = Role("target", avail) if n.pitch >= SPLIT_PITCH else Role("excluded", avail, "below G3 outside a chord window")
    return roles


RULES = {"g3": assign_g3, "top50": assign_top50, "grid300": assign_grid300}


def online_condition(notes, roles, d):
    return [n for n in notes if roles[n.nid].role == "cond" and roles[n.nid].available_at <= d and n.on <= d and n.off > d]


def offline_condition(notes, roles, t):
    return [n for n in notes if roles[n.nid].role == "cond" and n.on <= t < n.off]


def prefix_view(notes, d):
    """What exists at time d: notes that started by d; a release after d is not known yet (off = inf)."""
    return [Note(n.nid, n.pitch, n.on, n.off if n.off <= d else INF, n.vel) for n in notes if n.on <= d]


def pcs(cond):
    return frozenset(n.pitch % 12 for n in cond)
