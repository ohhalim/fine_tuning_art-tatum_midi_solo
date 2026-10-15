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
  recent_condition(d)   secondary, registered before the audit: condition notes whose role is known by d and that
                        started in [d - RECENT_MS, d], released or not. Uses no release information at all (comping hits
                        are short, so "still sounding at d" can be empty even right after a chord)

replay() is a second, independent implementation of the two online conditions: it feeds note-on/off events in time
order, closes rule windows when their time comes and answers each query from that state only. The audit compares it
with the batch functions above on every target.
"""
from __future__ import annotations

from dataclasses import dataclass

SPLIT_PITCH, GROUP_MS, GRID_MS, GRID_MIN = 55, 50.0, 300.0, 3
LEAD_MS = 468.75          # the runtime commit lead: 1 beat at 128 BPM
RECENT_MS = 937.5         # secondary condition window: one runtime block (2 beats at 128 BPM)
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


def recent_condition(notes, roles, d, w=RECENT_MS):
    return [n for n in notes if roles[n.nid].role == "cond" and roles[n.nid].available_at <= d and d - w <= n.on <= d]


def _decide(group, chord):
    """Roles for one closed window, written apart from assign_* on purpose. chord: the window counts as a chord."""
    out = {}
    if chord:
        top = sorted(group, key=lambda n: (-n.pitch, n.on, n.nid))[0]
        for n in group:
            out[n.nid] = "cond" if n is not top else ("target" if n.pitch >= SPLIT_PITCH else "excluded")
    else:
        for n in group:
            out[n.nid] = "target" if n.pitch >= SPLIT_PITCH else "excluded"
    return out


def replay(notes, rule, queries, w=RECENT_MS):
    """Event-stream answer to each query time: (sounding condition ids, recent condition ids). Never reads an event
    later than the query. rule: "g3" | "top50" | "grid300"."""
    events = sorted([(n.on, 0, n.nid) for n in notes] + [(n.off, 1, n.nid) for n in notes])
    by_id = {n.nid: n for n in notes}
    cond, sounding = set(), set()         # cond: ids whose window has closed with the condition role
    group, g0, windows = [], None, {}
    out, i = {}, 0

    def close_ready(d):
        nonlocal group, g0
        if rule == "top50" and group and g0 + GROUP_MS <= d:
            cond.update(k for k, r in _decide(group, True).items() if r == "cond")
            group, g0 = [], None
        if rule == "grid300":
            for k in sorted(windows):
                if (k + 1) * GRID_MS <= d:
                    g = windows.pop(k)
                    cond.update(k for k, r in _decide(g, len(g) >= GRID_MIN).items() if r == "cond")

    for qi, d in sorted(enumerate(queries), key=lambda x: x[1]):
        while i < len(events) and events[i][0] <= d:
            t, kind, nid = events[i]
            i += 1
            if kind == 1:
                sounding.discard(nid)
                continue
            close_ready(t)                # windows that ended by this onset close before it joins anything
            n = by_id[nid]
            sounding.add(nid)
            if rule == "g3":
                if n.pitch < SPLIT_PITCH:
                    cond.add(nid)
            elif rule == "top50":
                if not group:
                    g0 = t
                group.append(n)
            else:
                windows.setdefault(int(t // GRID_MS), []).append(n)
        close_ready(d)
        out[qi] = (frozenset(cond & sounding), frozenset(k for k in cond if d - w <= by_id[k].on <= d))
    return [out[k] for k in range(len(queries))]


def prefix_view(notes, d):
    """What exists at time d: notes that started by d; a release after d is not known yet (off = inf)."""
    return [Note(n.nid, n.pitch, n.on, n.off if n.off <= d else INF, n.vel) for n in notes if n.on <= d]


def pcs(cond):
    return frozenset(n.pitch % 12 for n in cond)
