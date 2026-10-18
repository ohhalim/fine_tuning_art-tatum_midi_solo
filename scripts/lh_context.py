"""Observed left hand in the Aria MIDI context (issue #1772, docs/experiments/LH_CONTEXT_CONTRACT.md). Pure Python.

For one block request R_k (scripts/block_request.py, session state of scripts/crop_contract.py) three inputs:
  a  the right-hand session context as today
  b  a + the human notes observed in [max(s_k - W, base_ms), s_k), inserted as Aria abs notes
  c  a + another dev song's human notes from a window of the same length, time-shifted to end at s_k
The continuation (window + boundary) is appended unchanged to all three, so the labels are the same tokens at the same
distance from the end. Human notes never become labels.

Rules
  window     onset in [max(s_k - W, base_ms), s_k); a note still sounding at s_k (off > s_k) is cut to off = s_k (P1),
             a virtual end forced by the representation, not a release. onset == s_k is left out, off == s_k is
             released. Request 0 (cold) gets no human notes. Right-hand history is never cut
  tokens     Aria abs rules (ariautils config: 10 ms grid, 5 s segments, 5 s max dur, velocity step 10, closest value by
             the tokenizer's own binary search, dur 0 -> 10), on integer-ms times (the input a 1 tick = 1 ms MIDI gives
             the official tokenizer). Velocity is the one recovered from the source MIDI (scripts/lh_velocity_recover.py)
  merge      right-hand context tokens are never changed; a human note goes into its 5 s segment before the first
             right-hand note with onset >= its onset (ties: human first). Its segment's <T> must already be in the
             context, else ContractViolation
  donor      another dev song only, request k' >= 1, window [s_k' - L, s_k') with L the target window length: same note
             count, same held flag, nearest total sounding ms (after the cut), ties by (|k' - k|, song, k'); cut at s_k',
             then shifted. Only human notes observable in the two windows choose it. c is "another song's context",
             not a wrong-harmony label
  crop       a's session (crop_contract.session_requests, right hand only) is the base. If b or c would need a crop,
             all three are cut again at the same later <T> (earliest that keeps the longest arm <= KEEP_TARGET, else
             the latest safe one), so base_ms and the first kept token stay equal
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass

from crop_contract import CAP, CAPACITY, HEADER, KEEP_TARGET, CapacityFailure, needs_crop
from stream_contract import SEGMENT_MS, GrammarState

W_MS = 7500.0
TIME_STEP_MS, MAX_DUR_MS, VELOCITY_STEP = 10, 5000, 10
ONSET_GRID = list(range(0, SEGMENT_MS, TIME_STEP_MS))
DUR_GRID = list(range(0, MAX_DUR_MS + 1, TIME_STEP_MS))
VELOCITY_GRID = [i * VELOCITY_STEP for i in range(int(127 / VELOCITY_STEP) + 1)]
ONSET_SLACK_MS = 5.5    # L1t: 0.5 integer rounding + 5 grid rounding
END_SLACK_MS = 15.5     # L1t: onset 5.5 + dur rounding up to 10 (dur 0 -> 10), measured from the cut raw off


class ContractViolation(Exception):
    pass


def closest(n, grid):
    """ariautils Tokenizer._find_closest_int: binary search, the first strictly closer value wins."""
    left, right, best = 0, len(grid) - 1, grid[0]
    while left <= right:
        mid = (left + right) // 2
        if abs(grid[mid] - n) < abs(best - n):
            best = grid[mid]
        if grid[mid] < n:
            left = mid + 1
        else:
            right = mid - 1
    return best


def q_onset(local_ms):
    return closest(local_ms, ONSET_GRID)


def q_dur(ms):
    d = closest(ms, DUR_GRID)
    return d if d != 0 else TIME_STEP_MS


def q_velocity(v):
    q = closest(v, VELOCITY_GRID)
    return VELOCITY_STEP if q == 0 and v != 0 else q


@dataclass(frozen=True)
class LHNote:
    pitch: int
    on: int          # integer ms
    off: int         # integer ms, cut at s if held
    held: bool
    raw_on: float
    raw_off: float   # cut at s, before rounding
    vel: int = 64
    source: str = "own"


class Human:
    """Human notes sorted by onset, for window lookups by bisect. notes: objects with pitch, on, off (ms)."""

    def __init__(self, notes):
        self.notes = sorted(notes, key=lambda n: (n.on, n.pitch))
        self.ons = [n.on for n in self.notes]


def window_notes(human, s, start):
    """Human notes with start <= onset < s, cut at s (P1)."""
    h = human if isinstance(human, Human) else Human(human)
    out = []
    for n in h.notes[bisect.bisect_left(h.ons, start):bisect.bisect_left(h.ons, s)]:
        held = n.off > s
        off = min(n.off, s)
        out.append(LHNote(n.pitch, int(round(n.on)), int(round(off)), held, float(n.on), float(off), int(n.vel)))
    return sorted(out, key=lambda x: (x.on, x.pitch))


def stats(notes):
    """(count, held flag, total sounding ms) of a window."""
    return len(notes), any(n.held for n in notes), sum(n.off - n.on for n in notes)


def note_tokens(n):
    """(absolute segment, quantized onset, [piano, onset, dur] tokens) under the Aria abs rules."""
    seg = n.on // SEGMENT_MS
    onset = q_onset(n.on - seg * SEGMENT_MS)
    return seg, onset, [("piano", n.pitch, q_velocity(n.vel)), ("onset", onset), ("dur", q_dur(n.off - n.on))]


def quantized_end(n):
    seg, onset, toks = note_tokens(n)
    return seg * SEGMENT_MS + onset + toks[2][1]


def quantized_on(n):
    seg, onset, _ = note_tokens(n)
    return seg * SEGMENT_MS + onset


def dur_capped(n):
    """True when the 5 s cap moved this note's end earlier (reported, not a violation)."""
    return n.off - n.on > MAX_DUR_MS


def _kind(t):
    return t[0] if isinstance(t, tuple) else t


def merge(ctx, base_ms, notes):
    """Insert human notes into a session context. Returns (tokens, tags) with tags in {"hdr", "rh", "lh"}."""
    if base_ms % SEGMENT_MS:
        raise ContractViolation(f"base_ms {base_ms} is not on a segment boundary")
    base_seg = int(base_ms // SEGMENT_MS)
    pending = sorted((note_tokens(n) for n in notes), key=lambda x: (x[0], x[1], x[2][0][1]))
    out, tags, seg, i, in_body = [], [], 0, 0, False

    def flush(upto_onset):
        nonlocal pending
        while pending and pending[0][0] - base_seg == seg and (upto_onset is None or pending[0][1] <= upto_onset):
            out.extend(pending[0][2])
            tags.extend(["lh"] * 3)
            pending = pending[1:]

    while i < len(ctx):
        t = ctx[i]
        k = _kind(t)
        if k == "piano":
            in_body = True
            if pending and pending[0][0] - base_seg < seg:
                raise ContractViolation(f"human note in segment {pending[0][0]} before the session segment {seg + base_seg}")
            flush(ctx[i + 1][1])
            out.extend(ctx[i:i + 3])
            tags.extend(["rh"] * 3)
            i += 3
        elif t == "<T>":
            in_body = True
            flush(None)
            if pending and pending[0][0] - base_seg <= seg:
                raise ContractViolation(f"human note left behind in segment {pending[0][0]}")
            out.append(t)
            tags.append("rh")
            seg += 1
            i += 1
        else:
            out.append(t)
            tags.append("rh" if in_body else "hdr")
            i += 1
    flush(None)
    if pending:
        raise ContractViolation(f"{len(pending)} human notes after the last context segment {seg + base_seg}")
    return out, tags


def strip_lh(tokens, tags):
    return [t for t, g in zip(tokens, tags) if g != "lh"]


def grammar_ok(tokens):
    st = GrammarState()
    try:
        for t in tokens:
            st = st.advance(t)
    except ValueError:
        return False
    return st.expect == "note"


def _ctx(tokens, srow, keep_from):
    return list(tokens[:srow["ctx_end"]]) if keep_from is None else list(HEADER) + list(tokens[keep_from:srow["ctx_end"]])


def extra_cut(tokens, srow, sizes_fn, cap=CAP, capacity=CAPACITY, keep_target=KEEP_TARGET):
    """(keep_from, base_ms) for a request whose longest arm needs a crop: a later <T> in a's session, the earliest
    that keeps the longest arm <= keep_target, else the latest safe one. sizes_fn(keep_from, base_ms) -> arm lengths."""
    ctx = _ctx(tokens, srow, srow["keep_from"])
    offset = 0 if srow["keep_from"] is None else srow["keep_from"] - len(HEADER)
    cands = []
    for j, t in enumerate(ctx):
        if t != "<T>":
            continue
        keep_from = offset + j + 1
        removed = ctx[:j + 1].count("<T>")
        base = srow["base_ms"] + removed * SEGMENT_MS
        cands.append((keep_from, base, max(sizes_fn(keep_from, base))))
    within = [c for c in cands if c[2] <= keep_target]
    if within:
        return within[0][:2]
    safe = [c for c in cands if not needs_crop(c[2], cap, capacity)]
    if safe:
        return safe[-1][:2]
    raise CapacityFailure(f"request {srow['k']}: no <T> fits the longest arm")


def donor_for(target_song, k, s, length, own, pool):
    """Best donor window for (target_song, k). pool: {song: (human notes, [(k', s'), ...])}. Returns
    (song, k', s', notes shifted to end at s) or None."""
    n, held, total = stats(own)
    best = None
    for song in sorted(pool):
        if song == target_song:
            continue
        human, reqs = pool[song]
        for k2, s2 in reqs:
            if k2 < 1:
                continue
            cand = window_notes(human, s2, s2 - length)
            n2, held2, total2 = stats(cand)
            if n2 != n or held2 != held:
                continue
            key = (abs(total2 - total), abs(k2 - k), song, k2)
            if best is None or key < best[0]:
                best = (key, song, k2, s2, cand)
    if best is None:
        return None
    _, song, k2, s2, cand = best
    shift = s - s2
    moved = [LHNote(x.pitch, int(round(x.raw_on + shift)), int(round(x.raw_off + shift)), x.held,
                    x.raw_on + shift, x.raw_off + shift, x.vel, f"{song}:{k2}") for x in cand]
    return song, k2, s2, moved


def build_request(tokens, srow, human, song, pool, w_ms=W_MS, cap=CAP, capacity=CAPACITY, keep_target=KEEP_TARGET, window_start=None):
    """The three inputs of one request. Returns a dict with per-arm tokens, tags, label start, plus metadata.
    window_start(srow, base) -> left-hand window start replaces max(s - w_ms, base) (issue #1831 offline variants)."""
    k, s = srow["k"], srow["s"]
    cont = list(tokens[srow["ctx_end"]:srow["cont_end"]])

    def arms_at(keep_from, base):
        ctx = _ctx(tokens, srow, keep_from)
        if k < 1:
            return ctx, [], None, w_ms
        start = window_start(srow, base) if window_start else max(s - w_ms, base)
        own = window_notes(human, s, start)
        donor = donor_for(song, k, s, s - start, own, pool) if own else None
        return ctx, own, donor, s - start

    keep_from, base = srow["keep_from"], srow["base_ms"]

    def sizes(kf, b):
        ctx, own, donor, _ = arms_at(kf, b)
        return [len(ctx), len(ctx) + 3 * len(own), len(ctx) + 3 * (len(donor[3]) if donor else len(own))]

    extra = None
    if needs_crop(max(sizes(keep_from, base)), cap, capacity):
        keep_from, base = extra_cut(tokens, srow, sizes, cap, capacity, keep_target)
        extra = {"keep_from": keep_from, "base_ms": base}
    ctx, own, donor, length = arms_at(keep_from, base)
    a_tok, a_tag = merge(ctx, base, [])
    b_tok, b_tag = merge(ctx, base, own)
    if not own:
        c_tok, c_tag, donor_meta = a_tok, a_tag, None
    elif donor is None:
        c_tok, c_tag, donor_meta = None, None, "missing"
    else:
        c_tok, c_tag = merge(ctx, base, donor[3])
        donor_meta = {"song": donor[0], "k": donor[1], "s": donor[2]}
    n, held, total = stats(own)
    layer = "empty" if n == 0 else ("held" if held else "released")
    eligible_c = donor is not None and stats(donor[3])[:2] == (n, held) if own else True
    out = {"k": k, "s": s, "cap": cap, "capacity": capacity, "keep_from": keep_from, "base_ms": base, "extra_crop": extra, "window_ms": length,
           "layer": layer, "lh_notes": n, "lh_total_ms": total, "donor": donor_meta, "eligible_c": eligible_c, "own": own,
           "donor_notes": donor[3] if donor else [], "cont": cont,
           "win_labels": srow["win_end"] - srow["ctx_end"], "boundary_labels": srow["cont_end"] - srow["win_end"]}
    for name, tok, tag in (("a", a_tok, a_tag), ("b", b_tok, b_tag), ("c", c_tok, c_tag)):
        out[name] = None if tok is None else {"tokens": tok + cont, "tags": tag + ["rh"] * len(cont),
                                               "label_start": len(tok)}
    return out


def check_request(r):
    """L1-L6, L8 violation counts for one built request (L7 needs the real tokenizer)."""
    v = {k: 0 for k in ("L1", "L1t", "L2", "L3", "L4", "L5", "L6", "L8")}
    s = r["s"]
    for n in r["own"] + r["donor_notes"]:
        if not (n.raw_on < s and n.raw_off <= s):
            v["L1"] += 1
        if quantized_on(n) > n.raw_on + ONSET_SLACK_MS or quantized_end(n) > n.raw_off + END_SLACK_MS:
            v["L1t"] += 1
    a = r["a"]
    for name in ("b", "c"):
        x = r[name]
        if x is None:
            continue
        if x["tokens"][x["label_start"]:] != a["tokens"][a["label_start"]:] or \
                len(x["tokens"]) - x["label_start"] != len(a["tokens"]) - a["label_start"]:
            v["L2"] += 1
        if strip_lh(x["tokens"], x["tags"]) != a["tokens"]:
            v["L3"] += 1
        if any(g == "lh" for g in x["tags"][x["label_start"]:]):
            v["L8"] += 1
        lh_runs = [x["tokens"][i] for i, g in enumerate(x["tags"]) if g == "lh"]
        if len(lh_runs) % 3 or any(_kind(t) != ("piano", "onset", "dur")[i % 3] for i, t in enumerate(lh_runs)):
            v["L8"] += 1
    for name in ("a", "b", "c"):
        x = r[name]
        if x is None:
            continue
        if not grammar_ok(x["tokens"]):
            v["L4"] += 1
        if needs_crop(x["label_start"], r["cap"], r["capacity"]):
            v["L5"] += 1
    if r["layer"] != "empty" and r["donor"] not in (None, "missing"):
        n, held, _ = stats(r["own"])
        n2, held2, _ = stats(r["donor_notes"])
        start = s - r["window_ms"]
        if n2 != n or held2 != held or any(not (start - 0.5 <= d.raw_on < s) for d in r["donor_notes"]):
            v["L6"] += 1
    return v
