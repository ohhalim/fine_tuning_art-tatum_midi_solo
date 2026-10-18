"""Live left-hand input state, block snapshot, session crop and the deadline rule (issue #1797, docs/experiments/LH_INPUT_SMOKE.md).

Pure Python, no MIDI library and no model: the producer feeds received messages in, takes a snapshot at each block
start and builds the #1773 b context from it.
"""
from __future__ import annotations

from dataclasses import dataclass

INF = float("inf")


@dataclass
class LiveNote:
    pitch: int
    on: float               # music ms
    off: float = INF        # music ms once a note-off was received, INF while held
    vel: int = 64


class LHState:
    """Note on/off bookkeeping in music time. Rules (doc 3): a re-struck pitch ends the previous note at the new onset;
    a velocity-0 note_on is a note_off; an off for a pitch never started is ignored and counted; CC64 is ignored and counted."""

    def __init__(self):
        self.notes, self.active = [], {}
        self.counts = {"on": 0, "off": 0, "retrigger": 0, "velocity0_off": 0, "unknown_off": 0, "cc64_ignored": 0, "other_ignored": 0}

    def feed(self, kind, m_ms, pitch=None, velocity=0, control=None):
        if kind == "note_on" and velocity > 0:
            if pitch in self.active:
                self.notes[self.active[pitch]].off = m_ms
                self.counts["retrigger"] += 1
            self.active[pitch] = len(self.notes)
            self.notes.append(LiveNote(pitch, m_ms, INF, velocity))
            self.counts["on"] += 1
        elif kind in ("note_off", "note_on"):
            if kind == "note_on":
                self.counts["velocity0_off"] += 1
            if pitch in self.active:
                self.notes[self.active.pop(pitch)].off = m_ms
                self.counts["off"] += 1
            else:
                self.counts["unknown_off"] += 1
        elif kind == "control_change" and control == 64:
            self.counts["cc64_ignored"] += 1
        else:
            self.counts["other_ignored"] += 1

    def human(self):
        """Objects for lh_context.Human / window_notes (pitch, on, off, vel). Held notes keep off = INF; window_notes
        cuts them at the snapshot time (virtual end, #1773 held rule)."""
        return [LiveNote(n.pitch, n.on, n.off, n.vel) for n in self.notes]


def is_late(ready_ns, deadline_ns):
    """Doc 5: a block whose commit/schedule happens after its deadline is not sent; the run stops (deadline_miss)."""
    return ready_ns > deadline_ns


def block_problem(ready_ns, deadline_ns, coverage_ms, horizon_ms):
    """Why a block may not be committed, or None. fill_block also calls on_ready when it ran out of samples before
    the horizon (issue #1802); such a block is incomplete and is not sent (horizon_underfill). A late block is
    reported as deadline_miss first. deadline_ns None (the cold block) skips the time check."""
    if deadline_ns is not None and is_late(ready_ns, deadline_ns):
        return "deadline_miss"
    if coverage_ms < horizon_ms:
        return "horizon_underfill"
    return None


def guarded_commit(ready_ns, deadline_ns, coverage_ms, horizon_ms, commit):
    """The producer's on_ready guard: run commit() (commit + send) only for a block with no problem.
    Returns the problem (deadline_miss, horizon_underfill) or None."""
    problem = block_problem(ready_ns, deadline_ns, coverage_ms, horizon_ms)
    if problem is None:
        commit()
    return problem


class SessionCrop:
    """Session state over the right-hand stream: keep_from (stream index of the first kept token, None = original
    header kept) and base_ms, as in crop_contract.session_requests."""

    def __init__(self, cap, capacity, keep_target):
        self.cap, self.capacity, self.keep_target = cap, capacity, keep_target
        self.keep_from, self.base_ms, self.crops = None, 0.0, 0

    def session(self, tokens, committed_end):
        from crop_contract import HEADER
        return list(tokens[:committed_end]) if self.keep_from is None else list(HEADER) + list(tokens[self.keep_from:committed_end])

    def maybe_crop(self, tokens, committed_end, merged_len=None):
        """Crop when the request built from the session would not fit. merged_len(session, base_ms) -> length of the
        merged request (right hand + the left hand windowed for that base); default: the session alone.
        The cut is chosen on that merged length, not on the right hand alone (issue #1803), in the order of
        crop_contract.choose_cut: the earliest <T> that leaves the request <= keep_target, else the latest that is
        still safe, else CapacityFailure. Returns True on a crop."""
        from crop_contract import HEADER, CapacityFailure, check_settings, needs_crop
        from stream_contract import SEGMENT_MS
        merged_len = merged_len or (lambda sess, base: len(sess))
        sess = self.session(tokens, committed_end)
        if not needs_crop(merged_len(sess, self.base_ms), self.cap, self.capacity):
            return False
        check_settings(self.cap, self.capacity, self.keep_target)
        cands = []
        for j, t in enumerate(sess):
            if t != "<T>":
                continue
            base = self.base_ms + sess[:j + 1].count("<T>") * SEGMENT_MS
            cands.append((j, base, merged_len(list(HEADER) + sess[j + 1:], base)))
        within = [c for c in cands if c[2] <= self.keep_target]
        safe = [c for c in cands if not needs_crop(c[2], self.cap, self.capacity)]
        if not within and not safe:
            raise CapacityFailure(f"no <T> among {len(cands)} leaves the merged request under {self.capacity}")
        j, base, _ = within[0] if within else safe[-1]
        cut = j if self.keep_from is None else self.keep_from + j - len(HEADER)
        self.keep_from, self.base_ms = cut + 1, base
        self.crops += 1
        return True


def cap_floor(onsets, position_ms, grid0, block_ms, cap):
    """Note cap per block window (issue #1819). Windows are [grid0 + i * block_ms, grid0 + (i + 1) * block_ms), the
    blocks' commit windows. When the window holding position_ms (the stream's current time) already has `cap` notes
    among the provisional onsets, the next onset must start in the next window: that window start is returned as a
    floor for the grammar mask. Otherwise -inf (no floor). cap 0 means off."""
    if not cap:
        return float("-inf")
    w = (position_ms - grid0) // block_ms
    n = sum(1 for o in onsets if (o - grid0) // block_ms == w)
    return grid0 + (w + 1) * block_ms if n >= cap else float("-inf")
