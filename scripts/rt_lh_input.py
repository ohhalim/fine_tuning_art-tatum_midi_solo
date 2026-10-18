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

    def maybe_crop(self, tokens, committed_end, extra_tokens=0):
        """Crop when the session plus extra_tokens (the left-hand tokens of the b context) would not fit. Returns True on a crop."""
        from crop_contract import HEADER, choose_cut, needs_crop
        from stream_contract import SEGMENT_MS
        sess = self.session(tokens, committed_end)
        if not needs_crop(len(sess) + extra_tokens, self.cap, self.capacity):
            return False
        j = choose_cut(sess, len(sess), self.cap, self.capacity, self.keep_target)
        removed_t = sess[:j + 1].count("<T>")
        cut = j if self.keep_from is None else self.keep_from + j - len(HEADER)
        self.keep_from, self.base_ms = cut + 1, self.base_ms + removed_t * SEGMENT_MS
        self.crops += 1
        return True
