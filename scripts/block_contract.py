"""Runtime block snapshot contract (issue #1754, docs/experiments/BLOCK_SNAPSHOT_LH_AUDIT.md). Pure Python.

What the real-time loop (scripts/rt_producer.py, #1747) can have observed when it commits a note, in music ms:

  block 0   cold prebuffer, made before playback starts. Commits onsets in [t0, t0 + B + beat).
            Observed no later than t0 - PREBUFFER_LEAD_MS (the producer starts playback 1 s after the prebuffer;
            the prebuffer's own run time makes the real observation even earlier, not modelled here). This is an
            upper bound on the cold observation time, not the time itself
  block k   k >= 1, starts at s_k = t0 + k*B and commits onsets in [t0 + k*B + beat, t0 + (k+1)*B + beat)
            (commit_until keeps onset < horizon, horizon = s_k + B + beat)
  onset < t0 is the primer, never generated

Rollback rule this contract requires (proposed, not implemented; today the producer rolls back only on fixed chord
blocks): at each block start the condition representation the model would get at s_k is computed again from what is
observed up to s_k and compared with the representation the uncommitted tokens were generated under (the one at
s_(k-1)). If they differ, the uncommitted tokens are dropped and regenerated from s_k. The comparison is on the whole
representation, not on input events: a recent-window expiry changes it with no new note-on, note-off or role event,
and so does any relative age or time feature put into the input (#1755 review M1). With this rule every committed
note's condition is the representation at its own block's start, which is what snapshot() returns.
Whether dropping tokens is enough, or the KV cache must be invalidated further back, depends on how the condition is
injected; that is checked when the injection is built, not guaranteed here.
"""
from __future__ import annotations

import math

BEAT, BLOCK = 468.75, 937.5       # scripts/aria_mlx_stream_loop.py (128 BPM, half-bar blocks)
PREBUFFER_LEAD_MS = 1000.0        # scripts/rt_producer.py LEAD_S
WORST_LEAD_MS = BLOCK + BEAT      # the largest onset - snapshot gap for k >= 1 (exclusive bound)


def block_of(t, t0):
    """The block that commits a note with onset t, or None for the primer (t < t0)."""
    if t < t0:
        return None
    return max(0, math.floor((t - t0 - BEAT) / BLOCK))


def horizon(k, t0):
    return t0 + (k + 1) * BLOCK + BEAT


def block_start(k, t0):
    """The observation time of block k: s_k = t0 + k*B, and t0 - PREBUFFER_LEAD_MS for the cold block 0."""
    return t0 - PREBUFFER_LEAD_MS if k == 0 else t0 + k * BLOCK


def snapshot(t, t0):
    """The latest time whose observations can shape the committed note with onset t (None for the primer)."""
    k = block_of(t, t0)
    return None if k is None else block_start(k, t0)


def must_regenerate(k, t0, condition_at):
    """True when block k has to drop its uncommitted tokens under the rollback rule above.

    condition_at(s) returns the full condition representation the model gets at time s (window expiry and any
    age or time feature included), as a comparable value. Block 0 has no earlier speculation."""
    return k > 0 and condition_at(block_start(k, t0)) != condition_at(block_start(k - 1, t0))
