"""Runtime block snapshot contract (issue #1754, docs/experiments/BLOCK_SNAPSHOT_LH_AUDIT.md). Pure Python.

What the real-time loop (scripts/rt_producer.py, #1747) can have observed when it commits a note, in music ms:

  block 0   cold prebuffer, made before playback starts. Commits onsets in [t0, t0 + B + beat).
            Observed no later than t0 - PREBUFFER_LEAD_MS (the producer starts playback 1 s after the prebuffer;
            the prebuffer's own run time makes the real observation even earlier, not modelled here)
  block k   k >= 1, starts at s_k = t0 + k*B and commits onsets in [t0 + k*B + beat, t0 + (k+1)*B + beat)
            (commit_until keeps onset < horizon, horizon = s_k + B + beat)
  onset < t0 is the primer, never generated

Rollback rule this contract requires (not implemented yet; today the producer rolls back only on fixed chord blocks):
at each block start, if the condition state at s_k differs from the state at s_(k-1) (a condition-role note started,
was released, or had its role confirmed), the uncommitted tokens are dropped and regenerated from s_k. Tokens kept
from speculation were made under an equal state. With this rule every committed note's condition is the state at its
own block's start, which is what snapshot() returns.
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


def snapshot(t, t0):
    """The latest time whose observations can shape the committed note with onset t (None for the primer)."""
    k = block_of(t, t0)
    if k is None:
        return None
    return t0 - PREBUFFER_LEAD_MS if k == 0 else t0 + k * BLOCK
