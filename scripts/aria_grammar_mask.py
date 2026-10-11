"""Format-only grammar mask for Aria piano generation (issue #1731, docs/experiments/MEHLDAU_ARIA_MASKED_GEN.md).

Same rules as aria_token_validity, applied before sampling instead of after:
- between notes: a piano note start, <T> (segment reset), <D> or <E>
- after a note start: an onset not before the last onset of the 5 s segment (equal is allowed: chords)
- after an onset: a dur
Nothing about content (chord tones, register, repetition, density) is constrained.

Sampling keeps the #1726 sampler: torch.multinomial(p, 1, generator=g) is argmax(p / Exp(1) noise) with one
noise vector over the vocab per step, so the masked pick is that argmax over allowed tokens only. Where the
unmasked pick was already allowed, the masked pick is the same token.
"""
from __future__ import annotations

import math

import torch

from aria_token_validity import segment_state


def kind(t):
    return t[0] if isinstance(t, tuple) else t


class GrammarMask:
    def __init__(self, vocab):
        self.note_ok = torch.tensor([kind(t) == "piano" or t in ("<T>", "<D>", "<E>") for t in vocab])
        self.dur_ok = torch.tensor([kind(t) == "dur" for t in vocab])
        self.onset_val = torch.tensor([t[1] if kind(t) == "onset" else -1 for t in vocab])
        self.last, self.expect = 0, "note"

    def start(self, prompt_tokens):
        self.last, self.expect = segment_state(prompt_tokens)

    def allowed(self):
        if self.expect == "note":
            return self.note_ok
        if self.expect == "dur":
            return self.dur_ok
        return self.onset_val >= self.last

    def advance(self, t):
        """Move past an allowed token. A disallowed token is a caller bug, not something to resync from."""
        k = kind(t)
        if self.expect == "note":
            if k == "<T>":
                self.last = 0
            elif k == "piano":
                self.expect = "onset"
            elif t not in ("<D>", "<E>"):
                raise ValueError(f"not allowed between notes: {t}")
        elif self.expect == "onset":
            if k != "onset" or t[1] < self.last:
                raise ValueError(f"not allowed after a note start (last onset {self.last}): {t}")
            self.last, self.expect = t[1], "dur"
        else:
            if k != "dur":
                raise ValueError(f"not allowed after an onset: {t}")
            self.expect = "note"


def masked_pick(p, ok, q):
    """(masked pick, unmasked pick) under the same noise q. The caller checks the allowed mass is > 0 first."""
    free = int(torch.argmax(p / q))
    pick = int(torch.argmax(torch.where(ok, p, torch.zeros_like(p)) / q))
    return pick, free


class MaskStop(Exception):
    """The allowed set is empty or its pre-mask mass is not finite / not positive."""


def masked_generate(next_probs, prompt_tokens, prompt_ids, vocab, seed, max_new, eos_id, buf, progress=None):
    """Fill buf["ids"], buf["pre_mask_allowed"], buf["mask_changed"] in place, one step at a time, so a stop in
    the middle (MaskStop, the wall alarm) leaves what was generated so far in buf.
    next_probs(seq) -> CPU probability vector at temperature 1. One Exp(1) noise vector per step, as in #1726."""
    gm = GrammarMask(vocab)
    gm.start(prompt_tokens)
    gen = torch.Generator().manual_seed(seed)
    seq = list(prompt_ids)
    for step in range(max_new):
        p = next_probs(seq)
        ok = gm.allowed()
        a = p[ok].sum().item()
        if not bool(ok.any()) or not math.isfinite(a) or a <= 0:
            raise MaskStop(f"step {step}: allowed set empty or allowed mass {a} ({gm.expect}, last onset {gm.last})")
        q = torch.empty(len(vocab)).exponential_(1, generator=gen)
        pick, free = masked_pick(p, ok, q)
        buf["pre_mask_allowed"].append(a)
        if pick != free:
            buf["mask_changed"].append({"step": step, "state": gm.expect, "unmasked_pick": str(vocab[free]), "masked_pick": str(vocab[pick])})
        seq.append(pick)
        buf["ids"].append(pick)
        if pick == eos_id:
            break
        gm.advance(vocab[pick])
        if progress and (step + 1) % 128 == 0:
            progress(step + 1)


def run_streams(specs, run_one, after, save):
    """run_one(spec, buf) fills buf; after(buf) records and may raise. save({"completed", "incomplete"}) always
    runs, also on a stop: the stream in progress is kept as "incomplete" with stop_step and stop_reason, apart
    from the completed ones (Astra review of #1732)."""
    completed, incomplete = [], None
    try:
        for spec in specs:
            buf = {**spec, "ids": [], "pre_mask_allowed": [], "mask_changed": []}
            try:
                run_one(spec, buf)
            except BaseException as e:
                incomplete = {**buf, "stop_step": len(buf["ids"]), "stop_reason": f"{type(e).__name__}: {e}"}
                raise
            completed.append(buf)
            after(buf)
    finally:
        save({"completed": completed, "incomplete": incomplete})
