"""Context crop + full-reset contract (issue #1758, docs/experiments/CROP_RESET_CONTRACT.md). Pure Python.

The block-request stream (scripts/block_request.py) grows past the real-time KV capacity (scripts/rt_producer.py,
4096 positions). At the start of a request, before anything is generated, the session may crop its committed history
and prefill again (stream_contract.TokenStream.request_full_reset). Only <T> boundaries are supported as cut points:
Aria abs onsets are relative to their 5 s segment, so cutting right after a <T> keeps every kept token unchanged.
Cutting at a note boundary would also be possible with a separately kept segment origin; that is left out here.

  decision   needs_crop(committed_len): committed_len + 2 * CAP > CAPACITY. Only the committed length is used, never
             the provisional or future tokens: in a training replay those are ground-truth future tokens, and the cut,
             offset and payload must not change when the future suffix changes (#1758 review 5). One fill_block call
             samples at most CAP tokens (ready + speculative); provisional tokens kept from the previous call are at most
             the CAP that call sampled. So committed_len + 2 * CAP <= CAPACITY keeps every fed position below CAPACITY
             (the last sampled token is fed by the next call, which checks again first)
  crop path  provisional tokens are dropped (as in a rollback) and regenerated after the reset, in training as in the
             runtime. The cut <T> is chosen by choose_cut():
               1. the earliest <T> that leaves header + kept <= KEEP_TARGET (keep the most history under the target)
               2. else the latest <T> that still satisfies the safety bound (target missed, still safe)
               3. else CapacityFailure: no <T> can make room. Never cut a note, never reset the clock
  time       base_ms accumulates: base_ms += (<T> tokens removed, the cut <T> included) * 5000. A token's absolute time is
             base_ms + local_segment * 5000 + onset. s_k, t0, block k and horizons stay on the session clock; the commit
             walk compares local times with h - base_ms. Scheduled note-offs belong to the scheduler and never move
  header     a synthetic [prefix, <S>]; the old header and the primer are not put back (that would revive removed time)
  conditions input position q carries the payload of the request that committed its label q + 1 (block_request).
             After a crop: the prefix position (label <S>) is cold; the <S> position predicts the first kept token and
             carries that token's committing request, i.e. what the cut <T> position carried before; kept positions keep
             their old provenance; p0 and the continuation carry the current request (empty kept history: <S> is p0).
             Replacing every kept condition by the current one would be a different hypothesis; not done here
"""
from __future__ import annotations

from block_request import COLD, committing_request, payload, primer_end, request_table
from block_contract import block_start, horizon
from stream_contract import SEGMENT_MS, TokenStream

CAP, CAPACITY, KEEP_TARGET = 64, 4096, 2048    # tokens per fill_block call, KV positions, kept length after a crop
HEADER = (("prefix", "instrument", "piano"), "<S>")


class CapacityFailure(Exception):
    """No <T> boundary can bring the committed history under the safety bound."""


def needs_crop(committed_len, cap=CAP, capacity=CAPACITY):
    return committed_len + 2 * cap > capacity


def choose_cut(session, committed_end, cap=CAP, capacity=CAPACITY, keep_target=KEEP_TARGET):
    """Index j of the <T> to cut after: session[:j + 1] is removed, session[j + 1:committed_end] is kept."""
    sizes = {j: len(HEADER) + committed_end - (j + 1) for j in range(committed_end) if session[j] == "<T>"}
    within = [j for j, n in sizes.items() if n <= keep_target]
    if within:
        return min(within)
    safe = [j for j, n in sizes.items() if not needs_crop(n, cap, capacity)]
    if safe:
        return max(safe)
    raise CapacityFailure(f"no <T> among {len(sizes)} leaves room under {capacity} with {committed_end} committed tokens")


def session_requests(tokens, t0, cap=CAP, capacity=CAPACITY, keep_target=KEEP_TARGET):
    """block_request.request_table rows plus the session state at each request start (after its crop, if any):
    keep_from (original index of the first kept token, None while the original header is kept), base_ms, ctx_len
    (session context length) and crop (None or {"cut": original index of the cut <T>, "removed_T": n})."""
    rows = request_table(tokens, t0)
    out, keep_from, base_ms = [], None, 0.0
    for r in rows:
        def session_ctx():
            return list(tokens[:r["ctx_end"]]) if keep_from is None else list(HEADER) + list(tokens[keep_from:r["ctx_end"]])
        sess = session_ctx()
        crop = None
        if needs_crop(len(sess), cap, capacity):
            j = choose_cut(sess, len(sess), cap, capacity, keep_target)
            removed_t = sess[:j + 1].count("<T>")
            cut = j if keep_from is None else keep_from + j - len(HEADER)
            keep_from, base_ms = cut + 1, base_ms + removed_t * SEGMENT_MS
            crop = {"cut": cut, "removed_T": removed_t}
            sess = session_ctx()
        out.append({**r, "keep_from": keep_from, "base_ms": base_ms, "ctx_len": len(sess), "crop": crop})
    return out


def session_tokens(tokens, srow):
    """The session token list of a request: context + continuation, in session coordinates."""
    if srow["keep_from"] is None:
        return list(tokens[:srow["cont_end"]])
    return list(HEADER) + list(tokens[srow["keep_from"]:srow["cont_end"]])


def session_position_requests(tokens, srows, k):
    """Which request's payload each session input position 0 .. len(session) - 2 of request k carries (-1 = cold)."""
    rows = [{key: r[key] for key in ("k", "ctx_end", "win_end", "cont_end")} for r in srows]
    r = srows[k]
    owner = committing_request(rows, r["cont_end"])
    p0 = r["ctx_len"] - 1
    n = len(session_tokens(tokens, r)) - 1
    if r["keep_from"] is None:
        return [owner[q + 1] for q in range(p0)] + [k] * (n - p0)
    a = r["keep_from"]
    return [-1 if q == 0 else owner[a + q - 1] for q in range(p0)] + [k] * (n - p0)


def absolute_times(session, base_ms):
    """Absolute ms of every note in a session token list: base_ms + local segment * 5000 + onset."""
    out, seg = [], 0
    for i, t in enumerate(session):
        if t == "<T>":
            seg += 1
        elif isinstance(t, tuple) and t[0] == "piano":
            out.append((base_ms + seg * SEGMENT_MS + session[i + 1][1], t[1], session[i + 2][1]))
    return out


def simulate_session_kv(tokens, t0, human_notes, cap=CAP, capacity=CAPACITY, keep_target=KEEP_TARGET, spec_ms=None):
    """Second, sequential implementation (for the checks): block_request.simulate_kv with the crop path. At each
    request start: payload; capacity check on the committed length; on a crop drop the provisional tokens, rebuild
    [prefix, <S>] + kept, carry the kept positions' tags over (the <S> position takes the cut <T> position's tag) and
    shift the clock by the removed <T> tokens; else roll back on a payload change as before. Returns
    {k: (payload per KV position q < p0, base_ms, original index of the cut <T> or None)}."""
    from block_contract import BLOCK
    spec_ms = BLOCK if spec_ms is None else spec_ms
    p_end = primer_end(tokens, t0)
    body = list(tokens[:tokens.index("<E>")] if "<E>" in tokens else tokens)
    stream = TokenStream(list(body[:p_end]))
    orig_at = list(range(p_end))          # original index of every session token; None for the synthetic header
    after_header = None                   # original index of the first token after a synthetic header
    kv = [COLD] * (p_end - 1)
    out, prev, base_ms, k = {}, None, 0.0, 0

    def next_orig():
        return orig_at[-1] + 1 if orig_at[-1] is not None else after_header

    def last_committed_orig():
        i = orig_at[stream.committed_end - 1]
        return i if i is not None else after_header - 1

    while last_committed_orig() < len(body) - 1:
        c = payload(human_notes, block_start(k, t0), k)
        cut = None
        if needs_crop(stream.committed_end, cap, capacity):
            stream.rollback_to_committed()
            ce = stream.committed_end
            j = choose_cut(stream.tokens, ce, cap, capacity, keep_target)
            removed_t = stream.tokens[:j + 1].count("<T>")
            kept, kept_at, cut = stream.tokens[j + 1:ce], orig_at[j + 1:ce], orig_at[j]
            new_kv = [COLD] + ([kv[j]] + kv[j + 1:ce - 1] if kept else [])
            stream = TokenStream(list(HEADER) + kept)
            stream.request_full_reset()       # the runtime path: re-prefill the whole kept context
            orig_at, kv, after_header = [None, None] + kept_at, new_kv, cut + 1
            base_ms += removed_t * SEGMENT_MS
        elif prev is not None and c != prev:
            stream.rollback_to_committed()
            del orig_at[stream.committed_end:]
            del kv[stream.committed_end - 1:]
        out[k] = (list(kv[:stream.committed_end - 1]), base_ms, cut)
        h = horizon(k, t0) - base_ms
        while True:
            st = stream.grammar_state()
            cov = st.segment * SEGMENT_MS + st.last_onset if st.expect == "note" else -1
            if cov >= h + spec_ms or next_orig() >= len(body):
                break
            i = next_orig()
            kv.extend([c] * (len(stream.tokens) - len(kv)))     # feed the last token, then sample the next
            stream.append_sampled(body[i])
            orig_at.append(i)
        stream.commit_until(h)
        prev, k = c, k + 1
    return out
