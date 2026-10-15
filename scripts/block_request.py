"""Block-request training input contract (issue #1756, docs/experiments/LH_BLOCK_REQUEST_INPUT.md). Pure Python.

One training example is one runtime block request R_k (scripts/rt_producer.py under the proposed block contract of
scripts/block_contract.py, rollback rule included). Token streams are decoded Aria tokens as in stream_contract.

  request     k, s_k = block_start(k, t0) and the commit window [h_(k-1), h_k), h_k = horizon(k, t0), h_(-1) = t0
  context     the model's adopted history: the tokens committed before R_k, i.e. TokenStream.commit_until(h_(k-1)) on
              the target stream, starting from the primer (every token before the first note with onset >= t0)
  continuation
              "window": the tokens R_k commits; then "boundary": the uncommitted <T> tokens and the first note after
              the window. The boundary says when the next note comes, so an empty window still has a label. <E> is
              never a continuation token; the last request of a song may have no boundary
  condition   payload(s_k), from the human notes observed up to s_k only. Every prediction position of R_k (from the
              last context position p0 = ctx_end - 1 to the end) gets this one value. The target onsets never choose
              it: choosing a snapshot by the onset of the note being predicted is a future-time leak (#1756 review Q2)
  context positions q < p0 carry the payload of the request that committed token q + 1; primer positions carry the
              cold payload. Under the rollback rule this is what the KV holds when R_k starts: a request whose payload
              changed recomputes from its own p0 on (the earliest input position whose condition changed, review Q3)

Payload (review Q1): g3 sounding at s_k as a 12-PC multihot, plus `available`. The cold block 0 and the primer are
"not observable" (available 0, the prebuffer runs before any human input is read); a warm block with nothing
sounding is "observed, empty" (available 1, all zeros). No age, velocity, octave or release after s_k.
Human notes never enter the token stream: model history and human history are separate inputs.
"""
from __future__ import annotations

from block_contract import block_start, horizon
from lh_role_proxy import assign_g3, online_condition
from stream_contract import SEGMENT_MS, TokenStream

COLD = ((0,) * 12, 0)


def _kind(t):
    return t[0] if isinstance(t, tuple) else t


def primer_end(tokens, t0):
    """Index of the first generated token: the first note with onset >= t0, or a <T> whose boundary is >= t0."""
    seg = 0
    for i, t in enumerate(tokens):
        k = _kind(t)
        if k == "<T>":
            if (seg + 1) * SEGMENT_MS >= t0:
                return i
            seg += 1
        elif k == "piano":
            if seg * SEGMENT_MS + tokens[i + 1][1] >= t0:
                return i
        elif k == "<E>":
            return i
    return len(tokens)


def request_table(tokens, t0):
    """One row per request until every token before <E> is committed (a stream cut after a <T> ends with that <T>).
    Rows hold token indexes (end-exclusive):
    ctx_end (= the context length), win_end (the window ends), cont_end (the boundary ends)."""
    p_end = primer_end(tokens, t0)
    stream = TokenStream(list(tokens[:p_end]))
    rest = list(tokens[p_end:])
    rest = rest[:rest.index("<E>")] if "<E>" in rest else rest
    st = stream.grammar_state()
    for t in rest:                       # validate the grammar once; append_sampled would re-walk it for every token
        st = st.advance(t)
    stream.tokens.extend(rest)
    body = stream.tokens
    rows, ctx_end, k = [], p_end, 0
    while ctx_end < len(body):
        h = horizon(k, t0)
        stream.commit_until(h)
        win_end = j = stream.committed_end
        while j < len(body) and body[j] == "<T>":
            j += 1
        boundary = j < len(body) and _kind(body[j]) == "piano"
        rows.append({"k": k, "s": block_start(k, t0), "h_prev": t0 if k == 0 else horizon(k - 1, t0), "h": h,
                     "ctx_end": ctx_end, "win_end": win_end, "cont_end": j + 3 if boundary else win_end, "boundary": boundary})
        ctx_end, k = win_end, k + 1
    return rows


def committing_request(rows, n_tokens):
    """For every token index: the request whose window committed it, -1 for the primer, None if never committed."""
    out = [None] * n_tokens
    for i in range(min(n_tokens, rows[0]["ctx_end"] if rows else n_tokens)):
        out[i] = -1
    for r in rows:
        for i in range(r["ctx_end"], min(r["win_end"], n_tokens)):
            out[i] = r["k"]
    return out


def position_requests(rows, k):
    """For request k: which request's payload each input position 0 .. cont_end - 2 carries (-1 = cold/primer)."""
    r = rows[k]
    p0 = r["ctx_end"] - 1
    owner = committing_request(rows, r["cont_end"])
    return [owner[q + 1] for q in range(p0)] + [k] * (r["cont_end"] - 1 - p0)


def payload(human_notes, s, k):
    """(12-PC multihot, available) the model gets for request k, from human notes observed up to s only."""
    if k <= 0:
        return COLD
    roles = assign_g3(human_notes)
    pcs = {n.pitch % 12 for n in online_condition(human_notes, roles, s)}
    return tuple(int(c in pcs) for c in range(12)), 1


def loss_class(rows, k):
    """For request k: (first label index, ["window"|"boundary"] per label token). Labels are tokens[ctx_end:cont_end];
    the label at index i is predicted by input position i - 1."""
    r = rows[k]
    return r["ctx_end"], ["window"] * (r["win_end"] - r["ctx_end"]) + ["boundary"] * (r["cont_end"] - r["win_end"])


def coverage(stream):
    """The producer's coverage (scripts/rt_producer.py): the onset of the last complete note, or the segment start
    right after a <T>; -1 inside a note."""
    st = stream.grammar_state()
    return st.segment * SEGMENT_MS + st.last_onset if st.expect == "note" else -1


def simulate_kv(tokens, t0, human_notes, spec_ms=None):
    """Second, sequential implementation (for the checks). Replays the producer's requests with teacher-forced tokens
    and the rollback rule: at each block start, if the payload changed, drop the provisional tokens and rewrite the KV
    from the earliest position whose condition changed (committed_end - 1). Returns {k: payload held by each KV
    position q < committed_end - 1 when request k starts}. A position is written when its token is fed, i.e. before
    the next token is sampled, so after sampling n tokens positions 0 .. n - 2 are in the KV."""
    from block_contract import BLOCK
    spec_ms = BLOCK if spec_ms is None else spec_ms
    p_end = primer_end(tokens, t0)
    body = [t for t in tokens[:next((i for i, t in enumerate(tokens) if t == "<E>"), len(tokens))]]
    stream = TokenStream(list(body[:p_end]))
    kv = [COLD] * (p_end - 1)             # the primer is prefilled; its last position is fed by request 0
    out, prev, k = {}, None, 0
    while stream.committed_end < len(body):
        c = payload(human_notes, block_start(k, t0), k)
        if prev is not None and c != prev:
            stream.rollback_to_committed()
            del kv[stream.committed_end - 1:]
        out[k] = list(kv[:stream.committed_end - 1])
        h = horizon(k, t0)
        while coverage(stream) < h + spec_ms and len(stream.tokens) < len(body):
            kv.extend([c] * (len(stream.tokens) - len(kv)))         # feed the last token, then sample the next
            stream.append_sampled(body[len(stream.tokens)])
        stream.commit_until(h)
        prev, k = c, k + 1
    return out
