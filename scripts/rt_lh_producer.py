#!/usr/bin/env python3
"""Opt-in real-clock producer with live left-hand input (issue #1797, docs/experiments/LH_INPUT_SMOKE.md).

A receiver thread opens the left-hand input port and only queues (monotonic ns, message). The main loop, at each
block start s_k: drains the queue into LHState, takes the #1773 window snapshot (onset < s_k, held cut at s_k), builds
the b context over the committed right hand, reuses the KV up to P_reuse = min(common prefix, KV valid end) (0 after a
crop, last token recomputed when S = 0), then fill_block. If a block's commit/schedule happens after its deadline
(W + k*B + one beat) it sends {"type": "abort", "reason": "deadline_miss"} instead of the notes and stops.

  rt_lh_producer.py <aria repo> <ckpt> <data json> <blocks> <lh port> <origin json> <progress jsonl> <report json>
      [--capacity N] [--keep-target N] [--slow-block K --slow-ms MS] [--check-blocks 4,8,12]
"""
from __future__ import annotations

import json
import math
import os
import queue
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_grammar_mask import GrammarMask, masked_pick  # noqa: E402
from aria_mlx_stream_loop import BEAT, BLOCK, CAP, SEED, fill_block  # noqa: E402
from aria_token_validity import check as token_check  # noqa: E402
from crop_contract import CAPACITY, KEEP_TARGET  # noqa: E402
from lh_cache_replay import reuse_start  # noqa: E402
from lh_context import W_MS, Human, merge, window_notes  # noqa: E402
from lh_kv_reuse import common_prefix, load_mlx  # noqa: E402
from rt_lh_input import LHState, SessionCrop, block_problem  # noqa: E402
from stream_contract import TokenStream  # noqa: E402

LEAD_S = 1.0


class Abort(Exception):
    """A block that may not be committed (deadline_miss, horizon_underfill); the run stops without sending it."""


def opt(argv, name, cast, default):
    return cast(argv[argv.index(name) + 1]) if name in argv else default


def main() -> None:
    import mido
    import numpy as np
    import torch
    aria_repo, ckpt, data_path, n_blocks, lh_port, origin_path, progress_path, report_path = sys.argv[1:9]
    rest = sys.argv[9:]
    n_blocks = int(n_blocks)
    capacity = opt(rest, "--capacity", int, CAPACITY)
    keep_target = opt(rest, "--keep-target", int, KEEP_TARGET)
    slow_block, slow_ms = opt(rest, "--slow-block", int, -1), opt(rest, "--slow-ms", float, 0.0)
    check_blocks = set(int(x) for x in opt(rest, "--check-blocks", str, "").split(",") if x)

    out, progress = sys.stdout, open(progress_path, "w")

    def emit(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()
        return time.monotonic_ns()

    def note_progress(obj):
        progress.write(json.dumps(obj) + "\n")
        progress.flush()

    tok, run = load_mlx(aria_repo, ckpt)
    V_SIZE = tok.vocab_size
    vocab = [tok.decode([i])[0] for i in range(V_SIZE)]
    to_id = {t: i for i, t in enumerate(vocab)}
    prompt = tok.decode(json.load(open(data_path))["prompts"][0]["ids"])
    stream = TokenStream(prompt)
    gm = GrammarMask(vocab)
    no_end = torch.tensor([t not in ("<D>", "<E>") for t in vocab])
    gen = torch.Generator().manual_seed(SEED)

    # model input actually in the KV (positions 0..V-1) and the tokens after it not written yet
    model = {"seq": [], "V": 0, "last": None}

    def write_pending():
        seq, V = model["seq"], model["V"]
        if V < len(seq):
            model["last"] = run([to_id[t] for t in seq[V:]], V)[-1]
            model["V"] = len(seq)
        return model["last"]

    samples = {"n": 0}                       # every sample, so an aborted block still has its count

    def sample_one():
        samples["n"] += 1
        p = torch.softmax(torch.from_numpy(write_pending()), -1)
        st = stream.grammar_state()
        gm.expect, gm.last = st.expect, st.last_onset
        ok = gm.allowed() & no_end
        a = p[ok].sum().item()
        if not (math.isfinite(a) and a > 0):
            raise RuntimeError(f"allowed mass {a}")
        pick, _ = masked_pick(p, ok, torch.empty(V_SIZE).exponential_(1, generator=gen))
        t = vocab[pick]
        stream.append_sampled(t)
        model["seq"].append(t)

    def coverage():
        st = stream.grammar_state()
        return st.segment * 5000 + st.last_onset if st.expect == "note" else -1

    def notes_json(notes):
        return [[n.note_id, n.pitch, n.velocity, int(round(n.on_ms)), int(round(n.off_ms))] for n in notes]

    lhq = queue.Queue()
    port = mido.open_input(lh_port, callback=lambda msg: lhq.put((time.monotonic_ns(), msg)))
    lh = LHState()
    received = []
    crop = SessionCrop(CAP, capacity, keep_target)
    t0 = coverage()
    origin = {}
    checks = []

    def drain():
        while True:
            try:
                ns, msg = lhq.get_nowait()
            except queue.Empty:
                break
            m_ms = t0 + (ns - origin["W"]) / 1e6 if origin else None
            rec = {"ns": ns, "type": msg.type, "m_ms": m_ms}
            if m_ms is None:                                  # before the session origin exists: recorded, not used
                rec["before_origin"] = True
                received.append(rec)
                continue
            if msg.type in ("note_on", "note_off"):
                rec.update(pitch=msg.note, velocity=msg.velocity)
                lh.feed(msg.type, m_ms, msg.note, msg.velocity)
            elif msg.type == "control_change":
                rec.update(control=msg.control)
                lh.feed(msg.type, m_ms, control=msg.control)
            else:
                lh.feed(msg.type, m_ms)
            received.append(rec)

    def prepare(k, s_k):
        """Snapshot, b context, KV reuse. Returns timing marks and reuse numbers."""
        marks = {"snapshot_ns": time.monotonic_ns()}
        drain()
        marks["input_received"] = len(received)
        dropped = stream.rollback_to_committed()
        if dropped:
            del model["seq"][len(model["seq"]) - dropped:]
            model["V"] = min(model["V"], len(model["seq"]))
        lh_notes = [] if k == 0 else window_notes(Human(lh.human()), s_k, max(s_k - W_MS, crop.base_ms))
        changed = crop.maybe_crop(stream.tokens, stream.committed_end, 3 * len(lh_notes))
        if changed and k > 0:
            lh_notes = window_notes(Human(lh.human()), s_k, max(s_k - W_MS, crop.base_ms))
        N, tags = merge(crop.session(stream.tokens, stream.committed_end), crop.base_ms, lh_notes)
        marks["context_ready_ns"] = time.monotonic_ns()
        if k == slow_block:
            time.sleep(slow_ms / 1000)
        V = model["V"]
        lcp = common_prefix(model["seq"][:V], N)
        start = reuse_start(V, lcp, len(N), changed or k == 0)
        if k in check_blocks:
            checks.append({"block": k, "prev_seq": list(model["seq"]), "V": V, "N": list(N), "start": start})
        model["last"] = run([to_id[t] for t in N[start:]], start, kv_end=len(N) - 1)[-1]
        model["seq"], model["V"] = list(N), len(N)
        marks["suffix_ready_ns"] = time.monotonic_ns()
        held = [n for n in lh_notes if n.held]
        return marks, {"crop": changed, "base_ms": crop.base_ms, "P_reuse": start, "V": V, "S": len(N) - start, "len_N": len(N),
                       "lh_notes": len(lh_notes), "lh_held": len(held), "lh_max_on": max((n.raw_on for n in lh_notes), default=None),
                       "lh_max_end": max((n.raw_off for n in lh_notes), default=None), "rollback_tokens": dropped}

    blocks, abort, snaps, reached = [], None, [], []
    # block 0 (cold): no left hand, no deadline, but it must reach its horizon (#1802)
    tc = time.perf_counter()
    marks0, info0 = prepare(0, t0)
    h0 = t0 + BLOCK + BEAT
    held0, cold = [], {}

    def on_ready0():
        cold["problem"] = block_problem(time.monotonic_ns(), None, coverage(), h0)
        if cold["problem"]:
            raise Abort()
        held0.extend(stream.commit_until(h0))

    n0 = samples["n"]
    try:
        res0 = fill_block(sample_one, coverage, on_ready0, h0, h0 + BLOCK, CAP)
    except Abort:
        res0 = {"reached": False}
    W = time.monotonic_ns() + int(LEAD_S * 1e9)
    origin.update(W=W)
    with open(origin_path, "w") as f:
        json.dump({"music_origin_ms": t0, "wall_origin_ns": W}, f)
    emit({"type": "start", "music_origin_ms": t0, "wall_origin_ns": W})
    if cold.get("problem"):
        abort = {"reason": cold["problem"], "block": 0}
        emit({"type": "abort", **abort})
    else:
        emit({"type": "notes", "block": 0, "notes": notes_json(held0)})
    snaps.append(list(stream.tokens[:stream.committed_end]))
    reached.append(res0["reached"])
    note_progress({"block": 0, "cold_s": round(time.perf_counter() - tc, 4), **info0, "tokens": samples["n"] - n0,
                   "reached": res0["reached"], "problem": cold.get("problem")})
    try:
        for k in range(1, n_blocks if abort is None else 1):
            wake = W + int(k * BLOCK * 1e6)
            while time.monotonic_ns() < wake:
                time.sleep(max(0.0, min(0.002, (wake - time.monotonic_ns()) / 1e9)))
            s_k = t0 + k * BLOCK
            h = s_k + BLOCK + BEAT
            deadline = W + int((k * BLOCK + BEAT) * 1e6)
            marks, info = prepare(k, s_k)
            ready = {}

            def on_ready():
                ready["ns"] = time.monotonic_ns()
                ready["coverage_ms"] = coverage()
                ready["problem"] = block_problem(ready["ns"], deadline, ready["coverage_ms"], h)
                if ready["problem"]:
                    raise Abort()
                notes = stream.commit_until(h)
                ready["notes"] = len(notes)
                ready["enqueued_ns"] = emit({"type": "notes", "block": k, "notes": notes_json(notes)})

            n_start = samples["n"]
            try:
                res = fill_block(sample_one, coverage, on_ready, h, h + BLOCK, CAP)
            except Abort:
                res = {"reached": None}
            row = {"block": k, **info, "late_start_ms": round((marks["snapshot_ns"] - wake) / 1e6, 3),
                   "context_ms": round((marks["context_ready_ns"] - marks["snapshot_ns"]) / 1e6, 3),
                   "suffix_ms": round((marks["suffix_ready_ns"] - marks["context_ready_ns"]) / 1e6, 3),
                   "ready_margin_ms": round((deadline - ready["ns"]) / 1e6, 3), "notes": ready.get("notes"),
                   "ready_s": round((ready["ns"] - marks["suffix_ready_ns"]) / 1e9, 4),   # suffix ready -> horizon covered (monotonic)
                   "enqueued_ns": ready.get("enqueued_ns"), "deadline_ns": deadline, "tokens": samples["n"] - n_start,
                   "reached": res["reached"], "problem": ready.get("problem"), "coverage_ms": ready.get("coverage_ms"),
                   "horizon_ms": h, "input_received": marks["input_received"],
                   "marks_ns": {"wake": wake, "snapshot": marks["snapshot_ns"], "context_ready": marks["context_ready_ns"],
                                "suffix_ready": marks["suffix_ready_ns"], "ready": ready.get("ns"), "enqueued": ready.get("enqueued_ns")}}
            blocks.append(row)
            note_progress(row)
            snaps.append(list(stream.tokens[:stream.committed_end]))
            if res["reached"] is not None:
                reached.append(res["reached"])
            if ready.get("problem"):
                abort = {"reason": ready["problem"], "block": k}
                emit({"type": "abort", **abort})
                break
        else:
            if abort is None:
                emit({"type": "end"})
    finally:
        final_drain_ns = time.monotonic_ns()   # messages sent after this are outside B1 (doc 9, deviation 2)
        drain()
        port.close()
    # post-run equivalence sample (outside the timed loop)
    eq = []
    for c in checks:
        prev, V, N, start = c["prev_seq"], c["V"], c["N"], c["start"]
        if V:
            run([to_id[t] for t in prev[:V]], 0)
        cached = run([to_id[t] for t in N[start:]], start, kv_end=len(N) - 1)
        full = run([to_id[t] for t in N], 0)[start:]
        top1 = int((cached.argmax(-1) == full.argmax(-1)).sum())
        diff = float(np.abs(cached[-1] - full[-1]).max())
        eq.append({"block": c["block"], "start": start, "S": len(N) - start, "top1_equal": top1, "positions": len(full),
                   "max_abs_last": diff, "ok": top1 == len(full) and diff <= 1e-3})
    import hashlib
    v = token_check(prompt, stream.tokens[len(prompt):stream.committed_end])
    report = {"issue": 1797, "t0_ms": t0, "wall_origin_ns": W, "capacity": capacity, "keep_target": keep_target,
              "checks": {"committed_prefix_never_changed": all(snaps[i] == snaps[i + 1][:len(snaps[i])] for i in range(len(snaps) - 1)),
                         "committed_tokens_format_valid": v["valid"] and v["tail"] is None,
                         "every_block_reached_horizon": all(reached)},
              "committed_token_sha256": hashlib.sha256(json.dumps([to_id[t] for t in stream.tokens[:stream.committed_end]]).encode()).hexdigest(),
              "slow_block": slow_block, "slow_ms": slow_ms, "blocks": blocks, "abort": abort, "crops": crop.crops,
              "lh_counts": lh.counts, "received": received, "final_drain_ns": final_drain_ns, "equivalence_sample": eq}
    with open(report_path, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
