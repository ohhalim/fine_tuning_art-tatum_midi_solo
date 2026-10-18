#!/usr/bin/env python3
"""Real-clock producer for the MIDI smoke (issue #1747, docs/experiments/REALTIME_MIDI_SMOKE.md).

The #1743 loop on a real clock: cold prebuffer (block 0's horizon + one speculative block, block 0's notes held),
then `start` with wall origin W = now + 1 s, then block k waits for W + k*B, rolls back on chord-input blocks, runs
fill_block (ready = commit + JSON line written), and continues speculatively. JSON lines go to stdout for the sender.
Per-block progress goes to a JSONL file as it happens, so the record survives a SIGKILL.

  rt_producer.py <aria repo> <checkpoint> <data json> <blocks> <progress jsonl> <report json>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aria_grammar_mask import GrammarMask, masked_pick  # noqa: E402
from aria_mlx_stream_loop import BEAT, BLOCK, CAP, CHORD_BLOCKS, SEED, fill_block  # noqa: E402
from aria_token_validity import check as token_check  # noqa: E402
from stream_contract import TokenStream  # noqa: E402

LEAD_S, CAPACITY = 1.0, 4096


def main() -> None:
    aria_repo, ckpt, data_path, n_blocks, progress_path, report_path = sys.argv[1:7]
    n_blocks = int(n_blocks)
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    import numpy as np
    import torch
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig

    out = sys.stdout
    progress = open(progress_path, "w")

    def emit(obj):
        out.write(json.dumps(obj) + "\n")
        out.flush()

    def note_progress(obj):
        progress.write(json.dumps(obj) + "\n")
        progress.flush()

    tok = AbsTokenizer()
    V = tok.vocab_size
    vocab = [tok.decode([i])[0] for i in range(V)]
    to_id = {t: i for i, t in enumerate(vocab)}
    prompt = tok.decode(json.load(open(data_path))["prompts"][0]["ids"])
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(V)
    model = TransformerLM(cfg)
    model.load_weights(ckpt, strict=True)
    mx.eval(model.parameters())
    model.eval()
    model.setup_cache(batch_size=1, max_seq_len=CAPACITY, dtype=mx.float32)

    def run(ids, start):
        if start < 0 or start + len(ids) > CAPACITY:
            raise ValueError(f"positions {start}..{start + len(ids) - 1} outside cache capacity {CAPACITY}")
        idxs = mx.array([ids], dtype=mx.int32)
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        o = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=start + len(ids) - 1)
        mx.eval(o)
        return np.array(o[0, -1].astype(mx.float32))

    stream = TokenStream(prompt)
    stream.request_full_reset()
    gm = GrammarMask(vocab)
    no_end = torch.tensor([t not in ("<D>", "<E>") for t in vocab])
    gen = torch.Generator().manual_seed(SEED)

    def next_logits():
        n = len(stream.tokens)
        start = stream.next_write_pos if stream.next_write_pos < n else n - 1
        lg = run([to_id[t] for t in stream.tokens[start:n]], start)
        stream.next_write_pos = start
        stream.mark_written(n)
        return lg

    def coverage():
        st = stream.grammar_state()
        return st.segment * 5000 + st.last_onset if st.expect == "note" else -1

    def sample_one():
        p = torch.softmax(torch.from_numpy(next_logits()), -1)
        st = stream.grammar_state()
        gm.sync(st, stream.onset_floor(st))
        ok = gm.allowed() & no_end
        a = p[ok].sum().item()
        if not (math.isfinite(a) and a > 0):
            raise RuntimeError(f"allowed mass {a}")
        pick, _ = masked_pick(p, ok, torch.empty(V).exponential_(1, generator=gen))
        stream.append_sampled(vocab[pick])

    def notes_json(notes):
        return [[n.note_id, n.pitch, n.velocity, int(round(n.on_ms)), int(round(n.off_ms))] for n in notes]

    t0 = coverage()
    snaps, blocks = [], []
    # cold prebuffer: block 0
    held = []
    tc = time.perf_counter()
    h0 = t0 + BLOCK + BEAT
    res0 = fill_block(sample_one, coverage, lambda: held.extend(stream.commit_until(h0)), h0, h0 + BLOCK, CAP)
    cold_s = time.perf_counter() - tc
    W = time.monotonic_ns() + int(LEAD_S * 1e9)
    emit({"type": "start", "music_origin_ms": t0, "wall_origin_ns": W})
    emit({"type": "notes", "block": 0, "notes": notes_json(held)})
    snaps.append(list(stream.tokens[:stream.committed_end]))
    note_progress({"block": 0, "cold_s": round(cold_s, 4), "tokens": res0["tokens"], "notes": len(held), "reached": res0["reached"]})
    for k in range(1, n_blocks):
        wake = W + int(k * BLOCK * 1e6)
        while time.monotonic_ns() < wake:
            time.sleep(max(0.0, min(0.002, (wake - time.monotonic_ns()) / 1e9)))
        late_start_ms = (time.monotonic_ns() - wake) / 1e6
        now = t0 + k * BLOCK
        h = now + BLOCK + BEAT
        dropped = 0
        if k in CHORD_BLOCKS:
            dropped = stream.rollback_to_committed()
            emit({"type": "rollback", "block": k})
        ready = {}

        def on_ready():
            notes = stream.commit_until(h)
            emit({"type": "notes", "block": k, "notes": notes_json(notes)})
            ready["ns"], ready["notes"] = time.monotonic_ns(), len(notes)

        res = fill_block(sample_one, coverage, on_ready, h, h + BLOCK, CAP)
        snaps.append(list(stream.tokens[:stream.committed_end]))
        deadline = W + int((k * BLOCK + BEAT) * 1e6)
        row = {"block": k, "chord_input": k in CHORD_BLOCKS, "dropped_tokens": dropped, "tokens": res["tokens"],
               "reached": res["reached"], "late_start_ms": round(late_start_ms, 3), "ready_s": round(res["ready_s"], 4),
               "ready_margin_ms": round((deadline - ready["ns"]) / 1e6, 3), "notes": ready["notes"], "block_s": round(res["block_s"], 4)}
        blocks.append(row)
        note_progress(row)
    emit({"type": "end"})
    v = token_check(prompt, stream.tokens[len(prompt):stream.committed_end])
    report = {"t0_ms": t0, "wall_origin_ns": W, "cold_prebuffer_s": round(cold_s, 4), "blocks": blocks,
              "checks": {"committed_prefix_never_changed": all(snaps[i] == snaps[i + 1][:len(snaps[i])] for i in range(len(snaps) - 1)),
                         "committed_tokens_format_valid": v["valid"] and v["tail"] is None,
                         "every_block_reached_horizon": res0["reached"] and all(b["reached"] for b in blocks)},
              "committed_token_sha256": hashlib.sha256(json.dumps([to_id[t] for t in stream.tokens[:stream.committed_end]]).encode()).hexdigest()}
    with open(report_path, "w") as f:
        json.dump(report, f)


if __name__ == "__main__":
    main()
