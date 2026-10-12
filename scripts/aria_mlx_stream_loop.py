#!/usr/bin/env python3
"""Local Aria MLX generation + TokenStream + Scheduler on a virtual clock (issue #1743,
docs/experiments/ARIA_MLX_STREAM_LOOP.md). A connection smoke: no MIDI port, no control tokens (a chord input is
only simulated as a rollback of the provisional tokens), no bias, reranking, CFG, quantisation, training, downloads.

Loop, 128 BPM: block B = half bar 937.5 ms, beat 468.75 ms. At block k (virtual time now = t0 + k*B):
  - if k is a chord-input block: rollback_to_committed() (provisional tokens dropped, grammar restored)
  - generate until the last complete note reaches the commit horizon H = now + B + beat ("needed" time, deadline:
    1 beat, because notes before now + beat were committed by the previous block), then keep generating one more
    block speculatively (until H + B); at most CAP tokens per block
  - commit_until(H), schedule the committed notes, tick the scheduler to now
Sampling: the #1726 sampler (one Exp(1) noise vector per step, temperature 1, no min_p) restricted by the format
mask (aria_grammar_mask). <D> and <E> are also masked out: a live partner keeps playing (operating condition).

  aria_mlx_stream_loop.py <aria repo> <checkpoint> <data json> <report json>
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from aria_grammar_mask import GrammarMask, masked_pick  # noqa: E402
from aria_token_validity import check as token_check  # noqa: E402
from stream_contract import Scheduler, TokenStream  # noqa: E402

BEAT, BLOCK = 468.75, 937.5
N_BLOCKS, CAP, SEED, CAPACITY = 48, 64, 1, 4096
CHORD_BLOCKS = (8, 16, 24, 32, 40)          # every 4 bars


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def thermal_level():
    try:
        return int(subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                                  capture_output=True, text=True, timeout=5).stdout.split()[-1])
    except Exception:
        return None


def nearest_rank(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def main() -> None:
    aria_repo, ckpt, data_path, report_path = sys.argv[1:5]
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    import numpy as np
    import torch
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig

    log = {"checks": {}, "thermal_start": thermal_level(), "result": None,
           "params": {"beat_ms": BEAT, "block_ms": BLOCK, "blocks": N_BLOCKS, "cap_tokens_per_block": CAP, "seed": SEED,
                      "chord_blocks": CHORD_BLOCKS, "capacity": CAPACITY}}
    tok = AbsTokenizer()
    V = tok.vocab_size
    vocab = [tok.decode([i])[0] for i in range(V)]
    prompt_ids = json.load(open(data_path))["prompts"][0]["ids"]
    prompt = tok.decode(prompt_ids)
    log["provenance"] = {"checkpoint_sha256": file_sha(ckpt), "data_sha256": file_sha(data_path), "mlx": getattr(mx, "__version__", None)}

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
        out = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=start + len(ids) - 1)
        mx.eval(out)
        return np.array(out[0, -1].astype(mx.float32))

    stream = TokenStream(prompt)
    stream.request_full_reset()                            # the KV is empty: the prompt must be prefilled first
    to_id = {t: i for i, t in enumerate(vocab)}
    gm = GrammarMask(vocab)
    no_end = torch.tensor([t not in ("<D>", "<E>") for t in vocab])
    gen = torch.Generator().manual_seed(SEED)
    sched = Scheduler()

    def next_logits():
        """Logits for the next position; writes every unwritten token (rewrites the last one if all are written)."""
        n = len(stream.tokens)
        start = stream.next_write_pos if stream.next_write_pos < n else n - 1
        ids = [to_id[t] for t in stream.tokens[start:n]]
        lg = run(ids, start)
        stream.next_write_pos = start                      # rewriting the last token is idempotent
        stream.mark_written(n)
        return lg

    def coverage():
        st = stream.grammar_state()
        return st.segment * 5000 + st.last_onset if st.expect == "note" else -1

    def sample_one():
        lg = next_logits()
        p = torch.softmax(torch.from_numpy(lg), -1)
        st = stream.grammar_state()
        gm.expect, gm.last = st.expect, st.last_onset
        ok = gm.allowed() & no_end
        a = p[ok].sum().item()
        if not (math.isfinite(a) and a > 0):
            raise RuntimeError(f"allowed mass {a}")
        q = torch.empty(V).exponential_(1, generator=gen)
        pick, free = masked_pick(p, ok, q)
        stream.append_sampled(vocab[pick])
        return pick != free

    t0 = coverage()
    log["t0_ms"] = t0
    blocks, committed_snap, mask_changes = [], [], 0
    stopped_reason = None
    try:
        for k in range(N_BLOCKS):
            now = t0 + k * BLOCK
            h = now + BLOCK + BEAT
            dropped = stream.rollback_to_committed() if k in CHORD_BLOCKS else 0
            t = time.perf_counter()
            n_tok, t_need, reached = 0, None, False
            while n_tok < CAP:
                c = coverage()
                if c >= h and not reached:
                    t_need, reached = time.perf_counter() - t, True
                if c >= h + BLOCK:
                    break
                mask_changes += sample_one()
                n_tok += 1
            if t_need is None:
                t_need = time.perf_counter() - t
            t_block = time.perf_counter() - t
            notes = stream.commit_until(h)
            sched.schedule(notes)
            sched.tick(now)
            committed_snap.append(list(stream.tokens[:stream.committed_end]))
            blocks.append({"block": k, "now_ms": now, "horizon_ms": h, "chord_input": k in CHORD_BLOCKS, "dropped_tokens": dropped,
                           "tokens_generated": n_tok, "reached_horizon": reached,
                           "time_to_horizon_s": round(t_need, 4), "block_s": round(t_block, 4), "notes_committed": len(notes),
                           "kv_valid_length": stream.kv_valid_length, "tokens": len(stream.tokens)})
    except (ValueError, RuntimeError) as e:
        stopped_reason = f"{type(e).__name__}: {e}"[:300]
    end_ms = t0 + len(blocks) * BLOCK + 10_000
    for x in range(int(t0), int(end_ms) + 1, 50):
        sched.tick(x)
    sounding_before_stop = dict(sched.sounding)
    sched.stop(end_ms)

    # checks
    log["stopped_reason"] = stopped_reason
    log["checks"]["ran_all_blocks"] = {"ok": len(blocks) == N_BLOCKS and stopped_reason is None, "blocks": len(blocks)}
    log["checks"]["every_block_reached_horizon"] = {"ok": all(b["reached_horizon"] for b in blocks),
                                                    "missed": [b["block"] for b in blocks if not b["reached_horizon"]]}
    log["checks"]["committed_prefix_never_changed"] = {"ok": all(committed_snap[i] == committed_snap[i + 1][:len(committed_snap[i])]
                                                                 for i in range(len(committed_snap) - 1))}
    log["checks"]["kv_tracks_stream"] = {"ok": all(b["tokens"] - b["kv_valid_length"] in (0, 1) for b in blocks)}
    v = token_check(prompt, stream.tokens[len(prompt):stream.committed_end])
    log["checks"]["committed_tokens_format_valid"] = {"ok": v["valid"] and v["tail"] is None, "first_error": v["first_error"]}
    ons = [x for x in sched.sent if x[1] == "on"]
    offs = [x for x in sched.sent if x[1] == "off"]
    on_ids = [x[2] for x in ons]
    off_ids = [x[2] for x in offs]
    log["checks"]["each_note_on_has_one_off"] = {"ok": len(set(on_ids)) == len(on_ids) and sorted(on_ids) == sorted(off_ids),
                                                 "note_on": len(ons), "note_off": len(offs)}
    log["checks"]["nothing_sounding_after_stop"] = {"ok": sched.sounding == {}, "sounding_before_stop": len(sounding_before_stop)}

    warm = [b for b in blocks if b["block"] > 0 and not b["chord_input"]]
    chord = [b for b in blocks if b["chord_input"]]

    def dist(xs):
        return {"n": len(xs), "median_s": round(statistics.median(xs), 4), "p99_nearest_rank_s": round(nearest_rank(xs, 0.99), 4),
                "max_s": round(max(xs), 4)} if xs else None

    log["latency"] = {
        "cold_first_block": {"time_to_horizon_s": blocks[0]["time_to_horizon_s"], "block_s": blocks[0]["block_s"]} if blocks else None,
        "warm_time_to_horizon": dist([b["time_to_horizon_s"] for b in warm]),
        "warm_block_incl_speculative": dist([b["block_s"] for b in warm]),
        "chord_input_time_to_horizon": dist([b["time_to_horizon_s"] for b in chord]),
        "deadline_misses_vs_beat": sum(1 for b in blocks[1:] if b["time_to_horizon_s"] > BEAT / 1000),
        "deadline_misses_vs_block": sum(1 for b in blocks[1:] if b["time_to_horizon_s"] > BLOCK / 1000),
        "note": "p99 by nearest rank; with n < 100 it equals one of the largest observations. Virtual clock: generation "
                "time is measured, music time is simulated. Not input-to-sound latency."}
    log["blocks"] = blocks
    log["totals"] = {"tokens": len(stream.tokens) - len(prompt), "committed_tokens": stream.committed_end - len(prompt),
                     "notes_sent": len(ons), "mask_changed_steps": mask_changes}
    log["thermal_end"] = thermal_level()
    log["result"] = "loop_done" if stopped_reason is None else "stopped"
    with open(report_path, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(json.dumps({"result": log["result"], "checks": {k: v["ok"] for k, v in log["checks"].items()},
                      "latency": {k: v for k, v in log["latency"].items() if k != "note"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
