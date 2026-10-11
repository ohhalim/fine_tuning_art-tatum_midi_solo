#!/usr/bin/env python3
"""Local Aria MLX: incremental KV cache vs full prefill, rollback, reset, and batch-1 latency (issue #1737,
docs/experiments/ARIA_MLX_KV_SMOKE.md). An infrastructure check only: no bias, CFG, quantisation, reranking,
training or downloads.

Token sequences are fixed and already saved: the #1731 masked generations (prompt + 512 new tokens), whose
base streams equal the #1726 unmasked ones. Weights and KV cache are float32, as aria/inference/sample_mlx.py.

Checks (fixed before the run):
  E1 equivalence   full prefill of seq A vs prefill(prompt) + one-token decode steps over the new tokens:
                   top-1 equal at every compared position and max |logit diff| <= 1e-3
  E2 rollback      after E1, write seq B from the first position where A and B differ (same cache, positions
                   overwritten, max_kv_pos limits what is visible) vs a fresh full prefill of B: same criteria
  E3 reset         reuse the cache from E2 for a fresh prefill of A from position 0 vs E1's full prefill: same
  E4 greedy        128 argmax tokens after the prompt, incremental KV vs full re-prefill at every step:
                   identical tokens and identical detokenized notes (pitch, onset, end, velocity)
  L  latency       cold first prefill; warm prefill of 128/256/512/1024 tokens (median of 5); one-token decode at
                   context 256/512/1024 (median of 64); a 24-token decode block after a 512-token prefix. Report only
  R  reference     MLX full-prefill logits vs the PyTorch MPS model on the same sequence. Report only

  aria_mlx_kv_smoke.py <aria repo> <checkpoint> <masked generations json> <data json> <report json>
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time

LOGIT_TOL, GREEDY_TOKENS = 1e-3, 128


def thermal_level():
    try:
        return int(subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                                  capture_output=True, text=True, timeout=5).stdout.split()[-1])
    except Exception:
        return None


def main() -> None:
    aria_repo, ckpt, gens_path, data_path, report_path = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    import numpy as np
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig

    log = {"checks": {}, "latency": {}, "reference": {}, "thermal_start": thermal_level(), "result": None}
    tok = AbsTokenizer()
    gens = json.load(open(gens_path))["completed"]
    prompts = json.load(open(data_path))["prompts"]
    a_gen = next(g for g in gens if g["prompt_index"] == 0 and g["seed"] == 1 and not g["adapter"])
    b_gen = next(g for g in gens if g["prompt_index"] == 0 and g["seed"] == 2 and not g["adapter"])
    prompt = prompts[0]["ids"]
    seq_a, seq_b = prompt + a_gen["ids"], prompt + b_gen["ids"]
    P = len(prompt)
    div = next(i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y)
    log["sequences"] = {"prompt_tokens": P, "len_a": len(seq_a), "len_b": len(seq_b), "first_difference": div}

    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_weights(ckpt, strict=True)
    mx.eval(model.parameters())
    model.eval()
    L = max(len(seq_a), len(seq_b)) + GREEDY_TOKENS + 8

    def run(ids, start):
        """Write ids at positions start.. and return logits for those positions (float32 numpy)."""
        idxs = mx.array([ids], dtype=mx.int32)
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        out = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=start + len(ids) - 1)
        mx.eval(out)
        return np.array(out[0].astype(mx.float32))

    def fresh():
        model.setup_cache(batch_size=1, max_seq_len=L, dtype=mx.float32)

    def compare(name, got, ref, positions):
        d = np.abs(got - ref)
        top_ok = (got.argmax(-1) == ref.argmax(-1))
        ok = bool(top_ok.all()) and float(d.max()) <= LOGIT_TOL
        log["checks"][name] = {"ok": ok, "positions": positions, "max_abs_logit_diff": float(d.max()),
                               "mean_abs_logit_diff": float(d.mean()), "top1_agree": int(top_ok.sum()), "of": int(top_ok.size)}

    # cold first prefill
    fresh()
    t = time.perf_counter()
    full_a = run(seq_a, 0)
    log["latency"]["cold_first_prefill_s"] = {"tokens": len(seq_a), "s": round(time.perf_counter() - t, 4)}

    # E1: prefill prompt, then one token at a time
    fresh()
    inc = [run(prompt, 0)[-1]]
    for i in range(P, len(seq_a) - 1):
        inc.append(run([seq_a[i]], i)[0])
    compare("E1_incremental_equals_full", np.stack(inc), full_a[P - 1:len(seq_a) - 1], f"{P - 1}..{len(seq_a) - 2}")

    # E2: rollback to the first difference and write seq B over the same cache
    roll = [run(seq_b[div - 1:div], div - 1)[0]] if div > 0 else []
    for i in range(div, len(seq_b) - 1):
        roll.append(run([seq_b[i]], i)[0])
    fresh()
    full_b = run(seq_b, 0)
    compare("E2_rollback_equals_fresh_full", np.stack(roll), full_b[div - 1:len(seq_b) - 1], f"{div - 1}..{len(seq_b) - 2}")

    # E3: reuse the (dirty) cache for a prefill of A from position 0
    compare("E3_reset_reuse_equals_full", run(seq_a, 0), full_a, f"0..{len(seq_a) - 1}")

    # E4: greedy, incremental vs full re-prefill
    fresh()
    g_inc, last = [], run(prompt, 0)[-1]
    for i in range(GREEDY_TOKENS):
        nxt = int(last.argmax())
        g_inc.append(nxt)
        last = run([nxt], P + i)[0]
    g_full = []
    for i in range(GREEDY_TOKENS):
        fresh()
        g_full.append(int(run(prompt + g_full, 0)[-1].argmax()))

    def notes(ids):
        md = tok.detokenize(tok.decode(prompt + ids))
        return [(n["data"]["pitch"], n["data"]["start"], n["data"]["end"], n["data"]["velocity"]) for n in md.note_msgs]

    first_diff = next((i for i, (x, y) in enumerate(zip(g_inc, g_full)) if x != y), None)
    log["checks"]["E4_greedy_tokens_and_notes_identical"] = {
        "ok": g_inc == g_full and notes(g_inc) == notes(g_full), "tokens": GREEDY_TOKENS,
        "first_token_difference": first_diff, "notes": len(notes(g_inc))}

    # latency, warm
    def timed(fn, n):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t)
        return {"median_s": round(statistics.median(ts), 5), "max_s": round(max(ts), 5), "n": n}

    long_seq = (seq_a + seq_b + seq_a)[:1024]
    for n in (128, 256, 512, 1024):
        fresh()
        log["latency"][f"warm_prefill_{n}"] = timed(lambda: run(long_seq[:n], 0), 5)
    for ctx in (256, 512, 1024):
        fresh()
        run(long_seq[:ctx - 1], 0)
        log["latency"][f"warm_decode_one_at_{ctx}"] = timed(lambda: run([long_seq[ctx - 1]], ctx - 1), 64)
    fresh()
    run(long_seq[:512], 0)

    def block():
        for i in range(24):
            run([long_seq[512 + i]], 512 + i)
    log["latency"]["warm_decode_block_24_after_512"] = timed(block, 5)
    log["latency"]["note"] = "128 BPM: 1 beat 0.469 s, half bar 0.9375 s. Previous runtime: PyTorch MPS full re-prefill 0.674 s per block"

    # reference: PyTorch MPS on the same sequence (report only)
    try:
        import torch
        from safetensors.torch import load_file
        from aria.model import TransformerLM as TorchLM
        tcfg = ModelConfig(**load_model_config(name="medium"))
        tcfg.set_vocab_size(tok.vocab_size)
        tm = TorchLM(tcfg)
        tm.load_state_dict(load_file(ckpt), strict=True)
        tm.to("mps").eval()
        tm.model.freqs_cis = None
        with torch.no_grad():
            ref = tm(torch.tensor([seq_a], device="mps")).float()[0].cpu().numpy()
        d = np.abs(ref - full_a)
        log["reference"] = {"max_abs_logit_diff_mlx_vs_mps": float(d.max()), "mean_abs_logit_diff": float(d.mean()),
                            "top1_agree": int((ref.argmax(-1) == full_a.argmax(-1)).sum()), "of": len(seq_a)}
    except Exception as e:  # report only
        log["reference"] = {"error": f"{type(e).__name__}: {e}"[:300]}

    log["thermal_end"] = thermal_level()
    log["result"] = "smoke_done"
    with open(report_path, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": {k: v["ok"] for k, v in log["checks"].items()}, "reference": log["reference"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
