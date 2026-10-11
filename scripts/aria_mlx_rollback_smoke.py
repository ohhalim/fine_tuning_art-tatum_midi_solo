#!/usr/bin/env python3
"""Local Aria MLX: mid-generation rollback, invisibility of cancelled tokens, and first-call cost per new length
(issue #1739, docs/experiments/ARIA_MLX_ROLLBACK.md). An infrastructure check only, like #1737: float32 weights and
cache, saved token sequences, no bias, CFG, quantisation, reranking, training or downloads.

Sequences: A = prompt 0 + base seed 1, B = prompt 0 + base seed 2 from the #1731 masked generations (614 tokens each).

  F  first-call cost   at the start of the process, before any other forward: for prefill lengths never used before,
                       the first call and a second call of the same length; then the first and later one-token decode
  R  rollback          fill the cache with all of A, roll back to R in {P+64, P+256, P+448}, write a replacement of
                       k in {8, 128} tokens from B in one chunk, then decode 16 more saved tokens one at a time.
                       Compare every written position's logits with a fresh full prefill of the same sequence:
                       top-1 equal and max |logit diff| <= 1e-3 (fixed before the run). The number of stale A positions
                       left beyond the new end is recorded; cases with stale positions test that cancelled tokens
                       stay invisible.

  aria_mlx_rollback_smoke.py <aria repo> <checkpoint> <masked generations json> <data json> <report json>
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time

LOGIT_TOL = 1e-3
ROLLBACK_OFFSETS, REPLACE_LENS, TAIL = (64, 256, 448), (8, 128), 16
FIRST_CALL_LENS = (96, 160, 224, 288, 352, 416, 480, 544)


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


def main() -> None:
    aria_repo, ckpt, gens_path, data_path, report_path = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    t_proc = time.perf_counter()
    import mlx.core as mx
    import numpy as np
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig

    log = {"checks": {}, "first_call": {}, "thermal_start": thermal_level(), "result": None}
    tok = AbsTokenizer()
    gens = json.load(open(gens_path))["completed"]
    prompt = json.load(open(data_path))["prompts"][0]["ids"]
    a = prompt + next(g for g in gens if g["prompt_index"] == 0 and g["seed"] == 1 and not g["adapter"])["ids"]
    b = prompt + next(g for g in gens if g["prompt_index"] == 0 and g["seed"] == 2 and not g["adapter"])["ids"]
    P = len(prompt)
    cases = [(P + off, k) for off in ROLLBACK_OFFSETS for k in REPLACE_LENS]
    L = max([len(a), max(FIRST_CALL_LENS)] + [r + k + TAIL for r, k in cases])
    assert all(P + k + TAIL <= len(b) for k in REPLACE_LENS)
    patch = subprocess.run(["git", "-C", aria_repo, "diff"], capture_output=True, text=True).stdout
    log["provenance"] = {"checkpoint_sha256": file_sha(ckpt), "generations_sha256": file_sha(gens_path), "data_sha256": file_sha(data_path),
                         "mlx": getattr(mx, "__version__", None),
                         "aria_commit": subprocess.run(["git", "-C", aria_repo, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                         "aria_local_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()}

    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_weights(ckpt, strict=True)
    mx.eval(model.parameters())
    model.eval()
    log["load_s"] = round(time.perf_counter() - t_proc, 3)

    def run(ids, start):
        if start < 0 or start + len(ids) > L:
            raise ValueError(f"positions {start}..{start + len(ids) - 1} outside cache capacity {L}")
        idxs = mx.array([ids], dtype=mx.int32)
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        out = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=start + len(ids) - 1)
        mx.eval(out)
        return np.array(out[0].astype(mx.float32))

    def fresh():
        model.setup_cache(batch_size=1, max_seq_len=L, dtype=mx.float32)

    fresh()
    log["cache"] = {"capacity": L, "k_cache": list(model.model.encode_layers[0].kv_cache.k_cache.shape)}

    # F: first call per new length, at the start of the process (no forward has run yet)
    filler = (a + b)[:max(FIRST_CALL_LENS)]
    rows = []
    for i, n in enumerate(FIRST_CALL_LENS):
        fresh()
        t = time.perf_counter()
        run(filler[:n], 0)
        first = time.perf_counter() - t
        fresh()
        t = time.perf_counter()
        run(filler[:n], 0)
        second = time.perf_counter() - t
        rows.append({"length": n, "first_s": round(first, 4), "second_s": round(second, 4),
                     "process_first_forward": i == 0})
    log["first_call"]["prefill"] = rows
    fresh()
    run(filler[:256], 0)
    dec = []
    for j in range(6):
        t = time.perf_counter()
        run([filler[256 + j]], 256 + j)
        dec.append(round(time.perf_counter() - t, 4))
    log["first_call"]["decode_one_after_prefill_256"] = {"first_s": dec[0], "later_s": dec[1:]}

    # R: rollback mid-generation
    results = []
    for r, k in cases:
        repl = b[P:P + k]
        tail = b[P + k:P + k + TAIL]
        seq = a[:r] + repl + tail
        fresh()
        run(a, 0)                                  # cache holds all of A (positions 0..len(a)-1)
        t = time.perf_counter()
        got = [run(repl, r)]                       # one chunk over positions r..r+k-1
        chunk_s = time.perf_counter() - t
        for i, x in enumerate(tail):
            got.append(run([x], r + k + i))
        got = np.concatenate(got)
        fresh()
        ref = run(seq, 0)[r:r + k + TAIL]
        d = np.abs(got - ref)
        top = got.argmax(-1) == ref.argmax(-1)
        new_end = r + k + TAIL
        results.append({"rollback_to": r, "replacement_tokens": k, "tail_tokens": TAIL, "positions": f"{r}..{new_end - 1}",
                        "stale_A_positions_left_beyond": max(0, len(a) - new_end),
                        "max_abs_logit_diff": float(d.max()), "top1_agree": int(top.sum()), "of": int(top.size),
                        "chunk_write_s": round(chunk_s, 4),
                        "ok": bool(top.all()) and float(d.max()) <= LOGIT_TOL})
    log["rollback"] = results
    log["checks"]["rollback_equals_fresh_all_cases"] = {"ok": all(x["ok"] for x in results), "cases": len(results)}
    log["checks"]["cancelled_tokens_invisible"] = {
        "ok": all(x["ok"] for x in results if x["stale_A_positions_left_beyond"] > 0),
        "cases_with_stale_positions": sum(1 for x in results if x["stale_A_positions_left_beyond"] > 0)}
    log["thermal_end"] = thermal_level()
    log["result"] = "smoke_done"
    with open(report_path, "w") as f:
        json.dump(log, f, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": {k: v["ok"] for k, v in log["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
