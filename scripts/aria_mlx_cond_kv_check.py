#!/usr/bin/env python3
"""KV check of the conditioned MLX forward before any training (issue #1760 step F, docs/experiments/COND_ADAPTER_MIN.md).

With a fixed non-zero condition weight (a zero weight would let a wrong injection pass), every block request of a few
train-song excerpts is run the way the runtime would run it, and compared with a full recompute of the same session
tokens and per-position conditions:
  runtime path  one KV cache for the session. A crop (crop_contract) resets the cache and prefills [prefix, <S>] + kept
                with the kept positions' old conditions; then the request feeds p0 (rewritten with the current
                condition, the rollback rule) and its continuation one token at a time
  reference     a fresh cache and one prefill of the session tokens up to the last continuation input
Pass: at every compared position top-1 agrees and max |logit diff| <= 1e-3 (the #1741 KV smoke tolerance). The run
also reports how many requests changed condition, cropped, were cold or warm-empty, and checks that the injection is
live (real vs zero condition differ). Run with the Aria venv:

  aria_mlx_cond_kv_check.py <aria repo> <checkpoint> <#1756 export dir> <report json>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_mlx_cond import COND_DIM, cond_forward, cond_rows  # noqa: E402
from block_contract import block_start  # noqa: E402
from block_request import payload  # noqa: E402
from crop_contract import HEADER, session_position_requests, session_requests, session_tokens  # noqa: E402
from lh_role_proxy import Note  # noqa: E402

SONGS = (0, 7)                    # train song indexes, fixed before the run
EXCERPT_NOTES = 500               # the first 500 notes of each song
CAP, CAPACITY, KEEP = 16, 384, 192   # small settings so that one excerpt crops several times
WEIGHT_SEED, WEIGHT_SCALE = 1760, 0.05
LOGIT_TOL = 1e-3


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def excerpt(tokens, ids, n_notes):
    """Tokens and ids up to the end of the n-th note, then <E>."""
    seen = 0
    for i, t in enumerate(tokens):
        if isinstance(t, tuple) and t[0] == "piano":
            seen += 1
            if seen == n_notes:
                end = i + 3
                eos = ids[tokens.index("<E>")]
                return list(tokens[:end]) + ["<E>"], list(ids[:end]) + [eos]
    return list(tokens), list(ids)


def main() -> None:
    aria_repo, ckpt, export_dir, report_path = sys.argv[1:5]
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    import numpy as np
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer

    t_start = time.time()
    tok = AbsTokenizer()
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_weights(ckpt, strict=True)
    mx.eval(model.parameters())
    model.eval()
    mx.random.seed(WEIGHT_SEED)
    weight = mx.random.normal((cfg.d_model, COND_DIM)) * WEIGHT_SCALE
    mx.eval(weight)

    def fresh():
        model.setup_cache(batch_size=1, max_seq_len=CAPACITY, dtype=mx.float32)

    def run(ids, start, rows):
        if start < 0 or start + len(ids) > CAPACITY:
            raise ValueError(f"positions {start}..{start + len(ids) - 1} outside capacity {CAPACITY}")
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        out = cond_forward(model, mx.array([ids], dtype=mx.int32), pos, start, start + len(ids) - 1,
                           mx.array([rows], dtype=mx.float32), weight)
        mx.eval(out)
        return np.array(out[0].astype(mx.float32))

    report = {"issue": 1760, "step": "F", "doc": "docs/experiments/COND_ADAPTER_MIN.md", "training": False, "quality_claimed": False,
              "params": {"songs": SONGS, "excerpt_notes": EXCERPT_NOTES, "cap": CAP, "capacity": CAPACITY, "keep_target": KEEP,
                         "weight_seed": WEIGHT_SEED, "weight_scale": WEIGHT_SCALE, "logit_tol": LOGIT_TOL},
              "run": {"git_head": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                      "checkpoint_sha256": sha256_file(ckpt), "mlx": mx.__version__,
                      "cond_module_sha256": sha256_file(os.path.join(HERE, "aria_mlx_cond.py")),
                      "check_script_sha256": sha256_file(os.path.abspath(__file__))},
              "songs": []}
    all_ok = True
    for idx in SONGS:
        d = json.load(gzip.open(os.path.join(export_dir, "requests", f"{idx:03d}.json.gz")))
        nj = json.load(open(os.path.join(export_dir, "songs", f"{idx:03d}.notes.json")))
        assert (nj["idx"], nj["source"], nj["sha1"]) == (d["idx"], d["source"], d["sha1"])
        tokens = [tuple(t) if isinstance(t, list) else t for t in d["tokens"]]
        tokens, ids = excerpt(tokens, d["token_ids"], EXCERPT_NOTES)
        hum = [Note(nid, p, on, off) for nid, p, on, off in nj["human"]]
        t0 = d["t0"]
        srows = session_requests(tokens, t0, CAP, CAPACITY, KEEP)
        pays = {r["k"]: payload(hum, block_start(r["k"], t0), r["k"]) for r in srows}
        header_ids = ids[:len(HEADER)]
        assert tuple(tokens[:len(HEADER)]) == HEADER
        song = {"idx": idx, "requests": len(srows), "crops": 0, "condition_changes": 0, "cold": 0, "warm_empty": 0,
                "compared_positions": 0, "max_abs_logit_diff": 0.0, "top1_disagree": 0, "live_max_diff": 0.0}
        prev = None
        for r in srows:
            k = r["k"]
            sess_ids = list(ids[:r["cont_end"]]) if r["keep_from"] is None else header_ids + list(ids[r["keep_from"]:r["cont_end"]])
            assert len(sess_ids) == len(session_tokens(tokens, r))
            rows = cond_rows(session_position_requests(tokens, srows, k), pays)
            p0, last = r["ctx_len"] - 1, len(sess_ids) - 1           # inputs p0 .. last - 1 predict the continuation
            if p0 >= last:
                continue
            song["crops"] += r["crop"] is not None
            song["condition_changes"] += prev is not None and pays[k] != prev
            song["cold"] += pays[k][1] == 0
            song["warm_empty"] += pays[k][1] == 1 and not any(pays[k][0])
            if k == 0 or r["crop"] is not None:
                fresh()
                if p0 > 0:
                    run(sess_ids[:p0], 0, rows[:p0])                   # reset: prefill the kept context with old conditions
            got = [run([sess_ids[q]], q, [rows[q]])[0] for q in range(p0, last)]   # p0 rewritten, then token by token
            caches = [b.kv_cache for b in model.model.encode_layers]            # keep the runtime cache aside
            fresh()
            ref = run(sess_ids[:last], 0, rows[:last])[p0:last]
            zero = run(sess_ids[:last], 0, rows[:p0] + [[0.0] * COND_DIM] * (last - p0))[p0:last]
            for b, c in zip(model.model.encode_layers, caches):              # restore the runtime cache
                b.kv_cache = c
            got = np.stack(got)
            diff = np.abs(got - ref)
            song["compared_positions"] += len(got)
            song["max_abs_logit_diff"] = max(song["max_abs_logit_diff"], float(diff.max()))
            song["top1_disagree"] += int((got.argmax(-1) != ref.argmax(-1)).sum())
            song["live_max_diff"] = max(song["live_max_diff"], float(np.abs(ref - zero).max()))
            prev = pays[k]
        song["ok"] = song["top1_disagree"] == 0 and song["max_abs_logit_diff"] <= LOGIT_TOL
        all_ok = all_ok and song["ok"]
        report["songs"].append(song)
        print(json.dumps(song))
    cover = {k: sum(s[k] for s in report["songs"]) for k in ("crops", "condition_changes", "cold", "warm_empty")}
    report["coverage"] = cover
    report["injection_live"] = all(s["live_max_diff"] > 1e-2 for s in report["songs"])
    report["coverage_ok"] = cover["crops"] >= 2 and cover["condition_changes"] > 0 and cover["cold"] > 0 and cover["warm_empty"] > 0
    report["pass"] = all_ok and report["injection_live"] and report["coverage_ok"]
    report["run"]["seconds"] = round(time.time() - t_start, 1)
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"pass": report["pass"], "coverage": cover, "injection_live": report["injection_live"],
                      "seconds": report["run"]["seconds"]}))


if __name__ == "__main__":
    main()
