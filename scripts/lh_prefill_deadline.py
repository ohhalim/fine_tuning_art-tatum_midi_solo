#!/usr/bin/env python3
"""Full-prefill cost per block with the observed left-hand context (issue #1791, docs/experiments/LH_PREFILL_DEADLINE.md).

Measures MLX Aria medium full prefill over a length grid and per-token decode after it, then applies the curve to the
b-context lengths of the dev requests (#1773 arms) to predict the share of blocks that fit in one beat at 128 BPM.

  lh_prefill_deadline.py <aria repo> <ckpt> <export dir> <arms dir> <report json>     (t3_aria venv)
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import resource
import statistics
import subprocess
import sys
import time

GRID = (128, 256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4000)
REPS, DECODE_N, CHECK_L = 5, 48, 2048
DEADLINE_MS = 60_000 / 128              # one beat at 128 BPM = 468.75 ms
CAPACITY = 4096 + 64


def interp(xs, ys, x):
    """Linear interpolation on a sorted grid, clamped at the ends."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if x0 <= x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    raise ValueError(x)


def extra_forwards(n):
    """Single-token forwards after the full prefill to generate n tokens: the prefill's last logits already give token 1."""
    return max(n - 1, 0)


def predict(lengths_tokens, prefill_curve, decode_first_ms, decode_rest_ms, deadline=DEADLINE_MS, decode_by_length=None):
    """[(context length, block tokens)] -> per-request predicted ms (generate the n block tokens) and the summary.

    decode_by_length: optional [(L, first_ms, rest_ms)] to interpolate the decode costs by context length (sensitivity)."""
    xs = [x for x, _ in prefill_curve]
    ys = [y for _, y in prefill_curve]
    ms = []
    for L, n in lengths_tokens:
        t = interp(xs, ys, L)
        first, rest = decode_first_ms, decode_rest_ms
        if decode_by_length:
            lx = [a for a, _, _ in decode_by_length]
            first = interp(lx, [b for _, b, _ in decode_by_length], L)
            rest = interp(lx, [c for _, _, c in decode_by_length], L)
        k = extra_forwards(n)
        if k:
            t += first + (k - 1) * rest
        ms.append(t)
    s = sorted(ms)
    return ms, {"n": len(s), "median_ms": round(statistics.median(s), 1), "p90_ms": round(s[int(0.9 * (len(s) - 1))], 1),
                "max_ms": round(s[-1], 1), "within_deadline": sum(1 for t in s if t <= deadline),
                "within_share": round(sum(1 for t in s if t <= deadline) / len(s), 3)}


def repredict(report_path, arms_dir, ckpt, aria_repo):
    """Recompute the prediction from the saved measurement (no model run), with provenance (review LOW)."""
    import hashlib
    import platform
    rep = json.load(open(report_path))
    files = sorted(glob.glob(os.path.join(arms_dir, "*.json.gz")))
    rows = requests_from_arms(arms_dir)
    if len(files) != 4 or len(rows) != 1314:
        sys.exit(f"unexpected arms input: {len(files)} files, {len(rows)} requests")
    sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()  # noqa: E731
    curve = [(int(L), v["prefill_median_ms"]) for L, v in rep["by_length"].items()]
    dec_first = statistics.median(v["decode_first_median_ms"] for v in rep["by_length"].values())
    dec_rest = statistics.median(v["decode_rest_median_ms"] for v in rep["by_length"].values())
    by_len = [(int(L), v["decode_first_median_ms"], v["decode_rest_median_ms"]) for L, v in rep["by_length"].items()]
    _, sb = predict([(r["Lb"], r["n"]) for r in rows], curve, dec_first, dec_rest)
    _, sa = predict([(r["La"], r["n"]) for r in rows], curve, dec_first, dec_rest)
    _, sb_len = predict([(r["Lb"], r["n"]) for r in rows], curve, dec_first, dec_rest, decode_by_length=by_len)
    rep["prediction_v1"] = rep.get("prediction_v1", rep.get("prediction"))
    rep["prediction"] = {"decode_first_ms": round(dec_first, 2), "decode_rest_ms": round(dec_rest, 2), "extra_forwards": "n - 1",
                         "b_context": sb, "a_context": sa, "b_context_decode_by_length": sb_len,
                         "clamped_below_grid": {"b": sum(1 for r in rows if r["Lb"] < GRID[0]), "a": sum(1 for r in rows if r["La"] < GRID[0]),
                                                "min_b": min(r["Lb"] for r in rows)},
                         "note": "prediction from the grid curve; p90 is over the request-length distribution, not real-time jitter; "
                                 "not an end-to-end measurement; usefulness of the left-hand conditioning is not tested"}
    rep["provenance"] = {"arms_files": {os.path.basename(f): sha(f) for f in files}, "requests": len(rows),
                         "checkpoint_sha256": sha(ckpt), "aria_repo_head": subprocess.run(["git", "-C", aria_repo, "rev-parse", "HEAD"],
                                                                                          capture_output=True, text=True).stdout.strip(),
                         "dtype": "float32", "machine": platform.platform(), "python": platform.python_version()}
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: rep["prediction"][k] for k in ("b_context", "a_context", "b_context_decode_by_length", "clamped_below_grid")}))


def thermal():
    out = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], capture_output=True, text=True).stdout
    try:
        return int(out.split()[-1])
    except (IndexError, ValueError):
        return -1


def requests_from_arms(arms_dir):
    rows = []
    for f in sorted(glob.glob(os.path.join(arms_dir, "*.json.gz"))):
        d = json.load(gzip.open(f))
        for r in d["requests"]:
            if r.get("status") != "ok" or not r.get("label_start"):
                continue
            rows.append({"song": d["idx"], "k": r["k"], "La": r["label_start"]["a"], "Lb": r["label_start"]["b"],
                         "n": r.get("win_labels", 0) + r.get("boundary_labels", 0)})
    return rows


def main() -> None:
    aria_repo, ckpt, export_dir, arms_dir, report_path = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    import numpy as np
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer

    songs = []
    for f in sorted(glob.glob(os.path.join(export_dir, "*.json.gz"))):
        d = json.load(gzip.open(f))
        songs.append((len(d["token_ids"]), d["idx"], d["token_ids"]))
    songs.sort(reverse=True)
    main_ids = songs[0][2] + songs[1][2]
    other_ids = songs[2][2] + songs[3][2]
    assert len(main_ids) >= GRID[-1] + DECODE_N and len(other_ids) >= CHECK_L

    tok = AbsTokenizer()
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_weights(ckpt, strict=True)
    mx.eval(model.parameters())
    model.eval()
    model.setup_cache(batch_size=1, max_seq_len=CAPACITY, dtype=mx.float32)

    def run(ids, start):
        idxs = mx.array([ids], dtype=mx.int32)
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        o = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=start + len(ids) - 1)
        mx.eval(o)
        return np.array(o[0, -1].astype(mx.float32))

    def one(ids, L, decode=True):
        t = time.perf_counter()
        run(ids[:L], 0)
        pre = (time.perf_counter() - t) * 1000
        dec = []
        if decode:
            for i in range(DECODE_N):
                t = time.perf_counter()
                run([ids[L + i]], L + i)
                dec.append((time.perf_counter() - t) * 1000)
        return pre, dec

    report = {"issue": 1791, "grid": GRID, "reps": REPS, "decode_n": DECODE_N, "deadline_ms": DEADLINE_MS,
              "token_source": {"main": [songs[0][1], songs[1][1]], "check": [songs[2][1], songs[3][1]]}, "by_length": {}}
    t_all = time.perf_counter()
    for L in GRID:
        th0 = thermal()
        one(main_ids, L)                                          # warm-up, discarded
        pres, firsts, rests = [], [], []
        for _ in range(REPS):
            pre, dec = one(main_ids, L)
            pres.append(pre)
            firsts.append(dec[0])
            rests += dec[1:]
        report["by_length"][L] = {"prefill_ms": [round(x, 1) for x in pres], "prefill_median_ms": round(statistics.median(pres), 1),
                                  "decode_first_median_ms": round(statistics.median(firsts), 2),
                                  "decode_rest_median_ms": round(statistics.median(rests), 2),
                                  "decode_rest_p90_ms": round(sorted(rests)[int(0.9 * (len(rests) - 1))], 2),
                                  "thermal_before": th0, "thermal_after": thermal()}
        print(L, report["by_length"][L]["prefill_median_ms"], report["by_length"][L]["decode_rest_median_ms"], flush=True)
    chk = [one(other_ids, CHECK_L, decode=False)[0] for _ in range(REPS)]
    report["content_check"] = {"L": CHECK_L, "other_prefill_median_ms": round(statistics.median(chk), 1),
                               "main_prefill_median_ms": report["by_length"][CHECK_L]["prefill_median_ms"]}
    report["rss_max_gib"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2)
    report["measure_s"] = round(time.perf_counter() - t_all, 1)

    curve = [(L, report["by_length"][L]["prefill_median_ms"]) for L in GRID]
    dec_first = statistics.median(report["by_length"][L]["decode_first_median_ms"] for L in GRID)
    dec_rest = statistics.median(report["by_length"][L]["decode_rest_median_ms"] for L in GRID)
    rows = requests_from_arms(arms_dir)
    _, sb = predict([(r["Lb"], r["n"]) for r in rows], curve, dec_first, dec_rest)
    _, sa = predict([(r["La"], r["n"]) for r in rows], curve, dec_first, dec_rest)
    report["prediction"] = {"decode_first_ms": round(dec_first, 2), "decode_rest_ms": round(dec_rest, 2), "b_context": sb, "a_context": sa,
                            "note": "prediction from the grid curve; not an end-to-end real-time measurement"}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["prediction"]))


if __name__ == "__main__":
    if sys.argv[1] == "--repredict":
        repredict(*sys.argv[2:6])
    else:
        main()
