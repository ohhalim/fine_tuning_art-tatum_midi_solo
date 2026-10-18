#!/usr/bin/env python3
"""Aria weight precision vs per-token time and next-token distribution (issue #1811, docs/experiments/ARIA_PRECISION.md).

Four arms of the same checkpoint: fp32 (the real-time path today), bf16, and bf16 with model.model quantized to 8 or 4
bits (group 32, lm_head left alone, as in Aria's demo_mlx.py). Timing uses the producer's forward call (model, eval,
logits to numpy). Distribution difference is teacher forced on real test windows and compared against fp32.

  aria_precision_probe.py <aria repo> <ckpt> <data json> <report json>
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

ARMS = (("fp32", None), ("bf16", None), ("q8", 8), ("q4", 4))   # (name, bits); bits None means no quantization
GROUP = 32
CAPACITY = 4096 + 64
CONTEXTS = (256, 512, 896)          # decode positions L..L+31, then a 48-token suffix at L+32; all inside one 1023-token window
DECODE_STEPS, SUFFIX, REPS, WARM_STEPS = 32, 48, 5, 8
TF_SPLIT = "test"                   # teacher-forcing windows: first window of every test recording


def thermal_level():
    try:
        return int(subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                                  capture_output=True, text=True, timeout=5).stdout.split()[-1])
    except Exception:
        return None


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def arm_order(rep):
    """Arms rotated by one per repetition, so no arm always runs first or last."""
    k = rep % len(ARMS)
    return [a for a, _ in ARMS[k:] + ARMS[:k]]


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def divergence(ref_logits, logits, targets):
    """Per position: KL(ref || arm), total variation, top-1 equal, and both NLLs of the real next token."""
    lr, la = log_softmax(ref_logits), log_softmax(logits)
    pr, pa = np.exp(lr), np.exp(la)
    rows = np.arange(len(targets))
    return {"kl": (pr * (lr - la)).sum(-1), "tv": 0.5 * np.abs(pr - pa).sum(-1), "top1": ref_logits.argmax(-1) == logits.argmax(-1),
            "nll_ref": -lr[rows, targets], "nll": -la[rows, targets], "entropy_ref": -(pr * lr).sum(-1)}


def summary(xs):
    xs = sorted(float(x) for x in xs)
    if not xs:
        return None
    return {"n": len(xs), "median": round(statistics.median(xs), 4), "mean": round(statistics.fmean(xs), 4),
            "p95": round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 4), "max": round(xs[-1], 4)}


def load_arm(aria_repo, ckpt, bits, name):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(AbsTokenizer().vocab_size)
    model = TransformerLM(cfg)
    dtype = mx.float32 if name == "fp32" else mx.bfloat16
    weights = mx.load(ckpt)
    model.load_weights([(k, v.astype(dtype)) for k, v in weights.items()], strict=True)
    if bits is not None:
        nn.quantize(model.model, group_size=GROUP, bits=bits)
    mx.eval(model.parameters())
    model.eval()
    model.setup_cache(batch_size=1, max_seq_len=CAPACITY, dtype=dtype)
    nbytes = sum(v.nbytes for _, v in tree_flatten(model.parameters()))

    def run(ids, start):
        """The producer's forward: positions start.., KV up to the last written; all logits as float32 numpy."""
        idxs = mx.array([ids], dtype=mx.int32)
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        o = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=start + len(ids) - 1)
        mx.eval(o)
        return np.array(o[0].astype(mx.float32))
    return run, nbytes


def timed(run, ids, start):
    t = time.perf_counter_ns()
    run(ids, start)
    return (time.perf_counter_ns() - t) / 1e6


def main() -> None:
    aria_repo, ckpt, data_path, report_path = sys.argv[1:5]
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    data = json.load(open(data_path))
    windows = [r["windows"][0]["ids"] for r in data["recordings"] if r["split"] == TF_SPLIT and r["windows"]]
    seq = windows[0]
    if max(CONTEXTS) + DECODE_STEPS + SUFFIX > len(seq):
        raise ValueError("timing window longer than the first test window")
    report = {"issue": 1811, "thermal_start": thermal_level(), "mlx": mx.__version__,
              "inputs": {"checkpoint_sha256": file_sha(ckpt), "data_sha256": file_sha(data_path)},
              "params": {"arms": [a for a, _ in ARMS], "group": GROUP, "contexts": CONTEXTS, "decode_steps": DECODE_STEPS,
                         "suffix": SUFFIX, "reps": REPS, "tf_split": TF_SPLIT, "tf_windows": len(windows)}}
    runs, sizes = {}, {}
    for name, bits in ARMS:
        runs[name], sizes[name] = load_arm(aria_repo, ckpt, bits, name)
        runs[name](seq[:256], 0)                                         # warm up: one prefill and a few decode steps
        for i in range(WARM_STEPS):
            runs[name]([seq[256 + i]], 256 + i)
    report["param_bytes"] = sizes

    # timing: arms interleaved, order rotated per repetition
    decode = {a: {L: [] for L in CONTEXTS} for a in runs}
    suffix = {a: {L: [] for L in CONTEXTS} for a in runs}
    thermal = []
    for rep in range(REPS):
        thermal.append({"rep": rep, "start": thermal_level()})
        for a in arm_order(rep):
            for L in CONTEXTS:
                runs[a](seq[:L], 0)
                decode[a][L] += [timed(runs[a], [seq[L + i]], L + i) for i in range(DECODE_STEPS)]
                s0 = L + DECODE_STEPS
                suffix[a][L].append(timed(runs[a], seq[s0:s0 + SUFFIX], s0))
        thermal[-1]["end"] = thermal_level()
    report["thermal_reps"] = thermal
    report["decode_ms"] = {a: {L: summary(v) for L, v in d.items()} for a, d in decode.items()}
    report["suffix_ms"] = {a: {L: summary(v) for L, v in d.items()} for a, d in suffix.items()}

    # distribution difference vs fp32, teacher forced
    acc = {a: {k: [] for k in ("kl", "tv", "top1", "nll_ref", "nll", "entropy_ref")} for a in runs if a != "fp32"}
    finite = {a: True for a in runs}
    repeat_max_abs = None
    for w, ids in enumerate(windows):
        ref = runs["fp32"](ids, 0)
        finite["fp32"] &= bool(np.isfinite(ref).all())
        if w == 0:
            repeat_max_abs = float(np.abs(runs["fp32"](ids, 0) - ref).max())
        targets = np.array(ids[1:])
        for a in acc:
            lg = runs[a](ids, 0)
            finite[a] &= bool(np.isfinite(lg).all())
            for k, v in divergence(ref[:-1], lg[:-1], targets).items():
                acc[a][k].extend(np.asarray(v, dtype=float).tolist())
    report["divergence"] = {}
    for a, d in acc.items():
        report["divergence"][a] = {"positions": len(d["kl"]), "kl": summary(d["kl"]), "tv": summary(d["tv"]),
                                   "top1_agree": round(float(np.mean(d["top1"])), 4),
                                   "nll_mean": round(float(np.mean(d["nll"])), 4),
                                   "nll_ref_mean": round(float(np.mean(d["nll_ref"])), 4),
                                   "entropy_ref_mean": round(float(np.mean(d["entropy_ref"])), 4)}
    report["thermal_end"] = thermal_level()
    report["checks"] = {"logits_finite": finite, "fp32_repeat_max_abs": repeat_max_abs,
                        "thermal_all_zero": all(t["start"] == 0 and t["end"] == 0 for t in thermal)}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps(report["checks"]))


if __name__ == "__main__":
    main()
