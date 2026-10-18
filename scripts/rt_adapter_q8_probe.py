#!/usr/bin/env python3
"""LoRA adapter on the q8 realtime Aria model: equivalence, cost and swap time (issue #1834, docs/experiments/RT_ADAPTER_Q8.md).

Arms on the same val windows: F0 fp32 base, FA fp32 + adapter (delta), FM fp32 + adapter (merge), Q0 q8 base,
QD q8 + adapter (delta on QuantizedLinear), QM q8 + adapter (float32 merge, bf16, 8-bit). Timing on q8 only.

  rt_adapter_q8_probe.py <aria repo> <ckpt> <data json> <adapter json-less .pt> <swap partner .pt> <report json>
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import aria_mlx_lora as al  # noqa: E402
from kv_equiv_precision import kl_tv, logp  # noqa: E402
from lh_kv_reuse import load_mlx  # noqa: E402

SCALE = 1.0                      # alpha / rank = 16 / 16 for every #1724-format adapter (aria_lora.py defaults, #1726, #1734)
SPLIT = "val"                    # #1726 dev split; its test split stays unopened
L_DECODE, DECODE_STEPS, SUFFIX, REPS = 512, 32, 48, 5
PREFILL_LENS = (512, 896)


def thermal():
    try:
        return int(subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"],
                                  capture_output=True, text=True, timeout=5).stdout.split()[-1])
    except Exception:
        return None


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def summ(xs):
    xs = sorted(float(x) for x in xs)
    return {"n": len(xs), "mean": statistics.fmean(xs), "median": statistics.median(xs),
            "p95": xs[min(len(xs) - 1, int(0.95 * len(xs)))], "max": xs[-1]} if xs else None


class Acc:
    """KL/TV between two arms accumulated over windows (positions of every window pooled)."""

    def __init__(self):
        self.kl, self.tv = [], []

    def add(self, ref, x):
        k, t = kl_tv(logp(ref), logp(x))
        self.kl += k.tolist()
        self.tv += t.tolist()

    def out(self):
        return {"kl": summ(self.kl), "tv": summ(self.tv)}


def nll(lg, targets):
    lp = logp(lg)
    return float(-lp[np.arange(len(targets)), targets].mean())


def free():
    """Release MLX buffers after the caller dropped its references."""
    import mlx.core as mx
    gc.collect()
    try:
        mx.clear_cache()
    except AttributeError:
        mx.metal.clear_cache()


def main() -> None:
    aria_repo, ckpt, data_path, ad_path, partner_path, report_path = sys.argv[1:7]
    import mlx.core as mx
    data = json.load(open(data_path))
    windows = [r["windows"][0]["ids"] for r in data["recordings"] if r["split"] == SPLIT and r["windows"]]
    adA, adB = al.load_adapter(ad_path), al.load_adapter(partner_path)
    rep = {"issue": 1834, "scale": SCALE, "split": SPLIT, "windows": len(windows), "thermal": {"start": thermal()},
           "inputs": {"checkpoint_sha256": sha(ckpt), "data_sha256": sha(data_path), "adapter_sha256": sha(ad_path),
                      "partner_sha256": sha(partner_path)}, "mlx": mx.__version__}

    # fp32 phase: F0, FA (delta), FM (merge); integrity checks
    tok, run, model = load_mlx(aria_repo, ckpt, "fp32", return_model=True)
    blk = model.model.encode_layers[0]
    al.check_against(adA, len(model.model.encode_layers),
                     {t: tuple(getattr(blk, t).weight.shape) for t in al.TARGETS})
    al.check_against(adB, len(model.model.encode_layers),
                     {t: tuple(getattr(blk, t).weight.shape) for t in al.TARGETS})
    base_w = al.base_target_weights(ckpt)
    base_mods = al.current_modules(model)
    zero = {k: (a, np.zeros_like(b)) for k, (a, b) in adA.items()}
    w0 = windows[0]
    ref0 = run(w0, 0)
    al.apply_delta(model, zero, SCALE, mx.float32)
    zero_diff = float(np.abs(run(w0, 0) - ref0).max())
    al.restore(model, base_mods)
    repeat_diff = float(np.abs(run(w0, 0) - ref0).max())
    fm_mods = al.build_merged(base_w, adA, SCALE, mx.float32)
    F0, FA, nl = [], [], {k: [] for k in ("F0", "FA", "FM", "Q0", "QD", "QM")}
    acc = {k: Acc() for k in ("F0_FA", "FA_FM", "F0_Q0", "FA_QD", "FA_QM", "Q0_QD", "Q0_QM", "QD_QM")}
    finite = True
    for ids in windows:
        tg = np.array(ids[1:])
        al.restore(model, base_mods)
        f0 = run(ids, 0)[:-1]
        al.apply_delta(model, adA, SCALE, mx.float32)
        fa = run(ids, 0)[:-1]
        al.restore(model, fm_mods)
        fm = run(ids, 0)[:-1]
        finite &= bool(np.isfinite(f0).all() and np.isfinite(fa).all() and np.isfinite(fm).all())
        acc["F0_FA"].add(f0, fa)
        acc["FA_FM"].add(fa, fm)
        nl["F0"].append(nll(f0, tg)); nl["FA"].append(nll(fa, tg)); nl["FM"].append(nll(fm, tg))
        F0.append(f0); FA.append(fa)
    rep["checks"] = {"zero_b_max_abs": zero_diff, "fp32_repeat_max_abs": repeat_diff}
    del model, run, fm_mods, base_mods
    free()
    rep["thermal"]["after_fp32"] = thermal()

    # q8 phase: Q0, QD, QM
    tok, run, model = load_mlx(aria_repo, ckpt, "q8", return_model=True)
    q_base = al.current_modules(model)
    al.apply_delta(model, adA, SCALE, mx.bfloat16)
    q_delta = al.current_modules(model)
    t = time.perf_counter()
    q_merge = al.build_merged(base_w, adA, SCALE, mx.bfloat16, bits=8)
    mx.eval([m.parameters() for m in q_merge.values()])
    first_merge_ms = (time.perf_counter() - t) * 1000
    cfg = {"Q0": q_base, "QD": q_delta, "QM": q_merge}
    for w_i, ids in enumerate(windows):
        tg = np.array(ids[1:])
        out = {}
        for name, mods in cfg.items():
            al.restore(model, mods)
            out[name] = run(ids, 0)[:-1]
            finite &= bool(np.isfinite(out[name]).all())
            nl[name].append(nll(out[name], tg))
        acc["F0_Q0"].add(F0[w_i], out["Q0"])
        acc["FA_QD"].add(FA[w_i], out["QD"])
        acc["FA_QM"].add(FA[w_i], out["QM"])
        acc["Q0_QD"].add(out["Q0"], out["QD"])
        acc["Q0_QM"].add(out["Q0"], out["QM"])
        acc["QD_QM"].add(out["QD"], out["QM"])
    rep["checks"]["logits_finite"] = finite
    rep["divergence"] = {k: v.out() for k, v in acc.items()}
    rep["nll_mean"] = {k: statistics.fmean(v) for k, v in nl.items()}
    del F0, FA
    gc.collect()
    rep["thermal"]["after_equiv"] = thermal()

    # timing on q8: decode at L 512, suffix 48, arms interleaved and rotated per rep
    seq = windows[0]
    names = list(cfg)
    dec = {n: [] for n in names}
    suf = {n: [] for n in names}
    for r in range(REPS):
        for n in names[r % 3:] + names[:r % 3]:
            al.restore(model, cfg[n])
            run(seq[:L_DECODE], 0)
            for i in range(DECODE_STEPS):
                t = time.perf_counter_ns()
                run([seq[L_DECODE + i]], L_DECODE + i)
                dec[n].append((time.perf_counter_ns() - t) / 1e6)
            s0 = L_DECODE + DECODE_STEPS
            t = time.perf_counter_ns()
            run(seq[s0:s0 + SUFFIX], s0)
            suf[n].append((time.perf_counter_ns() - t) / 1e6)
    rep["decode_ms"] = {n: summ(v) for n, v in dec.items()}
    rep["suffix_ms"] = {n: summ(v) for n, v in suf.items()}
    pre = {}
    for n in ("Q0", "QD"):
        al.restore(model, cfg[n])
        for L in PREFILL_LENS:
            xs = []
            for _ in range(REPS):
                t = time.perf_counter_ns()
                run(seq[:L], 0)
                xs.append((time.perf_counter_ns() - t) / 1e6)
            pre[f"{n}_{L}"] = summ(xs)
    rep["full_prefill_ms"] = pre

    # swap time
    al.restore(model, q_delta)
    load_ms, set_ms = [], []
    for r in range(6):
        path = partner_path if r % 2 == 0 else ad_path
        t = time.perf_counter()
        ad = al.load_adapter(path)
        load_ms.append((time.perf_counter() - t) * 1000)
        t = time.perf_counter()
        al.swap_delta(model, ad, mx.bfloat16)
        mx.eval(model.parameters())
        set_ms.append((time.perf_counter() - t) * 1000)
    merge_ms = []
    for r in range(3):
        ad = adB if r % 2 == 0 else adA
        t = time.perf_counter()
        mods = al.build_merged(base_w, ad, SCALE, mx.bfloat16, bits=8)
        al.restore(model, mods)
        mx.eval(model.parameters())
        merge_ms.append((time.perf_counter() - t) * 1000)
    rep["swap_ms"] = {"delta_load": summ(load_ms), "delta_set": summ(set_ms), "merge_requantize": summ(merge_ms),
                      "first_merge": first_merge_ms}
    rep["thermal"]["end"] = thermal()
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({"checks": rep["checks"], "thermal": rep["thermal"]}))


if __name__ == "__main__":
    main()
