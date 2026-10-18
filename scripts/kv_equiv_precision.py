#!/usr/bin/env python3
"""Cached-suffix equivalence inside one precision, with its own noise floor and defect controls
(issue #1815, docs/experiments/KV_EQUIV_PRECISION.md).

Per precision and case, R0 is a full prefill of N. The noise floor is the largest gap to R0 among computations that are
mathematically the same as R0: R1 full repeat, R2 two chunks split at the reuse start p, R3 token by token after p,
R4 two chunks split at p // 2. C is the producer path (Pseq fed, N[p:] from p, later KV masked). PC_future writes stale
KV after N and exposes it through max_kv_pos: the causal mask must hide it (positive control). Defect controls keep past
KV that N can attend to: NC_unfed (KV valid only up to lcp - 8 but reuse starts at lcp), NC_skip (no recompute after
the first change), NC_crop (old positions kept after a crop reset). Every variant starts from a poisoned cache.
The 2x floor and 10x detection rules are fixed operating rules, not a proof of equivalence.

  kv_equiv_precision.py <aria repo> <ckpt> <export dir> <velocity dir> <#1793 count json> <out json> [--precisions fp32,bf16,q8]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lh_kv_reuse import build_songs, common_prefix, file_sha, load_mlx, synthetic_cases  # noqa: E402

FLOOR_MULT, FLOOR_EPS = 2.0, {"kl": 1e-7, "tv": 1e-6}      # C and PC_future: within 2 x floor (+ eps for floors at 0)
DETECT_MULT, DETECT_EPS = 10.0, 1e-4                        # a defect control: max KL above 10 x floor + 1e-4
UNFED = 8                                                   # NC_unfed: the last 8 shared tokens were never written
METRICS = ("kl_last", "tv_last", "kl_max", "kl_p95", "tv_max", "masked_kl_last", "masked_tv_last")


def logp(x):
    x = x.astype(np.float64)
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def kl_tv(lr, lx):
    """Row-wise KL(ref || x) and total variation from log-probabilities (-inf where p = 0)."""
    pr, px = np.exp(lr), np.exp(lx)
    with np.errstate(invalid="ignore"):
        d = np.where(pr > 0, lr - lx, 0.0)
    return (pr * d).sum(-1), 0.5 * np.abs(pr - px).sum(-1)


def gaps(ref, x, allowed_last):
    """Gaps of x to ref over the same positions (rows), plus the last position under the sampler's mask.
    Non-finite logits or an allowed set with no tokens raise: they are failures, not small gaps."""
    if not (np.isfinite(ref).all() and np.isfinite(x).all()):
        raise FloatingPointError("non-finite logits")
    if not allowed_last.any():
        raise ValueError("no allowed token at the last position")
    kl, tv = kl_tv(logp(ref), logp(x))
    mref = np.where(allowed_last, ref[-1:].astype(np.float64), -np.inf)
    mx_ = np.where(allowed_last, x[-1:].astype(np.float64), -np.inf)
    mkl, mtv = kl_tv(logp(mref), logp(mx_))
    return {"kl_last": float(kl[-1]), "tv_last": float(tv[-1]), "kl_max": float(kl.max()),
            "kl_p95": float(np.sort(kl)[min(len(kl) - 1, int(0.95 * len(kl)))]), "tv_max": float(tv.max()),
            "masked_kl_last": float(mkl[0]), "masked_tv_last": float(mtv[0]),
            "top1_last_equal": bool(ref[-1].argmax() == x[-1].argmax()), "positions": int(len(kl))}


def top2_margin(row):
    """Probability gap between the two most likely tokens of the reference at the last position (top-1 context)."""
    p = np.exp(logp(row))
    a, b = np.sort(p)[-2:]
    return float(b - a)


def floor_of(refs):
    """Per metric, the largest gap among the references that apply (None = not applicable)."""
    present = [g for g in refs.values() if g is not None]
    return {m: max(g[m] for g in present) for m in METRICS}


def within_floor(g, floor):
    return {m: g[m] <= FLOOR_MULT * floor[m] + FLOOR_EPS["kl" if "kl" in m else "tv"] for m in METRICS}


def detected(g, floor):
    return g["kl_max"] > DETECT_MULT * floor["kl_max"] + DETECT_EPS


def judge(case):
    """case: {"R1_exact", "refs": {R1..R4}, "C", "PC_future", "controls": {name: {"gap", "refs"} | None}}.
    Controls are judged on their own positions against the references cut to the same positions."""
    floor = floor_of(case["refs"])
    c_in = within_floor(case["C"], floor)
    pc_in = within_floor(case["PC_future"], floor) if case.get("PC_future") is not None else None
    det = {k: detected(v["gap"], floor_of(v["refs"])) for k, v in case["controls"].items() if v is not None}
    return {"valid": case["R1_exact"], "floor": floor, "C_within": c_in, "C_ok": all(c_in.values()),
            "PC_future_ok": None if pc_in is None else all(pc_in.values()),
            "controls_detected": det, "controls_all_detected": all(det.values()) if det else None}


def build_cases(built, cnt):
    """#1793 --equiv cases (6 synthetic + first non-crop dev pair per cause) and #1795 cases (4 long P, S = 0, first crop)."""
    from lh_cache_replay import pick_cases

    def pair_of(x):
        reqs = built[x["song"]]
        i = next(j for j, r in enumerate(reqs) if r and r["k"] == x["k"])
        prev, cur = reqs[i - 1], reqs[i]
        changed = (prev["keep_from"], prev["base_ms"]) != (cur["keep_from"], cur["base_ms"])
        return prev["b"]["tokens"], cur["b"]["tokens"][:cur["b"]["label_start"]], changed

    cases = {}
    syn, syn_meta = synthetic_cases(built)
    for name, (P_tok, N_tok) in syn.items():
        cases[f"syn_{name}"] = (P_tok, N_tok, False)
    picked = {}
    for x in cnt["b"]["rows"]:
        if x["cause"] not in picked and x["cause"] != "crop":
            picked[x["cause"]] = x
    for c, x in picked.items():
        cases[f"dev_{c}"] = pair_of(x)
    for name, x in pick_cases(cnt["b"]["rows"]).items():
        cases[name] = pair_of(x)
    return cases, syn_meta


def control_points(Pi, Ni, lcp, p, changed):
    """Where each defect control restarts computing, or None when it would not keep any wrong past KV.
    NC_skip / NC_crop restart at s = min(len(Pseq), len(N)) - 1 and need a token before s that differs from N;
    NC_unfed restarts at lcp with KV valid only up to lcp - UNFED."""
    L = len(Ni)
    s = min(len(Pi), L) - 1
    differs = s > p and any(a != b for a, b in zip(Pi[p:s], Ni[p:s]))
    out = {"NC_unfed": lcp if (not changed and p == lcp and lcp - UNFED >= 1 and lcp < L) else None,
           "NC_skip": s if (not changed and differs) else None,
           "NC_crop": s if (changed and differs) else None}
    for k, v in out.items():
        assert v is None or 0 < v < L, (k, v, L)               # a non-empty suffix at a real position
    return out


def main() -> None:
    aria_repo, ckpt, export, velocity_dir, count_path, out_path = sys.argv[1:7]
    tail = sys.argv[7:]
    precisions = tail[tail.index("--precisions") + 1].split(",") if "--precisions" in tail else ["fp32", "bf16", "q8"]
    import torch
    from aria_grammar_mask import GrammarMask
    from lh_cache_replay import reuse_start
    from stream_contract import TokenStream
    cnt = json.load(open(count_path))
    built, fails = build_songs(export, velocity_dir)
    if fails or cnt.get("build_failures"):
        sys.exit("build failures: fail closed")
    cases, syn_meta = build_cases(built, cnt)
    rep = {"issue": 1815, "rules": {"floor_mult": FLOOR_MULT, "floor_eps": FLOOR_EPS, "detect_mult": DETECT_MULT,
                                    "detect_eps": DETECT_EPS, "unfed": UNFED},
           "inputs": {"checkpoint_sha256": file_sha(ckpt), "count_sha256": file_sha(count_path)},
           "synthetic_source": syn_meta, "precisions": {}}
    for prec in precisions:
        tok, run = load_mlx(aria_repo, ckpt, prec)
        vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
        gm = GrammarMask(vocab)
        no_end = torch.tensor([t not in ("<D>", "<E>") for t in vocab])
        enc = lambda toks: tok.encode(list(toks))  # noqa: E731
        last_134 = next(r for r in reversed(built[134]) if r)
        poison = (enc(last_134["b"]["tokens"]) * 4)[:4096]

        def fresh(ids=None):
            """Poison every cache position, then write ids from 0: each variant starts from the same kind of cache."""
            run(poison, 0)
            if ids:
                run(ids, 0)

        out = {}
        for name, (P_tok, N_tok, changed) in cases.items():
            Pi, Ni = enc(P_tok), enc(N_tok)
            L, lcp = len(Ni), common_prefix(Pi, Ni)
            p = reuse_start(len(Pi), lcp, L, changed)
            gm.sync(TokenStream(list(N_tok)).grammar_state())     # next-token state after the whole N, same for every variant
            allowed = (gm.allowed() & no_end).numpy()
            raw = {}
            fresh()
            raw["R0"] = run(Ni, 0)[p:]
            fresh()
            raw["R1"] = run(Ni, 0)[p:]
            if p > 0:
                fresh(Ni[:p])
                raw["R2"] = run(Ni[p:], p)
            fresh(Ni[:p] if p > 0 else None)
            raw["R3"] = np.stack([run([Ni[i]], i)[-1] for i in range(p, L)])
            if p // 2 >= 1:
                q = p // 2
                fresh(Ni[:q])
                raw["R4"] = run(Ni[q:], q)[p - q:]
            fresh(Pi)
            raw["C"] = run(Ni[p:], p, kv_end=L - 1)
            fresh(Ni)                                             # N's own KV, then stale poison after it, exposed by max_kv_pos
            raw["PC_future"] = run(Ni[p:], p, kv_end=len(poison) - 1) if len(poison) > L else None
            for k, s in control_points(Pi, Ni, lcp, p, changed).items():
                if s is None:
                    continue
                fresh(Pi[:lcp - UNFED] if k == "NC_unfed" else Pi)
                raw[k] = (s, run(Ni[s:], s, kv_end=L - 1))
            R0 = raw["R0"]
            refs = {r: (gaps(R0, raw[r], allowed) if r in raw else None) for r in ("R1", "R2", "R3", "R4")}
            controls = {}
            for k in ("NC_unfed", "NC_skip", "NC_crop"):
                if k not in raw:
                    controls[k] = None
                    continue
                s, rows = raw[k]
                cut = s - p                                       # the control and its references on the same positions
                controls[k] = {"start": s, "gap": gaps(R0[cut:], rows, allowed),
                               "refs": {r: (gaps(R0[cut:], raw[r][cut:], allowed) if r in raw else None)
                                        for r in ("R1", "R2", "R3", "R4")}}
            case = {"R1_exact": bool(np.array_equal(R0, raw["R1"])), "refs": refs, "C": gaps(R0, raw["C"], allowed),
                    "PC_future": gaps(R0, raw["PC_future"], allowed) if raw["PC_future"] is not None else None,
                    "controls": controls}
            out[name] = {"P_len": len(Pi), "N_len": L, "lcp": lcp, "p": p, "crop_changed": changed,
                         "allowed_last": int(allowed.sum()), "top2_margin_last": top2_margin(R0[-1]), **case, "judge": judge(case)}
            j = out[name]["judge"]
            print(prec, name, "p", p, "S", L - p, "valid", j["valid"], "C_ok", j["C_ok"], "PC", j["PC_future_ok"],
                  "det", j["controls_detected"], flush=True)
        js = [v["judge"] for v in out.values()]
        rep["precisions"][prec] = {"cases": out, "all_valid": all(j["valid"] for j in js),
                                   "all_C_ok": all(j["C_ok"] for j in js),
                                   "all_PC_future_ok": all(j["PC_future_ok"] is not False for j in js),
                                   "controls_undetected": [(n, k) for n, v in out.items()
                                                           for k, d in v["judge"]["controls_detected"].items() if not d],
                                   "controls_applied": sum(len(v["judge"]["controls_detected"]) for v in out.values())}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("all_valid", "all_C_ok", "all_PC_future_ok", "controls_undetected", "controls_applied")}
                      for k, v in rep["precisions"].items()}))


if __name__ == "__main__":
    main()
