#!/usr/bin/env python3
"""Replay consecutive requests on a real cache state (issue #1795, docs/experiments/LH_CACHE_REPLAY.md).

  lh_cache_replay.py <aria repo> <ckpt> <export dir> <velocity dir> <#1793 count json> <out json>     (t3_aria venv)
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from lh_kv_reuse import build_songs, common_prefix, file_sha, load_mlx  # noqa: E402

TARGET_P = (1692, 2142, 2877, 3343)
TARGET_S = {1692: 116, 2142: 248, 2877: 302, 3343: 248}
TOL, REPS, NS, SKIP = 1e-3, 5, (15, 30), 8
DEADLINE_MS = 60_000 / 128


def reuse_start(valid_end, lcp, n_len, crop_changed):
    """Doc 3: P_reuse = min(common prefix, KV valid end); 0 after a crop change; recompute the last token when S would be 0."""
    if crop_changed:
        return 0
    p = min(lcp, valid_end)
    return n_len - 1 if p >= n_len else p


def pick_cases(rows):
    nc = [x for x in rows if x["cause"] != "crop"]
    cases = {}
    for P in TARGET_P:
        best = min(nc, key=lambda x: (abs(x["P"] - P), abs(x["S"] - TARGET_S[P]), x["song"], x["k"]))
        cases[f"longP_{P}"] = best
    crop = sorted((x for x in rows if x["cause"] == "crop"), key=lambda x: (x["song"], x["k"]))
    if crop:
        cases["crop"] = crop[0]
    none = sorted((x for x in rows if x["cause"] == "none"), key=lambda x: (-x["P"], x["song"], x["k"]))
    if none:
        cases["s0"] = none[0]
    return cases


def main() -> None:
    import numpy as np
    aria_repo, ckpt, export, velocity_dir, count_path, out_path = sys.argv[1:7]
    cnt = json.load(open(count_path))
    if cnt.get("build_failures"):
        sys.exit("count has build failures: fail closed")
    built, fails = build_songs(export, velocity_dir)
    if fails:
        sys.exit("build failures: fail closed")
    tok, run = load_mlx(aria_repo, ckpt)
    enc = lambda toks: tok.encode(list(toks))  # noqa: E731
    cases = pick_cases(cnt["b"]["rows"])
    last_134 = next(r for r in reversed(built[134]) if r)
    poison = (enc(last_134["b"]["tokens"]) * 4)[:4096]  # stale content written before each replay

    def pair_tokens(x):
        reqs = built[x["song"]]
        i = next(j for j, r in enumerate(reqs) if r and r["k"] == x["k"])
        prev, cur = reqs[i - 1], reqs[i]
        changed = (prev["keep_from"], prev["base_ms"]) != (cur["keep_from"], cur["base_ms"])
        P = prev["b"]["tokens"]
        N = cur["b"]["tokens"][:cur["b"]["label_start"]]
        cont = cur["b"]["tokens"][cur["b"]["label_start"]:]
        return enc(P), enc(N), enc(cont), changed

    def state(P_ids, V):
        run(poison, 0)                                   # leave stale KV everywhere
        if V:
            run(P_ids[:V], 0)

    def check(P_ids, N_ids, V, start):
        state(P_ids, V)
        cached = run(N_ids[start:], start, kv_end=len(N_ids) - 1)
        full = run(N_ids, 0)
        ref = full[start:]
        top1 = int((cached.argmax(-1) == ref.argmax(-1)).sum())
        diff = float(np.abs(cached[-1] - ref[-1]).max())
        return {"start": start, "S": len(N_ids) - start, "top1_equal": top1, "positions": len(ref), "max_abs_last": diff,
                "ok": top1 == len(ref) and diff <= TOL}

    rep = {"issue": 1795, "tolerance": TOL, "cases": {}, "negative": {},
           "inputs": {"count_sha256": file_sha(count_path), "checkpoint_sha256": file_sha(ckpt)}}
    toks = {}
    for name, x in cases.items():
        P_ids, N_ids, cont, changed = pair_tokens(x)
        toks[name] = (P_ids, N_ids, cont, changed)
        lcp = common_prefix(P_ids, N_ids)
        for st, V in (("fed", len(P_ids)), ("unfed", len(P_ids) - 1)):
            start = reuse_start(V, lcp, len(N_ids), changed)
            r = check(P_ids, N_ids, V, start)
            rep["cases"][f"{name}/{st}"] = {"song": x["song"], "k": x["k"], "cause": x["cause"], "lcp": lcp, "V": V,
                                           "len_N": len(N_ids), **r}
            print(name, st, rep["cases"][f"{name}/{st}"], flush=True)
    # negative controls (must fail)
    P_ids, N_ids, cont, _ = toks["s0"] if "s0" in toks else toks[next(iter(toks))]
    N_ext = P_ids + cont[:1]                               # new context extends the previous input by one token
    V = len(P_ids) - 1                                     # unfed: the last previous token is not in the KV
    rep["negative"]["stale_beyond_valid"] = check(P_ids, N_ext, V, common_prefix(P_ids, N_ext))
    rep["negative"]["stale_beyond_valid_rule"] = check(P_ids, N_ext, V, reuse_start(V, common_prefix(P_ids, N_ext), len(N_ext), False))
    name = next(n for n in toks if n.startswith("longP") and common_prefix(toks[n][0], toks[n][1]) + SKIP < len(toks[n][1]))
    P_ids, N_ids, _, _ = toks[name]
    lcp = common_prefix(P_ids, N_ids)
    rep["negative"]["skip_change"] = {"case": name, **check(P_ids, N_ids, len(P_ids), lcp + SKIP)}
    for k, v in rep["negative"].items():
        print("negative", k, v, flush=True)
    positives_ok = all(v["ok"] for v in rep["cases"].values()) and rep["negative"]["stale_beyond_valid_rule"]["ok"]
    negatives_caught = not rep["negative"]["stale_beyond_valid"]["ok"] and not rep["negative"]["skip_change"]["ok"]
    rep["equivalence_ok"], rep["negatives_caught"] = positives_ok, negatives_caught
    if positives_ok and negatives_caught:
        rep["timing"] = {}
        song_ids = enc(next(r for r in reversed(built[44]) if r)["b"]["tokens"])
        for name in [n for n in toks if n.startswith("longP")]:
            P_ids, N_ids, cont, changed = toks[name]
            V = len(P_ids) - 1
            start = reuse_start(V, common_prefix(P_ids, N_ids), len(N_ids), changed)
            dec = (cont + song_ids)[:max(NS) - 1]
            for n in NS:
                times = []
                for r in range(REPS + 1):
                    state(P_ids, V)
                    t = time.perf_counter()
                    run(N_ids[start:], start, kv_end=len(N_ids) - 1)
                    for i in range(n - 1):
                        run([dec[i]], len(N_ids) + i)
                    if r:
                        times.append((time.perf_counter() - t) * 1000)
                rep["timing"][f"{name}/n{n}"] = {"P_reuse": start, "S": len(N_ids) - start, "n": n, "median_ms": round(statistics.median(times), 1),
                                                  "max_ms": round(max(times), 1), "runs_ms": [round(x, 1) for x in times],
                                                  "within_deadline_median": statistics.median(times) <= DEADLINE_MS}
                print(name, n, rep["timing"][f"{name}/n{n}"], flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f, indent=1)
    print("equivalence_ok", positives_ok, "negatives_caught", negatives_caught)
    if not (positives_ok and negatives_caught):
        sys.exit(1)


if __name__ == "__main__":
    main()
