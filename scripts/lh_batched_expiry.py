#!/usr/bin/env python3
"""Batched left-hand expiry, cost only (issue #1831, docs/experiments/LH_BATCHED_EXPIRY.md).

The current window start max(s - 7.5 s, crop base) moves every block, so the oldest left-hand note leaves the context
every block and the first divergence sits early. Here the start is rounded down to a coarser grid (every N blocks, or
5 s segment starts), so the window covers at least 7.5 s and expiry happens only when the grid point moves. Requests are
rebuilt with lh_context.build_request(window_start=...) and counted as #1793 / #1829 did. No model.

  lh_batched_expiry.py <export dir> <velocity dir> <#1793 count json> <#1793 cost json> <out json>
"""
from __future__ import annotations

import collections
import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_contract import BLOCK  # noqa: E402
from lh_kv_reuse import build_songs, file_sha, pair_rows, summary  # noqa: E402
from lh_window_suffix import cost_at, fit_cost, recompute_per_pair  # noqa: E402

W_MS = 7500.0
SEGMENT_MS = 5000.0
LARGE_MS = 100.0                        # a "large-S" pair: estimated suffix time at its own P >= 100 ms
RULE = {"non_expiry_S_p90_ratio": 0.5, "worst_S_ratio": 1.25, "crop_pairs_ratio": 1.5, "large_fraction_ratio": 0.5}


def grid_start(s, base, anchor, step, w_ms=W_MS):
    """Window start rounded down to the grid anchor + i*step, never after s - w_ms (window >= w_ms) nor before base."""
    q = anchor + math.floor((s - w_ms - anchor) / step) * step
    return max(q, base)


def block_grid(n):
    """Start rule for an N-block grid anchored at the block origin t0 = s_k - k*B (k >= 1)."""
    return lambda srow, base: grid_start(srow["s"], base, srow["s"] - srow["k"] * BLOCK, n * BLOCK)


def segment_grid(srow, base):
    """Start rule on 5 s segment starts (the tokenizer's <T> grid, which also holds every crop base)."""
    return grid_start(srow["s"], base, 0.0, SEGMENT_MS)


VARIANTS = (("W7.5", None), ("B2", block_grid(2)), ("B4", block_grid(4)), ("SEG5", segment_grid))


def tag_expiry(rows, reqs):
    """rows: pair_rows of one song; reqs: its built requests. Adds expiry = the window start moved between the pair."""
    by_k = {r["k"]: r for r in reqs if r is not None}
    out = []
    for x in rows:
        cur, prev = by_k[x["k"]], by_k[x["k"] - 1]
        start = lambda r: r["s"] - r["window_ms"]  # noqa: E731
        out.append(dict(x, start=start(cur), expiry=start(cur) != start(prev)))
    return out


def worst_rows(nc):
    """The largest-S pair (the gate) and the largest estimated-time pair, which can differ: the estimate grows with P too
    (Astra review of #1836)."""
    if not nc:
        return None, None
    return max(nc, key=lambda x: x["S"]), max(nc, key=lambda x: x["est_ms"])


def variant_report(built, fails, coef):
    rows = [dict(x, song=i) for i, reqs in built.items() for x in tag_expiry(pair_rows(reqs, "b"), reqs)]
    nc = [x for x in rows if x["cause"] != "crop"]
    reqs = [r for rs in built.values() for r in rs if r is not None]
    for x in nc:
        x["est_ms"] = cost_at(coef, x["P"], x["S"])
    worst, worst_est = worst_rows(nc)
    large = [x for x in nc if x["est_ms"] >= LARGE_MS]
    P_med = statistics.median(x["P"] for x in nc) if nc else None
    S = summary([x["S"] for x in nc])
    return {"build_failures": fails, "pairs": len(rows), "causes": dict(collections.Counter(x["cause"] for x in rows)),
            "crop_pairs": sum(1 for x in rows if x["cause"] == "crop"),
            "extra_crop_requests": sum(1 for r in reqs if r["extra_crop"] is not None),
            "P": summary([x["P"] for x in nc]), "S": S, "L": summary([x["L"] for x in rows]),
            "S_expiry": summary([x["S"] for x in nc if x["expiry"]]), "S_non_expiry": summary([x["S"] for x in nc if not x["expiry"]]),
            "expiry_pairs": sum(1 for x in nc if x["expiry"]), "non_expiry_pairs": sum(1 for x in nc if not x["expiry"]),
            "S_by_cause": {c: summary([x["S"] for x in nc if x["cause"] == c]) for c in sorted({x["cause"] for x in nc})},
            "worst": None if worst is None else {k: worst[k] for k in ("song", "k", "P", "S", "cause", "expiry", "est_ms")},
            "worst_est": None if worst_est is None else {k: worst_est[k] for k in ("song", "k", "P", "S", "cause", "expiry", "est_ms")},
            "large_pairs": len(large), "large_fraction": len(large) / len(nc) if nc else None,
            "window_ms": summary([r["window_ms"] for r in reqs if r["k"] >= 1]),
            "lh_notes": summary([r["lh_notes"] for r in reqs]),
            "recompute_tokens_per_pair": recompute_per_pair(rows),
            "est_suffix_ms": None if S is None else {"at_P_median": P_med, "S_p90": cost_at(coef, P_med, S["p90"]),
                                                     "S_max": cost_at(coef, P_med, S["max"])},
            "rows": rows}


def verdict(v, cur):
    """Doc 4 rule, fixed before the run: every condition must hold."""
    c = {"non_expiry_S_p90": v["S_non_expiry"] is not None
         and v["S_non_expiry"]["p90"] <= RULE["non_expiry_S_p90_ratio"] * cur["S"]["p90"],
         "worst_S": v["S"]["max"] <= RULE["worst_S_ratio"] * cur["S"]["max"],
         "crop_pairs": v["crop_pairs"] <= RULE["crop_pairs_ratio"] * cur["crop_pairs"],
         "large_fraction": v["large_fraction"] <= RULE["large_fraction_ratio"] * cur["large_fraction"]}
    return {"conditions": c, "worth_contract_experiment": all(c.values())}


def main() -> None:
    export, velocity_dir, count_path, cost_path, out_path = sys.argv[1:6]
    base = json.load(open(count_path))["b"]
    cost = json.load(open(cost_path))
    coef = fit_cost([(p["P"], p["S"], p["T_ms"]) for p in cost["points"]])
    rep = {"issue": 1831, "rule": RULE, "large_ms": LARGE_MS, "w_ms": W_MS,
           "inputs": {"count_sha256": file_sha(count_path), "cost_sha256": file_sha(cost_path)},
           "cost_fit": {"form": "T = a + b*S + c*S*P (ms, fp32 teacher forcing, #1793 4 points, as #1829)", "coef": coef},
           "variants": {}}
    for name, rule in VARIANTS:
        built, fails = build_songs(export, velocity_dir, window_start=rule)
        v = variant_report(built, fails, coef)
        rep["variants"][name] = v
        print(name, v["pairs"], "crop", v["crop_pairs"], "S", v["S"], "exp", v["S_expiry"], "non", v["S_non_expiry"],
              "large", v["large_pairs"], round(v["large_fraction"], 3), flush=True)
    cur = rep["variants"]["W7.5"]
    rep["validation_matches_1793"] = (cur["pairs"] == base["pairs"] and cur["causes"] == base["causes"] and cur["S"] == base["S"]
                                      and cur["P"] == base["P"] and cur["L"] == base["L"])
    for name in ("B2", "B4", "SEG5"):
        rep["variants"][name]["verdict"] = verdict(rep["variants"][name], cur)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f)
    print(json.dumps({"validation": rep["validation_matches_1793"],
                      **{n: rep["variants"][n]["verdict"] for n in ("B2", "B4", "SEG5")}}))
    if not rep["validation_matches_1793"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
