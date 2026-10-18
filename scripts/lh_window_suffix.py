#!/usr/bin/env python3
"""Suffix recompute under left-hand window variants, cost only (issue #1820, docs/experiments/LH_WINDOW_SUFFIX.md).

Rebuilds the #1793 requests with lh_context.build_request(w_ms=W) for W = 7.5 s (current), 15 s and no expiry
(window start = crop base) and counts consecutive pairs as #1793 did. No model.

  lh_window_suffix.py <export dir> <velocity dir> <#1793 count json> <#1793 cost json> <out json>
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
from lh_kv_reuse import build_songs, file_sha, pair_rows, summary  # noqa: E402

VARIANTS = (("W7.5", 7500.0), ("W15", 15000.0), ("no_expiry", math.inf))


def recompute_per_pair(rows):
    """Mean tokens recomputed per pair: S for a pair that keeps a prefix, the whole request L after a crop change."""
    return statistics.fmean(x["L"] if x["cause"] == "crop" else x["S"] for x in rows) if rows else None


def fit_cost(points):
    """Least squares T = a + b*S + c*S*P over the #1793 measured points [(P, S, T_ms)]. Returns (a, b, c)."""
    rows = [(1.0, s, s * p) for p, s, _ in points]
    t = [x[2] for x in points]
    n = 3
    ata = [[sum(r[i] * r[j] for r in rows) for j in range(n)] for i in range(n)]
    atb = [sum(r[i] * y for r, y in zip(rows, t)) for i in range(n)]
    for i in range(n):                                          # Gauss-Jordan on the 3x3 normal equations
        piv = max(range(i, n), key=lambda r: abs(ata[r][i]))
        ata[i], ata[piv], atb[i], atb[piv] = ata[piv], ata[i], atb[piv], atb[i]
        for r in range(n):
            if r != i:
                f = ata[r][i] / ata[i][i]
                ata[r] = [x - f * y for x, y in zip(ata[r], ata[i])]
                atb[r] -= f * atb[i]
    return tuple(atb[i] / ata[i][i] for i in range(n))


def cost_at(coef, P, S):
    a, b, c = coef
    return a + b * S + c * S * P


def variant_report(built, fails, coef):
    rows = [dict(x, song=i) for i, reqs in built.items() for x in pair_rows(reqs, "b")]
    nc = [x for x in rows if x["cause"] != "crop"]
    reqs = [r for rs in built.values() for r in rs if r is not None]
    P_med = statistics.median(x["P"] for x in nc) if nc else None
    S = summary([x["S"] for x in nc])
    return {"build_failures": fails, "pairs": len(rows), "causes": dict(collections.Counter(x["cause"] for x in rows)),
            "crop_pairs": sum(1 for x in rows if x["cause"] == "crop"),
            "extra_crop_requests": sum(1 for r in reqs if r["extra_crop"] is not None),
            "P": summary([x["P"] for x in nc]), "S": S, "L": summary([x["L"] for x in rows]),
            "S_by_cause": {c: summary([x["S"] for x in nc if x["cause"] == c]) for c in sorted({x["cause"] for x in nc})},
            "lh_notes": summary([r["lh_notes"] for r in reqs]),
            "recompute_tokens_per_pair": recompute_per_pair(rows),
            "est_suffix_ms": None if S is None else {"at_P_median": P_med, "S_p90": cost_at(coef, P_med, S["p90"]),
                                                     "S_max": cost_at(coef, P_med, S["max"])},
            "rows": rows}


def main() -> None:
    export, velocity_dir, count_path, cost_path, out_path = sys.argv[1:6]
    base = json.load(open(count_path))["b"]
    cost = json.load(open(cost_path))
    coef = fit_cost([(p["P"], p["S"], p["T_ms"]) for p in cost["points"]])
    rep = {"issue": 1820, "inputs": {"count_sha256": file_sha(count_path), "cost_sha256": file_sha(cost_path)},
           "cost_fit": {"form": "T = a + b*S + c*S*P (ms, fp32 teacher forcing, #1793 4 points)", "coef": coef,
                        "residuals_ms": [p["T_ms"] - cost_at(coef, p["P"], p["S"]) for p in cost["points"]]},
           "crop_full_prefill_ms_1793": cost["crop_pairs_full_prefill_ms"], "variants": {}}
    for name, w in VARIANTS:
        built, fails = build_songs(export, velocity_dir, w_ms=w)
        v = variant_report(built, fails, coef)
        rep["variants"][name] = v
        print(name, v["pairs"], v["causes"], "S", v["S"], "crop", v["crop_pairs"], "extra", v["extra_crop_requests"],
              "recompute/pair", round(v["recompute_tokens_per_pair"], 1), flush=True)
    cur = rep["variants"]["W7.5"]
    rep["validation_matches_1793"] = (cur["pairs"] == base["pairs"] and cur["causes"] == base["causes"] and cur["S"] == base["S"]
                                      and cur["P"] == base["P"] and cur["L"] == base["L"])
    if not rep["validation_matches_1793"]:
        print("W7.5 does not reproduce #1793 count: stop", file=sys.stderr)
    for name in ("W15", "no_expiry"):
        v = rep["variants"][name]
        v["vs_W7.5"] = {"S_p90_ratio": v["S"]["p90"] / cur["S"]["p90"],
                        "recompute_per_pair_ratio": v["recompute_tokens_per_pair"] / cur["recompute_tokens_per_pair"]}
        v["worth_contract_experiment"] = v["vs_W7.5"]["S_p90_ratio"] <= 0.5 and v["vs_W7.5"]["recompute_per_pair_ratio"] <= 0.7
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f)
    print(json.dumps({"validation": rep["validation_matches_1793"],
                      **{n: rep["variants"][n].get("vs_W7.5") for n in ("W15", "no_expiry")}}))
    if not rep["validation_matches_1793"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
