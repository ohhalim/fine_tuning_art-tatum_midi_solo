#!/usr/bin/env python3
"""Is q8 sparser than fp32 in free generation? (issue #1832, docs/experiments/Q8_DENSITY.md)

Generation is scripts/q8_free_gen.py --generate unchanged (same sampler and settings as #1823), on new seeds.
This script only compares: one primary statistic (notes per second) with a Hodges–Lehmann paired difference and its
99% signed-rank confidence interval, and the two 2-beat window counts as secondary statistics.

  q8_density.py --compare <fp32 jsonl> <q8 jsonl> <report json>
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from q8_free_gen import load, wilcoxon  # noqa: E402

PRIMARY = "notes_per_s"
SECONDARY = ("win_max", "win_median")
ALPHA = 0.01
Z_995 = 2.5758293035489004              # two-sided 99%
MARGIN = 0.3                            # notes per second: the smallest difference treated as practical (doc 5)
PROMPTS = (0, 1, 2, 3)
SEEDS = tuple(range(1000, 1192))        # disjoint from #1823 (1-32)


def hodges_lehmann(diffs, z=Z_995):
    """Median of the Walsh averages (d_i + d_j) / 2, i <= j, and the signed-rank confidence interval from the normal
    approximation: with M = n(n+1)/2 sorted Walsh averages and k = floor(n(n+1)/4 - z * sqrt(n(n+1)(2n+1)/24)),
    the interval is [W_(k+1), W_(M-k)] (1-based). All differences are used, zeros included."""
    n = len(diffs)
    if n == 0:
        return None, (None, None)
    walsh = sorted((diffs[i] + diffs[j]) / 2 for i in range(n) for j in range(i, n))
    m = len(walsh)
    k = math.floor(n * (n + 1) / 4 - z * math.sqrt(n * (n + 1) * (2 * n + 1) / 24))
    k = max(0, min(k, (m - 1) // 2))
    return statistics.median(walsh), (walsh[k], walsh[m - 1 - k])


def verdict(lo, hi, margin=MARGIN):
    """Decision on the primary 99% interval, fixed before the run (doc 5)."""
    within = lo >= -margin and hi <= margin
    if hi < 0:
        return "q8 성김 검출, 실용 차이 범위 안" if within else "q8 성김 검출, 실용 차이 범위 밖 가능"
    if lo > 0:
        return "q8 조밀함 검출, 실용 차이 범위 안" if within else "q8 조밀함 검출, 실용 차이 범위 밖 가능"
    return "실용 차이 없음" if within else "미결"


def stat_block(A, B, keys, s):
    rows = [(k, A[k]["stats"][s], B[k]["stats"][s]) for k in keys
            if A[k]["stats"][s] is not None and B[k]["stats"][s] is not None]
    diffs = [y - x for _, x, y in rows]
    hl, (lo, hi) = hodges_lehmann(diffs)
    w = wilcoxon(diffs)
    return {"pairs": len(rows), "fp32_median": statistics.median(x for _, x, _ in rows),
            "q8_median": statistics.median(y for _, _, y in rows), "hl_diff": hl, "ci99": [lo, hi], "wilcoxon": w,
            "significant": w["p"] < ALPHA,
            "per_prompt_hl": {str(p): hodges_lehmann([y - x for k, x, y in rows if k[0] == p])[0]
                              for p in sorted({k[0] for k, _, _ in rows})}}


def compare(fp32_path, q8_path, report_path):
    A, B = load(fp32_path, "fp32"), load(q8_path, "q8")
    expected = {(p, s) for p in PROMPTS for s in SEEDS}
    keys = sorted(set(A) & set(B) & expected)
    rep = {"issue": 1832, "rule": {"primary": PRIMARY, "secondary": SECONDARY, "alpha": ALPHA, "ci": 0.99, "margin": MARGIN},
           "pairs": len(keys), "missing": sorted(expected - (set(A) & set(B))),
           "valid": {"fp32": sum(A[k]["valid"] for k in keys), "q8": sum(B[k]["valid"] for k in keys)},
           "stop": {"fp32": sorted({A[k]["stop"] for k in keys}), "q8": sorted({B[k]["stop"] for k in keys})},
           "seconds_mean": {"fp32": statistics.mean(A[k]["seconds"] for k in keys), "q8": statistics.mean(B[k]["seconds"] for k in keys)},
           "stats": {s: stat_block(A, B, keys, s) for s in (PRIMARY,) + SECONDARY}}
    p = rep["stats"][PRIMARY]
    lo, hi = p["ci99"]
    rep["verdict"] = {"primary": verdict(lo, hi), "replicates_1823_direction": p["hl_diff"] < 0,
                      "all_valid": rep["valid"]["fp32"] == rep["valid"]["q8"] == len(keys), "complete": not rep["missing"]}
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"pairs": len(keys), "hl": p["hl_diff"], "ci99": p["ci99"], "p": p["wilcoxon"]["p"],
                      "r": p["wilcoxon"]["r"], **rep["verdict"]}, ensure_ascii=False))
    return rep


def main() -> None:
    if sys.argv[1] == "--compare":
        compare(*sys.argv[2:5])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
