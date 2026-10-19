#!/usr/bin/env python3
"""Calibration summary and paired test evaluation for the density tilt (issue #1844, docs/experiments/DENSITY_TILT.md).

  density_tilt_eval.py --calib <run dir> <out json>
  density_tilt_eval.py --test <run dir> <out json>
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

LAMS = ("0", "0.1", "0.25", "0.5", "1.0")
CALIB_SEEDS = tuple(range(101, 109))
TEST_SEEDS = tuple(range(1, 17))
SCENARIOS = ("A_normal", "B_crop_early")
CHORD_MS = 30.0


def warm(prod):
    return [b for b in prod["blocks"] if b.get("samples_to_ready") is not None]


def pick_lam(rows, lo=0.15, hi=0.25, max_notes=12):
    """doc 4: the smallest lam with a 15-25% median drop and max <= 12; else max <= 12 closest to 20%; else closest."""
    base = rows["0"]["median_notes"]
    drop = {k: (base - v["median_notes"]) / base for k, v in rows.items() if k != "0"}
    ok = [k for k in drop if lo <= drop[k] <= hi and rows[k]["max_notes"] <= max_notes]
    if ok:
        return min(ok, key=float), drop, "15~25% 감소·최대 12 이하 중 최소"
    capped = [k for k in drop if rows[k]["max_notes"] <= max_notes]
    pool, why = (capped, "최대 12 이하 중 20%에 가장 가까움(감소율 기준 미달)") if capped else (list(drop), "20%에 가장 가까움(두 기준 미달)")
    return min(pool, key=lambda k: abs(drop[k] - 0.20)), drop, why


def calib(run_dir, out):
    rows = {}
    for lam in LAMS:
        notes, smp = [], []
        for s in CALIB_SEEDS:
            p = json.load(open(os.path.join(run_dir, f"cal_l{lam}_s{s}", "A_normal", "producer.json")))
            for b in warm(p):
                if b.get("notes") is not None:
                    notes.append(b["notes"])
                smp.append(b["samples_to_ready"])
        rows[lam] = {"blocks": len(notes), "median_notes": statistics.median(notes), "max_notes": max(notes),
                     "mean_notes": statistics.fmean(notes), "max_samples": max(smp)}
    lam, drop, why = pick_lam(rows)
    rep = {"issue": 1844, "rows": rows, "drop_vs_off": drop, "chosen_lam": lam, "why": why}
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False))


def first_block_diff(a, b):
    for x, y in zip(a, b):
        if (x["tokens"], x.get("notes"), x["S"]) != (y["tokens"], y.get("notes"), y["S"]):
            return x["block"]
    return None


def note_ons(snd):
    return [(r["pitch"], r["velocity"], r["due_ms"], r.get("block")) for r in snd["sent"] if r["kind"] == "on"]


def first_note_on_diff(a, b):
    for x, y in zip(a, b):
        if x[:3] != y[:3]:
            blocks = [v for v in (x[3], y[3]) if v is not None]
            return min(blocks) if blocks else None
    return None


def texture(snd):
    """Chord share (note-ons within 30 ms of the previous) and the median longest rest per block window."""
    ons = sorted(r["due_ms"] for r in snd["sent"] if r["kind"] == "on")
    chord = sum(1 for a, b in zip(ons, ons[1:]) if b - a < CHORD_MS) / max(1, len(ons) - 1)
    by_block = {}
    for r in snd["sent"]:
        if r["kind"] == "on" and r.get("block") is not None:
            by_block.setdefault(r["block"], []).append(r["due_ms"])
    rests = [max((b - a for a, b in zip(sorted(v), sorted(v)[1:])), default=0.0) for v in by_block.values() if len(v) > 1]
    return {"chord_share": chord, "median_longest_rest_ms": statistics.median(rests) if rests else None}


def sign_test_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def test(run_dir, out):
    pairs, arms = [], {a: {"notes": [], "samples": [], "k1": True, "complete": 0, "misses": 0, "capacity": 0,
                           "tilted_blocks": 0, "warm_blocks": 0, "ms": [], "chord": [], "rest": []} for a in ("off", "tilt")}
    for s in TEST_SEEDS:
        for sc in SCENARIOS:
            row = {}
            for arm in ("off", "tilt"):
                d = os.path.join(run_dir, f"test_{arm}_s{s}", sc)
                rep = json.load(open(os.path.join(run_dir, f"test_{arm}_s{s}.json")))["scenarios"][sc]
                prod, snd = json.load(open(os.path.join(d, "producer.json"))), json.load(open(os.path.join(d, "sender.json")))
                A = arms[arm]
                A["k1"] &= all(v is True for v in rep["rerun_checks"].values())
                ab = prod["abort"]
                wb = warm(prod)
                A["complete"] += ab is None and len(wb) == 16
                A["misses"] += bool(ab and ab["reason"] == "deadline_miss")
                A["capacity"] += bool(ab and ab["reason"] == "capacity_failure")
                A["notes"] += [b["notes"] for b in wb if b.get("notes") is not None]
                A["samples"] += [b["samples_to_ready"] for b in wb]
                A["tilted_blocks"] += sum(1 for b in wb if b.get("tilted_to_ready"))
                A["warm_blocks"] += len(wb)
                if rep.get("loop_speed", {}).get("median_ms_per_sample") is not None:
                    A["ms"].append(rep["loop_speed"]["median_ms_per_sample"])
                tx = texture(snd)
                A["chord"].append(tx["chord_share"])
                if tx["median_longest_rest_ms"] is not None:
                    A["rest"].append(tx["median_longest_rest_ms"])
                row[arm] = (prod, snd, ab)
            (po, so, ao), (pt, st, at) = row["off"], row["tilt"]
            first_tilt = next((b["block"] for b in pt["blocks"] if b.get("tilted_samples")), None)
            fb, fn = first_block_diff(po["blocks"], pt["blocks"]), first_note_on_diff(note_ons(so), note_ons(st))
            ok = lambda f: f is None or (first_tilt is not None and f >= first_tilt)  # noqa: E731
            pairs.append({"seed": s, "scenario": sc, "first_tilt_block": first_tilt, "first_block_row_diff": fb,
                          "first_note_on_diff_block": fn, "p0": ok(fb) and ok(fn),
                          "off": ao and [ao["reason"], ao["block"]], "tilt": at and [at["reason"], at["block"]]})
    valid = [p for p in pairs if p["p0"]]
    miss = lambda x: bool(x) and x[0] == "deadline_miss"  # noqa: E731
    off_only = sum(1 for p in valid if p["first_tilt_block"] is not None and miss(p["off"]) and not miss(p["tilt"]))
    tilt_only = sum(1 for p in valid if p["first_tilt_block"] is not None and miss(p["tilt"]) and not miss(p["off"]))
    summ = {}
    for a, A in arms.items():
        sm = sorted(A["samples"])
        summ[a] = {"k1_all": A["k1"], "complete16": A["complete"], "deadline_misses": A["misses"], "capacity_failures": A["capacity"],
                   "median_notes": statistics.median(A["notes"]), "dense_share": sum(1 for x in A["notes"] if x >= 11) / len(A["notes"]), "max_notes": max(A["notes"]), "mean_notes": statistics.fmean(A["notes"]),
                   "max_samples": sm[-1], "p95_samples": sm[int(0.95 * (len(sm) - 1))],
                   "tilted_block_share": A["tilted_blocks"] / A["warm_blocks"] if A["warm_blocks"] else None,
                   "median_ms_per_sample": statistics.median(A["ms"]) if A["ms"] else None,
                   "chord_share_median": statistics.median(A["chord"]), "longest_rest_ms_median": statistics.median(A["rest"]) if A["rest"] else None}
    o, t = summ["off"], summ["tilt"]
    rep = {"issue": 1844, "pairs": pairs, "p0_violations": [p for p in pairs if not p["p0"]], "arms": summ,
           "judge": {"K1": o["k1_all"] and t["k1_all"],
                     "T1": t["dense_share"] < o["dense_share"] and t["max_notes"] <= o["max_notes"],      # doc 6 revision
                     "T2": t["max_samples"] < o["max_samples"],
                     "T3": {"off_only_miss": off_only, "tilt_only_miss": tilt_only, "sign_p": sign_test_p(min(off_only, tilt_only), off_only + tilt_only)}}}
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "pairs"}, ensure_ascii=False))


if __name__ == "__main__":
    {"--calib": calib, "--test": test}[sys.argv[1]](*sys.argv[2:4])
