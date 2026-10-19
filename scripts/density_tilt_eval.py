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


REQUIRED_CHECKS = ("no_note_before_previous_horizon", "no_late_intake_rejection", "forwarded_blocks_fit_capacity",
                   "stop_is_recorded_abort_or_end", "safe_stop", "existing_checks")


def first_block_diff(a, b):
    """First block whose row differs; a row present on one side only counts as a difference (Astra review of #1845)."""
    for x, y in zip(a, b):
        if (x["tokens"], x.get("notes"), x["S"]) != (y["tokens"], y.get("notes"), y["S"]):
            return x["block"]
    if len(a) != len(b):
        return (a if len(a) > len(b) else b)[min(len(a), len(b))]["block"]
    return None


def note_ons(snd):
    return [(r["pitch"], r["velocity"], r["due_ms"], r.get("block")) for r in snd["sent"] if r["kind"] == "on"]


def first_note_on_diff(a, b, comparable_until=None):
    """First block whose sent note-on differs. An extra note-on on one side counts only before comparable_until (the
    earlier stop of the two runs: past it the shorter run simply ended)."""
    for x, y in zip(a, b):
        if x[:3] != y[:3]:
            blocks = [v for v in (x[3], y[3]) if v is not None]
            return min(blocks) if blocks else -1
    if len(a) != len(b):
        extra = (a if len(a) > len(b) else b)[min(len(a), len(b))]
        if comparable_until is None or (extra[3] is not None and extra[3] < comparable_until):
            return extra[3] if extra[3] is not None else -1
    return None


def texture(snd):
    """Chord share (note-ons within 30 ms of the previous) and, per block, the largest gap between note-on times. The
    gap is an onset interval, not silence: sounding notes and block edges are ignored (Astra review of #1845)."""
    ons = sorted(r["due_ms"] for r in snd["sent"] if r["kind"] == "on")
    chord = sum(1 for a, b in zip(ons, ons[1:]) if b - a < CHORD_MS) / max(1, len(ons) - 1)
    by_block = {}
    for r in snd["sent"]:
        if r["kind"] == "on" and r.get("block") is not None:
            by_block.setdefault(r["block"], []).append(r["due_ms"])
    rests = [max((b - a for a, b in zip(sorted(v), sorted(v)[1:])), default=0.0) for v in by_block.values() if len(v) > 1]
    return {"chord_share": chord, "median_max_onset_gap_ms": statistics.median(rests) if rests else None}


def sign_test_p(k, n):
    """Two-sided exact sign test."""
    if n == 0:
        return None
    tail = sum(math.comb(n, i) for i in range(0, min(k, n - k) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def run_record(run_dir, arm, s, sc):
    d = os.path.join(run_dir, f"test_{arm}_s{s}", sc)
    rep = json.load(open(os.path.join(run_dir, f"test_{arm}_s{s}.json")))["scenarios"][sc]
    prod, snd = json.load(open(os.path.join(d, "producer.json"))), json.load(open(os.path.join(d, "sender.json")))
    checks = rep.get("rerun_checks") or {}
    return {"rep": rep, "prod": prod, "snd": snd, "abort": prod["abort"], "warm": warm(prod),
            "k1": all(checks.get(k) is True for k in REQUIRED_CHECKS)}


def pair_record(run_dir, s, sc):
    o, t = run_record(run_dir, "off", s, sc), run_record(run_dir, "tilt", s, sc)
    first_tilt = next((b["block"] for b in t["prod"]["blocks"] if b.get("tilted_samples")), None)
    stop = min(len(o["warm"]), len(t["warm"])) + 1          # past the earlier stop the shorter run simply ended
    fb = first_block_diff(o["prod"]["blocks"], t["prod"]["blocks"])
    fn = first_note_on_diff(note_ons(o["snd"]), note_ons(t["snd"]), comparable_until=stop)
    ok = lambda f: f is None or (first_tilt is not None and f >= first_tilt)  # noqa: E731
    return {"seed": s, "scenario": sc, "first_tilt_block": first_tilt, "first_block_row_diff": fb,
            "first_note_on_diff_block": fn, "p0": ok(fb) and ok(fn), "runs": {"off": o, "tilt": t}}


def arm_summary(runs):
    notes = [b["notes"] for r in runs for b in r["warm"] if b.get("notes") is not None]
    sm = sorted(b["samples_to_ready"] for r in runs for b in r["warm"])
    warm_n = sum(len(r["warm"]) for r in runs)
    ms = [r["rep"]["loop_speed"]["median_ms_per_sample"] for r in runs
          if r["rep"].get("loop_speed", {}).get("median_ms_per_sample") is not None]
    tx = [texture(r["snd"]) for r in runs]
    gaps = [x["median_max_onset_gap_ms"] for x in tx if x["median_max_onset_gap_ms"] is not None]
    reason = lambda r, why: bool(r["abort"]) and r["abort"]["reason"] == why  # noqa: E731
    return {"runs": len(runs), "k1_all": all(r["k1"] for r in runs),
            "complete16": sum(1 for r in runs if r["abort"] is None and len(r["warm"]) == 16),
            "deadline_misses": sum(1 for r in runs if reason(r, "deadline_miss")),
            "capacity_failures": sum(1 for r in runs if reason(r, "capacity_failure")),
            "warm_blocks": warm_n, "median_notes": statistics.median(notes), "mean_notes": statistics.fmean(notes),
            "dense_share": sum(1 for x in notes if x >= 11) / len(notes), "max_notes": max(notes),
            "max_samples": sm[-1], "p95_samples": sm[int(0.95 * (len(sm) - 1))],
            "tilted_block_share": sum(1 for r in runs for b in r["warm"] if b.get("tilted_to_ready")) / warm_n if warm_n else None,
            "median_ms_per_sample": statistics.median(ms) if ms else None,
            "chord_share_median": statistics.median(x["chord_share"] for x in tx),
            "max_onset_gap_ms_median": statistics.median(gaps) if gaps else None}


def test(run_dir, out):
    """Fail closed (Astra review of #1845): K1 needs every required check key True; pairs failing P0 are left out of
    T1, T2 and T3; T3 counts only pairs where neither run stopped on capacity_failure (an incomplete partner is not a
    deadline avoided)."""
    pairs = [pair_record(run_dir, s, sc) for s in TEST_SEEDS for sc in SCENARIOS]
    valid = [p for p in pairs if p["p0"]]
    summ = {a: arm_summary([p["runs"][a] for p in valid]) for a in ("off", "tilt")}
    k1 = all(p["runs"][a]["k1"] for p in pairs for a in ("off", "tilt"))
    miss = lambda r: bool(r["abort"]) and r["abort"]["reason"] == "deadline_miss"  # noqa: E731
    cap = lambda r: bool(r["abort"]) and r["abort"]["reason"] == "capacity_failure"  # noqa: E731
    t3 = [p for p in valid if p["first_tilt_block"] is not None and not cap(p["runs"]["off"]) and not cap(p["runs"]["tilt"])]
    off_only = sum(1 for p in t3 if miss(p["runs"]["off"]) and not miss(p["runs"]["tilt"]))
    tilt_only = sum(1 for p in t3 if miss(p["runs"]["tilt"]) and not miss(p["runs"]["off"]))
    o, t = summ["off"], summ["tilt"]
    strip = lambda p: {k: v for k, v in p.items() if k != "runs"} | {  # noqa: E731
        a: (p["runs"][a]["abort"] and [p["runs"][a]["abort"]["reason"], p["runs"][a]["abort"]["block"]]) for a in ("off", "tilt")}
    rep = {"issue": 1844, "pairs": [strip(p) for p in pairs], "p0_violations": [strip(p) for p in pairs if not p["p0"]],
           "pairs_valid": len(valid), "t3_pairs": len(t3), "arms": summ,
           "judge": {"K1": k1, "T1": t["dense_share"] < o["dense_share"] and t["max_notes"] <= o["max_notes"],
                     "T2": t["max_samples"] < o["max_samples"],
                     "T3": {"off_only_miss": off_only, "tilt_only_miss": tilt_only,
                            "sign_p": sign_test_p(min(off_only, tilt_only), off_only + tilt_only)}}}
    json.dump(rep, open(out, "w"), ensure_ascii=False, indent=1)
    print(json.dumps({k: v for k, v in rep.items() if k != "pairs"}, ensure_ascii=False))


if __name__ == "__main__":
    {"--calib": calib, "--test": test}[sys.argv[1]](*sys.argv[2:4])
