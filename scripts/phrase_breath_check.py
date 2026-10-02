#!/usr/bin/env python3
"""Phrase breath A/B on the bebop runtime (docs/experiments/PHRASE_BREATH.md).

A and B are each the preregistered run set (``bebop_<tag>_s<seed>`` under
``<runs-dir>/A`` and ``<runs-dir>/B``); B must have used ``--phrase-breath 24``
and A no breath. Metrics are on the scheduled time axis (#1623).
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    sys.path.insert(0, str(p))

from scripts.runtime_rh_check import DATA, check_run_set, pooled, solo_stats, timing  # noqa: E402

BREATH = 24


def breath_problems(run_dirs, expect) -> list[str]:
    out = []
    for d in run_dirs:
        pb = json.loads(Path(d, "continuous_report.json").read_text()).get("phrase_breath")
        got = pb["max_notes"] if pb else None
        if got != expect:
            out.append(f"{Path(d).name}: phrase_breath {got}, expected {expect}")
    return out


def extras(run_dirs) -> dict:
    rs = [json.loads(Path(d, "continuous_report.json").read_text()) for d in run_dirs]
    kept = sum(r["played_note_count"] for r in rs)
    dropped = sum((r.get("phrase_breath") or {}).get("dropped_notes", 0) for r in rs)
    return {"gen_ms_max": max(r["production"]["generation_ms"]["maximum"] for r in rs),
            "rendered_invalid": sum((r.get("solo_line_render") or {}).get("rendered_invalid", 0) for r in rs),
            "fallback_total": sum(r["production"]["fallback_bar_count"] for r in rs),
            "dropped_share": dropped / (kept + dropped) if kept + dropped else None}


def judge(a: dict, b: dict, ax: dict, bx: dict) -> dict:
    ok = {
        "phrase_median_le_24": b["phrase_median"] is not None and b["phrase_median"] <= BREATH,
        "rest_more_than_A_and_half_data": b["rest_share"] > a["rest_share"] and b["rest_share"] >= 0.5 * DATA["rest_share"],
        "density_0.67_1.5x_data": 0.67 * DATA["notes_per_s"] <= b["notes_per_s"] <= 1.5 * DATA["notes_per_s"],
        "chord_tone_ge_A_minus_0.03": b["solo_chord_tone"] >= a["solo_chord_tone"] - 0.03,
        "render_ok_and_gen_time": bx["rendered_invalid"] == 0 and bx["gen_ms_max"] <= 1.1 * ax["gen_ms_max"],
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    dirs = {arm: sorted(d for d in glob.glob(str(args.runs_dir / arm / "bebop_*")) if Path(d).is_dir())
            for arm in ("A", "B")}
    problems = {arm: check_run_set(ds, "bebop") + breath_problems(ds, None if arm == "A" else BREATH)
                for arm, ds in dirs.items()}
    if any(problems.values()):
        print(json.dumps({"refused": problems}, indent=1))
        return 2
    stats = {arm: [solo_stats(json.loads(Path(d, "continuous_report.json").read_text())) for d in ds]
             for arm, ds in dirs.items()}
    a, b = pooled(stats["A"]), pooled(stats["B"])
    ax, bx = extras(dirs["A"]), extras(dirs["B"])
    blues = {arm: pooled([s for s, d in zip(stats[arm], dirs[arm]) if "_blues_" in d])["phrase_median"]
             for arm in dirs}
    out = {"schema": "phrase_breath_check_v1", "time_axis": "scheduled (dispatch target times)",
           "data_targets": DATA, "A": {**a, **ax}, "B": {**b, **bx}, "blues_phrase_median": blues,
           "timing": {arm: timing(ds) for arm, ds in dirs.items()},
           "verdict": judge(a, b, ax, bx), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
