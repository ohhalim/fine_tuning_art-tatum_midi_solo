#!/usr/bin/env python3
"""The listening configuration, all parts together (docs/experiments/COMBINED_CHECK.md).

bebop + --solo-line --comp --comp-style varied --phrase-breath 24 --candidates 2
--context-carry-tokens 48 --context-carry-position after, on the held-out
progressions. Runtime budget, boundary continuity, harmony, comp delivery.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from scripts.boundary_check import PROGRESSIONS, SEEDS, pooled, run_stats  # noqa: E402

DATA_NOTES_PER_S, DATA_REST, REAL_GE9 = 3.71, 0.0756, 0.11
EXPECT = {"bpm": 128, "bars": 16, "candidates": 2, "comp_style": "varied", "solo_line": True,
          "context_carry_tokens": 48, "context_carry_position": "after", "temperature": 1.0,
          "context_history": False}


def check_set(dirs) -> list[str]:
    problems, seen = [], set()
    for d in dirs:
        name = Path(d).name
        try:
            _, tag, s = name.split("_")
            r = json.loads(Path(d, "continuous_report.json").read_text())
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        if tag not in PROGRESSIONS or ",".join(r.get("chords", [])) != PROGRESSIONS[tag]:
            problems.append(f"{name}: not preregistered")
        for k, v in {**EXPECT, "seed": int(s.lstrip("s"))}.items():
            if r.get(k) != v:
                problems.append(f"{name}: {k} = {r.get(k)!r}, expected {v!r}")
        if (r.get("phrase_breath") or {}).get("max_notes") != 24 or not r.get("comp_trace"):
            problems.append(f"{name}: breath / comp trace missing")
        if not r.get("run_completed") or r.get("completed_bars") != r.get("bars"):
            problems.append(f"{name}: not completed")
        if (tag, s) in seen:
            problems.append(f"{name}: duplicate")
        seen.add((tag, s))
    problems += [f"missing {t}_s{s}" for t in PROGRESSIONS for s in SEEDS if (t, f"s{s}") not in seen]
    return problems


def comp_stats(reports) -> dict:
    from scripts.comp_source_check import tag_run
    rows = [tag_run(r) for r in reports]
    emitted = sum(x["emitted"] for x in rows)
    return {"delivery": sum(x["played_comp"] for x in rows) / emitted if emitted else None,
            "composite_alignment": sum(x["union_ok"] for x in rows) / max(1, sum(x["union_n"] for x in rows)),
            "comp_only_alignment": sum(x["align_ok"] for x in rows) / max(1, sum(x["align_n"] for x in rows)),
            "unknown_notes": sum(x["unknown"] for x in rows)}


def judge(p: dict, c: dict) -> dict:
    ok = {
        "runtime_ok": p["fallback_total"] == 0 and p["misses_total"] <= 2 and p["invalid"] == 0
                      and p["gen_ms_p99"] is not None and p["gen_ms_p99"] <= 419,
        "boundary_ge9_le_2x_real": p["across_ge9"] <= 2 * REAL_GE9,
        "boundary_p50_le_4": p["across_p50"] <= 4,
        "on_beat_clash_le_0.397": p["on_beat_clash"] <= 0.397,
        "density_and_rest": (0.67 * DATA_NOTES_PER_S <= p["notes_per_s"] <= 1.5 * DATA_NOTES_PER_S
                             and p["rest_share"] >= 0.5 * DATA_REST),
        "copy_guard": p["copied_block_share"] <= 0.05,
        "comp_delivery_ge_0.99": c["delivery"] is not None and c["delivery"] >= 0.99,
        "composite_alignment_ge_0.95": c["composite_alignment"] >= 0.95,
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    dirs = sorted(d for d in glob.glob(str(args.runs_dir / "bebop_*")) if Path(d).is_dir())
    problems = check_set(dirs)
    if problems:
        print(json.dumps({"refused": problems}, indent=1))
        return 2
    reports = [json.loads(Path(d, "continuous_report.json").read_text()) for d in dirs]
    p = pooled([run_stats(r) for r in reports])
    c = comp_stats(reports)
    report = {"schema": "combined_check_v1", "solo": p, "comp": c, "verdict": judge(p, c),
              "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
