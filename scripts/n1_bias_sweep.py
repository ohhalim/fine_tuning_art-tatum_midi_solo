#!/usr/bin/env python3
"""N=1 + carry 48 + avoid bias 2: preregistered repeat-penalty sweep, then confirmation (docs/experiments/N1_BIAS_SWEEP.md).

Stage 1 (selection, seeds 42/43): arms R0, R0.5, R1 under ``<runs-dir>/select/R<r>``.
Selection rule (fixed in advance): the smallest R whose arm has same-note share <= .109
and within-block step p90 <= 12; none -> stop. Stage 2 (confirmation, seeds 44/45):
the chosen R under ``<runs-dir>/confirm/R<r>`` against every gate. Reference values are
the bias0 arm of #1656 (the current listening setup: candidates 2, no bias).
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

from scripts.boundary_check import PROGRESSIONS  # noqa: E402
from scripts.dissonance_check import arm  # noqa: E402

ARMS = (0.0, 0.5, 1.0)
REF = {"avoid_share": 0.161, "clash_share": 0.122, "on_beat_clash": 0.281, "same_note_share": 0.109,
       "within_p90": 10, "distinct_openings": 0.892}
DATA_NOTES_PER_S, DATA_REST, REAL_GE9 = 3.71, 0.0756, 0.11
EXPECT = {"bpm": 128, "bars": 16, "candidates": 1, "comp_style": "varied", "solo_line": True,
          "context_carry_tokens": 48, "context_carry_position": "after", "temperature": 1.0,
          "context_history": False, "harmony_bias": 2.0}


def check_set(dirs, seeds, repeat: float) -> list[str]:
    problems, seen = [], set()
    for d in dirs:
        name = Path(d).name
        try:
            _, tag, s = name.split("_")
            r = json.loads(Path(d, "continuous_report.json").read_text())
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        if tag not in PROGRESSIONS or int(s.lstrip("s")) not in seeds or ",".join(r["chords"]) != PROGRESSIONS[tag]:
            problems.append(f"{name}: not preregistered")
        for k, v in {**EXPECT, "seed": int(s.lstrip("s")), "repeat_penalty": repeat}.items():
            if r.get(k) != v:
                problems.append(f"{name}: {k} = {r.get(k)!r}, expected {v!r}")
        if (r.get("phrase_breath") or {}).get("max_notes") != 24 or not r.get("comp_trace"):
            problems.append(f"{name}: breath / comp trace missing")
        if not r.get("run_completed") or r.get("completed_bars") != r.get("bars"):
            problems.append(f"{name}: not completed")
        if (tag, s) in seen:
            problems.append(f"{name}: duplicate")
        seen.add((tag, s))
    problems += [f"missing {t}_s{s}" for t in PROGRESSIONS for s in seeds if (t, f"s{s}") not in seen]
    return problems


def select(stats: dict) -> float | None:
    ok = [r for r in ARMS if stats[r]["same_note_share"] <= REF["same_note_share"]
          and stats[r]["within_p90"] <= REF["within_p90"] + 2]
    return min(ok) if ok else None


def judge(b: dict) -> dict:
    ok = {
        "avoid_le_half_ref": b["avoid_share"] <= 0.5 * REF["avoid_share"],
        "clash_le_0.7_ref": b["clash_share"] <= 0.7 * REF["clash_share"],
        "on_beat_clash_le_ref": b["on_beat_clash"] <= REF["on_beat_clash"],
        "boundary_ok": b["across_ge9"] <= 2 * REAL_GE9 and b["across_p50"] <= 4,
        "same_note_le_ref": b["same_note_share"] <= REF["same_note_share"],
        "within_p90_le_ref_plus_2": b["within_p90"] <= REF["within_p90"] + 2,
        "variety": b["distinct_openings"] >= 0.9 * REF["distinct_openings"] and b["copied_block_share"] <= 0.05,
        "density_and_rest": (0.67 * DATA_NOTES_PER_S <= b["notes_per_s"] <= 1.5 * DATA_NOTES_PER_S
                             and b["rest_share"] >= 0.5 * DATA_REST),
        "runtime_ok": b["fallback_total"] == 0 and b["misses_total"] <= 2 and b["invalid"] == 0
                      and b["gen_ms_p99"] is not None and b["gen_ms_p99"] <= 419,
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--stage", choices=["select", "confirm"], required=True)
    ap.add_argument("--repeat", type=float, default=None, help="confirm: the selected R")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.stage == "select":
        stats = {}
        for r in ARMS:
            dirs = sorted(d for d in glob.glob(str(args.runs_dir / "select" / f"R{r:g}" / "bebop_*")) if Path(d).is_dir())
            problems = check_set(dirs, (42, 43), r)
            if problems:
                print(json.dumps({"refused": problems}, indent=1))
                return 2
            stats[r] = arm(dirs)
        out = {"schema": "n1_bias_sweep_select_v1", "arms": {f"R{r:g}": v for r, v in stats.items()},
               "selected": select(stats), "musical_quality_verified": False}
    else:
        dirs = sorted(d for d in glob.glob(str(args.runs_dir / "confirm" / f"R{args.repeat:g}" / "bebop_*"))
                      if Path(d).is_dir())
        problems = check_set(dirs, (44, 45), args.repeat)
        if problems:
            print(json.dumps({"refused": problems}, indent=1))
            return 2
        b = arm(dirs)
        out = {"schema": "n1_bias_sweep_confirm_v1", "repeat": args.repeat, "arm": b, "verdict": judge(b),
               "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
