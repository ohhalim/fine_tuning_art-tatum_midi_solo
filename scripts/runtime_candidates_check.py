#!/usr/bin/env python3
"""Runtime budget and played-line checks for --candidates (docs/experiments/RUNTIME_CANDIDATES.md).

Run dirs ``<runs-dir>/N<n>/bebop_<tag>_s<seed>``. Per arm: block-ready
(generation_ms) p95/p99 pooled over blocks, deadline misses, fallback, the
played top line's density and on-beat clash against the chord of its bar.
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

PROGRESSIONS = {"iiVIF": "Gm7,C7,Fmaj7,Fmaj7", "rhythmA": "Bbmaj7,G7,Cm7,F7", "minorA": "Bm7b5,E7,Am7,Am7"}
SEEDS = (42, 43)
DATA_NOTES_PER_S = 3.71


def played_line(report):
    import pretty_midi
    from inference.control.solo_line import top_notes

    bar = 60.0 / report["bpm"] * report.get("beats_per_bar", 4)
    notes = [pretty_midi.Note(velocity=80, pitch=n[0], start=b["bar"] * bar + n[1], end=b["bar"] * bar + n[2])
             for b in report["played_bars"] for n in b["notes"]]
    return [n for n in top_notes(notes) if n.pitch >= 55], bar


def run_stats(report) -> dict:
    from inference.app.fallback import parse_chord

    line, bar = played_line(report)
    beat = bar / report.get("beats_per_bar", 4)
    chords = report["chords"]
    on, clash = 0, 0
    for n in line:
        k = round(n.start / beat)
        if abs(n.start - k * beat) > 0.03:
            continue
        root, iv = parse_chord(chords[int(n.start // bar) % len(chords)])
        pcs = {(root + i) % 12 for i in iv}
        on += 1
        clash += int(min(min((n.pitch - q) % 12, (q - n.pitch) % 12) for q in pcs) == 1)
    return {"notes": len(line), "seconds": len(report["played_bars"]) * bar, "on_beat": on, "on_beat_clash": clash,
            "misses": report["scheduler_dispatch_deadline_miss_count"],
            "fallback": report["production"]["fallback_bar_count"],
            "completed": bool(report["run_completed"]) and report["completed_bars"] == report["bars"]}


EXPECT = {"bpm": 128, "bars": 16, "temperature": 1.0, "context_carry_tokens": 0, "context_history": False,
          "pattern_cache": False, "start_budget_bars": 0.5, "fetch_margin_ms": 50.0}


def check_set(dirs, n: int) -> list[str]:
    """Exact paired set, completed, preregistered settings (#1640 review: completion was not in the verdict)."""
    from scripts.play_personalized import CHECKPOINTS

    problems, seen = [], set()
    for d in dirs:
        name = Path(d).name
        try:
            _, tag, s = name.split("_")
            r = json.loads(Path(d, "continuous_report.json").read_text())
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        if tag not in PROGRESSIONS or int(s.lstrip("s")) not in SEEDS:
            problems.append(f"{name}: not preregistered")
        elif ",".join(r["chords"]) != PROGRESSIONS[tag]:
            problems.append(f"{name}: chords differ")
        if r.get("candidates") != n or not r.get("solo_line") or r.get("comp") or r.get("phrase_breath"):
            problems.append(f"{name}: settings differ (candidates {r.get('candidates')})")
        for k, v in EXPECT.items():
            if r.get(k) != v:
                problems.append(f"{name}: {k} = {r.get(k)!r}, expected {v!r}")
        if not r.get("run_completed") or r.get("completed_bars") != r.get("bars"):
            problems.append(f"{name}: not completed")
        if r.get("seed") != int(s.lstrip("s")):
            problems.append(f"{name}: report seed {r.get('seed')}")
        if not str(r.get("checkpoint", "")).endswith(CHECKPOINTS["bebop"]):
            problems.append(f"{name}: checkpoint {r.get('checkpoint')}")
        if (tag, s) in seen:
            problems.append(f"{name}: duplicate")
        seen.add((tag, s))
    problems += [f"missing {t}_s{s}" for t in PROGRESSIONS for s in SEEDS if (t, f"s{s}") not in seen]
    return problems


def pctl(vals, q):
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))] if vals else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--candidate-arm", type=int, default=3, help="N of the arm compared with N=1")
    ap.add_argument("--budget-ms", type=float, default=None,
                    help="block-ready p99 limit; default 0.6 x block (the #1639 registration)")
    args = ap.parse_args(argv)
    out = {}
    for n in (1, args.candidate_arm):
        dirs = sorted(d for d in glob.glob(str(args.runs_dir / f"N{n}" / "bebop_*")) if Path(d).is_dir())
        problems = check_set(dirs, n)
        if problems:
            print(json.dumps({"refused": problems}, indent=1))
            return 2
        reports = [json.loads(Path(d, "continuous_report.json").read_text()) for d in dirs]
        stats = [run_stats(r) for r in reports]
        gen = [b["generation_ms"] for r in reports for b in r["bars_detail"] if b.get("generation_ms") is not None]
        out[f"N{n}"] = {
            "runs": len(stats), "completed": sum(s["completed"] for s in stats),
            "fallback_total": sum(s["fallback"] for s in stats), "misses_total": sum(s["misses"] for s in stats),
            "gen_ms_p50": pctl(gen, 0.5), "gen_ms_p95": pctl(gen, 0.95), "gen_ms_p99": pctl(gen, 0.99),
            "gen_samples": len(gen),
            "notes_per_s": sum(s["notes"] for s in stats) / sum(s["seconds"] for s in stats),
            "on_beat_clash": (sum(s["on_beat_clash"] for s in stats) / sum(s["on_beat"] for s in stats)
                              if sum(s["on_beat"] for s in stats) else None),
            "on_beat_clash_n": sum(s["on_beat"] for s in stats),
            "candidate_stats": [r.get("candidate_stats") for r in reports],
        }
    block_ms = 60000.0 / 128 * 2
    a, b = out["N1"], out[f"N{args.candidate_arm}"]
    budget = args.budget_ms if args.budget_ms is not None else 0.6 * block_ms
    if not a["on_beat_clash_n"] or not b["on_beat_clash_n"]:
        print(json.dumps({"indeterminate": "no on-beat notes in an arm"}))
        return 3
    verdict = {
        "fallback_0": b["fallback_total"] == 0,
        "gen_p99_le_budget": b["gen_ms_p99"] is not None and b["gen_ms_p99"] <= budget,
        "misses_le_N1_plus_2": b["misses_total"] <= a["misses_total"] + 2,
        "on_beat_clash_below_N1": b["on_beat_clash"] < a["on_beat_clash"],
        "density_0.67_1.5x_data": 0.67 * DATA_NOTES_PER_S <= b["notes_per_s"] <= 1.5 * DATA_NOTES_PER_S,
    }
    verdict["pass"] = all(verdict.values())
    report = {"schema": "runtime_candidates_check_v1", "block_ms": block_ms, "budget_ms": budget, "arms": out,
              "verdict": verdict,
              "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "candidate_stats"} for k, v in out.items()}, indent=1))
    print(json.dumps(verdict, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
