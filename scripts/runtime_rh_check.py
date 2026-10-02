#!/usr/bin/env python3
"""Runtime solo-line check for the bebop preset (docs/experiments/BEBOP_RUNTIME.md).

Top line of the dispatched events on their scheduled time axis (the solo), per
run set: density, rests,
phrase length, chord-tone of the solo, share of phrase-final notes on chord
tones (the "clear what it says over which chord" property the user liked in
the textbook-lick example), fallback and misses.
"""
from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

REST_S = 0.3
DATA = {"notes_per_s": 3.71, "rest_share": 0.0756, "phrase_median": 16}     # data/bebop_rh train medians


def solo_stats(report) -> dict:
    import pretty_midi
    from inference.app.fallback import parse_chord
    from inference.control.solo_line import top_notes

    bar = 60.0 / report["bpm"] * report.get("beats_per_bar", 4)
    chords = report["chords"]
    notes = [pretty_midi.Note(velocity=80, pitch=n[0], start=i * bar + n[1], end=i * bar + n[2])
             for b in report["played_bars"] for n in b["notes"] for i in [b["bar"]]]
    line = top_notes(notes)
    total = len(report["played_bars"]) * bar

    def pcs(t):
        root, iv = parse_chord(chords[int(t // bar) % len(chords)])
        return {(root + k) % 12 for k in iv}
    gaps, phrases, cur, prev_end, finals = [], [], 0, 0.0, []
    for k, n in enumerate(line):
        if n.start - prev_end >= REST_S:
            gaps.append(n.start - prev_end)
            if cur:
                phrases.append(cur)
                finals.append(line[k - 1])
            cur = 0
        cur += 1
        prev_end = max(prev_end, n.end)
    if total - prev_end >= REST_S:
        gaps.append(total - prev_end)
    if cur:
        phrases.append(cur)
        finals.append(line[-1])
    ct = sum(1 for n in line if n.pitch % 12 in pcs(n.start))
    fin = sum(1 for n in finals if n.pitch % 12 in pcs(n.start))
    return {"notes": len(line), "seconds": total, "rest_time": sum(gaps), "phrases": phrases,
            "chord_tone_hits": ct, "final_hits": fin, "finals": len(finals),
            "fallback": report["production"]["fallback_bar_count"],
            "misses": report["scheduler_dispatch_deadline_miss_count"]}


PROGRESSIONS = {"iiVI": ("Dm7,G7,Cmaj7,Cmaj7", 16, 128),
                "blues": ("F7,Bb7,F7,F7,Bb7,Bb7,F7,F7,Gm7,C7,F7,C7", 12, 120),
                "minor": ("Dm7b5,G7,Cm7,Cm7", 16, 128)}
SEEDS = (42, 43)


def check_run_set(run_dirs, preset: str) -> list[str]:
    """Problems with a run set against the preregistered one (empty = usable).

    Exactly one completed solo-line run per progression and seed, without comp,
    from ``<preset>_<tag>_s<seed>``; seed and checkpoint are checked when the
    report records them (#1621 review M1: a single run used to pass)."""
    from scripts.play_personalized import CHECKPOINTS

    problems, seen, legacy = [], {}, []
    for d in run_dirs:
        name = Path(d).name
        try:
            r = json.loads(Path(d, "continuous_report.json").read_text())
        except (OSError, ValueError) as exc:
            problems.append(f"{name}: no report ({exc})")
            continue
        try:
            p, tag, s = name.rsplit("_", 2)
            seed = int(s.lstrip("s"))
        except ValueError:
            problems.append(f"{name}: name is not <preset>_<tag>_s<seed>")
            continue
        if p != preset or tag not in PROGRESSIONS or seed not in SEEDS:
            problems.append(f"{name}: not in the preregistered set")
            continue
        chords, bars, bpm = PROGRESSIONS[tag]
        if ",".join(r.get("chords") or []) != chords or r.get("bars") != bars or r.get("bpm") != bpm:
            problems.append(f"{name}: chords/bars/bpm differ")
        if not r.get("run_completed") or r.get("completed_bars") != r.get("bars"):
            problems.append(f"{name}: not completed")
        if not r.get("solo_line") or r.get("comp"):
            problems.append(f"{name}: needs --solo-line without --comp")
        # Preregistered decoding (#1627 review M2): T 1.0, no carried context, no pattern cache.
        if (r.get("temperature") != 1.0 or r.get("context_carry_tokens") != 0
                or r.get("context_history") is not False or r.get("pattern_cache") is not False):
            problems.append(f"{name}: decoding settings differ from the preregistered default")
        if "seed" not in r or "checkpoint" not in r:
            legacy.append(name)
        if "seed" in r and r["seed"] != seed:
            problems.append(f"{name}: report seed {r['seed']}")
        if r.get("checkpoint") and not str(r["checkpoint"]).endswith(CHECKPOINTS[preset]):
            problems.append(f"{name}: checkpoint {r['checkpoint']}")
        if (tag, seed) in seen:
            problems.append(f"{name}: duplicates {seen[(tag, seed)]}")
        seen[(tag, seed)] = name
    missing = [f"{preset}_{t}_s{s}" for t in PROGRESSIONS for s in SEEDS if (t, s) not in seen]
    problems += [f"{m}: missing" for m in missing]
    if legacy:
        # Runs from before seed/checkpoint were recorded (#1620): checked by name only.
        print(f"legacy runs (seed/checkpoint checked by directory name only): {legacy}", file=sys.stderr)
    return problems


def timing(run_dirs) -> dict:
    """Actual dispatch timing, apart from the scheduled-axis metrics (#1621 review M2)."""
    rs = [json.loads(Path(d, "continuous_report.json").read_text()) for d in run_dirs]
    return {"deadline_miss_total": sum(r["scheduler_dispatch_deadline_miss_count"] for r in rs),
            "max_lateness_ms": max((r["dispatch_attempt_lateness_summary_ms"] or {}).get("maximum") or 0
                                   for r in rs) if rs else None,
            "runs_with_miss": sum(1 for r in rs if r["scheduler_dispatch_deadline_miss_count"])}


def pooled(rows) -> dict:
    secs = sum(r["seconds"] for r in rows)
    ph = [x for r in rows for x in r["phrases"]]
    notes = sum(r["notes"] for r in rows)
    return {"notes_per_s": notes / secs, "rest_share": sum(r["rest_time"] for r in rows) / secs,
            "phrase_median": statistics.median(ph) if ph else None,
            "solo_chord_tone": sum(r["chord_tone_hits"] for r in rows) / notes if notes else None,
            "phrase_final_chord_tone": (sum(r["final_hits"] for r in rows) / sum(r["finals"] for r in rows))
            if sum(r["finals"] for r in rows) else None,
            "fallback_total": sum(r["fallback"] for r in rows), "misses_total": sum(r["misses"] for r in rows),
            "runs": len(rows)}


def judge(c: dict, base: dict) -> dict:
    ok = {
        "density_0.67_1.5x_data": 0.67 * DATA["notes_per_s"] <= c["notes_per_s"] <= 1.5 * DATA["notes_per_s"],
        "rest_half_of_data": c["rest_share"] >= 0.5 * DATA["rest_share"],
        "phrase_at_most_2x_data": c["phrase_median"] is not None and c["phrase_median"] <= 2 * DATA["phrase_median"],
        "chord_tone_guard": c["solo_chord_tone"] is not None and base["solo_chord_tone"] is not None
        and c["solo_chord_tone"] >= base["solo_chord_tone"] - 0.03,
        "no_fallback": c["fallback_total"] == 0,
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True, help="directory with <preset>_<tag>_s<seed> runs")
    ap.add_argument("--candidate", default="bebop")
    ap.add_argument("--baseline", default="tatum")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    dirs = {p: sorted(d for d in glob.glob(str(args.runs_dir / f"{p}_*")) if Path(d).is_dir())
            for p in (args.candidate, args.baseline)}
    problems = {p: check_run_set(ds, p) for p, ds in dirs.items()}
    if any(problems.values()):
        print(json.dumps({"refused": problems}, indent=1))
        return 2
    load = lambda ds: [solo_stats(json.loads(Path(d, "continuous_report.json").read_text())) for d in ds]
    cand, base = pooled(load(dirs[args.candidate])), pooled(load(dirs[args.baseline]))
    out = {"schema": "runtime_rh_check_v2", "time_axis": "scheduled (dispatch target times)",
           "data_targets": DATA, "candidate": cand, "baseline": base,
           "timing": {p: timing(ds) for p, ds in dirs.items()},
           "verdict": judge(cand, base), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
