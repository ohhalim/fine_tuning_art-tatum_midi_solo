#!/usr/bin/env python3
"""Fixed shell comp vs varied comp vs a random negative control, over the same solos (docs/experiments/COMPING.md).

Solos: the played top lines of the N=1 bebop runtime runs on the held-out
progressions (outputs/runtime_candidates/N1). Comp is generated per half-bar
block for the same chords. The random control (R) copies the varied comp's hit
count, lengths and velocities but puts three random pitches at random times:
the gates are only meaningful if R fails them.
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

BPM, BLOCKS_PER_BAR = 128, 2
BEAT = 60.0 / BPM
HALF = 2 * BEAT


def solo_of(report):
    from scripts.runtime_candidates_check import played_line
    line, _ = played_line(report)
    return [(n.pitch, n.start, n.end) for n in line]


def comp_run(chords, bars: int, arm: str, seed: int, ref=None):
    """Hits [(start s, [(pitch, start, end, vel)])] over the run for one arm."""
    from inference.control.comping import comp_half, shell_half

    hits, state, rng = [], {}, random.Random(seed)
    for block in range(bars * BLOCKS_PER_BAR):
        chord = chords[(block // BLOCKS_PER_BAR) % len(chords)]
        t0 = block * HALF
        if arm == "F":
            notes = shell_half(chord, sub_index=block % 2, bpm=BPM)
        else:
            notes, _ = comp_half(chord, block=block, bpm=BPM, seed=seed, state=state)
        groups = {}
        for p, s, e, v in notes:
            groups.setdefault(round(s, 4), []).append((p, t0 + s, t0 + e, v))
        hits += [(t0 + s, g) for s, g in sorted(groups.items())]
    if arm == "R":                                    # same count / lengths / velocities, random everything else
        total = bars * BLOCKS_PER_BAR * HALF
        out = []
        for _, g in hits:
            length = g[0][2] - g[0][1]
            s = rng.uniform(0, total - length)
            out.append((s, [(rng.randint(52, 64), s, s + length, g[0][3]) for _ in range(3)]))
        hits = sorted(out)
    return hits


def metrics(hits, chords, bars: int, solo) -> dict:
    from inference.control.comping import chord_tones

    # chord alignment: each chord change has a hit within its first beat carrying its 3rd and 7th, no foreign pc
    changes, ok = 0, 0
    for block in range(bars * BLOCKS_PER_BAR):
        chord = chords[(block // BLOCKS_PER_BAR) % len(chords)]
        prev = chords[((block - 1) // BLOCKS_PER_BAR) % len(chords)] if block else None
        if chord == prev:
            continue
        changes += 1
        root, (third, fifth, seventh, ninth) = chord_tones(chord)
        allowed = {root, third, fifth, seventh, ninth}
        t0 = block * HALF
        for s, g in hits:
            pcs = {p % 12 for p, _, _, _ in g}
            if t0 - 1e-6 <= s < t0 + BEAT and {third, seventh} <= pcs and pcs <= allowed:
                ok += 1
                break
    upper = [[p for p, _, _, _ in g if p >= 48] for _, g in hits]
    motion = [abs(statistics.mean(a) - statistics.mean(b)) for a, b in zip(upper, upper[1:]) if a and b]
    bar_rhythms = []
    for bar in range(bars):
        bar_rhythms.append(tuple(round((s - bar * 2 * HALF) / BEAT, 2) for s, _ in hits
                                 if bar * 2 * HALF <= s < (bar + 1) * 2 * HALF))
    run, max_run = 1, 1
    for a, b in zip(bar_rhythms, bar_rhythms[1:]):
        run = run + 1 if a == b else 1
        max_run = max(max_run, run)
    vels = [g[0][3] for _, g in hits]
    comp_time = clash_time = 0.0
    for _, g in hits:
        for p, s, e, _ in g:
            comp_time += e - s
            for q, qs, qe in solo:
                ov = min(e, qe) - max(s, qs)
                if ov > 0 and (p == q or (p - q) % 12 in (1, 11)):
                    clash_time += ov
    return {"hits": len(hits), "hits_per_s": len(hits) / (bars * 2 * HALF),
            "chord_alignment": ok / changes if changes else None,
            "voice_motion_p50": statistics.median(motion) if motion else None,
            "max_identical_bar_rhythm_run": max_run, "velocity_std": statistics.pstdev(vels) if vels else None,
            "solo_clash_share": clash_time / comp_time if comp_time else None}


def judge(f: dict, v: dict, r: dict) -> dict:
    ok = {
        "V_alignment_1": v["chord_alignment"] == 1.0,
        "V_motion_le_3": v["voice_motion_p50"] <= 3.0,
        "V_rhythm_run_le_2": v["max_identical_bar_rhythm_run"] <= 2,
        "V_velocity_std_3_10": 3.0 <= v["velocity_std"] <= 10.0,
        "V_hits_per_s_1_3.7": 1.0 <= v["hits_per_s"] <= 3.7,
        "V_clash_le_F_plus_0.02": v["solo_clash_share"] <= f["solo_clash_share"] + 0.02,
    }
    r_fails = r["chord_alignment"] < 1.0 or r["voice_motion_p50"] > 3.0
    return {**ok, "R_fails_alignment_or_motion": r_fails, "pass": all(ok.values()) and r_fails}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solo-runs", default="outputs/runtime_candidates/N1/bebop_*")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    per_arm = {"F": [], "V": [], "R": []}
    dirs = sorted(d for d in glob.glob(str(ROOT / args.solo_runs)) if Path(d).is_dir())
    # Exact preregistered solo set (#1647 review M2): the 6 N=1 runs, completed, settings checked
    from scripts.runtime_candidates_check import check_set
    problems = check_set(dirs, 1)
    if problems:
        print(json.dumps({"refused": problems}, indent=1))
        return 2
    for d in dirs:
        r = json.loads(Path(d, "continuous_report.json").read_text())
        solo, chords, bars, seed = solo_of(r), r["chords"], r["bars"], r["seed"]
        for arm in per_arm:
            per_arm[arm].append(metrics(comp_run(chords, bars, arm, seed), chords, bars, solo))
    pooled = {}
    for arm, rows in per_arm.items():
        pooled[arm] = {k: (statistics.mean(x[k] for x in rows) if k != "max_identical_bar_rhythm_run"
                           else max(x[k] for x in rows)) for k in rows[0]}
    report = {"schema": "comping_eval_v1", "runs": len(per_arm["F"]), "arms": pooled, "per_run": per_arm,
              "verdict": judge(pooled["F"], pooled["V"], pooled["R"]), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"arms": pooled, "verdict": report["verdict"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
