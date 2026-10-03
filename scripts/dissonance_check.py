#!/usr/bin/env python3
"""Harmony bias A/B on the listening configuration (docs/experiments/HARMONY_BIAS.md).

Arms ``<runs-dir>/bias0`` and ``<runs-dir>/bias<S>``, each the combined setup
(bebop, solo line, varied comp, breath 24, candidates 2, carry 48 after) on the
held-out progressions. Solo = played notes not joined to the emitted comp.
Primary: solo time on the chord's avoid notes; solo time in a simultaneous
interval-class-1 clash (minor 2nd / major 7th / minor 9th) with a sounding comp note.
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
from scripts.combined_check import EXPECT as COMBINED  # noqa: E402

DATA_NOTES_PER_S, DATA_REST, REAL_GE9 = 3.71, 0.0756, 0.11


def dissonance(r) -> dict:
    from inference.app.fallback import parse_chord
    from inference.control.harmony_bias import AVOID
    from scripts.comp_source_check import tag_run

    t = tag_run(r)
    bar = 60.0 / r["bpm"] * r.get("beats_per_bar", 4)
    played = [(n[0], b["bar"] * bar + n[1], b["bar"] * bar + n[2]) for b in r["played_bars"] for n in b["notes"]]
    solo = t["solo"]
    sset = set(solo)
    comp = [n for n in played if n not in sset]
    tot = avoid = clash = 0.0
    for p, s, e in solo:
        d = max(e - s, 0.0)
        tot += d
        root, iv = parse_chord(r["chords"][int(s // bar) % len(r["chords"])])
        if (p - root) % 12 in AVOID.get(tuple(iv), set()):
            avoid += d
        ov = max([min(e, qe) - max(s, qs) for q, qs, qe in comp if (p - q) % 12 in (1, 11)] + [0.0])
        clash += max(ov, 0.0)
    return {"solo_time": tot, "avoid_time": avoid, "clash_time": clash}


def openings(r) -> list:
    import pretty_midi
    from inference.control.solo_line import top_notes
    from scripts.comp_source_check import tag_run

    half = 60.0 / r["bpm"] * 2
    line = top_notes([pretty_midi.Note(velocity=80, pitch=p, start=s, end=e) for p, s, e in tag_run(r)["solo"]])
    blocks = {}
    for n in line:
        blocks.setdefault(int(n.start // half + 1e-9), []).append(n.pitch)
    return [tuple(b - a for a, b in zip(ps, ps[1:]))[:3] for ps in blocks.values() if len(ps) >= 2]


def arm(dirs) -> dict:
    reports = [json.loads(Path(d, "continuous_report.json").read_text()) for d in dirs]
    p = pooled([run_stats(r) for r in reports])
    dd = [dissonance(r) for r in reports]
    st = sum(x["solo_time"] for x in dd)
    ops = [o for r in reports for o in openings(r)]
    p.update(avoid_share=sum(x["avoid_time"] for x in dd) / st, clash_share=sum(x["clash_time"] for x in dd) / st,
             distinct_openings=len(set(ops)) / max(1, len(ops)))
    return p


def check_set(dirs, bias: float) -> list[str]:
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
        for k, v in {**COMBINED, "seed": int(s.lstrip("s")), "harmony_bias": bias}.items():
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


def judge(a: dict, b: dict) -> dict:
    ok = {
        "avoid_share_le_half": b["avoid_share"] <= 0.5 * a["avoid_share"],
        "clash_share_le_0.7x": b["clash_share"] <= 0.7 * a["clash_share"],
        "on_beat_clash_not_worse": b["on_beat_clash"] <= a["on_beat_clash"] + 0.01,
        "boundary_ok": b["across_ge9"] <= 2 * REAL_GE9 and b["across_p50"] <= 4,
        "variety_kept": b["distinct_openings"] >= 0.9 * a["distinct_openings"]
                        and b["same_note_share"] <= a["same_note_share"] + 0.05 and b["copied_block_share"] <= 0.05,
        "density_and_rest": (0.67 * DATA_NOTES_PER_S <= b["notes_per_s"] <= 1.5 * DATA_NOTES_PER_S
                             and b["rest_share"] >= 0.5 * DATA_REST),
        "runtime_ok": b["fallback_total"] == 0 and b["misses_total"] <= 2 and b["invalid"] == 0
                      and b["gen_ms_p99"] is not None and b["gen_ms_p99"] <= 419,
    }
    return {**ok, "pass": all(ok.values())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--bias", type=float, default=2.0)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    arms = {}
    for label, bias in (("bias0", 0.0), (f"bias{args.bias:g}", args.bias)):
        dirs = sorted(d for d in glob.glob(str(args.runs_dir / label / "bebop_*")) if Path(d).is_dir())
        problems = check_set(dirs, bias)
        if problems:
            print(json.dumps({"refused": problems}, indent=1))
            return 2
        arms[label] = arm(dirs)
    a, b = arms["bias0"], arms[f"bias{args.bias:g}"]
    report = {"schema": "dissonance_check_v1", "arms": arms, "verdict": judge(a, b), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
