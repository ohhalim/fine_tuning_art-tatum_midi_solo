#!/usr/bin/env python3
"""Block-boundary continuity of the played solo, carry 0 vs carry K (docs/experiments/SOLO_CARRY.md).

Solo = played notes not joined to the emitted comp (comp_source_check.tag_run).
Consecutive top-line steps with a gap <= 0.3 s are split into within-block and
across-boundary (half-bar blocks); real bebop right hands: p50 2, >= 9 semitones 11%.
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

PROGRESSIONS = {"iiVIF": "Gm7,C7,Fmaj7,Fmaj7", "rhythmA": "Bbmaj7,G7,Cm7,F7", "minorA": "Bm7b5,E7,Am7,Am7"}
SEEDS = (42, 43)
REAL_GE9 = 0.11
DATA_NOTES_PER_S, DATA_REST = 3.71, 0.0756


def run_stats(r) -> dict:
    import pretty_midi
    from inference.app.fallback import parse_chord
    from inference.control.solo_line import top_notes
    from scripts.comp_source_check import solo_metrics, tag_run

    tagged = tag_run(r)
    half = 60.0 / r["bpm"] * 2
    beat = 60.0 / r["bpm"]
    bar = 2 * half
    line = top_notes([pretty_midi.Note(velocity=80, pitch=p, start=s, end=e) for p, s, e in tagged["solo"]])
    within, across, same = [], [], 0
    for a, b in zip(line, line[1:]):
        if b.start - a.end > 0.3:
            continue
        step = abs(b.pitch - a.pitch)
        same += int(step == 0)
        (across if int(b.start // half + 1e-9) != int(a.start // half + 1e-9) else within).append(step)
    blocks = {}
    for n in line:
        blocks.setdefault(int(n.start // half + 1e-9), []).append(n)
    first_rel, copies, prev_seq = [], 0, None
    for b in sorted(blocks):
        ns = blocks[b]
        seq = tuple(n.pitch for n in ns)
        copies += int(seq == prev_seq and len(seq) >= 3)
        prev_seq = seq
        if len(ns) >= 3:
            first_rel.append(ns[0].pitch - statistics.median(n.pitch for n in ns))
    on = clash = 0
    for n in line:
        if abs(n.start - round(n.start / beat) * beat) > 0.03:
            continue
        root, iv = parse_chord(r["chords"][int(n.start // bar) % len(r["chords"])])
        pcs = {(root + i) % 12 for i in iv}
        on += 1
        clash += int(min(min((n.pitch - q) % 12, (q - n.pitch) % 12) for q in pcs) == 1)
    sm = solo_metrics(tagged["solo"], tagged["seconds"])
    return {"within": within, "across": across, "same_steps": same, "steps": len(within) + len(across),
            "first_rel": first_rel, "copies": copies, "blocks": len(blocks), "on": on, "clash": clash,
            "notes": sm["notes"], "rest_time": sm["rest_time"], "seconds": sm["seconds"],
            "fallback": tagged["fallback"], "misses": tagged["misses"], "gen_ms": tagged["gen_ms"],
            "render": tagged["render"]}


def pooled(rows) -> dict:
    q = lambda v, p: sorted(v)[int(p * (len(v) - 1))] if v else None
    within = [x for r in rows for x in r["within"]]
    across = [x for r in rows for x in r["across"]]
    first = [x for r in rows for x in r["first_rel"]]
    gen = sorted(g for r in rows for g in r["gen_ms"])
    return {"across_n": len(across), "across_p50": q(across, .5), "across_p90": q(across, .9),
            "across_ge9": sum(x >= 9 for x in across) / len(across) if across else None,
            "within_p50": q(within, .5), "within_p90": q(within, .9),
            "first_minus_median_p50": q(first, .5),
            "same_note_share": sum(r["same_steps"] for r in rows) / max(1, sum(r["steps"] for r in rows)),
            "copied_block_share": sum(r["copies"] for r in rows) / max(1, sum(r["blocks"] for r in rows)),
            "on_beat_clash": sum(r["clash"] for r in rows) / max(1, sum(r["on"] for r in rows)),
            "notes_per_s": sum(r["notes"] for r in rows) / sum(r["seconds"] for r in rows),
            "rest_share": sum(r["rest_time"] for r in rows) / sum(r["seconds"] for r in rows),
            "fallback_total": sum(r["fallback"] for r in rows), "misses_total": sum(r["misses"] for r in rows),
            "invalid": sum(r["render"].get("rendered_invalid", 0) + r["render"].get("raw_invalid", 0) for r in rows),
            "gen_ms_p99": gen[min(len(gen) - 1, int(0.99 * len(gen)))] if gen else None}


def judge(a: dict, b: dict) -> dict:
    ok = {
        "across_ge9_le_2x_real": b["across_ge9"] <= 2 * REAL_GE9,
        "across_p50_le_4": b["across_p50"] <= 4,
        "within_p90_not_worse": b["within_p90"] <= a["within_p90"] + 2,
        "on_beat_clash_not_worse": b["on_beat_clash"] <= a["on_beat_clash"] + 0.03,
        "same_note_guard": b["same_note_share"] <= a["same_note_share"] + 0.05,
        "copy_guard": b["copied_block_share"] <= 0.05,
        "density_and_rest": (0.67 * DATA_NOTES_PER_S <= b["notes_per_s"] <= 1.5 * DATA_NOTES_PER_S
                             and b["rest_share"] >= 0.5 * DATA_REST),
        "runtime_ok": b["fallback_total"] == 0 and b["misses_total"] <= 2 and b["invalid"] == 0
                      and b["gen_ms_p99"] is not None and b["gen_ms_p99"] <= 419,
    }
    return {**ok, "pass": all(ok.values())}


def check_set(dirs, carry: int) -> list[str]:
    problems, seen = [], set()
    for d in dirs:
        name = Path(d).name
        try:
            _, tag, s = name.split("_")
            r = json.loads(Path(d, "continuous_report.json").read_text())
        except (ValueError, OSError) as exc:
            problems.append(f"{name}: {exc}")
            continue
        exp = {"bpm": 128, "bars": 16, "seed": int(s.lstrip("s")), "context_carry_tokens": carry,
               "candidates": 1, "comp_style": "varied", "solo_line": True, "temperature": 1.0,
               "context_history": False}
        if tag not in PROGRESSIONS or ",".join(r.get("chords", [])) != PROGRESSIONS[tag]:
            problems.append(f"{name}: not preregistered")
        for k, v in exp.items():
            if r.get(k) != v:
                problems.append(f"{name}: {k} = {r.get(k)!r}, expected {v!r}")
        if carry and r.get("context_carry_position") != "after":
            problems.append(f"{name}: carry position {r.get('context_carry_position')}")
        if (r.get("phrase_breath") or {}).get("max_notes") != 24 or not r.get("comp_trace"):
            problems.append(f"{name}: breath / comp trace missing")
        if not r.get("run_completed") or r.get("completed_bars") != r.get("bars"):
            problems.append(f"{name}: not completed")
        if (tag, s) in seen:
            problems.append(f"{name}: duplicate")
        seen.add((tag, s))
    problems += [f"missing {t}_s{s}" for t in PROGRESSIONS for s in SEEDS if (t, f"s{s}") not in seen]
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", type=Path, required=True)
    ap.add_argument("--carry", type=int, default=48)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    arms = {}
    for label, carry in (("carry0", 0), (f"carry{args.carry}", args.carry)):
        dirs = sorted(d for d in glob.glob(str(args.runs_dir / label / "bebop_*")) if Path(d).is_dir())
        problems = check_set(dirs, carry)
        if problems:
            print(json.dumps({"refused": problems}, indent=1))
            return 2
        arms[label] = pooled([run_stats(json.loads(Path(d, "continuous_report.json").read_text())) for d in dirs])
    a, b = arms["carry0"], arms[f"carry{args.carry}"]
    report = {"schema": "boundary_check_v1", "arms": arms, "verdict": judge(a, b), "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
