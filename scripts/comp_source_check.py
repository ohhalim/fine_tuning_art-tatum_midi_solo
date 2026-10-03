#!/usr/bin/env python3
"""Played notes tagged by source, for the combined listening setup (docs/experiments/COMP_SOURCE.md).

Each run's ``comp_trace`` (per generated half-bar block: planned / emitted /
dropped comp) is joined to ``played_bars``: a played note is comp when a note
emitted for its block has the same pitch and starts within 12 ms. Then, on
model-played blocks only: chord alignment of the emitted comp, delivery
(played comp / emitted comp), drop reasons, and solo metrics on the untagged
(solo) notes alone.
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

TOL = 0.012
PROGRESSIONS = {"iiVIF": "Gm7,C7,Fmaj7,Fmaj7", "rhythmA": "Bbmaj7,G7,Cm7,F7", "minorA": "Bm7b5,E7,Am7,Am7"}
SEEDS = (42, 43)
EXPECT = {"bpm": 128, "bars": 16, "candidates": 2, "comp_style": "varied", "solo_line": True}


def tag_run(r) -> dict:
    from inference.control.comping import chord_tones

    bpm, bars = r["bpm"], r["bars"]
    bar_s = 60.0 / bpm * r.get("beats_per_bar", 4)
    half = bar_s / 2
    beat = 60.0 / bpm
    trace = r["comp_trace"]
    sources = [b["source"] for b in r["bars_detail"]]          # per scheduler block (half bar)
    emitted_abs, planned_n, dropped = [], 0, {}
    align_ok = align_n = 0
    prev_chord = None
    for i, t in enumerate(trace):
        if t is None or i >= len(sources) or sources[i] != "model":
            prev_chord = t["chord"] if t else prev_chord
            continue
        planned_n += len(t.get("planned", []))
        for k, v in t.get("dropped", {}).items():
            dropped[k] = dropped.get(k, 0) + v
        t0 = i * half
        emitted_abs += [(p, t0 + s, t0 + e) for p, s, e, _ in t.get("emitted", [])]   # "rest" figure: no comp
        if t["chord"] != prev_chord:
            align_n += 1
            root, (third, fifth, seventh, ninth) = chord_tones(t["chord"])
            by_onset = {}
            for p, s, e, _ in t.get("emitted", []):
                if s < beat:
                    by_onset.setdefault(round(s, 3), set()).add(p % 12)
            if any({third, seventh} <= pcs for pcs in by_onset.values()):
                align_ok += 1
        prev_chord = t["chord"]
    played = [(n[0], b["bar"] * bar_s + n[1], b["bar"] * bar_s + n[2]) for b in r["played_bars"] for n in b["notes"]]
    used, comp, solo = set(), [], []
    for n in played:
        hit = next((j for j, c in enumerate(emitted_abs)
                    if j not in used and c[0] == n[0] and abs(c[1] - n[1]) <= TOL), None)
        if hit is None:
            solo.append(n)
        else:
            used.add(hit)
            comp.append(n)
    return {"emitted": len(emitted_abs), "planned": planned_n, "played_comp": len(comp), "dropped": dropped,
            "align_ok": align_ok, "align_n": align_n, "solo": solo, "seconds": bars * bar_s,
            "fallback": r["production"]["fallback_bar_count"], "misses": r["scheduler_dispatch_deadline_miss_count"],
            "render": r.get("solo_line_render") or {},
            "gen_ms": [b["generation_ms"] for b in r["bars_detail"] if b.get("generation_ms") is not None]}


def solo_metrics(solo, seconds: float) -> dict:
    from scripts.runtime_rh_check import REST_S
    from inference.control.solo_line import top_notes
    import pretty_midi

    line = top_notes([pretty_midi.Note(velocity=80, pitch=p, start=s, end=e) for p, s, e in solo])
    gaps, phrases, cur, prev_end = [], [], 0, 0.0
    for n in line:
        if n.start - prev_end >= REST_S:
            gaps.append(n.start - prev_end)
            if cur:
                phrases.append(cur)
            cur = 0
        cur += 1
        prev_end = max(prev_end, n.end)
    if cur:
        phrases.append(cur)
    return {"notes": len(line), "rest_time": sum(gaps), "phrases": phrases, "seconds": seconds}


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
        if tag not in PROGRESSIONS or int(s.lstrip("s")) not in SEEDS or ",".join(r["chords"]) != PROGRESSIONS[tag]:
            problems.append(f"{name}: not preregistered")
        for k, v in EXPECT.items():
            if r.get(k) != v:
                problems.append(f"{name}: {k} = {r.get(k)!r}")
        if not r.get("comp") or (r.get("phrase_breath") or {}).get("max_notes") != 24:
            problems.append(f"{name}: comp / breath settings differ")
        if not r.get("run_completed") or r.get("completed_bars") != r.get("bars") or r.get("seed") != int(s.lstrip("s")):
            problems.append(f"{name}: not completed or seed differs")
        if not r.get("comp_trace"):
            problems.append(f"{name}: no comp_trace")
        if (tag, s) in seen:
            problems.append(f"{name}: duplicate")
        seen.add((tag, s))
    problems += [f"missing {t}_s{s}" for t in PROGRESSIONS for s in SEEDS if (t, f"s{s}") not in seen]
    return problems


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
    rows = [tag_run(json.loads(Path(d, "continuous_report.json").read_text())) for d in dirs]
    sm = [solo_metrics(r["solo"], r["seconds"]) for r in rows]
    gen = sorted(g for r in rows for g in r["gen_ms"])
    dropped = {}
    for r in rows:
        for k, v in r["dropped"].items():
            dropped[k] = dropped.get(k, 0) + v
    phrases = sorted(x for s in sm for x in s["phrases"])
    out = {
        "runs": len(rows), "fallback_total": sum(r["fallback"] for r in rows),
        "misses_total": sum(r["misses"] for r in rows),
        "rendered_invalid": sum(r["render"].get("rendered_invalid", 0) for r in rows),
        "raw_invalid": sum(r["render"].get("raw_invalid", 0) for r in rows),
        "gen_ms_p99": gen[min(len(gen) - 1, int(0.99 * len(gen)))] if gen else None,
        "comp_planned": sum(r["planned"] for r in rows), "comp_emitted": sum(r["emitted"] for r in rows),
        "comp_played": sum(r["played_comp"] for r in rows), "dropped": dropped,
        "emitted_alignment": sum(r["align_ok"] for r in rows) / sum(r["align_n"] for r in rows),
        "solo_notes_per_s": sum(s["notes"] for s in sm) / sum(s["seconds"] for s in sm),
        "solo_rest_share": sum(s["rest_time"] for s in sm) / sum(s["seconds"] for s in sm),
        "solo_phrase_median": phrases[len(phrases) // 2] if phrases else None,
    }
    out["delivery"] = out["comp_played"] / out["comp_emitted"] if out["comp_emitted"] else None
    verdict = {
        "fallback_0": out["fallback_total"] == 0,
        "render_valid": out["rendered_invalid"] == 0 and out["raw_invalid"] == 0,
        "misses_le_2": out["misses_total"] <= 2,
        "gen_p99_le_419": out["gen_ms_p99"] is not None and out["gen_ms_p99"] <= 419,
        "emitted_alignment_ge_0.95": out["emitted_alignment"] >= 0.95,
        "delivery_ge_0.98": out["delivery"] is not None and out["delivery"] >= 0.98,
    }
    verdict["pass"] = all(verdict.values())
    report = {"schema": "comp_source_check_v1", "summary": out, "verdict": verdict, "musical_quality_verified": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
