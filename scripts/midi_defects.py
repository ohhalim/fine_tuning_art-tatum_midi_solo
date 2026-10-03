#!/usr/bin/env python3
"""Defects you can read off the played MIDI, per run set (docs/experiments/CLEAN_MODE.md).

Solo = played notes not joined to the emitted comp. Counted per solo note:
  harsh      a sounding comp note a minor 2nd / major 7th / minor 9th away
  under      a sounding comp note at or above the solo note
  low        solo note below C4 (bass notes leaking into the line)
  outside    solo note outside the chord's strict scale that is not followed (within 0.3 s)
             by a chord tone at most 2 semitones away
  repeat     same pitch as the previous solo note (gap <= 0.3 s)
  leap       more than an octave from the previous solo note (gap <= 0.3 s)
Real bebop right hands: repeat .055, leaps > 12 about .06.
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


def run_defects(r) -> dict:
    from inference.app.fallback import parse_chord
    from inference.control.harmony_bias import SCALE
    from scripts.comp_source_check import tag_run

    t = tag_run(r)
    bar = 60.0 / r["bpm"] * r.get("beats_per_bar", 4)
    played = [(n[0], b["bar"] * bar + n[1], b["bar"] * bar + n[2]) for b in r["played_bars"] for n in b["notes"]]
    solo = sorted(t["solo"], key=lambda x: x[1])
    sset = set(solo)
    comp = [n for n in played if n not in sset]
    c = {"notes": len(solo), "harsh": 0, "under": 0, "low": 0, "outside": 0, "repeat": 0, "leap": 0, "steps": 0}
    for i, (p, s, e) in enumerate(solo):
        ov = [q for q, qs, qe in comp if min(e, qe) - max(s, qs) > 0.02]
        c["harsh"] += any(abs(p - q) in (1, 11, 13) for q in ov)
        c["under"] += any(q >= p for q in ov)
        c["low"] += p < 60
        root, iv = parse_chord(r["chords"][int(s // bar) % len(r["chords"])])
        scale = {(root + k) % 12 for k in SCALE.get(tuple(iv), set(iv))}
        tones = {(root + k) % 12 for k in iv}
        if p % 12 not in scale:
            nxt = solo[i + 1] if i + 1 < len(solo) else None
            resolved = nxt is not None and nxt[1] - e <= 0.3 and abs(nxt[0] - p) <= 2 and nxt[0] % 12 in tones
            c["outside"] += not resolved
        if i and s - solo[i - 1][2] <= 0.3:
            c["steps"] += 1
            c["repeat"] += p == solo[i - 1][0]
            c["leap"] += abs(p - solo[i - 1][0]) > 12
    c["fallback"] = r["production"]["fallback_bar_count"]
    c["misses"] = r["scheduler_dispatch_deadline_miss_count"]
    c["gen_ms"] = [b["generation_ms"] for b in r["bars_detail"] if b.get("generation_ms") is not None]
    return c


def pooled(rows) -> dict:
    n = sum(r["notes"] for r in rows)
    st = sum(r["steps"] for r in rows)
    gen = sorted(g for r in rows for g in r["gen_ms"])
    out = {k: sum(r[k] for r in rows) / n for k in ("harsh", "under", "low", "outside")}
    out.update(repeat=sum(r["repeat"] for r in rows) / st, leap=sum(r["leap"] for r in rows) / st, notes=n,
               fallback=sum(r["fallback"] for r in rows), misses=sum(r["misses"] for r in rows),
               gen_ms_p99=gen[min(len(gen) - 1, int(0.99 * len(gen)))] if gen else None)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="globs of run dirs")
    args = ap.parse_args(argv)
    dirs = sorted(d for pat in args.runs for d in glob.glob(str(ROOT / pat)) if Path(d, "continuous_report.json").exists())
    print(json.dumps({"runs": len(dirs), **pooled([run_defects(json.loads(Path(d, "continuous_report.json").read_text()))
                                                     for d in dirs])}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
