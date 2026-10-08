#!/usr/bin/env python3
"""CMT continuous baseline analysis (docs/experiments/CMT_CONTINUOUS.md).

Reads the regenerated raw outputs (CMT grid, 120 BPM equivalent), writes 128 BPM conversions
(time x 120/128, velocity 90, no swing, no comp), the manifest, and the preregistered metrics.
"""
import hashlib, json, platform, statistics, subprocess, sys
from pathlib import Path

import numpy
import pretty_midi
import torch

R = Path(sys.argv[1] if len(sys.argv) > 1 else "/Users/ohhalim/git_box/t1_models/cmt_runs/head")
CMT = Path("/Users/ohhalim/git_box/t1_models/cmt")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/experiments/cmt_continuous")
BAR = 2.0                       # CMT grid: one bar = 2 s (120 BPM)
K = 120 / 128
SCORED = {3: "F", 4: "F", 5: "G", 7: "F", 8: "F"}           # bars (1-based) where P and Q differ, primer bar 1 excluded
P_TONES = {"F": {9, 4}, "G": {2}}                          # Fmaj7 A E / Gm7 D
Q_TONES = {"F": {8, 3}, "G": {1}}                          # Fm7 Ab Eb / Gm7b5 Db
sha1 = lambda p: hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]


def events(notes, win=0.03):
    ev, cur = [], []
    for n in sorted(notes, key=lambda n: (n.start, n.pitch)):
        if cur and n.start - cur[0].start > win:
            ev.append((cur[0].start, tuple(sorted(m.pitch for m in cur))))
            cur = []
        cur.append(n)
    if cur:
        ev.append((cur[0].start, tuple(sorted(m.pitch for m in cur))))
    return ev


def longest_repeat(ev, kmax=24):
    best = 0.0
    for k in range(1, kmax + 1):
        run = 0
        for i in range(len(ev) - k):
            if ev[i][1] == ev[i + k][1]:
                run += 1
                if (run + k) / k >= 3:
                    best = max(best, ev[i + k][0] - ev[i - run + 1][0])
            else:
                run = 0
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    conv_dir = R / "conv128"
    conv_dir.mkdir(exist_ok=True)
    report = json.loads((R / "report.json").read_text())
    rows, scale_err = [], 0.0
    for run in report["runs"]:
        raw = pretty_midi.PrettyMIDI(str(R / run["file"])).instruments[0].notes
        conv = [pretty_midi.Note(velocity=90, pitch=n.pitch, start=n.start * K, end=n.end * K) for n in raw]
        pm = pretty_midi.PrettyMIDI(initial_tempo=128)
        inst = pretty_midi.Instrument(program=0, name="cmt")
        inst.notes = conv
        pm.instruments.append(inst)
        cpath = conv_dir / run["file"].replace("_raw.mid", "_128.mid")
        pm.write(str(cpath))
        back = pretty_midi.PrettyMIDI(str(cpath)).instruments[0].notes
        for a, b in zip(sorted(raw, key=lambda n: (n.start, n.pitch)), sorted(back, key=lambda n: (n.start, n.pitch))):
            scale_err = max(scale_err, abs((b.end - b.start) - (a.end - a.start) * K))
        # bar scoring on the raw grid
        bars = {}
        for b_, key in SCORED.items():
            ns = [n for n in raw if (b_ - 1) * BAR - 1e-6 <= n.start < b_ * BAR - 1e-6]
            p = sum(n.pitch % 12 in P_TONES[key] for n in ns)
            q = sum(n.pitch % 12 in Q_TONES[key] for n in ns)
            bars[b_] = {"n": len(ns), "P": p, "Q": q}
        c7 = {b_: sum(1 for n in raw if (b_ - 1) * BAR - 1e-6 <= n.start < b_ * BAR - 1e-6) for b_ in (2, 6)}
        # phrase metrics on the 128 BPM conversion, generated part (bars 2-8)
        gen = sorted([n for n in back if n.start >= BAR * K - 1e-6], key=lambda n: n.start)
        span0, span1 = gen[0].start, max(n.end for n in gen)
        gaps = [b.start - a.end for a, b in zip(gen, gen[1:]) if b.start - a.end >= 0.3]
        durs = sorted(n.end - n.start for n in gen)
        rows.append({"name": run["file"].replace("_raw.mid", ""), "progression": run["progression"], "seed": run["seed"],
                     "notes_total": run["notes"], "gen_notes": len(gen), "gen_seconds_model": run["gen_seconds"],
                     "bars": bars, "c7_bar_notes": c7,
                     "notes_per_s": round(len(gen) / (span1 - span0), 2), "rests_ge_0.3s": len(gaps),
                     "rest_time_share": round(sum(gaps) / (span1 - span0), 3),
                     "dur_median_s": round(statistics.median(durs), 3), "dur_p90_s": round(durs[int(0.9 * (len(durs) - 1))], 3),
                     "dur_lt_0.1s_share": round(sum(d < 0.1 for d in durs) / len(durs), 3),
                     "longest_repeat_s": round(longest_repeat(events(gen)), 2),
                     "raw_sha1": sha1(R / run["file"]), "conv128_sha1": sha1(cpath)})
    # paired differences per scored bar
    by = {(r["progression"], r["seed"]): r for r in rows}
    pairs = []
    for s in (1, 2, 3, 4):
        a, b = by[("P", s)], by[("Q", s)]
        d = {}
        for b_ in SCORED:
            fa = (a["bars"][b_]["P"] - a["bars"][b_]["Q"]) / a["bars"][b_]["n"] if a["bars"][b_]["n"] else None
            fb = (b["bars"][b_]["P"] - b["bars"][b_]["Q"]) / b["bars"][b_]["n"] if b["bars"][b_]["n"] else None
            d[b_] = None if fa is None or fb is None else round(fa - fb, 3)
        pairs.append({"seed": s, "d_by_bar": d})
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    manifest = {"checkpoint": str(CMT / "ckpt/best_jazz_model_8bars.pth.tar"),
                "checkpoint_sha256_16": hashlib.sha256((CMT / "ckpt/best_jazz_model_8bars.pth.tar").read_bytes()).hexdigest()[:16],
                "hparams_sha1": sha1(CMT / "ckpt/hparams.yaml"), "epoch": report.get("epoch"),
                "cmt_code_sha1": {f: sha1(CMT / f) for f in ("model.py", "layers.py")},
                "generate_script_blob": git("hash-object", "scripts/t1_cmt_generate.py"),
                "env": {"python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "pretty_midi": pretty_midi.__version__ if hasattr(pretty_midi, "__version__") else "?"},
                "command": "python scripts/t1_cmt_generate.py --cmt-dir /Users/ohhalim/git_box/t1_models/cmt --prime head --output-dir /Users/ohhalim/git_box/t1_models/cmt_runs/head",
                "prime": "head (BebopNet head bar 1: G4 A4 Bb4 C5 D5 C5 Bb4 A4, eighths)", "topk": report.get("topk"),
                "progressions": {"P": "Gm7 C7 Fmaj7 Fmaj7 x2", "Q": "Gm7b5 C7 Fm7 Fm7 x2"}, "seeds": [1, 2, 3, 4],
                "raw_dir": str(R), "generated_representation": str(R / "report.json"),
                "conversion": "time x 120/128, velocity 90, no swing, no comp", "duration_scale_max_error_s": round(scale_err, 6)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    (OUT / "metrics.json").write_text(json.dumps({"runs": rows, "pairs": pairs}, indent=1) + "\n")
    print("duration scale max error (s):", round(scale_err, 6))
    for r in rows:
        print(r["name"], {b_: (v["n"], v["P"], v["Q"]) for b_, v in r["bars"].items()}, "c7", r["c7_bar_notes"], "| nps", r["notes_per_s"], "rests", r["rests_ge_0.3s"], r["rest_time_share"],
              "dur med", r["dur_median_s"], "<0.1", r["dur_lt_0.1s_share"], "rep", r["longest_repeat_s"], "gen", r["gen_seconds_model"])
    for p in pairs:
        print("seed", p["seed"], "d by bar", p["d_by_bar"])


if __name__ == "__main__":
    main()
