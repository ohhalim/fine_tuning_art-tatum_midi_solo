#!/usr/bin/env python3
"""Post-hoc calibration for #1575 (not part of the preregistered gate).

How often does the reappearance measure of ``motif_transfer_ab`` fire on real
playing and on the runtime's own played blocks? Same rules as the experiment:
session = top line in the 8 s before the block, block = one half bar (0.9375 s
for real songs, the report's own block length for runtime sessions).
docs/experiments/MOTIF_TRANSFER.md, "사후 진단".
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))
W = str(ROOT)
MEHLDAU_SONGS = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/data/mehldau_full/train")
from scripts.coherence_metrics import report_line, song_line
from scripts.motif_transfer_ab import reappearance, SETS


def scan(line, block_s, first_t=8.0, ks=None):
    tot = {"windows": 0, "variant": 0, "exact": 0, "interval": 0, "blocks": 0, "blocks_with_window": 0}
    end = line[-1][0] if line else 0
    starts = [k * block_s for k in ks] if ks is not None else \
        [first_t + i * block_s for i in range(int((end - first_t) // block_s))]
    for t in starts:
        ses = [(a, p) for a, p in line if t - 8.0 <= a < t]
        gen = [(a - t, p) for a, p in line if t <= a < t + block_s]
        r = reappearance(gen, ses)
        tot["blocks"] += 1
        tot["blocks_with_window"] += r["windows"] > 0
        for key in ("windows", "variant", "exact", "interval"):
            tot[key] += r[key]
    w = tot["windows"]
    tot.update({k.upper()[0] + "R": (tot[k] / w if w else None) for k in ("variant", "exact", "interval")})
    return tot


out = {"note": "post-hoc calibration, not part of the gate", "sets": {}}
for name, d in (("tatum_real", Path(W) / "data/tatum_full/train"),
                ("mehldau_real", MEHLDAU_SONGS)):
    agg = None
    for f in sorted(d.glob("*.npy")):
        r = scan(song_line(f), 0.9375)
        agg = r if agg is None else {k: agg[k] + r[k] for k in ("windows", "variant", "exact", "interval", "blocks", "blocks_with_window")}
    w = agg["windows"]
    agg.update({"VR": agg["variant"] / w, "XR": agg["exact"] / w, "IR": agg["interval"] / w})
    out["sets"][name] = agg
# the runtime context sessions themselves: played block k vs its own previous 8 s, same ks
for m, sessions in SETS["dev"]["sessions"].items():
    agg = None
    for s in sessions:
        p = Path(W) / s / "continuous_report.json"
        line, block_s, _ = report_line(p)
        nb = json.loads(p.read_text())["bars"] * 2
        r = scan(line, block_s, ks=[k for k in (6, 10, 14, 18, 22, 26, 30) if k < nb])
        agg = r if agg is None else {k: agg[k] + r[k] for k in ("windows", "variant", "exact", "interval", "blocks", "blocks_with_window")}
    w = agg["windows"]
    agg.update({"VR": agg["variant"] / w if w else None, "XR": agg["exact"] / w if w else None, "IR": agg["interval"] / w if w else None})
    out["sets"][f"runtime_played_{m}"] = agg
for k, v in out["sets"].items():
    print(f"{k:24s} blocks {v['blocks']:5d} with-window {v['blocks_with_window']:5d} windows {v['windows']:6d} "
          f"VR {v['VR']:.4f} XR {v['XR']:.4f} IR {v['IR']:.4f}")
json.dump(out, open(W + "/outputs/motif_transfer/dev/calibration_posthoc.json", "w"), indent=2)
