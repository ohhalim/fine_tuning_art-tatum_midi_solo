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
from bisect import bisect_left

from scripts.coherence_metrics import generation_block_s, top_line
from scripts.motif_transfer_ab import reappearance, SETS

KEYS = ("windows", "variant", "exact", "interval", "blocks", "blocks_with_window")


def scan(notes, block_s, first_t=8.0, ks=None):
    """``notes``: sorted (start, pitch). Each window's top line is built from the
    notes inside it, so a cluster cannot borrow a pitch across t (Astra review)."""
    tot = dict.fromkeys(KEYS, 0)
    starts_of = [n[0] for n in notes]
    end = notes[-1][0] if notes else 0
    starts = [k * block_s for k in ks] if ks is not None else \
        [first_t + i * block_s for i in range(int((end - first_t) // block_s))]
    for t in starts:
        lo, mid, hi = (bisect_left(starts_of, x) for x in (t - 8.0, t, t + block_s))
        ses = top_line(notes[lo:mid])
        gen = [(a - t, p) for a, p in top_line(notes[mid:hi])]
        r = reappearance(gen, ses)
        tot["blocks"] += 1
        tot["blocks_with_window"] += r["windows"] > 0
        for key in ("windows", "variant", "exact", "interval"):
            tot[key] += r[key]
    return tot


def rates(agg):
    w = agg["windows"]
    agg.update({"VR": agg["variant"] / w if w else None, "XR": agg["exact"] / w if w else None,
                "IR": agg["interval"] / w if w else None})
    return agg


def song_notes(path):
    from scripts.style_distance import tokens_to_notes
    from scripts.validate_style_distance import load
    return sorted((n.start, n.pitch) for n in tokens_to_notes(load(path)))


def report_notes(path):
    r = json.loads(path.read_text())
    bar_s = 60.0 / r["bpm"] * r.get("beats_per_bar", 4)
    return sorted((i * bar_s + n[1], n[0]) for i, b in enumerate(r["played_bars"]) for n in b["notes"]), r


out = {"note": "post-hoc calibration, not part of the gate; v2: per-window top lines", "sets": {}}
for name, d in (("tatum_real", Path(W) / "data/tatum_full/train"),
                ("mehldau_real", MEHLDAU_SONGS)):
    agg = dict.fromkeys(KEYS, 0)
    for f in sorted(d.glob("*.npy")):
        r = scan(song_notes(f), 0.9375)
        agg = {k: agg[k] + r[k] for k in KEYS}
    out["sets"][name] = rates(agg)
# the runtime context sessions themselves: played block k vs its own previous 8 s, same ks
for m, sessions in SETS["dev"]["sessions"].items():
    agg = dict.fromkeys(KEYS, 0)
    for sess in sessions:
        notes, r = report_notes(Path(W) / sess / "continuous_report.json")
        nb = r["bars"] * 2
        res = scan(notes, generation_block_s(r), ks=[k for k in (6, 10, 14, 18, 22, 26, 30) if k < nb])
        agg = {k: agg[k] + res[k] for k in KEYS}
    out["sets"][f"runtime_played_{m}"] = rates(agg)
Path(W, "outputs/motif_transfer/dev_v2").mkdir(parents=True, exist_ok=True)
for k, v in out["sets"].items():
    print(f"{k:24s} blocks {v['blocks']:5d} with-window {v['blocks_with_window']:5d} windows {v['windows']:6d} "
          f"VR {v['VR']:.4f} XR {v['XR']:.4f} IR {v['IR']:.4f}")
json.dump(out, open(W + "/outputs/motif_transfer/dev_v2/calibration_posthoc.json", "w"), indent=2)
