#!/usr/bin/env python3
"""T3: fast-run density of generations vs reference corpora.

Pre-registered in docs/experiments/TATUM_PERSONALIZATION.md. Onsets within
30 ms are one cluster; the run ratio is the share of consecutive cluster
intervals in [40, 120] ms. Also reports cluster onsets per second.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

import numpy as np

CLUSTER_S = 0.03
RUN_MIN_S, RUN_MAX_S = 0.04, 0.12


def cluster_onsets(starts) -> list[float]:
    onsets = []
    for s in sorted(starts):
        if not onsets or s - onsets[-1] >= CLUSTER_S:
            onsets.append(s)
    return onsets


def run_stats(starts) -> dict:
    onsets = cluster_onsets(starts)
    if len(onsets) < 2:
        return {"run_ratio": None, "onsets_per_s": None, "onsets": len(onsets)}
    gaps = np.diff(onsets)
    span = onsets[-1] - onsets[0]
    return {"run_ratio": float(np.mean((gaps >= RUN_MIN_S) & (gaps <= RUN_MAX_S))),
            "onsets_per_s": float((len(onsets) - 1) / span) if span > 0 else None,
            "onsets": len(onsets)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--updates", default="0,518")
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--target-val-dir", type=Path, required=True)
    ap.add_argument("--generic-list", type=Path, required=True)
    ap.add_argument("--jazz-dir", type=Path,
                    default=Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/data/jazz_full/train"))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    import pretty_midi
    from scripts.eval_mehldau_snapshots import bootstrap_diff
    from scripts.style_distance import tokens_to_notes
    from scripts.validate_style_distance import load, middle_chunk

    def from_tokens(tokens):
        return run_stats([n.start for n in tokens_to_notes(tokens)])

    out = {"schema": "run_density_v1", "cluster_s": CLUSTER_S,
           "run_window_s": [RUN_MIN_S, RUN_MAX_S], "references": {}, "snapshots": {}}
    refs = {"target_val": [from_tokens(middle_chunk(load(f), 1024))
                           for f in sorted(args.target_val_dir.glob("*.npy"))],
            "generic_probe": [from_tokens(middle_chunk(load(args.jazz_dir / n), 1024))
                              for n in json.loads(args.generic_list.read_text())["generic_probe_files"]]}
    for name, rows in refs.items():
        vals = [r["run_ratio"] for r in rows if r["run_ratio"] is not None]
        rates = [r["onsets_per_s"] for r in rows if r["onsets_per_s"] is not None]
        out["references"][name] = {"n": len(vals), "mean_run_ratio": float(np.mean(vals)),
                                   "median_run_ratio": float(np.median(vals)),
                                   "mean_onsets_per_s": float(np.mean(rates))}
    seeds = [int(s) for s in args.seeds.split(",")]
    for u in (int(x) for x in args.updates.split(",")):
        per = []
        for seed in seeds:
            pm = pretty_midi.PrettyMIDI(str(args.eval_dir / f"gen_u{u:03d}_s{seed}.mid"))
            per.append({"seed": seed, **run_stats([n.start for i in pm.instruments for n in i.notes])})
        out["snapshots"][str(u)] = {
            "per_seed": per,
            "mean_run_ratio": float(np.mean([p["run_ratio"] for p in per])),
            "mean_onsets_per_s": float(np.mean([p["onsets_per_s"] for p in per]))}
    ups = list(out["snapshots"])
    a = [p["run_ratio"] for p in out["snapshots"][ups[0]]["per_seed"]]
    b = [p["run_ratio"] for p in out["snapshots"][ups[-1]]["per_seed"]]
    ci = bootstrap_diff(a, b)
    out["diff"] = {"from": ups[0], "to": ups[-1], "mean_diff": float(np.mean(b) - np.mean(a)),
                   "ci95": ci, "moved_toward_runs": bool(np.mean(b) > np.mean(a) and ci[0] > 0)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"references": out["references"],
                      "snapshots": {k: {kk: vv for kk, vv in v.items() if kk != "per_seed"}
                                    for k, v in out["snapshots"].items()},
                      "diff": out["diff"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
