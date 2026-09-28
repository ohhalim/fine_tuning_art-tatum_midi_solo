#!/usr/bin/env python3
"""Pooled pitch-jump medians per model: within bars and across bar boundaries.

For every ``<dir>/<model>/bpm*_seed*/continuous_report.json`` the played notes
of each bar are ordered by onset (then pitch). Within-bar jumps are |Δpitch|
between consecutive notes of a bar; boundary jumps are |first note of bar b+1 -
last note of bar b| for non-empty neighbours. Values are pooled over seeds and
the median is reported (docs/experiments/USAGE_PATH_16BAR.md §6-§8).
Observation only: not a quality or style claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def pooled_jumps(d: Path) -> dict:
    out = {}
    for mdir in sorted(p for p in d.iterdir() if p.is_dir()):
        within, boundary = [], []
        for rep in sorted(mdir.glob("bpm*_seed*/continuous_report.json")):
            bars = json.loads(rep.read_text())["played_bars"]
            seqs = [[n[0] for n in sorted(b["notes"], key=lambda n: (n[1], n[0]))] for b in bars]
            for s in seqs:
                within += [abs(b - a) for a, b in zip(s, s[1:])]
            boundary += [abs(b[0] - a[-1]) for a, b in zip(seqs, seqs[1:]) if a and b]
        if within and boundary:
            out[mdir.name] = {"within_median": statistics.median(within),
                              "boundary_median": statistics.median(boundary),
                              "n_within": len(within), "n_boundary": len(boundary)}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="append", required=True, metavar="TAG=DIR")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {}
    for spec in args.sweep:
        tag, d = spec.split("=", 1)
        out[tag] = pooled_jumps(Path(d))
        print(tag, {m: (v["boundary_median"], v["within_median"]) for m, v in out[tag].items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
