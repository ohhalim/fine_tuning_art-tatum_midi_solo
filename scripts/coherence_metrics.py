#!/usr/bin/env python3
"""Does the top line carry across block boundaries, and does it reuse motifs?

docs/experiments/COHERENCE_GOAL.md. Two observations of "sentences that connect":

* boundary ratio: mean |pitch jump| of the top line across block boundaries
  divided by the mean jump inside blocks. Blocks generated from scratch jump at
  their seams; a continuous line does not care where the seams are (~1).
* motif reuse: share of top-line interval 3-grams (4 notes) that already
  occurred in the preceding ``horizon`` seconds.

The top line is the highest pitch of each onset cluster (50 ms, as in
``diversity_metrics.group_voicings``), so left-hand chords on downbeats do not
count as melodic jumps. Observations only: no quality or style claim.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))


def top_line(notes, window_s: float = 0.05) -> list[tuple[float, int]]:
    """(onset, highest pitch) per onset cluster; ``notes`` are (start, pitch)."""
    out, first, best = [], None, None
    for start, pitch in sorted(notes):
        if first is None or start - first > window_s:
            if first is not None:
                out.append((first, best))
            first, best = start, pitch
        else:
            best = max(best, pitch)
    if first is not None:
        out.append((first, best))
    return out


def jumps(line, block_s: float) -> tuple[list[int], list[int]]:
    """(boundary jumps, within-block jumps) of consecutive top-line notes."""
    boundary, within = [], []
    for (t0, p0), (t1, p1) in zip(line, line[1:]):
        (boundary if int(t0 // block_s) != int(t1 // block_s) else within).append(abs(p1 - p0))
    return boundary, within


def motif_reuse(line, horizon_s: float = 8.0) -> tuple[int, int]:
    """(reused, total) interval 3-grams, reuse looked up in the previous ``horizon_s``."""
    from collections import Counter, deque

    grams = [(line[i][0], tuple(line[i + k + 1][1] - line[i + k][1] for k in range(3)))
             for i in range(len(line) - 3)]
    reused = 0
    pending, window, counts = deque(), deque(), Counter()
    for i, (t, g) in enumerate(grams):
        while pending and pending[0][0] <= i - 4:      # earlier grams that do not overlap gram i
            _, tj, gj = pending.popleft()
            window.append((tj, gj))
            counts[gj] += 1
        while window and t - window[0][0] > horizon_s:
            _, old = window.popleft()
            counts[old] -= 1
        if counts[g] > 0:
            reused += 1
        pending.append((i, t, g))
    return reused, len(grams)


def summarize(lines, block_s: float) -> dict:
    b, w, reused, total = [], [], 0, 0
    for line in lines:
        bj, wj = jumps(line, block_s)
        b += bj
        w += wj
        r, t = motif_reuse(line)
        reused += r
        total += t
    mean = lambda x: sum(x) / len(x) if x else None
    return {"boundary_mean": mean(b), "within_mean": mean(w),
            "boundary_ratio": (mean(b) / mean(w)) if b and w and mean(w) else None,
            "boundary_median": statistics.median(b) if b else None,
            "within_median": statistics.median(w) if w else None,
            "motif_reuse": reused / total if total else None,
            "n_boundary": len(b), "n_within": len(w), "n_grams": total}


def report_line(path: Path) -> tuple[list[tuple[float, int]], float, float]:
    """Top line of a runtime report and its generation block length (half bar)."""
    r = json.loads(path.read_text())
    bar_s = 240.0 / r["bpm"]
    notes = [(i * bar_s + n[1], n[0]) for i, b in enumerate(r["played_bars"]) for n in b["notes"]]
    return top_line(notes), bar_s / 2, bar_s


def song_line(path: Path) -> list[tuple[float, int]]:
    from scripts.style_distance import tokens_to_notes
    from scripts.validate_style_distance import load
    return top_line([(n.start, n.pitch) for n in tokens_to_notes(load(path))])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--songs", action="append", default=[], metavar="NAME=DIR",
                    help="real performances (.npy tokens), cut into --block-s windows")
    ap.add_argument("--reports", action="append", default=[], metavar="NAME=GLOB",
                    help="runtime continuous_report.json files; blocks are half bars")
    ap.add_argument("--block-s", type=float, default=0.9375, help="window for real songs (half bar at 128 BPM)")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    out = {"schema": "coherence_v1", "style_verified": False, "musical_quality_verified": False, "sets": {}}
    for spec in args.songs:
        name, d = spec.split("=", 1)
        lines = [song_line(f) for f in sorted(Path(d).glob("*.npy"))]
        out["sets"][name] = {"kind": "real", "items": len(lines), **summarize(lines, args.block_s)}
    grouped: dict[str, list[Path]] = {}
    for spec in args.reports:                     # the same name may repeat: files accumulate
        name, pattern = spec.split("=", 1)
        grouped.setdefault(name, []).extend(sorted(ROOT.glob(pattern)))
    for name, files in grouped.items():
        lines, block = [], None
        for f in files:
            line, block, _ = report_line(f)
            lines.append(line)
        out["sets"][name] = {"kind": "runtime", "items": len(files), **summarize(lines, block or args.block_s)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for name, s in out["sets"].items():
        fmt = lambda v: "-" if v is None else f"{v:.3f}"
        print(f"{name:24s} n={s['items']:3d} boundary/within {fmt(s['boundary_ratio'])} "
              f"(mean {fmt(s['boundary_mean'])}/{fmt(s['within_mean'])}) motif reuse {fmt(s['motif_reuse'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
