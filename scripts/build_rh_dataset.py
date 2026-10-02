#!/usr/bin/env python3
"""Right-hand lines of bebop pianists as a training set (docs/experiments/BEBOP_RH_DATA.md).

The solo-piano recordings keep both hands going, so a model trained on them
plays continuous solo piano, not lick-like solo lines with breaths (user
listening, 2026-10-02). This keeps, per 50 ms onset cluster, the highest note
when it is at or above ``--split`` (G3 = 55); clusters whose top note is lower
are left-hand comping and become rests. Kept notes are made monophonic (each
ends by the next onset). Tokens use the same encoder as the existing data
(``scripts.generate.encode_notes_simple``). Splits are per song, deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

MIDI_ROOT = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/midi_dataset/midi")
PIANISTS = ("Barry Harris", "Cedar Walton", "Hank Jones", "Duke Jordan", "Kenny Barron", "Tommy Flanagan",
            "Junior Mance", "Harold Mabern", "Mulgrew Miller", "Red Garland", "Benny Green", "Kenny Drew",
            "Bill Charlap")
SPLIT = 55
REST_S = 0.3


def right_hand(notes, split: int = SPLIT, window_s: float = 0.05):
    """Monophonic right-hand line: top note per onset cluster if >= split."""
    import pretty_midi

    clusters, first, best = [], None, None
    for n in sorted(notes, key=lambda n: (n.start, -n.pitch)):
        if first is None or n.start - first > window_s:
            if best is not None:
                clusters.append(best)
            first, best = n.start, n
        elif n.pitch > best.pitch:
            best = n
    if best is not None:
        clusters.append(best)
    line = [n for n in clusters if n.pitch >= split]
    out = []
    for i, n in enumerate(line):
        end = min(n.end, line[i + 1].start) if i + 1 < len(line) else n.end
        if end - n.start >= 0.02:
            out.append(pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=end))
    return out


def line_stats(line) -> dict | None:
    if len(line) < 8:
        return None
    gaps = [b.start - a.end for a, b in zip(line, line[1:])]
    iois = [b.start - a.start for a, b in zip(line, line[1:])]
    dur = line[-1].end - line[0].start
    lens, cur = [], 1
    for g in gaps:
        if g >= REST_S:
            lens.append(cur)
            cur = 1
        else:
            cur += 1
    lens.append(cur)
    rests = [g for g in gaps if g >= REST_S]
    return {"notes_per_s": len(line) / dur, "ioi_median_ms": statistics.median(iois) * 1000,
            "rests_per_10s": len(rests) / dur * 10, "rest_time": sum(rests) / dur,
            "phrase_notes_median": statistics.median(lens)}


def split_of(key: str) -> str:
    h = int(hashlib.sha1(key.encode()).hexdigest(), 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def main(argv=None) -> int:
    import numpy as np
    import pretty_midi
    from scripts.generate import encode_notes_simple

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/bebop_rh")
    ap.add_argument("--split", type=int, default=SPLIT)
    args = ap.parse_args(argv)
    for s in ("train", "val", "test"):
        (args.output_dir / s).mkdir(parents=True, exist_ok=True)
    seen, rows, idx = set(), [], {"train": 0, "val": 0, "test": 0}
    for pianist in PIANISTS:
        files = sorted(f for sub in ("live", "studio") for f in (MIDI_ROOT / sub / pianist).rglob("*.mid*"))
        for f in files:
            digest = hashlib.sha1(f.read_bytes()).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            try:
                pm = pretty_midi.PrettyMIDI(str(f))
            except Exception as exc:                       # unreadable file: record and skip
                rows.append({"pianist": pianist, "source": str(f), "status": f"unreadable: {exc}"[:120]})
                continue
            notes = [n for i in pm.instruments if not i.is_drum for n in i.notes]
            line = right_hand(notes, args.split)
            tokens = encode_notes_simple(line) if line else []
            if len(tokens) < 100:
                rows.append({"pianist": pianist, "source": str(f), "status": "too_short"})
                continue
            split = split_of(digest)
            name = f"{idx[split]:05d}.npy"
            idx[split] += 1
            np.save(args.output_dir / split / name, np.asarray(tokens, dtype=np.int32))
            rows.append({"pianist": pianist, "source": str(f), "sha1": digest, "split": split, "file": name,
                         "rh_notes": len(line), "all_notes": len(notes), "tokens": len(tokens),
                         "rh_stats": line_stats(line), "status": "ok"})
    ok = [r for r in rows if r["status"] == "ok"]
    summary = {}
    for s in ("train", "val", "test"):
        rs = [r["rh_stats"] for r in ok if r["split"] == s and r["rh_stats"]]
        summary[s] = {"songs": sum(1 for r in ok if r["split"] == s),
                      **({k: statistics.median(x[k] for x in rs) for k in rs[0]} if rs else {})}
    manifest = {"schema": "bebop_rh_v1", "split_pitch": args.split, "rest_s": REST_S, "pianists": list(PIANISTS),
                "summary": summary, "per_pianist": {p: sum(1 for r in ok if r["pianist"] == p) for p in PIANISTS},
                "skipped": [r for r in rows if r["status"] != "ok"], "songs": ok}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": summary, "per_pianist": manifest["per_pianist"],
                      "skipped": len(manifest["skipped"])}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
