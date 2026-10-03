#!/usr/bin/env python3
"""Bebop solo lines with the harmony guide in front of every window (docs/experiments/GUIDE_ADAPTER.md, #1631).

Same songs and splits as data/bebop_rh. Each song becomes a stream of
``[guide][solo window]`` examples over contiguous 0.9375 s windows (the runtime
half-bar block at 128 BPM), serialised with inference/control/harmony_contract.
The guide is the window's accompaniment proxy; when it has fewer than two pitch
classes the previous guide is carried, and before any guide the window has none.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

WINDOW_S = 0.9375
MIN_PCS = 2


def song_stream(notes) -> tuple[list[int], dict]:
    from inference.control.harmony_contract import (SOLO_SPLIT, accompaniment_proxy, guide_notes, serialize,
                                                    solo_window)
    from inference.control.solo_line import top_notes

    notes = sorted(notes, key=lambda n: (n.start, n.pitch))
    line = [n for n in top_notes(notes) if n.pitch >= SOLO_SPLIT]
    if not line:
        return [], {"windows": 0}
    t, end = line[0].start, max(n.end for n in notes)
    toks, prev = [], None
    st = {"windows": 0, "new_guide": 0, "carried": 0, "no_guide": 0, "solo_notes": 0}
    while t < end:
        bass, pcs, _ = accompaniment_proxy(notes, t, t + WINDOW_S)
        if bass is not None and len(pcs) >= MIN_PCS:
            prev = (bass, pcs)
            st["new_guide"] += 1
        elif prev is not None:
            st["carried"] += 1
        else:
            st["no_guide"] += 1
        guide = guide_notes(prev[0], prev[1], WINDOW_S) if prev else []
        solo = solo_window(line, t, t + WINDOW_S)
        part, _ = serialize(guide, solo, WINDOW_S)
        toks += part
        st["windows"] += 1
        st["solo_notes"] += len(solo)
        t += WINDOW_S
    return toks, st


def main(argv=None) -> int:
    import numpy as np
    import pretty_midi

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-manifest", type=Path, default=ROOT / "data/bebop_rh/manifest.json")
    ap.add_argument("--output-dir", type=Path, default=ROOT / "data/bebop_guide")
    args = ap.parse_args(argv)
    src = json.loads(args.source_manifest.read_text())
    out_songs, total = [], {"windows": 0, "new_guide": 0, "carried": 0, "no_guide": 0, "solo_notes": 0}
    for s in src["songs"]:
        if s.get("status") != "ok":
            continue
        pm = pretty_midi.PrettyMIDI(s["source"])
        toks, st = song_stream([n for i in pm.instruments for n in i.notes])
        if not toks:
            continue
        d = args.output_dir / s["split"]
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / s["file"], np.asarray(toks, dtype=np.int32))
        out_songs.append({**{k: s[k] for k in ("pianist", "source", "sha1", "split", "file")},
                          "tokens": len(toks), **st})
        for k in total:
            total[k] += st[k]
    manifest = {"schema": "bebop_guide_v1", "window_s": WINDOW_S, "min_pcs": MIN_PCS,
                "source_manifest": str(args.source_manifest.relative_to(ROOT)),
                "summary": {**total, "songs": len(out_songs),
                            **{f"{k}_share": total[k] / total["windows"] for k in ("new_guide", "carried", "no_guide")},
                            "split_songs": {sp: sum(1 for x in out_songs if x["split"] == sp)
                                            for sp in ("train", "val", "test")}},
                "songs": out_songs}
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps(manifest["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
