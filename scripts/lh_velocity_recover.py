"""Recover human-note velocities for the dev songs of #1772 (docs/experiments/LH_CONTEXT_CONTRACT.md). Main venv.

The #1756 export stored human notes as [nid, pitch, on, off] only. For each requested song: check the source SHA1
against the export's split_meta (= the manifest record), reload with the export's own loader
(lh_audit_stats.load_notes), and take the velocity of the note with the same nid and the same (pitch, on, off).
Any missing or different note stops the run: no default velocity is filled in. The export is never written; output
goes to a new folder with provenance.

  python scripts/lh_velocity_recover.py --export /Users/ohhalim/git_box/lh_request_export_run3 \
      --out /Users/ohhalim/git_box/lh_context_run1 --songs 44 68 185 134
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def recover(export, out, songs):
    from lh_audit_stats import load_notes, sha1_file
    meta = json.load(open(os.path.join(export, "songs", "split_meta.json")))
    rec = {s["idx"]: s for s in meta["songs"]}
    os.makedirs(os.path.join(out, "human_velocity"), exist_ok=True)
    report = {"export": export, "split_meta_sha256": sha256_file(os.path.join(export, "songs", "split_meta.json")),
              "manifest": meta["manifest"], "manifest_sha256": meta["manifest_sha256"], "loader": "lh_audit_stats.load_notes",
              "songs": []}
    for i in songs:
        r = rec[i]
        if sha1_file(r["source"]) != r["sha1"]:
            sys.exit(f"song {i}: source sha1 differs from the export record")
        stored = json.load(open(os.path.join(export, "songs", f"{i:03d}.notes.json")))
        if stored["sha1"] != r["sha1"]:
            sys.exit(f"song {i}: notes.json sha1 differs from split_meta")
        notes, _ = load_notes(r["source"])
        by_id = {n.nid: n for n in notes}
        rows = []
        for nid, p, on, off in stored["human"]:
            n = by_id.get(nid)
            if n is None or (n.pitch, n.on, n.off) != (p, on, off):
                sys.exit(f"song {i}: human note {nid} not found with the same pitch/on/off; no default is used")
            rows.append([nid, p, on, off, n.vel])
        path = os.path.join(out, "human_velocity", f"{i:03d}.json")
        with open(path, "w") as f:
            json.dump({"idx": i, "source": r["source"], "sha1": r["sha1"], "human": rows}, f)
        vels = [x[4] for x in rows]
        report["songs"].append({"idx": i, "sha1": r["sha1"], "human_notes": len(rows), "matched": len(rows),
                                "velocity_min": min(vels), "velocity_max": max(vels), "file_sha256": sha256_file(path)})
    with open(os.path.join(out, "human_velocity", "manifest.json"), "w") as f:
        json.dump(report, f, indent=1)
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--songs", type=int, nargs="+", required=True)
    a = ap.parse_args()
    if os.path.abspath(a.out).startswith(os.path.abspath(a.export)):
        sys.exit("--out must not be inside the export")
    rep = recover(a.export, a.out, a.songs)
    for s in rep["songs"]:
        print(s)
