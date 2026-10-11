#!/usr/bin/env python3
"""C-pool training data for the same-budget control adapter (issue #1733, docs/experiments/CONTROL_ADAPTER_DESIGN.md).

50 recordings (studio 13 / live 37, the composition of the #1726 Mehldau train split), at most one recording
per pianist, from the PiJAMA `midi` tree. Fixed once with seed 0; training seeds change later, this does not.

Excluded before sampling:
- pianists: Brad Mehldau, Art Tatum and the 4 #1726 probe pianists
- normalised titles of the 72 Mehldau recordings and of the 20 probe recordings
Sampling: candidates sorted by path, live then studio shuffled with one random.Random(0). Walk the order and
accept a recording when its pianist is unused and its interval 16-gram containment against every Mehldau and
probe recording is below 0.2 (the #1726 merge threshold). No model score is used.
Windows: exactly the #1726 rules (mehldau_aria_data.windows_of).

  control_pool_data.py <aria repo> <mehldau manifest json> <out data json> <out manifest json>
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from mehldau_aria_data import (MERGE_AT, PIJAMA, Encoder, containment, interval_ngrams,  # noqa: E402
                               recordings_of, sha_ids, windows_of)

SEED, N_STUDIO, N_LIVE = 0, 13, 37


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    aria_repo, m_manifest_path, out_data, out_manifest = sys.argv[1:5]
    enc = Encoder(aria_repo)
    mm = json.load(open(m_manifest_path))
    probe_pianists = set(mm["probe"]["pianists"])
    excluded_pianists = {"Brad Mehldau", "Art Tatum"} | probe_pianists
    reference = [r for r in mm["recordings"] if r["pianist"] == "Brad Mehldau" or r["split"] == "probe"]
    excluded_titles = {r["title"] for r in reference}
    ref_grams = {r["path"]: interval_ngrams(enc.MidiDict.from_midi(str(PIJAMA / "midi" / r["path"])).note_msgs) for r in reference}

    pianists = sorted(set(os.listdir(PIJAMA / "midi" / "studio")) | set(os.listdir(PIJAMA / "midi" / "live")))
    cands = [r for p in pianists if p not in excluded_pianists for r in recordings_of(p) if r["title"] not in excluded_titles]
    live = sorted((r for r in cands if r["kind"] == "live"), key=lambda r: r["path"])
    studio = sorted((r for r in cands if r["kind"] == "studio"), key=lambda r: r["path"])
    rng = random.Random(SEED)
    rng.shuffle(live)
    rng.shuffle(studio)

    used, chosen, skipped = set(), [], {"pianist_used": 0, "content_overlap": 0, "no_windows": 0}
    overlaps = []
    for pool, need in ((live, N_LIVE), (studio, N_STUDIO)):
        got = 0
        for r in pool:
            if got == need:
                break
            if r["pianist"] in used:
                skipped["pianist_used"] += 1
                continue
            t, d = enc.full(r["path"])
            g = interval_ngrams(d.note_msgs)
            worst_path, worst = max(((p, containment(g, rg)) for p, rg in ref_grams.items()), key=lambda x: x[1])
            if worst >= MERGE_AT:
                skipped["content_overlap"] += 1
                overlaps.append({"path": r["path"], "vs": worst_path, "containment": round(worst, 4)})
                continue
            ws, dropped = windows_of(enc, d)
            if not ws:
                skipped["no_windows"] += 1
                continue
            assert all(w["renormalised_roundtrip_exact"] and not w["has_eos_or_dim"] and w["n_tokens"] <= 1024 for w in ws), r["path"]
            used.add(r["pianist"])
            chosen.append({**r, "split": "train", "n_notes": len(d.note_msgs),
                           "duration_s": round(max(x["data"]["end"] for x in d.note_msgs) / 1000, 2),
                           "max_containment_vs_reference": round(worst, 4), "windows": ws, "dropped_tail_notes": dropped,
                           "full_tokens_sha256": sha_ids(enc.tok.encode(t))})
            got += 1
        assert got == need, f"only {got} of {need} {pool[0]['kind']} recordings available"

    data = {"recordings": [{"path": r["path"], "split": "train", "windows": [{"ids": w["ids"], "s_index": w["s_index"]} for w in r["windows"]]}
                           for r in chosen]}
    with open(out_data, "w") as f:
        json.dump(data, f)
    manifest = {
        "schema": "control_pool_v1", "issue": 1733, "seed": SEED, "data_json": out_data, "data_sha256": file_sha(out_data),
        "mehldau_manifest_sha256": file_sha(m_manifest_path),
        "rules": {"composition": {"studio": N_STUDIO, "live": N_LIVE}, "max_per_pianist": 1,
                  "excluded_pianists": sorted(excluded_pianists), "excluded_titles": len(excluded_titles),
                  "content_threshold": MERGE_AT, "content_reference": f"{len(reference)} recordings (Mehldau 72 + probe 20)",
                  "windows": "mehldau_aria_data.windows_of (same as #1726)"},
        "pool": {"pianists": len({r['pianist'] for r in cands}), "candidates": len(cands),
                 "live": len(live), "studio": len(studio)},
        "skipped_while_walking": skipped, "content_overlaps": overlaps,
        "recordings": [{k: v for k, v in r.items() if k != "windows"} | {"n_windows": len(r["windows"]),
                        "window_tokens": [w["n_tokens"] for w in r["windows"]]} for r in chosen],
        "totals": {"recordings": len(chosen), "pianists": len(used), "windows": sum(len(r["windows"]) for r in chosen),
                   "studio": sum(r["kind"] == "studio" for r in chosen), "live": sum(r["kind"] == "live" for r in chosen)},
    }
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    print(json.dumps({"totals": manifest["totals"], "skipped": skipped, "pool": manifest["pool"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
