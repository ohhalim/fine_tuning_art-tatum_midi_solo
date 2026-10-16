#!/usr/bin/env python3
"""Crop + full-reset contract on the #1756 export (issue #1758, docs/experiments/CROP_RESET_CONTRACT.md). No model.

  crop_check.py <#1756 export dir (run 3)> <report json>

For every song (production settings CAP 64, CAPACITY 4096, KEEP_TARGET 2048) it builds the session requests and
checks, registered before the run:
  R1 no CapacityFailure
  R2 every request's kept context notes keep their original absolute (onset, pitch, dur) after base_ms is added back
  R3 every request's session position payloads equal the sequential session replay (simulate_session_kv), and both
     give the same base_ms and cut
  R4 no request's session context fails the configured bound after its crop (the same needs_crop: "the setting holds",
     not a proof that real calls stay below capacity; that is the CAP-aware real fill_block fixtures' job)
Reported only: crops per song, base_ms, session context length, requests with a crop, kept length after a crop.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from block_contract import block_start  # noqa: E402
from block_request import COLD, payload  # noqa: E402
from crop_contract import (CAP, CAPACITY, KEEP_TARGET, CapacityFailure, absolute_times, needs_crop,  # noqa: E402
                           session_position_requests, session_requests, session_tokens, simulate_session_kv)
from lh_role_proxy import Note  # noqa: E402


def sha256_file(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def dist(xs):
    if not xs:
        return None
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]
    return {"n": len(s), "mean": round(statistics.fmean(s), 2), "min": s[0], "median": q(0.5), "p90": q(0.9), "max": s[-1]}


def check_song(toks, t0, hum):
    out = {"capacity_failure": 0, "r2_time_mismatch": 0, "r3_replay_mismatch": 0, "r4_unsafe": 0, "requests": 0, "crops": 0,
           "base_ms_last": 0.0, "ctx_len": [], "kept_after_crop": []}
    try:
        srows = session_requests(toks, t0)
        kv = simulate_session_kv(toks, t0, hum)
    except CapacityFailure:
        out["capacity_failure"] = 1
        return out
    pays = {}
    for r in srows:
        k = r["k"]
        out["requests"] += 1
        out["ctx_len"].append(r["ctx_len"])
        if r["crop"]:
            out["crops"] += 1
            out["kept_after_crop"].append(r["ctx_len"])
        out["r4_unsafe"] += needs_crop(r["ctx_len"])
        sess = session_tokens(toks, r)[:r["ctx_len"]]
        kept = absolute_times(sess, r["base_ms"])
        orig = absolute_times(list(toks[:r["ctx_end"]]), 0.0)
        out["r2_time_mismatch"] += kept != orig[len(orig) - len(kept):]
        tags = session_position_requests(toks, srows, k)[:r["ctx_len"] - 1]
        for j in set(tags):
            if j not in pays:
                pays[j] = COLD if j < 0 else payload(hum, block_start(j, t0), j)
        got, base, cut = kv[k]
        out["r3_replay_mismatch"] += (got != [pays[j] for j in tags] or base != r["base_ms"]
                                      or cut != (r["crop"]["cut"] if r["crop"] else None))
    out["base_ms_last"] = srows[-1]["base_ms"] if srows else 0.0
    return out


def main() -> None:
    export_dir, report_path = sys.argv[1:3]
    t_start = time.time()
    totals, ctx, kept, crops_per_song, bases, songs = collections.Counter(), [], [], [], [], []
    for i in range(189):
        d = json.load(gzip.open(os.path.join(export_dir, "requests", f"{i:03d}.json.gz")))
        nj = json.load(open(os.path.join(export_dir, "songs", f"{i:03d}.notes.json")))
        assert (nj["idx"], nj["source"], nj["sha1"]) == (d["idx"], d["source"], d["sha1"]) == (i, d["source"], d["sha1"])
        toks = [tuple(t) if isinstance(t, list) else t for t in d["tokens"]]
        hum = [Note(nid, p, on, off) for nid, p, on, off in nj["human"]]
        res = check_song(toks, d["t0"], hum)
        for key in ("capacity_failure", "r2_time_mismatch", "r3_replay_mismatch", "r4_unsafe", "requests", "crops"):
            totals[key] += res[key]
        ctx += res["ctx_len"]
        kept += res["kept_after_crop"]
        crops_per_song.append(res["crops"])
        bases.append(res["base_ms_last"])
        songs.append({"idx": i, "requests": res["requests"], "crops": res["crops"], "capacity_failure": res["capacity_failure"],
                      "r2": res["r2_time_mismatch"], "r3": res["r3_replay_mismatch"], "r4": res["r4_unsafe"]})
    checks = {k: totals[k] for k in ("capacity_failure", "r2_time_mismatch", "r3_replay_mismatch", "r4_unsafe")}
    report = {"issue": 1758, "doc": "docs/experiments/CROP_RESET_CONTRACT.md", "quality_claimed": False, "training": False,
              "run": {"git_head": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                      "crop_contract_sha256": sha256_file(os.path.join(HERE, "crop_contract.py")),
                      "check_script_sha256": sha256_file(os.path.abspath(__file__)), "export_dir": export_dir,
                      "seconds": round(time.time() - t_start, 1)},
              "params": {"cap": CAP, "capacity": CAPACITY, "keep_target": KEEP_TARGET, "decision": "committed_len + 2*CAP + 3 > CAPACITY"},
              "checks": checks, "all_checks_zero": not any(checks.values()), "songs_checked": len(songs),
              "requests": totals["requests"], "crops": totals["crops"],
              "requests_with_crop_share": round(totals["crops"] / totals["requests"], 4) if totals["requests"] else None,
              "crops_per_song": dist(crops_per_song), "songs_with_a_crop": sum(1 for c in crops_per_song if c),
              "session_ctx_len": dist(ctx), "kept_len_after_crop": dist(kept), "base_ms_at_song_end": dist(bases),
              "songs": songs}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": checks, "requests": totals["requests"], "crops": totals["crops"], "seconds": report["run"]["seconds"]}))


if __name__ == "__main__":
    main()
