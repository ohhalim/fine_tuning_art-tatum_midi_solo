#!/usr/bin/env python3
"""Train-only export of block-request examples (issue #1756, docs/experiments/LH_BLOCK_REQUEST_INPUT.md).
No model, no training. Two stages because the note loader (pretty_midi) and the Aria tokenizer live in different venvs.

  split  (main venv):  lh_request_export.py split <bebop_rh manifest> <out dir>
         train songs only (val/test never opened, SHA1 checked). g3 split: pitch < 55 is the human condition history,
         pitch >= 55 the model's target stream. Writes <out>/songs/NNN.target.mid (target notes only, original velocity,
         no pedal CC) and NNN.human.json (human notes in ms, the audit's loader)
  build  (Aria venv):  lh_request_export.py build <out dir> <aria repo> <report json>
         tokenizes each target MIDI with ariautils AbsTokenizer, builds the request table (scripts/block_request.py) with
         t0 = PHASES[song index % 4], the payload of every request, and runs the checks below on every song. Writes
         <out>/requests/NNN.json.gz and the summary report

Checks (registered before the run): C1 every request's context payloads equal the sequential runtime replay
(simulate_kv); C2 a prefix-only stream (tokens before the request's context end, then <E>) gives the same committing
requests for the context; C3 the payload from human notes observed up to s_k only (later notes dropped, releases after
s_k unknown) equals the payload from all human notes; C4 every token pitch is >= 55 (no human note in the model stream).
The tokenizer may merge or drop notes; the token note count against the target note count is reported, not judged.
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
from block_contract import BLOCK, block_start  # noqa: E402
from block_request import (COLD, committing_request, loss_class, payload, position_requests, primer_end,  # noqa: E402
                           request_table, simulate_kv)
from lh_role_proxy import INF, SPLIT_PITCH, Note  # noqa: E402

PHASES = (0.0, BLOCK / 4, BLOCK / 2, 3 * BLOCK / 4)
CAPACITY, CAP = 4096, 64          # scripts/rt_producer.py cache capacity, scripts/aria_mlx_stream_loop.py tokens per block


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split(manifest_path, out_dir):
    import pretty_midi
    from lh_audit_stats import load_notes, sha1_file
    manifest = json.load(open(manifest_path))
    train = [s for s in manifest["songs"] if s["split"] == "train"]
    assert len(train) == 189, len(train)
    os.makedirs(os.path.join(out_dir, "songs"), exist_ok=True)
    meta = []
    for i, s in enumerate(train):
        assert sha1_file(s["source"]) == s["sha1"], s["source"]
        notes, info = load_notes(s["source"])
        hum = [n for n in notes if n.pitch < SPLIT_PITCH]
        tgt = [n for n in notes if n.pitch >= SPLIT_PITCH]
        pm = pretty_midi.PrettyMIDI(initial_tempo=120)
        ins = pretty_midi.Instrument(0)
        for n in tgt:
            ins.notes.append(pretty_midi.Note(velocity=n.vel, pitch=n.pitch, start=n.on / 1000.0, end=n.off / 1000.0))
        pm.instruments.append(ins)
        pm.write(os.path.join(out_dir, "songs", f"{i:03d}.target.mid"))
        rec = {"idx": i, "source": s["source"], "sha1": s["sha1"], "pianist": s["pianist"], "notes": len(notes),
               "target_notes": len(tgt), "human_notes": len(hum), "nonpositive_length": info["nonpositive_length"]}
        with open(os.path.join(out_dir, "songs", f"{i:03d}.human.json"), "w") as f:
            json.dump({**rec, "human": [[n.nid, n.pitch, n.on, n.off] for n in hum]}, f)
        meta.append(rec)
    with open(os.path.join(out_dir, "songs", "split_meta.json"), "w") as f:
        json.dump({"manifest": manifest_path, "manifest_sha256": sha256_file(manifest_path), "split_pitch": SPLIT_PITCH,
                   "songs": meta}, f, indent=1)
    print(f"split {len(meta)} train songs into {out_dir}/songs")


def dist(xs):
    if not xs:
        return None
    s = sorted(xs)

    def q(p):
        return s[min(len(s) - 1, max(0, math.ceil(p * len(s)) - 1))]
    return {"n": len(s), "mean": round(statistics.fmean(s), 3), "min": s[0], "median": q(0.5), "p90": q(0.9), "max": s[-1]}


def observed_only(humans, s):
    """Human notes as the runtime knows them at s: later note-ons dropped, releases after s unknown."""
    return [Note(n.nid, n.pitch, n.on, n.off if n.off <= s else INF) for n in humans if n.on <= s]


def build_song(idx, toks, humans):
    t0 = PHASES[idx % len(PHASES)]
    rows = request_table(toks, t0)
    pays = {r["k"]: payload(humans, r["s"], r["k"]) for r in rows}
    chk = {"c1_replay_mismatch": 0, "c2_prefix_mismatch": 0, "c3_observed_only_mismatch": 0}
    kv = simulate_kv(toks, t0, humans)
    for r in rows:
        k = r["k"]
        owner = position_requests(rows, k)[:r["ctx_end"] - 1]
        chk["c1_replay_mismatch"] += kv[k] != [COLD if j < 0 else pays[j] for j in owner]
        pre = request_table(list(toks[:r["ctx_end"]]) + ["<E>"], t0)
        chk["c2_prefix_mismatch"] += committing_request(pre, r["ctx_end"]) != committing_request(rows, r["ctx_end"])
        chk["c3_observed_only_mismatch"] += payload(observed_only(humans, r["s"]), r["s"], k) != pays[k]
    piano = [t for t in toks if isinstance(t, tuple) and t[0] == "piano"]
    chk["c4_low_pitch_tokens"] = sum(1 for t in piano if t[1] < 55)
    reqs = []
    for r in rows:
        pc, avail = pays[r["k"]]
        _, classes = loss_class(rows, r["k"])
        reqs.append({**r, "pc12": list(pc), "available": avail, "window_tokens": classes.count("window"),
                     "boundary_tokens": classes.count("boundary")})
    return {"t0": t0, "primer_end": primer_end(toks, t0), "token_notes": len(piano), "requests": reqs, "checks": chk}


def build(out_dir, aria_repo, report_path):
    sys.path.insert(0, aria_repo)
    from ariautils.midi import MidiDict
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    t_start = time.time()
    meta = json.load(open(os.path.join(out_dir, "songs", "split_meta.json")))
    os.makedirs(os.path.join(out_dir, "requests"), exist_ok=True)
    songs, totals = [], collections.Counter()
    ctx_lens, win_toks, bnd_toks, pc_sets = [], [], [], collections.Counter()
    for rec in meta["songs"]:
        i = rec["idx"]
        hj = json.load(open(os.path.join(out_dir, "songs", f"{i:03d}.human.json")))
        humans = [Note(nid, p, on, off) for nid, p, on, off in hj["human"]]
        toks = tok.tokenize(MidiDict.from_midi(os.path.join(out_dir, "songs", f"{i:03d}.target.mid")))
        res = build_song(i, toks, humans)
        out = {"idx": i, "source": rec["source"], "sha1": rec["sha1"], "pianist": rec["pianist"], "t0": res["t0"],
               "primer_end": res["primer_end"], "tokens": [list(t) if isinstance(t, tuple) else t for t in toks],
               "token_ids": tok.encode(toks), "requests": res["requests"]}
        path = os.path.join(out_dir, "requests", f"{i:03d}.json.gz")
        with open(path, "wb") as fb, gzip.GzipFile(filename="", fileobj=fb, mode="wb", mtime=0) as gz:
            gz.write(json.dumps(out, separators=(",", ":")).encode())
        reqs = res["requests"]
        warm = [r for r in reqs if r["available"]]
        for k, v in res["checks"].items():
            totals[k] += v
        totals["requests"] += len(reqs)
        totals["cold_requests"] += sum(1 for r in reqs if not r["available"])
        totals["warm_requests"] += len(warm)
        totals["warm_nonempty_payload"] += sum(1 for r in warm if any(r["pc12"]))
        totals["empty_windows"] += sum(1 for r in reqs if r["window_tokens"] == 0)
        totals["no_boundary"] += sum(1 for r in reqs if not r["boundary"])
        totals["windows_over_cap"] += sum(1 for r in reqs if r["window_tokens"] + r["boundary_tokens"] > CAP)
        totals["requests_over_capacity"] += sum(1 for r in reqs if r["cont_end"] > CAPACITY)
        totals["tokens"] += len(toks)
        totals["token_notes"] += res["token_notes"]
        totals["target_notes"] += rec["target_notes"]
        ctx_lens += [r["ctx_end"] for r in reqs]
        win_toks += [r["window_tokens"] for r in reqs]
        bnd_toks += [r["boundary_tokens"] for r in reqs]
        pc_sets.update(tuple(r["pc12"]) for r in warm if any(r["pc12"]))
        songs.append({"idx": i, "pianist": rec["pianist"], "t0": res["t0"], "tokens": len(toks), "token_notes": res["token_notes"],
                      "target_notes": rec["target_notes"], "requests": len(reqs),
                      "warm_nonempty_ratio": round(sum(1 for r in warm if any(r["pc12"])) / len(warm), 4) if warm else None,
                      "checks": res["checks"], "file_sha256": sha256_file(path)})
    checks = {k: totals[k] for k in ("c1_replay_mismatch", "c2_prefix_mismatch", "c3_observed_only_mismatch", "c4_low_pitch_tokens")}
    report = {"issue": 1756, "doc": "docs/experiments/LH_BLOCK_REQUEST_INPUT.md", "quality_claimed": False, "training": False,
              "run": {"git_head": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                      "block_request_sha256": sha256_file(os.path.join(HERE, "block_request.py")),
                      "export_script_sha256": sha256_file(os.path.abspath(__file__)), "out_dir": out_dir,
                      "manifest_sha256": meta["manifest_sha256"], "seconds": round(time.time() - t_start, 1)},
              "params": {"phases": PHASES, "t0_rule": "PHASES[song index % 4]", "payload": "g3 sounding at s_k, 12-PC multihot + available",
                         "capacity_tokens": CAPACITY, "cap_tokens_per_block": CAP, "split_pitch": SPLIT_PITCH},
              "checks": checks, "all_checks_zero": not any(checks.values()),
              "totals": dict(totals),
              "warm_nonempty_ratio": round(totals["warm_nonempty_payload"] / totals["warm_requests"], 4) if totals["warm_requests"] else None,
              "distinct_pc12_sets_warm_nonempty": len(pc_sets), "top5_pc12_sets": [[list(k), v] for k, v in pc_sets.most_common(5)],
              "context_tokens": dist(ctx_lens), "window_tokens": dist(win_toks), "boundary_tokens": dist(bnd_toks),
              "songs": songs}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": checks, "requests": totals["requests"], "seconds": report["run"]["seconds"]}))


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "split":
        split(*args)
    elif cmd == "build":
        build(*args)
    else:
        sys.exit(f"unknown command {cmd}")
