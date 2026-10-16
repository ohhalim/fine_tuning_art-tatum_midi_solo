"""Build and check the a/b/c inputs of #1772 on the dev songs (docs/experiments/LH_CONTEXT_CONTRACT.md). Aria venv.

No model, no scoring. For every request of the given songs: build a/b/c with scripts/lh_context.py, run L1-L6 and
L8, the future-invariance check L1f (change only human onsets/offs after s_k), and L7 (every human note used is
tokenized alone by the official AbsTokenizer from a 1 tick = 1 ms MIDI and compared with the pure-Python tokens).
Arms are not stored as token lists: they are rebuilt from the stored notes and crop state, and each arm's sha256 is
kept so a later scoring run can check it rebuilt the same input.

  .../t3_aria/.venv/bin/python scripts/lh_context_build.py --export /Users/ohhalim/git_box/lh_request_export_run3 \
      --velocity /Users/ohhalim/git_box/lh_context_run1/human_velocity --out /Users/ohhalim/git_box/lh_context_run1 \
      --report docs/experiments/lh_context/report.json --songs 44 68 185 134
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import statistics
import sys
import tempfile

sys.path.insert(0, os.path.dirname(__file__))
from crop_contract import CapacityFailure, session_requests  # noqa: E402
from lh_context import (W_MS, ContractViolation, Human, build_request, check_request, dur_capped,  # noqa: E402
                        note_tokens, quantized_end)
from lh_role_proxy import Note  # noqa: E402


def tup(t):
    return tuple(t) if isinstance(t, list) else t


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def arm_sha(tokens):
    return hashlib.sha256(json.dumps(tokens, ensure_ascii=False).encode()).hexdigest()


def load_song(export, velocity_dir, i, manifest):
    d = json.load(gzip.open(os.path.join(export, "requests", f"{i:03d}.json.gz")))
    vpath = os.path.join(velocity_dir, f"{i:03d}.json")
    want = {s["idx"]: s["file_sha256"] for s in manifest["songs"]}
    if sha256_file(vpath) != want[i]:
        sys.exit(f"song {i}: velocity file sha256 differs from its manifest")
    v = json.load(open(vpath))
    if v["sha1"] != d["sha1"]:
        sys.exit(f"song {i}: velocity file belongs to another source")
    human = [Note(nid, p, on, off, vel) for nid, p, on, off, vel in v["human"]]
    return [tup(t) for t in d["tokens"]], d["t0"], human


def perturb_future(human, s):
    """L1f: same notes up to s, but every held note lasts 1 s longer and five new notes start at or after s."""
    out = [Note(n.nid, n.pitch, n.on, n.off + 1000 if n.off > s else n.off, n.vel) for n in human]
    out += [Note(10 ** 6 + j, 43 + j, s + 37.0 * j, s + 37.0 * j + 400, 50) for j in range(5)]
    return out


def official_tokens(tok, notes):
    """{(pitch, on, off, vel): (n_T, triple) | None} from AbsTokenizer, one note per 1 tick = 1 ms MIDI file."""
    import mido
    from ariautils.midi import MidiDict
    kw = {"remove_preceding_silence": False, "add_dim_tok": False, "add_eos_tok": True}
    out = {}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "n.mid")
        for key in sorted(notes):
            p, on, off, vel = key
            mf = mido.MidiFile(ticks_per_beat=1000)
            tr = mido.MidiTrack()
            tr.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
            tr.append(mido.Message("program_change", program=0, channel=0, time=0))
            tr.append(mido.Message("note_on", note=p, velocity=vel, channel=0, time=on))
            tr.append(mido.Message("note_off", note=p, velocity=0, channel=0, time=off - on))
            mf.tracks.append(tr)
            mf.save(path)
            toks = tok.tokenize(MidiDict.from_midi(path), **kw)
            body = [t for t in toks if not (isinstance(t, tuple) and t[0] == "prefix") and t not in ("<S>", "<E>")]
            n_t = body.count("<T>")
            notes_only = [t for t in body if t != "<T>"]
            out[key] = (n_t, notes_only) if len(notes_only) == 3 else None
    return out


def summarize(xs):
    return {"n": len(xs), "mean": round(statistics.mean(xs), 2), "median": statistics.median(xs)} if xs else {"n": 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--velocity", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--songs", type=int, nargs="+", required=True)
    a = ap.parse_args()
    from ariautils.tokenizer import AbsTokenizer
    import ariautils.tokenizer.absolute
    tok = AbsTokenizer()

    manifest = json.load(open(os.path.join(a.velocity, "manifest.json")))
    songs = {i: load_song(a.export, a.velocity, i, manifest) for i in a.songs}
    pool = {i: (Human(h), [(r["k"], r["s"]) for r in session_requests(t, t0)]) for i, (t, t0, h) in songs.items()}

    checks = {k: 0 for k in ("L1", "L1t", "L1f", "L2", "L3", "L4", "L5", "L6", "L7", "L8")}
    counts = {"requests": 0, "cold": 0, "empty": 0, "held": 0, "released": 0, "donor_missing": 0, "ineligible_c": 0,
              "b_equals_c": 0, "extra_crop": 0, "capacity_failure": 0, "contract_violation": 0, "lh_end_past_s": 0,
              "lh_dur_capped": 0, "l7_zero_length_skipped": 0}
    lens = {"a": [], "b": [], "c": []}
    imbalance = {"b_mean_pitch": [], "c_mean_pitch": [], "b_mean_vel": [], "c_mean_vel": [], "b_total_ms": [], "c_total_ms": []}
    used = set()
    os.makedirs(os.path.join(a.out, "arms"), exist_ok=True)
    per_song = []
    for i, (tokens, t0, human) in songs.items():
        srows = session_requests(tokens, t0)
        h_idx = Human(human)
        rows_out = []
        for srow in srows:
            counts["requests"] += 1
            try:
                r = build_request(tokens, srow, h_idx, i, pool)
            except CapacityFailure:
                counts["capacity_failure"] += 1
                rows_out.append({"k": srow["k"], "status": "capacity_failure"})
                continue
            except ContractViolation as e:
                counts["contract_violation"] += 1
                rows_out.append({"k": srow["k"], "status": f"contract_violation: {e}"})
                continue
            for key, val in check_request(r).items():
                checks[key] += val
            r2 = build_request(tokens, srow, Human(perturb_future(human, srow["s"])), i, pool)
            same = all((r[x] and r[x]["tokens"]) == (r2[x] and r2[x]["tokens"]) for x in "abc") and \
                (r["donor"], r["keep_from"], r["base_ms"]) == (r2["donor"], r2["keep_from"], r2["base_ms"])
            checks["L1f"] += not same
            counts["cold"] += srow["k"] == 0
            counts[r["layer"]] += 1
            counts["donor_missing"] += r["donor"] == "missing"
            counts["ineligible_c"] += not r["eligible_c"]
            counts["extra_crop"] += r["extra_crop"] is not None
            if r["c"] is not None and r["layer"] != "empty" and r["b"]["tokens"] == r["c"]["tokens"]:
                counts["b_equals_c"] += 1
            for n in r["own"] + r["donor_notes"]:
                counts["lh_end_past_s"] += quantized_end(n) > r["s"]
                counts["lh_dur_capped"] += dur_capped(n)
                used.add((n.pitch, n.on, n.off, n.vel))
            for x in "abc":
                if r[x] is not None:
                    lens[x].append(r[x]["label_start"])
            if r["own"] and isinstance(r["donor"], dict):
                for arm, ns in (("b", r["own"]), ("c", r["donor_notes"])):
                    imbalance[f"{arm}_mean_pitch"].append(statistics.mean(n.pitch for n in ns))
                    imbalance[f"{arm}_mean_vel"].append(statistics.mean(n.vel for n in ns))
                    imbalance[f"{arm}_total_ms"].append(sum(n.off - n.on for n in ns))
            note = lambda n: [n.pitch, n.on, n.off, n.vel, n.held, n.raw_on, n.raw_off]  # noqa: E731
            rows_out.append({"k": r["k"], "s": r["s"], "status": "ok", "keep_from": r["keep_from"], "base_ms": r["base_ms"],
                             "extra_crop": r["extra_crop"], "window_ms": r["window_ms"], "layer": r["layer"],
                             "donor": r["donor"], "eligible_c": r["eligible_c"],
                             "own": [note(n) for n in r["own"]], "donor_notes": [note(n) for n in r["donor_notes"]],
                             "label_start": {x: (r[x]["label_start"] if r[x] else None) for x in "abc"},
                             "win_labels": r["win_labels"], "boundary_labels": r["boundary_labels"],
                             "sha256": {x: (arm_sha(r[x]["tokens"]) if r[x] else None) for x in "abc"}})
        with gzip.open(os.path.join(a.out, "arms", f"{i:03d}.json.gz"), "wt") as f:
            json.dump({"idx": i, "t0": t0, "w_ms": W_MS, "requests": rows_out}, f)
        per_song.append({"idx": i, "requests": len(srows)})

    zero = [k for k in used if k[1] == k[2]]
    counts["l7_zero_length_skipped"] = len(zero)
    official = official_tokens(tok, [k for k in used if k[1] != k[2]])
    l7_bad = []
    for key, got in official.items():
        p, on, off, vel = key
        mine = note_tokens(type("N", (), {"pitch": p, "on": on, "off": off, "vel": vel})())
        if got is None or got[0] != mine[0] or got[1] != mine[2]:
            l7_bad.append({"note": key, "official": got, "mine": [mine[0], mine[2]]})
    checks["L7"] = len(l7_bad)

    report = {"issue": 1772, "songs": a.songs, "w_ms": W_MS, "checks": checks,
              "contract_ok": all(v == 0 for v in checks.values()) and counts["capacity_failure"] == 0
              and counts["contract_violation"] == 0,
              "counts": counts, "l7_notes_compared": len(official), "l7_mismatch_examples": l7_bad[:10],
              "session_length": {x: summarize(v) for x, v in lens.items()},
              "b_vs_c_summary": {k: summarize(v) for k, v in imbalance.items()},
              "inputs": {"export": a.export, "velocity_manifest_sha256": sha256_file(os.path.join(a.velocity, "manifest.json")),
                         "ariautils_absolute_py_sha256": sha256_file(ariautils.tokenizer.absolute.__file__)},
              "per_song": per_song}
    os.makedirs(os.path.dirname(a.report), exist_ok=True)
    with open(a.report, "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: report[k] for k in ("checks", "contract_ok", "counts", "l7_notes_compared")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
