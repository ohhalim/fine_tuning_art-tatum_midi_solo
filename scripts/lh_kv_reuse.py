#!/usr/bin/env python3
"""Reusable common prefix under the current #1773 contract (issue #1793, docs/experiments/LH_KV_REUSE.md).

  lh_kv_reuse.py --count <export dir> <velocity dir> <out json>                       (no model)
  lh_kv_reuse.py --equiv <aria repo> <ckpt> <export dir> <velocity dir> <count json> <out json>   (MLX)
  lh_kv_reuse.py --cost  <aria repo> <ckpt> <export dir> <count json> <#1791 report json> <out json>  (MLX)
Run with the t3_aria venv.
"""
from __future__ import annotations

import glob
import gzip
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SONGS = (44, 68, 134, 185)
REPS = 5
CAPACITY = 4096 + 64
DEADLINE_MS = 60_000 / 128


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def lh_triplet(tokens, tags, i):
    """(start, key, triplet) of the left-hand note whose token is at i, or None. key = (pitch, segment, onset)."""
    if i >= len(tags) or tags[i] != "lh":
        return None
    m = 0
    while i - m - 1 >= 0 and tags[i - m - 1] == "lh":
        m += 1
    j = i - (m % 3)
    trip = tuple(tokens[j:j + 3])
    seg = sum(1 for t in tokens[:j] if t == "<T>")
    return j, (trip[0][1], seg, trip[1][1]), trip


def lh_keys(tokens, tags):
    keys, i = set(), 0
    while i < len(tokens):
        if tags[i] == "lh":
            _, key, _ = lh_triplet(tokens, tags, i)
            keys.add(key)
            i += 3
        else:
            i += 1
    return keys


def cause(P_tok, P_tag, N_tok, N_tag, d):
    """Why the inputs part at position d (the common prefix length)."""
    if d == len(N_tok):
        return "none"
    if d == len(P_tok):
        return "append"
    tp, tn = lh_triplet(P_tok, P_tag, d), lh_triplet(N_tok, N_tag, d)
    if tp and tn and tp[1] == tn[1] and tp[2] != tn[2]:
        return "held_update"
    if tp and tp[1] not in lh_keys(N_tok, N_tag):
        return "expired"
    if tn and tn[1] not in lh_keys(P_tok, P_tag):
        return "insert"
    return "other"


def pair_rows(reqs, arm):
    """reqs: built requests of one song in order (dicts from build_request, or None for failures).
    -> [{k, P, S, L, n, cause}] for consecutive pairs with k >= 2 and both requests built."""
    out = []
    for prev, cur in zip(reqs, reqs[1:]):
        if prev is None or cur is None or cur["k"] < 2 or prev[arm] is None or cur[arm] is None:
            continue
        P_tok, P_tag = prev[arm]["tokens"], prev[arm]["tags"]
        ls = cur[arm]["label_start"]
        N_tok, N_tag = cur[arm]["tokens"][:ls], cur[arm]["tags"][:ls]
        if (prev["keep_from"], prev["base_ms"]) != (cur["keep_from"], cur["base_ms"]):
            P, why = 0, "crop"
        else:
            P = common_prefix(P_tok, N_tok)
            why = cause(P_tok, P_tag, N_tok, N_tag, P)
        out.append({"k": cur["k"], "P": P, "S": len(N_tok) - P, "L": len(N_tok),
                    "n": cur["win_labels"] + cur["boundary_labels"], "cause": why})
    return out


def summary(xs):
    s = sorted(xs)
    return {"n": len(s), "median": statistics.median(s), "p90": s[int(0.9 * (len(s) - 1))], "max": s[-1]} if s else None


def build_songs(export, velocity_dir, w_ms=None):
    """Built requests per dev song. w_ms: left-hand window (default lh_context.W_MS; #1820 varies it)."""
    from crop_contract import CapacityFailure, session_requests
    from lh_context import W_MS, ContractViolation, Human, build_request
    w_ms = W_MS if w_ms is None else w_ms
    from lh_context_build import load_song
    manifest = json.load(open(os.path.join(velocity_dir, "manifest.json")))
    songs = {i: load_song(export, velocity_dir, i, manifest) for i in SONGS}
    pool = {i: (Human(h), [(r["k"], r["s"]) for r in session_requests(t, t0)]) for i, (t, t0, h) in songs.items()}
    built, fails = {}, 0
    for i, (tokens, t0, human) in songs.items():
        h = Human(human)
        built[i] = []
        for srow in session_requests(tokens, t0):
            try:
                built[i].append(build_request(tokens, srow, h, i, pool, w_ms=w_ms))
            except (CapacityFailure, ContractViolation):
                built[i].append(None)
                fails += 1
    return built, fails


def count(export, velocity_dir, out_path):
    import collections
    built, fails = build_songs(export, velocity_dir)
    rep = {"issue": 1793, "songs": SONGS, "contract": "#1773 default (window 7.5 s)", "build_failures": fails}
    for arm in ("a", "b"):
        rows = [dict(x, song=i) for i, reqs in built.items() for x in pair_rows(reqs, arm)]
        nc = [x for x in rows if x["cause"] != "crop"]
        rep[arm] = {"pairs": len(rows), "causes": dict(collections.Counter(x["cause"] for x in rows)),
                    "P": summary([x["P"] for x in nc]), "S": summary([x["S"] for x in nc]), "L": summary([x["L"] for x in rows]),
                    "S_by_cause": {c: summary([x["S"] for x in nc if x["cause"] == c]) for c in sorted({x["cause"] for x in nc})},
                    "rows": rows}
        print(arm, rep[arm]["pairs"], rep[arm]["causes"], "P", rep[arm]["P"], "S", rep[arm]["S"])
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(rep, f)


PRECISIONS = {"fp32": None, "bf16": None, "q8": 8, "q4": 4}     # quantization bits; #1811 recipe (group 32, model.model only)


def load_mlx(aria_repo, ckpt, precision="fp32"):
    if precision not in PRECISIONS:
        raise ValueError(f"precision {precision!r} not in {sorted(PRECISIONS)}")
    sys.path.insert(0, aria_repo)
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np
    from aria.config import load_model_config
    from aria.inference.model_mlx import TransformerLM
    from aria.model import ModelConfig
    from ariautils.tokenizer import AbsTokenizer
    tok = AbsTokenizer()
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    dtype = mx.float32 if precision == "fp32" else mx.bfloat16      # weights and KV cache together, as in Aria's demo_mlx.py
    if precision == "fp32":
        model.load_weights(ckpt, strict=True)
    else:
        model.load_weights([(k, v.astype(dtype)) for k, v in mx.load(ckpt).items()], strict=True)
    if PRECISIONS[precision] is not None:
        nn.quantize(model.model, group_size=32, bits=PRECISIONS[precision])
    mx.eval(model.parameters())
    model.eval()
    model.setup_cache(batch_size=1, max_seq_len=CAPACITY, dtype=dtype)

    def run(ids, start, kv_end=None):
        """Forward ids at positions start.. ; KV beyond kv_end (default: last written) is invisible. Returns all logits."""
        idxs = mx.array([ids], dtype=mx.int32)
        pos = mx.arange(start, start + len(ids), dtype=mx.int32)
        last = start + len(ids) - 1 if kv_end is None else kv_end
        o = model(idxs=idxs, input_pos=pos, offset=pos[0], max_kv_pos=last)
        mx.eval(o)
        return np.array(o[0].astype(mx.float32))
    return tok, run


def compare(run, P_ids, N_ids, P):
    """Full prefill of N vs cached suffix (Pseq KV kept up to P, N[P:] recomputed, later old KV masked)."""
    import numpy as np
    S = len(N_ids) - P
    if S > 0:
        run(P_ids, 0)
        cached = run(N_ids[P:], P, kv_end=len(N_ids) - 1)
    else:
        cached = run(P_ids, 0)[len(N_ids) - 1:len(N_ids)]
    full = run(N_ids, 0)
    ref = full[P:] if S > 0 else full[len(N_ids) - 1:len(N_ids)]
    top1 = int((cached.argmax(-1) == ref.argmax(-1)).sum())
    return {"P": P, "S": S, "positions": len(ref), "top1_equal": top1, "max_abs_last": float(np.abs(cached[-1] - ref[-1]).max()),
            "ok": top1 == len(ref) and float(np.abs(cached[-1] - ref[-1]).max()) <= 1e-3}


def synthetic_cases(built):
    """Six edited copies of one real request with held left hand (insert, expired, held_update, none, empty, same_onset)."""
    import dataclasses
    from lh_context import merge
    base_req = next(r for reqs in built.values() for r in reqs
                    if r and r["layer"] == "held" and r["lh_notes"] >= 3 and r["extra_crop"] is None)
    a = base_req["a"]
    ctx = a["tokens"][:a["label_start"]]
    cont = a["tokens"][a["label_start"]:]
    own = list(base_req["own"])
    held = next(n for n in own if n.held)
    shorter = dataclasses.replace(held, off=held.off - 200, raw_off=held.raw_off - 200)
    twin = dataclasses.replace(own[0], pitch=own[0].pitch + 12 if own[0].pitch + 12 < 55 else own[0].pitch - 12)
    cases = {"insert": (own[:-1], own), "expired": (own, own[1:]),
             "held_update": ([shorter if n is held else n for n in own], own), "none": (own, own),
             "empty": ([], []), "same_onset": (own, sorted(own + [twin], key=lambda n: (n.on, n.pitch)))}
    out = {}
    for name, (prev, new) in cases.items():
        P_tok = merge(ctx, base_req["base_ms"], prev)[0] + cont
        N_tok = merge(ctx, base_req["base_ms"], new)[0]
        out[name] = (P_tok, N_tok)
    return out, {"song_k": base_req["k"], "lh_notes": base_req["lh_notes"]}


def equiv(aria_repo, ckpt, export, velocity_dir, count_path, out_path):
    built, _ = build_songs(export, velocity_dir)
    tok, run = load_mlx(aria_repo, ckpt)
    enc = lambda toks: tok.encode(list(toks))  # noqa: E731
    res = {"issue": 1793, "tolerance": 1e-3, "synthetic": {}, "dev": {}}
    cases, meta = synthetic_cases(built)
    res["synthetic_source"] = meta
    for name, (P_tok, N_tok) in cases.items():
        res["synthetic"][name] = compare(run, enc(P_tok), enc(N_tok), common_prefix(P_tok, N_tok))
        print("synthetic", name, res["synthetic"][name], flush=True)
    cnt = json.load(open(count_path))
    picked = {}
    for x in cnt["b"]["rows"]:
        if x["cause"] not in picked and x["cause"] != "crop":
            picked[x["cause"]] = x
    for c, x in picked.items():
        reqs = built[x["song"]]
        cur_i = next(i for i, r in enumerate(reqs) if r and r["k"] == x["k"])
        prev, cur = reqs[cur_i - 1], reqs[cur_i]
        P_tok = prev["b"]["tokens"]
        N_tok = cur["b"]["tokens"][:cur["b"]["label_start"]]
        res["dev"][c] = {"song": x["song"], "k": x["k"], **compare(run, enc(P_tok), enc(N_tok), x["P"])}
        print("dev", c, res["dev"][c], flush=True)
    res["all_ok"] = all(v["ok"] for v in list(res["synthetic"].values()) + list(res["dev"].values()))
    res["inputs"] = {"count_sha256": file_sha(count_path), "checkpoint_sha256": file_sha(ckpt)}
    with open(out_path, "w") as f:
        json.dump(res, f, indent=1)
    print("all_ok", res["all_ok"])
    if not res["all_ok"]:
        sys.exit(1)


def file_sha(path):
    import hashlib
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def resummarize(cost_path, full_path, count_path):
    """Post-hoc sensitivity on the stored measurement (no model): length-based decode at L = P + S and joint (P, S, n) counts."""
    import lh_prefill_deadline as lp
    c, full, cnt = json.load(open(cost_path)), json.load(open(full_path)), json.load(open(count_path))
    by = sorted((int(L), v["decode_first_median_ms"], v["decode_rest_median_ms"]) for L, v in full["by_length"].items())
    xs = [a for a, _, _ in by]
    nc = [x for x in cnt["b"]["rows"] if x["cause"] != "crop"]
    for pt in c["points"]:
        L = pt["P"] + pt["S"]
        f, r = lp.interp(xs, [b for _, b, _ in by], L), lp.interp(xs, [cc for _, _, cc in by], L)
        pt["decode_by_length_at"] = L
        pt["request_ms_n50_by_length"] = round(pt["T_ms"] + f + (c["n50"] - 2) * r, 1)
        pt["request_ms_n90_by_length"] = round(pt["T_ms"] + f + (c["n90"] - 2) * r, 1)
        pt["dominated_and_n_le_n90"] = sum(1 for x in nc if x["P"] <= pt["P"] and x["S"] <= pt["S"] and x["n"] <= c["n90"])
    c["sensitivity_note"] = ("post-hoc: decode interpolated by context length at L = P + S (not growing during decode); "
                             "joint counts are dominated pairs with n <= n90; no pass-rate claim from these")
    c["inputs"] = {"count_sha256": file_sha(count_path), "full_report_sha256": file_sha(full_path)}
    with open(cost_path, "w") as f:
        json.dump(c, f, indent=1)
    print(json.dumps([{k: pt[k] for k in ("label", "request_ms_n90", "request_ms_n90_by_length", "dominated_pairs", "dominated_and_n_le_n90")}
                      for pt in c["points"]]))


def representative_points(rows):
    """Fixed rule (doc 5): S quantiles 50/90/99 with the median P of pairs near each, plus (P p90, S p90)."""
    nc = [x for x in rows if x["cause"] != "crop"]
    S = sorted(x["S"] for x in nc)
    P = sorted(x["P"] for x in nc)
    q = lambda xs, f: xs[int(f * (len(xs) - 1))]  # noqa: E731
    pts = []
    for f in (0.5, 0.9, 0.99):
        sq = q(S, f)
        near = [x["P"] for x in nc if 0.9 * sq <= x["S"] <= 1.1 * sq] or \
            [x["P"] for x in sorted(nc, key=lambda x: abs(x["S"] - sq))[:10]]
        pts.append({"label": f"S{int(f * 100)}", "P": int(statistics.median(near)), "S": sq})
    pts.append({"label": "P90_S90", "P": q(P, 0.9), "S": q(S, 0.9)})
    return pts


def cost(aria_repo, ckpt, export, count_path, full_path, out_path):
    import lh_prefill_deadline as lp
    cnt, full = json.load(open(count_path)), json.load(open(full_path))
    eq_path = os.path.join(os.path.dirname(out_path), "equiv.json")
    eq = json.load(open(eq_path)) if os.path.exists(eq_path) else {}
    if not eq.get("all_ok") or eq.get("inputs", {}).get("count_sha256") not in (None, file_sha(count_path)):
        sys.exit("equivalence missing, failed or built from another count: no cost conclusion (doc 4)")
    if cnt.get("build_failures"):
        sys.exit("count has build failures: fail closed")
    rows = cnt["b"]["rows"]
    pts = representative_points(rows)
    songs = sorted(((len(d["token_ids"]), d["token_ids"]) for d in
                    (json.load(gzip.open(f)) for f in glob.glob(os.path.join(export, "requests", "*.json.gz")))), reverse=True)
    ids = songs[0][1] + songs[1][1]
    _, run = load_mlx(aria_repo, ckpt)
    first, rest = full["prediction"]["decode_first_ms"], full["prediction"]["decode_rest_ms"]
    ns = sorted(x["n"] for x in rows)
    n50, n90 = ns[int(0.5 * (len(ns) - 1))], ns[int(0.9 * (len(ns) - 1))]
    dec = lambda n: (first + (lp.extra_forwards(n) - 1) * rest) if lp.extra_forwards(n) else 0.0  # noqa: E731
    nc = [x for x in rows if x["cause"] != "crop"]
    for pt in pts:
        times = []
        for rep in range(REPS + 1):
            run(ids[:pt["P"]], 0)
            t = time.perf_counter()
            if pt["S"] > 0:
                run(ids[pt["P"]:pt["P"] + pt["S"]], pt["P"])
            if rep:
                times.append((time.perf_counter() - t) * 1000)
        pt["T_ms"] = round(statistics.median(times), 1)
        pt["T_runs_ms"] = [round(x, 1) for x in times]
        pt["request_ms_n50"] = round(pt["T_ms"] + dec(n50), 1)
        pt["request_ms_n90"] = round(pt["T_ms"] + dec(n90), 1)
        pt["dominated_pairs"] = sum(1 for x in nc if x["P"] <= pt["P"] and x["S"] <= pt["S"])
        print(pt, flush=True)
    crop = [x for x in rows if x["cause"] == "crop"]
    curve = [(int(L), v["prefill_median_ms"]) for L, v in full["by_length"].items()]
    out = {"issue": 1793, "points": pts, "n50": n50, "n90": n90, "decode_first_ms": first, "decode_rest_ms": rest,
           "deadline_ms": DEADLINE_MS, "pairs_non_crop": len(nc),
           "crop_pairs_full_prefill_ms": [round(lp.interp([a for a, _ in curve], [b for _, b in curve], x["L"]), 1) for x in crop],
           "note": "teacher-forced token sequences; not a free-generation success rate; not end-to-end"}
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    mode, args = sys.argv[1], sys.argv[2:]
    {"--count": count, "--equiv": equiv, "--cost": cost, "--resummarize": resummarize}[mode](*args)
