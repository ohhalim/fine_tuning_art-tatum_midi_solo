#!/usr/bin/env python3
"""fp32 vs q8 Aria free (sampled) generation, compared across seeds (issue #1823, docs/experiments/Q8_FREE_GEN.md).

The realtime sampler (format mask, temperature 1, one Exp(1) noise vector per step, <D>/<E> masked) continues each
prompt for 30 s of music with no wall clock, no left hand and no block rollbacks. Same (prompt, seed) under both
precisions shares the noise sequence, so a pair is identical until the first different pick; distributions are compared
across pairs with a paired signed-rank test fixed before the run.

  q8_free_gen.py --generate <aria repo> <ckpt> <data json> <precision> <out jsonl> [--prompts 0,1] [--seeds 1-32]
  q8_free_gen.py --compare <fp32 jsonl> <q8 jsonl> <report json>
  q8_free_gen.py --render <fp32 jsonl> <q8 jsonl> <out dir> <report json>
"""
from __future__ import annotations

import json
import math
import os
import random
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

PROMPTS = (0, 1, 2, 3)
SEEDS = tuple(range(1, 33))
DUR_MS, MAX_TOKENS = 30_000, 2400
BLOCK_MS = 937.5                       # 2 beats at 128 BPM
CHORD_MS = 30                          # onsets within 30 ms of a cluster start are one cluster
STATS = ("notes_per_s", "pitch_median", "pitch_range", "chord_share", "ioi_median_s", "dur_median_s",
         "vel_median", "rest_share", "win_max", "win_median")
ALPHA, MIN_EFFECT = 0.05 / len(STATS), 0.3     # Bonferroni over the 10 stats; |rank-biserial| >= 0.3
LISTEN = ((0, 1), (0, 2), (3, 1), (3, 2))       # (prompt, seed) pairs rendered for listening, fixed before the run


# ---------------------------------------------------------------- statistics (pure, tested)
def run_stats(notes, t0, dur_ms=DUR_MS):
    """notes: [(on_ms, off_ms, pitch, velocity)]. Stats over notes starting in [t0, t0 + dur_ms)."""
    ns = sorted(n for n in notes if t0 <= n[0] < t0 + dur_ms)
    out = {"notes": len(ns), "notes_per_s": len(ns) / (dur_ms / 1000)}
    if not ns:
        return {**out, **{k: None for k in STATS if k != "notes_per_s"}}
    pitches = [n[2] for n in ns]
    clusters, start = [], None
    for on, *_ in ns:
        if start is None or on - start > CHORD_MS:
            clusters.append([on])
            start = on
        else:
            clusters[-1].append(on)
    starts = [c[0] for c in clusters]
    gaps = [(b - a) / 1000 for a, b in zip(starts, starts[1:])]
    covered, cur = 0.0, None                              # union of sounding intervals inside the span
    for on, off, *_ in sorted((max(n[0], t0), min(n[1], t0 + dur_ms)) for n in ns):
        if off <= on:
            continue
        if cur is None or on > cur[1]:
            if cur:
                covered += cur[1] - cur[0]
            cur = [on, off]
        else:
            cur[1] = max(cur[1], off)
    if cur:
        covered += cur[1] - cur[0]
    wins = [0] * math.ceil(dur_ms / BLOCK_MS)
    for n in ns:
        wins[int((n[0] - t0) // BLOCK_MS)] += 1
    return {**out, "pitch_median": statistics.median(pitches), "pitch_range": max(pitches) - min(pitches),
            "chord_share": sum(len(c) for c in clusters if len(c) >= 2) / len(ns),
            "ioi_median_s": statistics.median(gaps) if gaps else None,
            "dur_median_s": statistics.median((n[1] - n[0]) / 1000 for n in ns),
            "vel_median": statistics.median(n[3] for n in ns), "rest_share": 1 - covered / dur_ms,
            "win_max": max(wins), "win_median": statistics.median(wins)}


def ranks(xs):
    """Average ranks (1-based) with ties."""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def wilcoxon(diffs):
    """Signed-rank test on paired differences (zeros dropped), normal approximation with tie correction, no continuity
    correction. Returns n, W+, z, two-sided p, matched-pairs rank-biserial r = (W+ - W-) / (W+ + W-)."""
    d = [x for x in diffs if x != 0]
    n = len(d)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "z": 0.0, "p": 1.0, "r": 0.0}
    rk = ranks([abs(x) for x in d])
    wp = sum(r for r, x in zip(rk, d) if x > 0)
    wm = sum(r for r, x in zip(rk, d) if x < 0)
    ties = {}
    for a in (abs(x) for x in d):
        ties[a] = ties.get(a, 0) + 1
    var = n * (n + 1) * (2 * n + 1) / 24 - sum(t ** 3 - t for t in ties.values()) / 48
    z = (wp - n * (n + 1) / 4) / math.sqrt(var) if var > 0 else 0.0
    return {"n": n, "w_plus": wp, "z": z, "p": math.erfc(abs(z) / math.sqrt(2)), "r": (wp - wm) / (wp + wm)}


def cliffs_delta(a, b):
    """P(b > a) - P(b < a) over all pairs (b = q8, a = fp32)."""
    gt = sum(1 for x in a for y in b if y > x)
    lt = sum(1 for x in a for y in b if y < x)
    return (gt - lt) / (len(a) * len(b))


def bootstrap_median_ci(diffs, reps=2000, seed=0):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(diffs, k=len(diffs))) for _ in range(reps))
    return meds[int(0.025 * reps)], meds[int(0.975 * reps) - 1]


def compare_stat(pairs):
    """pairs: [(fp32 value, q8 value)] with both present. Primary signed-rank verdict plus secondary context."""
    a, b = [p[0] for p in pairs], [p[1] for p in pairs]
    diffs = [y - x for x, y in pairs]
    w = wilcoxon(diffs)
    lo, hi = bootstrap_median_ci(diffs) if diffs else (None, None)
    return {"pairs": len(pairs), "fp32_median": statistics.median(a) if a else None, "q8_median": statistics.median(b) if b else None,
            "median_diff": statistics.median(diffs) if diffs else None, "median_diff_ci95": [lo, hi], "wilcoxon": w,
            "cliffs_delta": cliffs_delta(a, b) if a else None,
            "detected": w["p"] < ALPHA and abs(w["r"]) >= MIN_EFFECT}


# ---------------------------------------------------------------- generation (MLX)
def parse_seeds(s):
    if "-" in s:
        a, b = s.split("-")
        return tuple(range(int(a), int(b) + 1))
    return tuple(int(x) for x in s.split(","))


def generate(aria_repo, ckpt, data_path, precision, out_path, prompts, seeds):
    import torch
    from aria_grammar_mask import GrammarMask, masked_pick
    from aria_token_validity import check as token_check
    from lh_kv_reuse import load_mlx
    from stream_contract import TokenStream
    done = set()
    if os.path.exists(out_path):
        done = {(r["prompt"], r["seed"]) for r in map(json.loads, open(out_path)) if r.get("precision") == precision}
    tok, run = load_mlx(aria_repo, ckpt, precision)
    V = tok.vocab_size
    vocab = [tok.decode([i])[0] for i in range(V)]
    to_id = {t: i for i, t in enumerate(vocab)}
    no_end = torch.tensor([t not in ("<D>", "<E>") for t in vocab])
    data = json.load(open(data_path))
    with open(out_path, "a") as out:
        for pi in prompts:
            ids = data["prompts"][pi]["ids"]
            prompt = tok.decode(ids)
            for seed in seeds:
                if (pi, seed) in done:
                    continue
                t_start = time.perf_counter()
                stream = TokenStream(prompt)
                gm = GrammarMask(vocab)
                gen = torch.Generator().manual_seed(seed)

                def position():
                    st = stream.grammar_state()
                    return st.segment * 5000 + st.last_onset if st.expect == "note" else -1

                t0 = position()
                if t0 < 0:
                    raise ValueError(f"prompt {pi} ends inside a note")
                logits = run([to_id[t] for t in prompt], 0)[-1]
                pos, picks, stop = len(prompt), [], "duration"
                while True:
                    if position() >= t0 + DUR_MS:
                        break
                    if len(picks) >= MAX_TOKENS:
                        stop = "max_tokens"
                        break
                    p = torch.softmax(torch.from_numpy(logits), -1)
                    st = stream.grammar_state()
                    gm.sync(st, stream.onset_floor(st))
                    ok = gm.allowed() & no_end
                    a = p[ok].sum().item()
                    if not (math.isfinite(a) and a > 0):
                        stop = f"allowed mass {a}"
                        break
                    pick, _ = masked_pick(p, ok, torch.empty(V).exponential_(1, generator=gen))
                    stream.append_sampled(vocab[pick])
                    picks.append(pick)
                    logits = run([pick], pos)[-1]
                    pos += 1
                new = stream.tokens[len(prompt):]
                v = token_check(prompt, new)
                notes = [(n.on_ms, n.off_ms, n.pitch, n.velocity) for n in stream.commit_until(float("inf"))]
                rec = {"prompt": pi, "seed": seed, "precision": precision, "t0_ms": t0, "stop": stop, "tokens": picks,
                       "notes": notes, "valid": bool(v["valid"] and v["tail"] is None), "first_error": v["first_error"],
                       "stats": run_stats(notes, t0), "seconds": round(time.perf_counter() - t_start, 2)}
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(precision, pi, seed, stop, len(picks), rec["stats"]["notes"], rec["valid"], rec["seconds"], flush=True)


# ---------------------------------------------------------------- compare / render
def load(path, precision):
    return {(r["prompt"], r["seed"]): r for r in map(json.loads, open(path)) if r["precision"] == precision}


def input_checks(fp32_path, q8_path, expected):
    """Preconditions before any verdict (Astra review of #1830): every expected (prompt, seed) exactly once per
    precision, each run format-valid and stopped by reaching the duration. Any failure means no verdict."""
    out = {}
    for prec, path in (("fp32", fp32_path), ("q8", q8_path)):
        rows = [r for r in map(json.loads, open(path)) if r["precision"] == prec]
        keys = [(r["prompt"], r["seed"]) for r in rows]
        seen, dup = set(), set()
        for k in keys:
            (dup if k in seen else seen).add(k)
        out[prec] = {"rows": len(rows), "duplicates": sorted(dup), "missing": sorted(set(expected) - seen),
                     "unexpected": sorted(seen - set(expected)),
                     "invalid": sorted((r["prompt"], r["seed"]) for r in rows if not r["valid"]),
                     "not_duration": sorted((r["prompt"], r["seed"]) for r in rows if r["stop"] != "duration")}
    ok = all(not v[k] for v in out.values() for k in ("duplicates", "missing", "unexpected", "invalid", "not_duration"))
    return ok, out


def seed_cluster(A, B, keys, stat):
    """Post-hoc sensitivity (not the registered verdict): the four prompts reuse the same seeds, so their pairs share
    sampling noise. Average the q8 - fp32 differences over prompts per seed and test those per-seed means."""
    per = {}
    for k in keys:
        x, y = A[k]["stats"][stat], B[k]["stats"][stat]
        if x is not None and y is not None:
            per.setdefault(k[1], []).append(y - x)
    means = [statistics.fmean(v) for _, v in sorted(per.items())]
    return {"seeds": len(means), "median_of_seed_means": statistics.median(means) if means else None, "wilcoxon": wilcoxon(means)}


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def compare(fp32_path, q8_path, report_path, expected=None):
    expected = expected if expected is not None else [(p, s) for p in PROMPTS for s in SEEDS]
    inputs_ok, inputs = input_checks(fp32_path, q8_path, expected)
    A, B = load(fp32_path, "fp32"), load(q8_path, "q8")
    keys = sorted(set(A) & set(B))
    rep = {"issue": 1823, "rule": {"alpha_bonferroni": ALPHA, "min_abs_rank_biserial": MIN_EFFECT, "stats": STATS},
           "pairs": len(keys), "missing": sorted(set(A) ^ set(B)),
           "valid": {"fp32": sum(A[k]["valid"] for k in keys), "q8": sum(B[k]["valid"] for k in keys)},
           "stop": {"fp32": sorted({A[k]["stop"] for k in keys}), "q8": sorted({B[k]["stop"] for k in keys})},
           "stats": {}}
    for s in STATS:
        pairs = [(A[k]["stats"][s], B[k]["stats"][s]) for k in keys if A[k]["stats"][s] is not None and B[k]["stats"][s] is not None]
        rep["stats"][s] = compare_stat(pairs)
        rep["stats"][s]["per_prompt_median"] = {
            str(p): {"fp32": statistics.median([A[k]["stats"][s] for k in keys if k[0] == p and A[k]["stats"][s] is not None] or [0]),
                     "q8": statistics.median([B[k]["stats"][s] for k in keys if k[0] == p and B[k]["stats"][s] is not None] or [0])}
            for p in sorted({k[0] for k in keys})}
    div = [first_divergence(A[k]["tokens"], B[k]["tokens"]) for k in keys]
    dv = [d for d in div if d is not None]
    rep["divergence"] = {"never": sum(1 for d in div if d is None), "median_first_index": statistics.median(dv) if dv else None,
                         "min": min(dv) if dv else None, "max": max(dv) if dv else None}
    detected = [s for s in STATS if rep["stats"][s]["detected"]]
    all_valid = rep["valid"]["fp32"] == rep["valid"]["q8"] == len(keys)
    rep["inputs"] = inputs
    rep["sensitivity_seed_cluster"] = {s: seed_cluster(A, B, keys, s) for s in STATS}
    summary = ("차이 검출" if detected else "이 n에서 검출된 차이 없음") + ("" if all_valid else ", 형식 무효 실행 있음")
    rep["verdict"] = {"detected_stats": detected, "all_valid": all_valid, "inputs_ok": inputs_ok,
                      "summary": summary if inputs_ok else "판단 불가(입력 조건 미충족)"}
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep["verdict"], ensure_ascii=False))
    if not inputs_ok:
        sys.exit(2)
    return rep


def render(fp32_path, q8_path, out_dir, report_path):
    import bebop_fragment_retrieval as bf
    A, B = load(fp32_path, "fp32"), load(q8_path, "q8")
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(1823)                              # blind A/B order, key kept in the report only
    items = []
    for pi, seed in LISTEN:
        sides = [("fp32", A[(pi, seed)]), ("q8", B[(pi, seed)])]
        if rng.random() < 0.5:
            sides.reverse()
        row = {"prompt": pi, "seed": seed, "key": {}}
        for label, (prec, r) in zip("AB", sides):
            t0 = r["t0_ms"]
            rows = [{"pitch": p, "velocity": max(1, v), "start": (on - t0) / 1000, "end": (max(on + 10, off) - t0) / 1000}
                    for on, off, p, v in r["notes"] if t0 <= on < t0 + 20_000]
            name = f"p{pi}_s{seed}_{label}.mid"
            f = bf.write_midi(rows, os.path.join(out_dir, name))
            bf.render(os.path.join(out_dir, name))
            row["key"][label] = prec
            row[label] = {"audio": name[:-4] + ".m4a", "notes": f["notes"], "roundtrip_diffs": f["roundtrip_diffs"]}
        items.append(row)
    html = ["<!doctype html><html lang='ko'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>",
            "<title>정밀도 청취 비교</title><style>:root{--bg:#fff;--fg:#1d1d1f;--mu:#6e6e73;--card:#f5f5f7}"
            "@media (prefers-color-scheme:dark){:root{--bg:#111;--fg:#eee;--mu:#9a9aa0;--card:#1c1c1e}}"
            "body{background:var(--bg);color:var(--fg);font:16px/1.5 -apple-system,sans-serif;margin:0 auto;max-width:760px;padding:16px}"
            ".c{background:var(--card);border-radius:12px;padding:12px 16px;margin:12px 0}audio{width:100%}small{color:var(--mu)}"
            "textarea{width:100%;height:140px}</style></head><body>",
            "<h1>정밀도 청취 비교 (#1823)</h1><p>같은 프롬프트·같은 seed로 두 정밀도가 이어 친 20초. A·B 중 어느 쪽이 어떤 정밀도인지 숨김(무작위 순서).",
            " 같은 seed라도 첫 다른 선택 뒤에는 서로 다른 연주라 '같은 곡의 두 녹음'이 아님. 품질 주장 없음.</p>"]
    for i, it in enumerate(items, 1):
        html.append(f"<div class='c'><b>{i}. 프롬프트 {it['prompt']} · seed {it['seed']}</b>")
        for label in "AB":
            html.append(f"<p>{label} <small>({it[label]['notes']}음)</small><br><audio controls preload='none' src='{it[label]['audio']}'></audio></p>")
        html.append("</div>")
    form = "\n".join(f"{i}. 더 나은 쪽: A / B / 차이 모름 — 메모:" for i in range(1, len(items) + 1))
    html.append(f"<div class='c'><b>답 (복사해서 붙여 주세요)</b><textarea readonly>[#1823 정밀도 청취]\n{form}</textarea></div></body></html>")
    open(os.path.join(out_dir, "index.html"), "w").write("\n".join(html))
    rep = json.load(open(report_path))
    rep["listening"] = {"dir": out_dir, "items": items}
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps([{k: it[k] for k in ("prompt", "seed", "key")} for it in items]))


def main() -> None:
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "--generate":
        aria_repo, ckpt, data_path, precision, out_path = args[:5]
        sys.path.insert(0, aria_repo)
        prompts = tuple(int(x) for x in args[args.index("--prompts") + 1].split(",")) if "--prompts" in args else PROMPTS
        seeds = parse_seeds(args[args.index("--seeds") + 1]) if "--seeds" in args else SEEDS
        generate(aria_repo, ckpt, data_path, precision, out_path, prompts, seeds)
    elif mode == "--compare":
        compare(*args[:3])
    elif mode == "--render":
        render(*args[:4])
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
