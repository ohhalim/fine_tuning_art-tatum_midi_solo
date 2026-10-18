#!/usr/bin/env python3
"""Liked-lick prompts vs the realtime prompt: free generation toward the liked licks? (issue #1827,
docs/experiments/LICK_PROMPT.md). quality_claimed false: measurement and a listening set only.

  lick_prompt.py --prompts <aria repo> <clips.json> <pijama root> <ref licks report> <data json> <out prompts json>   (t3 venv)
  lick_prompt.py --generate <aria repo> <ckpt> <prompts json> <run dir> [--precision q8]               (t3 venv)
  lick_prompt.py --analyze <prompts json> <run dir> <report json>                                      (repo venv)
  lick_prompt.py --listen <prompts json> <run dir> <listen dir>                                        (repo venv)
"""
from __future__ import annotations

import json
import os
import random
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(HERE, "..")):
    sys.path.insert(0, _p)

LIKED = (("P18", "18", 1), ("P06", "06", 1), ("P16", "16", 1))   # (prompt, clip, liked lick index): first liked lick per clip
LEAD_S = 0.5                       # the prompt window starts this long before the liked lick's first onset
SEEDS = tuple(range(1, 13))
BLOCKS, CAP = 16, 64
BEAT, BLOCK = 468.75, 937.5        # 128 BPM block contract, as in the realtime producer
LIKED_SPEED = (5.77, 10.41)        # liked licks, top-line notes per second (#1809)
LIKED_MEDIAN_SPEED = 7.86
G3 = 55
WIN_S = BLOCK / 1000
PRIMARY = ("licks_per_min", "melody_notes_per_s", "liked_speed_run_share")
MIN_EFFECT = {"licks_per_min": 2.0, "melody_notes_per_s": 1.0, "liked_speed_run_share": 0.15}
ALPHA = 0.05 / 9                   # 3 liked prompts x 3 primary metrics, Bonferroni
N_RESAMPLE = 10000
LISTEN_SEEDS = (1, 2, 3)


# ---------------------------------------------------------------------------------------------- prompts
def prefix_within(token_counts, budget):
    """Largest n such that token_counts[n - 1] <= budget (token_counts[i] = tokens of the first i + 1 notes)."""
    n = 0
    for i, c in enumerate(token_counts):
        if c <= budget:
            n = i + 1
    return n


def prompt_notes(enc, ids):
    """[[pitch, velocity, start_ms, end_ms]] of a prompt, from its own tokens (copy indicator in the analysis)."""
    d = enc.tok.detokenize(enc.tok.decode(ids))
    return sorted([[x["data"]["pitch"], x["data"]["velocity"], x["data"]["start"], x["data"]["end"]] for x in d.note_msgs],
                  key=lambda r: (r[2], r[0]))


def build_prompts(aria_repo, clips_path, pijama_root, ref_report, data_path, out_path):
    import mehldau_aria_data as mad
    mad.PIJAMA = type(mad.PIJAMA)(pijama_root)
    enc = mad.Encoder(aria_repo)
    data = json.load(open(data_path))
    p0 = data["prompts"][0]
    budget = len(p0["ids"])
    clips = {c["id"]: c for c in json.load(open(clips_path))["clips"]}
    ref = json.load(open(ref_report))
    licks = {(x["clip"], x["index"]): x for x in ref["licks"]}
    out = {"issue": 1827, "token_budget": budget, "lead_s": LEAD_S,
           "prompts": [{"name": "P0", "source": p0["path"], "ids": p0["ids"], "notes": p0["prompt_notes"], "tokens": budget,
                        "rule": "data.json prompts[0]"}]}
    for name, clip, idx in LIKED:
        lk = licks[(clip, idx)]
        _, d = enc.full(clips[clip]["source"])
        start_ms = (lk["t0_s"] - LEAD_S) * 1000
        notes = sorted((x for x in d.note_msgs if x["data"]["start"] >= start_ms), key=lambda x: (x["data"]["start"], x["data"]["pitch"]))
        counts, n = [], 0
        for i in range(1, min(len(notes), 80) + 1):
            counts.append(len(enc.sub(d, notes[:i])[0]))
        n = prefix_within(counts, budget)
        toks, shifted = enc.sub(d, notes[:n])
        span = (notes[n - 1]["data"]["start"] - notes[0]["data"]["start"]) / 1000
        lick_end = lk["t0_s"] + lk["span_s"]
        out["prompts"].append({"name": name, "source": clips[clip]["source"], "clip": clip, "liked_lick": f"{clip}-{idx}",
                               "window_start_s": round(start_ms / 1000, 4), "notes": n, "tokens": len(toks), "span_s": round(span, 3),
                               "lick_fully_inside": notes[n - 1]["data"]["start"] / 1000 >= lick_end - 1e-3,
                               "pitch_min": min(x["data"]["pitch"] for x in notes[:n]), "pitch_max": max(x["data"]["pitch"] for x in notes[:n]),
                               "ids": enc.tok.encode(toks)})
    for P in out["prompts"]:
        P["prompt_notes"] = prompt_notes(enc, P["ids"])
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(json.dumps([{k: v for k, v in p.items() if k not in ("ids", "prompt_notes")} for p in out["prompts"]], ensure_ascii=False, indent=1))


# ------------------------------------------------------------------------------------------- generation
def generate(aria_repo, ckpt, prompts_path, run_dir, precision="q8"):
    import math
    import time
    import torch
    sys.path.insert(0, aria_repo)
    from aria_grammar_mask import GrammarMask, masked_pick
    from aria_token_validity import check as token_check
    from lh_kv_reuse import load_mlx
    from stream_contract import SEGMENT_MS, TokenStream
    os.makedirs(run_dir, exist_ok=True)
    prompts = json.load(open(prompts_path))["prompts"]
    tok, run = load_mlx(aria_repo, ckpt, precision)
    V_SIZE = tok.vocab_size
    vocab = [tok.decode([i])[0] for i in range(V_SIZE)]
    to_id = {t: i for i, t in enumerate(vocab)}
    gm = GrammarMask(vocab)
    no_end = torch.tensor([t not in ("<D>", "<E>") for t in vocab])
    for P in prompts:
        prompt = tok.decode(P["ids"])
        for seed in SEEDS:
            path = os.path.join(run_dir, f"{P['name']}_s{seed}.json")
            if os.path.exists(path):
                continue
            stream = TokenStream(prompt)
            m = {"seq": list(prompt), "V": 0, "last": None}
            gen = torch.Generator().manual_seed(seed)

            def coverage():
                st = stream.grammar_state()
                return st.segment * SEGMENT_MS + st.last_onset if st.expect == "note" else -1

            def sample_one():
                if m["V"] < len(m["seq"]):
                    m["last"] = run([to_id[t] for t in m["seq"][m["V"]:]], m["V"])[-1]
                    m["V"] = len(m["seq"])
                p = torch.softmax(torch.from_numpy(m["last"]), -1)
                st = stream.grammar_state()
                gm.sync(st, stream.onset_floor(st))
                ok = gm.allowed() & no_end
                a = p[ok].sum().item()
                if not (math.isfinite(a) and a > 0):
                    raise RuntimeError(f"allowed mass {a}")
                pick, _ = masked_pick(p, ok, torch.empty(V_SIZE).exponential_(1, generator=gen))
                t = vocab[pick]
                stream.append_sampled(t)
                m["seq"].append(t)

            t0 = coverage()
            committed, samples, underfill = [], [], None
            tc = time.perf_counter()
            for k in range(BLOCKS):
                h = t0 + (k + 1) * BLOCK + BEAT
                n = 0
                while coverage() < h and n < CAP:
                    sample_one()
                    n += 1
                samples.append(n)
                if coverage() < h:
                    underfill = k
                    break
                committed += stream.commit_until(h)
            v = token_check(prompt, stream.tokens[len(prompt):stream.committed_end])
            rec = {"prompt": P["name"], "seed": seed, "precision": precision, "t0_ms": t0,
                   "blocks_done": BLOCKS if underfill is None else underfill, "underfill_block": underfill,
                   "samples_per_block": samples, "token_valid": bool(v["valid"] and v["tail"] is None), "gen_s": round(time.perf_counter() - tc, 2),
                   "notes": [[n.pitch, n.velocity, n.on_ms, n.off_ms] for n in committed]}
            with open(path, "w") as f:
                json.dump(rec, f)
            print(P["name"], seed, "notes", len(committed), "underfill", underfill, "valid", rec["token_valid"], rec["gen_s"], "s", flush=True)


# --------------------------------------------------------------------------------------------- analysis
def to_pm(notes, t0_ms):
    import pretty_midi
    return [(f"g{i}", pretty_midi.Note(velocity=max(1, v), pitch=p, start=(on - t0_ms) / 1000, end=max(off, on + 10) / 1000 - t0_ms / 1000))
            for i, (p, v, on, off) in enumerate(sorted(notes, key=lambda x: (x[2], x[0])))]


def window_max(onsets, win):
    """Most onsets inside any half-open window of length win."""
    s = sorted(onsets)
    best, j = 0, 0
    for i in range(len(s)):
        while s[i] - s[j] >= win:
            j += 1
        best = max(best, i - j + 1)
    return best


def max_shared_ngram(a, b):
    """Longest common contiguous run between two interval sequences (copy indicator)."""
    best = 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
                best = max(best, cur[j])
        prev = cur
    return best


def run_metrics(notes, t0_ms, span_s, prompt_intervals=()):
    import bebop_fragment_retrieval as bf
    from inference.control.lick_bank import lick_from_run, runs
    pm = to_pm(notes, t0_ms)
    line, _, _, _ = bf.split_song(pm)
    mel = [t for _, t in line]
    rs = runs(mel)
    long_runs = [r for r in rs if len(r) >= 8]
    licks = [r for r in long_runs if lick_from_run(r) is not None]

    def speed(r):
        span = r[-1].start - r[0].start
        return (len(r) - 1) / span if span > 0 else None
    sp = [s for s in (speed(r) for r in long_runs) if s is not None]
    iois = [b.start - a.start for a, b in zip(mel, mel[1:])]
    intervals = [b.pitch - a.pitch for a, b in zip(mel, mel[1:])]
    return {"notes": len(pm), "melody_notes": len(mel), "runs_8plus": len(long_runs), "licks": len(licks),
            "licks_per_min": len(licks) / span_s * 60, "melody_notes_per_s": len(mel) / span_s, "all_notes_per_s": len(pm) / span_s,
            "liked_speed_run_share": (sum(1 for s in sp if LIKED_SPEED[0] <= s <= LIKED_SPEED[1]) / len(sp)) if sp else None,
            "lick_speeds": [round(speed(r), 2) for r in licks if speed(r)],
            "register_share_g3": (sum(1 for _, n in pm if n.pitch >= G3) / len(pm)) if pm else None,
            "melody_median_ioi_s": statistics.median(iois) if iois else None,
            "melody_window_max": window_max([t.start for t in mel], WIN_S),
            "copy_ngram_with_prompt": max_shared_ngram(intervals, list(prompt_intervals))}


def median_diff_stats(a, b, n=N_RESAMPLE, seed=0):
    """Median(b) - median(a): bootstrap 95% CI and a two-sided permutation p (difference of medians)."""
    rng = random.Random(seed)
    obs = statistics.median(b) - statistics.median(a)
    boots = sorted(statistics.median(rng.choices(b, k=len(b))) - statistics.median(rng.choices(a, k=len(a))) for _ in range(n))
    pooled, na = list(a) + list(b), len(a)
    hits = 0
    for _ in range(n):
        rng.shuffle(pooled)
        d = statistics.median(pooled[na:]) - statistics.median(pooled[:na])
        hits += abs(d) >= abs(obs) - 1e-12
    return {"diff": obs, "ci95": [boots[int(0.025 * n)], boots[int(0.975 * n) - 1]], "p": (hits + 1) / (n + 1)}


def verdict(tests):
    """Toward the liked licks: licks/min or liked-speed share up, significant at ALPHA and at least MIN_EFFECT."""
    def sig(m, sign):
        t = tests.get(m)
        return t is not None and t["p"] <= ALPHA and sign * t["diff"] >= MIN_EFFECT[m]
    closer = sig("licks_per_min", 1) or sig("liked_speed_run_share", 1)
    farther = sig("licks_per_min", -1) or sig("liked_speed_run_share", -1)
    return "섞임" if closer and farther else "가까워짐" if closer else "멀어짐" if farther else "차이 미검출"


def melody_intervals(notes):
    """Top-line (G3+) intervals of [[pitch, velocity, start_ms, end_ms]] notes."""
    import bebop_fragment_retrieval as bf
    line, _, _, _ = bf.split_song(to_pm(notes, 0))
    mel = [t for _, t in line]
    return [b.pitch - a.pitch for a, b in zip(mel, mel[1:])]


def analyze(prompts_path, run_dir, report_path):
    prompts = json.load(open(prompts_path))
    rows = {}
    for P in prompts["prompts"]:
        rows[P["name"]] = []
        pint = melody_intervals(P["prompt_notes"])
        for seed in SEEDS:
            r = json.load(open(os.path.join(run_dir, f"{P['name']}_s{seed}.json")))
            span = (r["blocks_done"] * BLOCK + BEAT) / 1000 if r["blocks_done"] else 0.0
            met = run_metrics(r["notes"], r["t0_ms"], span, pint) if span > 0 else None
            rows[P["name"]].append({"seed": seed, "underfill_block": r["underfill_block"], "token_valid": r["token_valid"],
                                    "span_s": span, "metrics": met})
    rep = {"issue": 1827, "alpha": ALPHA, "min_effect": MIN_EFFECT, "runs": rows, "summary": {}, "tests": {}, "verdict": {}}
    keys = ("licks_per_min", "melody_notes_per_s", "liked_speed_run_share", "all_notes_per_s", "register_share_g3",
            "melody_median_ioi_s", "melody_window_max", "copy_ngram_with_prompt")
    for name, rs in rows.items():
        ms = [x["metrics"] for x in rs if x["metrics"]]
        rep["summary"][name] = {"runs": len(rs), "underfill": sum(1 for x in rs if x["underfill_block"] is not None),
                                "token_invalid": sum(1 for x in rs if not x["token_valid"]),
                                "licks_total": sum(m["licks"] for m in ms),
                                **{k: {"median": statistics.median(v), "min": min(v), "max": max(v)} if v else None
                                   for k, v in ((k, [m[k] for m in ms if m[k] is not None]) for k in keys)}}
    base = rows["P0"]
    for name in [p["name"] for p in prompts["prompts"] if p["name"] != "P0"]:
        rep["tests"][name] = {}
        for m in PRIMARY:
            a = [x["metrics"][m] for x in base if x["metrics"] and x["metrics"][m] is not None]
            b = [x["metrics"][m] for x in rows[name] if x["metrics"] and x["metrics"][m] is not None]
            rep["tests"][name][m] = median_diff_stats(a, b) if len(a) >= 3 and len(b) >= 3 else None
        rep["verdict"][name] = verdict(rep["tests"][name])
    with open(report_path, "w") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps({"summary": {n: {k: (v["median"] if isinstance(v, dict) else v) for k, v in s.items()} for n, s in rep["summary"].items()},
                      "verdict": rep["verdict"]}, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------------------------------------- listening
def listen(prompts_path, run_dir, out_dir):
    import bebop_fragment_retrieval as bf
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "key"), exist_ok=True)
    prompts = json.load(open(prompts_path))["prompts"]
    items = []
    for P in prompts:
        for seed in LISTEN_SEEDS:
            r = json.load(open(os.path.join(run_dir, f"{P['name']}_s{seed}.json")))
            items.append((P["name"], seed, r))
    order = list(range(len(items)))
    random.Random(1827).shuffle(order)
    key = []
    for pos, i in enumerate(order, 1):
        name, seed, r = items[i]
        rows = [{"pitch": p, "velocity": max(1, v), "start": (on - r["t0_ms"]) / 1000, "end": max(off, on + 10) / 1000 - r["t0_ms"] / 1000}
                for p, v, on, off in r["notes"]]
        mid = os.path.join(out_dir, f"x{pos:02d}.mid")
        bf.write_midi(rows, mid)
        bf.render(mid)
        key.append({"excerpt": f"x{pos:02d}", "prompt": name, "seed": seed})
    with open(os.path.join(out_dir, "key", "key.json"), "w") as f:
        json.dump(key, f, indent=1)
    cards = "\n".join(f"""<section class="card"><h3>x{k:02d}</h3><audio controls preload="none" src="x{k:02d}.m4a"></audio>
<div class="opts" data-id="x{k:02d}"><label><input type="radio" name="x{k:02d}" value="원하는 릭에 가까움">원하는 릭에 가까움</label>
<label><input type="radio" name="x{k:02d}" value="아님">아님</label><label><input type="radio" name="x{k:02d}" value="모름">모름</label>
<input type="text" class="note" placeholder="메모(선택)"></div></section>""" for k in range(1, len(items) + 1))
    html = f"""<!doctype html><html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>릭 프롬프트 청취</title><style>
:root{{--bg:#fafaf8;--fg:#1d1d1b;--muted:#6b6b66;--card:#fff;--line:#e3e3de;--accent:#2f5d8a}}
@media (prefers-color-scheme:dark){{:root{{--bg:#161615;--fg:#ecece8;--muted:#a3a39d;--card:#20201e;--line:#34342f;--accent:#8fb6dd}}}}
body{{background:var(--bg);color:var(--fg);font:16px/1.55 -apple-system,system-ui,sans-serif;margin:0;padding:16px;max-width:760px;margin-inline:auto}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:12px 0}}
h1{{font-size:1.3rem}} h3{{margin:0 0 6px}} audio{{width:100%}} .opts label{{display:inline-block;margin:6px 12px 6px 0}}
.note{{width:100%;box-sizing:border-box;margin-top:6px;padding:6px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}}
.muted{{color:var(--muted)}} button{{background:var(--accent);color:#fff;border:0;border-radius:8px;padding:10px 16px;font-size:1rem}}
textarea{{width:100%;box-sizing:border-box;height:160px;margin-top:8px;background:transparent;color:var(--fg);border:1px solid var(--line);border-radius:6px}}
</style></head><body>
<h1>릭 프롬프트 청취 (#1827)</h1>
<p>모델이 이어서 만든 오른손·왼손 연주 {len(items)}개(각 약 15초). 프롬프트 4종 × seed 3개를 섞은 순서이고, 어느 프롬프트인지는 숨김. 원하는 릭(#1809, 클립 18 Barry Harris 같은 라인)에 가까운지 골라 주세요.</p>
<p class="muted">품질 주장 아님. 고른 결과는 아래 버튼으로 복사해 붙여 주시면 됩니다.</p>
{cards}
<section class="card"><button id="copy">답 복사</button><textarea id="out" readonly></textarea></section>
<script>
document.getElementById('copy').onclick=function(){{var lines=['[릭 프롬프트 청취 #1827]'];
document.querySelectorAll('.opts').forEach(function(d){{var id=d.dataset.id;var r=d.querySelector('input[type=radio]:checked');var n=d.querySelector('.note').value.trim();
lines.push(id+': '+(r?r.value:'-')+(n?' / '+n:''));}});var t=lines.join('\\n');var o=document.getElementById('out');o.value=t;
try{{navigator.clipboard.writeText(t);}}catch(e){{}}o.select();}};
</script></body></html>"""
    with open(os.path.join(out_dir, "index.html"), "w") as f:
        f.write(html)
    print("listening page", os.path.join(out_dir, "index.html"), len(items), "excerpts")


def main() -> None:
    mode, args = sys.argv[1], sys.argv[2:]
    if mode == "--prompts":
        build_prompts(*args[:6])
    elif mode == "--generate":
        prec = args[args.index("--precision") + 1] if "--precision" in args else "q8"
        generate(*args[:4], precision=prec)
    elif mode == "--analyze":
        analyze(*args[:3])
    elif mode == "--listen":
        listen(*args[:3])
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
