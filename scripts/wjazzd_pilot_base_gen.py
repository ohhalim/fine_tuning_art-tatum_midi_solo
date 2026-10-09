#!/usr/bin/env python3
"""Base (no adapter) generations for the WJazzD pilot and validity breakdown of all 24 samples
(docs/experiments/WJAZZD_PILOT.md).

The 8 base samples use the same prompts, seeds, temperature 1, min_p 0, 96 new tokens and <E> stop
as the 16 adapter samples (kept in generation_regen.json). No adapter is attached at all. Every
sample is then checked with aria_token_validity.check; raw tokens are not changed.
Run with the Aria venv: wjazzd_pilot_base_gen.py <aria repo> <checkpoint> <wjazzd.db> <pilot dir> <summary json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract import prefix_times  # noqa: E402
from aria_cond_contract_v2 import features  # noqa: E402
from aria_token_validity import check  # noqa: E402
from wjazzd_pilot_data import chroma, plan_for  # noqa: E402
from wjazzd_pilot_run import GEN_SEEDS, GEN_SONGS, GEN_TOKENS  # noqa: E402


def sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def longest_repeat(pitches):
    best = 0
    for per in range(1, 9):
        for s0 in range(len(pitches)):
            k = 0
            while s0 + per + k < len(pitches) and pitches[s0 + k] == pitches[s0 + per + k]:
                k += 1
            best = max(best, k + per if k >= per else 0)
    return best


def main() -> None:
    aria_repo, ckpt, db, pilot, summary_out = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    import torch
    from safetensors.torch import load_file
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM

    dev = torch.device("mps")
    tok = AbsTokenizer()
    vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
    with open(os.path.join(pilot, "data.json")) as f:
        data = json.load(f)
    regen_path = os.path.join(pilot, "generation_regen.json")
    with open(regen_path) as f:
        regen = json.load(f)
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_state_dict(load_file(ckpt), strict=True)
    model.eval()
    model.to(dev)
    model.model.freqs_cis = None
    con = sqlite3.connect(db)
    test = [ch for ch in data["chunks"] if ch["split"] == "test"]
    songs = []
    for ch in test:
        if ch["melid"] not in songs:
            songs.append(ch["melid"])
    base = []
    for m in songs[:GEN_SONGS]:
        ch = next(c for c in test if c["melid"] == m)
        for seed in GEN_SEEDS:
            g_ = torch.Generator().manual_seed(seed)
            prefix_ids = ch["token_ids"][: ch["first_target"]]
            ids = torch.tensor([prefix_ids], device=dev)
            new_ids = []
            for _ in range(GEN_TOKENS):
                with torch.no_grad():
                    lg = model(ids)[0, -1].float().cpu()
                nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=g_).item()
                new_ids.append(nxt)
                ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
                if vocab[nxt] == "<E>":
                    break
            base.append({"melid": m, "seed": seed, "plan": "base", "condition_shift_semitones": None,
                         "chunk_t0": ch["t0"], "prefix_ids": prefix_ids, "generated_ids": new_ids})
    base_path = os.path.join(pilot, "generation_base.json")
    with open(base_path, "w") as f:
        json.dump({"note": "base model, no adapter", "samples": base}, f)

    rows = []
    for s in base + regen["samples"]:
        prefix = [vocab[i] for i in s["prefix_ids"]]
        new = [vocab[i] for i in s["generated_ids"]]
        v = check(prefix, new)
        plan = plan_for(con, s["melid"], s["chunk_t0"])
        toks = prefix + new
        feats = features(toks, plan)
        times = prefix_times(toks)
        stop = len(prefix) + (v["first_error"]["index"] if v["first_error"] else len(new))
        pitches, fit_orig, fit_input = [], [], []
        for i in range(len(prefix), min(stop, len(toks) - 1)):
            if isinstance(toks[i], tuple) and toks[i][0] == "piano" and isinstance(toks[i + 1], tuple) and toks[i + 1][0] == "onset":
                t_ms = feats[i + 1]["t_ms"]
                seg = next((p_ for p_ in plan if p_[0] <= t_ms < p_[1]), None)
                pitches.append(toks[i][1])
                if seg is not None:
                    cr = chroma(seg[2])
                    pc = toks[i][1] % 12
                    fit_orig.append(cr[pc] == 1.0)
                    if s["plan"] != "base":
                        given = cr if s["plan"] == "correct" else cr[11:12] + cr[0:11]
                        fit_input.append(given[pc] == 1.0)
        rows.append({"melid": s["melid"], "seed": s["seed"], "plan": s["plan"], "new_tokens": len(new), **v,
                     "music_time_s": round((times[-1] - times[len(prefix) - 1]) / 1000.0, 3),
                     "longest_repeat_notes_valid_part": longest_repeat(pitches),
                     "original_plan_fit_valid_part": (sum(fit_orig) / len(fit_orig)) if fit_orig else None,
                     "input_plan_fit_valid_part": (sum(fit_input) / len(fit_input)) if fit_input else None,
                     "notes_in_fit": len(fit_orig)})
    commit = subprocess.run(["git", "-C", os.path.join(HERE, ".."), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    summary = {"files": {"generation_regen.json": sha(regen_path), "generation_base.json": sha(base_path)},
               "code_commit_at_run": commit, "rows": rows}
    with open(summary_out, "w") as f:
        json.dump(summary, f, indent=1, ensure_ascii=False)
    for r in rows:
        print(r["melid"], r["seed"], r["plan"].ljust(7), "valid" if r["valid"] else f"first {r['first_error']['type']}@{r['first_error']['index']}",
              "notes_before", r["notes_before_first_error"], "cascade", r["cascade_errors"], "eos", r["ended_by_eos"], "tail", r["tail"],
              "time", r["music_time_s"], "fit", r["original_plan_fit_valid_part"], r["input_plan_fit_valid_part"])


if __name__ == "__main__":
    main()
