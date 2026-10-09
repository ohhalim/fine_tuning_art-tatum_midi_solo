#!/usr/bin/env python3
"""Regeneration of the 16 WJazzD pilot free generations with token ids kept (docs/experiments/WJAZZD_PILOT.md).

The pilot run stored only aggregates. This script reloads the saved pilot adapter (no training),
repeats the generation loop of wjazzd_pilot_run.py with the same inputs and RNG, saves prefix and
generated token ids locally, and checks that the aggregates match the original run. It adds one
post-hoc information metric: fit to the plan given to the model (for the wrong plan, the chord
tones moved up a semitone), next to the original-progression fit the run reported.
Run with the Aria venv: wjazzd_pilot_regen.py <aria repo> <checkpoint> <wjazzd.db> <pilot dir>
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract_v2 import features  # noqa: E402
from aria_cond_loss_arms import grammar_check  # noqa: E402
from aria_cond_smoke import file_sha  # noqa: E402
from wjazzd_pilot_data import chroma, plan_for, row  # noqa: E402
from wjazzd_pilot_run import DIM, GEN_SEEDS, GEN_SONGS, GEN_TOKENS, wrong_rows  # noqa: E402


def main() -> None:
    aria_repo, ckpt, db, pilot = sys.argv[1:5]
    sys.path.insert(0, aria_repo)
    import torch
    from safetensors.torch import load_file
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM

    dev = torch.device("mps")
    with open(os.path.join(pilot, "data.json")) as f:
        data = json.load(f)
    with open(os.path.join(pilot, "result.json")) as f:
        original = json.load(f)
    adapter_path = os.path.join(pilot, "adapter_wjazzd_pilot_diagnostic_only.pt")
    adapter_sha = file_sha(adapter_path)
    assert adapter_sha == original["adapter_sha256"], "adapter file differs from the run"
    tok = AbsTokenizer()
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_state_dict(load_file(ckpt), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(dev)
    model.model.freqs_cis = None
    adapter = torch.nn.Linear(DIM, cfg.d_model, bias=False).to(dev)
    adapter.weight.data.copy_(torch.load(adapter_path)["weight"].to(dev))
    adapter.weight.requires_grad_(False)
    state = {"cond": None, "adapter": None}

    def pre_hook(_mod, args, kwargs):
        if state["adapter"] is None:
            return None
        return (args[0] + state["adapter"](state["cond"]),) + args[1:], kwargs

    model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)

    def run(ids, rows=None, ad=None):
        state["adapter"] = ad
        state["cond"] = None if rows is None else torch.tensor([rows], dtype=torch.float32, device=dev)
        return model(ids)

    vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
    con = sqlite3.connect(db)
    test = [ch for ch in data["chunks"] if ch["split"] == "test"]
    songs = []
    for ch in test:
        if ch["melid"] not in songs:
            songs.append(ch["melid"])
    samples, compare = [], []
    for m in songs[:GEN_SONGS]:
        ch = next(c for c in test if c["melid"] == m)
        plan = plan_for(con, m, ch["t0"])
        for seed in GEN_SEEDS:
            for name in ("correct", "wrong"):
                g_ = torch.Generator().manual_seed(seed)
                prefix_ids = ch["token_ids"][: ch["first_target"]]
                ids = torch.tensor([prefix_ids], device=dev)
                toks = [vocab[i] for i in prefix_ids]
                new, new_ids = [], []
                for _ in range(GEN_TOKENS):
                    rows = [row(f) for f in features(toks, plan)]
                    rows = wrong_rows(rows) if name == "wrong" else rows
                    with torch.no_grad():
                        lg = run(ids, rows, adapter)[0, -1].float().cpu()
                    nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=g_).item()
                    t = vocab[nxt]
                    new.append(t)
                    new_ids.append(nxt)
                    toks.append(t)
                    ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
                    if t == "<E>":
                        break
                feats = features(toks, plan)
                pitches, fit_orig, fit_given = [], [], []
                base_len = ch["first_target"]
                for i in range(base_len, len(toks) - 1):
                    if isinstance(toks[i], tuple) and toks[i][0] == "piano" and isinstance(toks[i + 1], tuple) and toks[i + 1][0] == "onset":
                        t_ms = feats[i + 1]["t_ms"]
                        seg = next((p_ for p_ in plan if p_[0] <= t_ms < p_[1]), None)
                        pc = toks[i][1] % 12
                        pitches.append(toks[i][1])
                        if seg is not None:
                            cr = chroma(seg[2])
                            fit_orig.append(cr[pc] == 1.0)
                            given = cr if name == "correct" else cr[11:12] + cr[0:11]     # same +1 semitone shift as the input
                            fit_given.append(given[pc] == 1.0)
                longest = 0
                for per in range(1, 9):
                    for s0 in range(len(pitches)):
                        k = 0
                        while s0 + per + k < len(pitches) and pitches[s0 + k] == pitches[s0 + per + k]:
                            k += 1
                        longest = max(longest, k + per if k >= per else 0)
                agg = {"melid": m, "seed": seed, "plan": name, "tokens": len(new), "eos": new[-1] == "<E>", "notes": len(pitches),
                       "grammar_errors": len(grammar_check(toks[:base_len], new)), "longest_repeat_notes": longest,
                       "chord_tone_ratio_descriptive": (sum(fit_orig) / len(fit_orig)) if fit_orig else None}
                orig = next(g for g in original["generation"] if g["melid"] == m and g["seed"] == seed and g["plan"] == name)
                compare.append(all(orig[k] == agg[k] for k in agg))
                samples.append({**agg, "original_progression_fit": agg["chord_tone_ratio_descriptive"],
                                "given_plan_fit_posthoc": (sum(fit_given) / len(fit_given)) if fit_given else None,
                                "condition_shift_semitones": 0 if name == "correct" else 1, "chunk_t0": ch["t0"],
                                "prefix_ids": prefix_ids, "generated_ids": new_ids,
                                "grammar_error_detail": grammar_check(toks[:base_len], new)})
    out = {"note": "regenerated with the saved pilot adapter, same inputs and RNG; not the original process",
           "adapter_sha256": adapter_sha, "matches_original_aggregates": sum(compare), "of": len(compare), "samples": samples}
    with open(os.path.join(pilot, "generation_regen.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print(json.dumps({"matches": sum(compare), "of": len(compare)}))


if __name__ == "__main__":
    main()
