#!/usr/bin/env python3
"""Injection-position probe at fixed prefixes of the WJazzD pilot (docs/experiments/WJAZZD_PILOT.md).

No training, no sampling, no new generation. For 15 fixed prefixes the next-token distribution is
read under three arms:
  A  the pilot adapter at every input position (as in the run)
  B  adapter off
  C  adapter at every position except the last input position (its condition row set to zero)
Prefixes (rules fixed before the run):
  - failure: the 7 adapter samples with an internal error, cut right before the first error token,
    with the plan they were generated under (correct or wrong)
  - control: the 8 base samples, cut right after the pitch token of the 16th complete generated note
    (the last complete note if there are fewer than 16), correct plan only
Allowed onset at that position: onset >= the last onset of the current 5 s segment (same onset
allowed, <T> resets). Run with the Aria venv:
wjazzd_pilot_inject_probe.py <aria repo> <checkpoint> <wjazzd.db> <pilot dir> <summary json>
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract_v2 import features  # noqa: E402
from aria_token_validity import segment_state  # noqa: E402
from wjazzd_pilot_data import plan_for, row  # noqa: E402
from wjazzd_pilot_run import DIM, wrong_rows  # noqa: E402

CONTROL_NOTE = 16


def ttype(t):
    if isinstance(t, tuple):
        if t[0] in ("piano", "onset", "dur"):
            return t[0]
        if t[0] == "prefix":
            return "special"
        return "other_instrument"
    return {"<D>": "<D>", "<T>": "<T>", "<E>": "<E>"}.get(t, "special")


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
    load = lambda name: json.load(open(os.path.join(pilot, name)))
    regen, base = load("generation_regen.json"), load("generation_base.json")
    validity = json.load(open(os.path.join(HERE, "..", "docs", "experiments", "wjazzd_pilot", "generation_validity.json")))
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
    adapter.weight.data.copy_(torch.load(os.path.join(pilot, "adapter_wjazzd_pilot_diagnostic_only.pt"))["weight"].to(dev))
    state = {"cond": None, "on": False}

    def pre_hook(_mod, args, kwargs):
        if not state["on"]:
            return None
        return (args[0] + adapter(state["cond"]),) + args[1:], kwargs

    model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)
    con = sqlite3.connect(db)
    types = sorted({ttype(t) for t in vocab})
    idx_type = {k: torch.tensor([i for i, t in enumerate(vocab) if ttype(t) == k], device=dev) for k in types}
    onset_ids = [i for i, t in enumerate(vocab) if isinstance(t, tuple) and t[0] == "onset"]

    cases = []
    for s in regen["samples"]:
        v = next(r for r in validity["rows"] if r["melid"] == s["melid"] and r["seed"] == s["seed"] and r["plan"] == s["plan"])
        if v["valid"]:
            continue
        k = v["first_error"]["index"]
        cases.append({"kind": "failure", "melid": s["melid"], "seed": s["seed"], "plan": s["plan"], "t0": s["chunk_t0"],
                      "ids": s["prefix_ids"] + s["generated_ids"][:k], "next_actual": s["generated_ids"][k]})
    for s in base["samples"]:
        new = [vocab[i] for i in s["generated_ids"]]
        pitch_pos, complete = [], 0
        for i in range(len(new) - 2):
            if isinstance(new[i], tuple) and new[i][0] == "piano" and isinstance(new[i + 1], tuple) and new[i + 1][0] == "onset" \
                    and isinstance(new[i + 2], tuple) and new[i + 2][0] == "dur":
                complete += 1
                pitch_pos.append(i)
        cut = pitch_pos[CONTROL_NOTE - 1] if len(pitch_pos) >= CONTROL_NOTE else pitch_pos[-1]
        cases.append({"kind": "control", "melid": s["melid"], "seed": s["seed"], "plan": "correct", "t0": s["chunk_t0"],
                      "ids": s["prefix_ids"] + s["generated_ids"][: cut + 1], "next_actual": s["generated_ids"][cut + 1],
                      "control_note": min(CONTROL_NOTE, len(pitch_pos))})

    rows_out = []
    for c in cases:
        toks = [vocab[i] for i in c["ids"]]
        assert ttype(toks[-1]) == "piano", "prefix must end on a pitch token"
        plan = plan_for(con, c["melid"], c["t0"])
        rows = [row(f) for f in features(toks, plan)]
        rows = wrong_rows(rows) if c["plan"] == "wrong" else rows
        last, expect = segment_state(toks)
        allowed = torch.tensor([i for i in onset_ids if last <= vocab[i][1]], device=dev)
        ids = torch.tensor([c["ids"]], device=dev)
        arms = {}
        for arm in "ABC":
            r = [list(x) for x in rows]
            if arm == "C":
                r[-1] = [0.0] * DIM
            state["on"] = arm != "B"
            state["cond"] = torch.tensor([r], dtype=torch.float32, device=dev)
            with torch.no_grad():
                prob = torch.softmax(model(ids)[0, -1].float(), -1)
            actual = c["next_actual"]
            rank = int((prob > prob[actual]).sum().item()) + 1
            arms[arm] = {"onset_mass": prob[idx_type["onset"]].sum().item(),
                         "allowed_onset_mass": prob[allowed].sum().item(),
                         "actual_next_token": str(vocab[actual]), "actual_next_prob": prob[actual].item(), "actual_next_rank": rank,
                         "type_mass": {k: prob[v].sum().item() for k, v in idx_type.items()}}
        rows_out.append({k: c[k] for k in ("kind", "melid", "seed", "plan")} | {"prefix_len": len(c["ids"]),
                        "segment_last_onset": last, "expect": expect, "arms": arms, "control_note": c.get("control_note")})
        a, b, cc = arms["A"]["allowed_onset_mass"], arms["B"]["allowed_onset_mass"], arms["C"]["allowed_onset_mass"]
        print(c["kind"], c["melid"], c["seed"], c["plan"], "allowed onset A %.3f B %.3f C %.3f" % (a, b, cc),
              "| actual", arms["A"]["actual_next_token"], "p A %.3g B %.3g C %.3g" % (arms["A"]["actual_next_prob"], arms["B"]["actual_next_prob"], arms["C"]["actual_next_prob"]),
              "rank", arms["A"]["actual_next_rank"], arms["B"]["actual_next_rank"], arms["C"]["actual_next_rank"])
    with open(summary_out, "w") as f:
        json.dump({"control_rule": f"pitch token of the {CONTROL_NOTE}th complete generated note (last complete note if fewer)",
                   "allowed_onset_rule": "onset >= last onset of the current 5 s segment (same onset allowed, <T> resets)",
                   "cases": rows_out}, f, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
