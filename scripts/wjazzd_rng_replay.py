#!/usr/bin/env python3
"""Exact replay of the three seed-2 generations that emitted ('organ', 96, 120) at step 38
(docs/experiments/WJAZZD_PILOT.md). Same prompt, same plan, same adapter, same path as the run:
MPS forward -> .float().cpu() -> CPU float32 softmax -> torch.multinomial(generator). Each step keeps
the probability checksum, shape, dtype and the generator state before the draw; the drawn id must
equal the stored token. At step 38 the state before the draw is restored and the draw repeated;
the A/B/C distributions at that prefix (A as generated, B adapter off, C last position off) are
checked for finiteness, sum, zero/negative entries and the vocab id of the drawn token.
Run with the Aria venv: wjazzd_rng_replay.py <aria repo> <checkpoint> <wjazzd.db> <pilot dir> <out json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract_v2 import features  # noqa: E402
from wjazzd_pilot_data import plan_for, row  # noqa: E402
from wjazzd_pilot_run import DIM, wrong_rows  # noqa: E402

CASES = [(12, 2, "wrong"), (92, 2, "wrong"), (292, 2, "wrong")]
STEP = 38


def h(t):
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()[:16]


def main() -> None:
    aria_repo, ckpt, db, pilot, out = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    import torch
    from safetensors.torch import load_file
    from ariautils.tokenizer import AbsTokenizer
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM

    dev = torch.device("mps")
    tok = AbsTokenizer()
    vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
    regen = json.load(open(os.path.join(pilot, "generation_regen.json")))
    probe = json.load(open(os.path.join(HERE, "..", "docs", "experiments", "wjazzd_pilot", "inject_probe.json")))
    cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    model.load_state_dict(load_file(ckpt), strict=True)
    model.eval()
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

    def logits(ids, rows, on):
        state["on"], state["cond"] = on, torch.tensor([rows], dtype=torch.float32, device=dev)
        with torch.no_grad():
            return model(ids)[0, -1].float().cpu()

    con = sqlite3.connect(db)
    report = {"torch": torch.__version__, "device_path": "MPS forward, logits .float().cpu(), CPU float32 softmax, torch.multinomial on CPU generator",
              "cases": []}
    for melid, seed, plan_name in CASES:
        s = next(x for x in regen["samples"] if x["melid"] == melid and x["seed"] == seed and x["plan"] == plan_name)
        plan = plan_for(con, melid, s["chunk_t0"])
        g = torch.Generator().manual_seed(seed)
        ids = torch.tensor([s["prefix_ids"]], device=dev)
        toks = [vocab[i] for i in s["prefix_ids"]]
        steps, match = [], True
        for k in range(STEP + 1):
            rows = [row(f) for f in features(toks, plan)]
            rows = wrong_rows(rows) if plan_name == "wrong" else rows
            prob = torch.softmax(logits(ids, rows, True), -1)
            gstate = g.get_state().clone()
            if k == STEP:
                state_before = gstate
                prob_a = prob
            nxt = torch.multinomial(prob, 1, generator=g).item()
            ok = nxt == s["generated_ids"][k]
            match = match and ok
            steps.append({"step": k, "prob_sha16": h(prob), "shape": list(prob.shape), "dtype": str(prob.dtype),
                          "sum": float(prob.sum()), "gen_state_sha16": h(gstate), "drawn": nxt, "stored": s["generated_ids"][k], "match": ok})
            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
            toks.append(vocab[nxt])
        g2 = torch.Generator()
        g2.set_state(state_before)
        redraw = torch.multinomial(prob_a, 1, generator=g2).item()
        # A/B/C at the step-38 prefix (the prefix before the step-38 token)
        pre_ids = torch.tensor([s["prefix_ids"] + s["generated_ids"][:STEP]], device=dev)
        pre_toks = [vocab[i] for i in s["prefix_ids"] + s["generated_ids"][:STEP]]
        rows = [row(f) for f in features(pre_toks, plan)]
        rows = wrong_rows(rows) if plan_name == "wrong" else rows
        arms = {}
        for arm in "ABC":
            r = [list(x) for x in rows]
            if arm == "C":
                r[-1] = [0.0] * DIM
            p = torch.softmax(logits(pre_ids, r, arm != "B"), -1)
            tid = s["generated_ids"][STEP]
            arms[arm] = {"finite": bool(torch.isfinite(p).all()), "sum": float(p.sum()), "negatives": int((p < 0).sum()),
                         "zeros": int((p == 0).sum()), "p_token": float(p[tid]), "token": str(vocab[tid])}
        probe_case = next(c for c in probe["cases"] if c["kind"] == "failure" and c["melid"] == melid and c["seed"] == seed and c["plan"] == plan_name)
        report["cases"].append({"melid": melid, "seed": seed, "plan": plan_name, "replay_all_steps_match": match,
                                "redraw_from_restored_state": redraw, "redraw_equals_stored": redraw == s["generated_ids"][STEP],
                                "token_at_step": str(vocab[s["generated_ids"][STEP]]), "token_id": s["generated_ids"][STEP],
                                "cpu_softmax_p_token_A": arms["A"]["p_token"],
                                "probe_mps_softmax_p_token_A": probe_case["arms"]["A"]["actual_next_prob"],
                                "arms_cpu_softmax": arms, "steps": steps})
        print(melid, seed, plan_name, "match", match, "redraw", redraw == s["generated_ids"][STEP], "p cpu %.3g mps %.3g" % (arms["A"]["p_token"], probe_case["arms"]["A"]["actual_next_prob"]),
              "state38", steps[STEP]["gen_state_sha16"])
    with open(out, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
