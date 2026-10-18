#!/usr/bin/env python3
"""Minimal condition-adapter run: 32 updates, then off / real / wrong condition NLL (issue #1760,
docs/experiments/COND_ADAPTER_MIN.md). Frozen Aria medium on MPS, fp32, batch 1. Run once, after review, with the Aria venv:

  cond_adapter_train.py <aria repo> <checkpoint> <#1756 export dir> <split json> <out dir (outside the repo)> <report json>

Training: STEPS examples picked with SEED from the eligible requests of the 8 training songs; window cross-entropy;
Adam on the adapter only. Evaluation at step 0 and STEPS only, on EVAL_PER_SONG evenly spaced eligible requests of each
of the 4 development songs: (a) adapter off, (b) real condition, (c) the current request's condition replaced by
wrong_payload (context rows kept), per window token, split by token kind and by song. Generation check (G): 2 primers x
2 seeds x (a, b, c), GEN_TOKENS sampled tokens each, grammar violations and <E> counted.
"""
from __future__ import annotations

import collections
import hashlib
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_mlx_cond import COND_DIM, cond_vector  # noqa: E402
from cond_adapter_data import SEED, load_song, pick_evenly, pick_training, song_examples, with_current_condition, wrong_payload  # noqa: E402
from stream_contract import GrammarState  # noqa: E402

STEPS, LR, EVAL_PER_SONG = 32, 1e-3, 48
GEN_PRIMERS, GEN_SEEDS, GEN_TOKENS = ((0, 10), (1, 10)), (1, 2), 48     # (dev song position, example position)
WALL_LIMIT_S = 3600     # training-loop limit; evaluation and generation after it are not bounded (expected/watch limit only)


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def step_guard(loss_value, elapsed_s, limit_s=WALL_LIMIT_S):
    """Reason to stop before the optimizer step, or None (non-finite loss, or over the training wall limit)."""
    if not math.isfinite(loss_value):
        return f"nonfinite loss {loss_value}"
    if elapsed_s > limit_s:
        return f"wall limit {limit_s} s"
    return None


def stop(report, report_path, reason):
    """Save the report with the reason and exit non-zero; nothing runs after a stop (no evaluation, no generation)."""
    report["stopped_at"] = reason
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"stopped_at": reason}))
    sys.exit(1)


def main() -> None:
    aria_repo, ckpt, export_dir, split_path, out_dir, report_path = sys.argv[1:7]
    sys.path.insert(0, aria_repo)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM
    from ariautils.tokenizer import AbsTokenizer

    t_start = time.time()
    split = json.load(open(split_path))
    train_ex, dev_ex, counts = [], {}, {}
    for s in split["train"]:
        exs, c = song_examples(*load_song(export_dir, s["idx"]), song=s["idx"])
        train_ex += exs
        counts[s["idx"]] = {**c, "eligible": len(exs)}
    for s in split["dev"]:
        exs, c = song_examples(*load_song(export_dir, s["idx"]), song=s["idx"])
        dev_ex[s["idx"]] = pick_evenly(exs, EVAL_PER_SONG)
        counts[s["idx"]] = {**c, "eligible": len(exs), "evaluated": len(dev_ex[s["idx"]])}
    train_pick = pick_training(train_ex, STEPS)

    dev = torch.device("mps")
    tok = AbsTokenizer()
    dry = os.environ.get("COND_ADAPTER_DRYRUN") == "1"     # plumbing only: a tiny random model, no checkpoint, no result
    if dry:
        cfg = ModelConfig(d_model=64, n_heads=4, n_layers=2, ff_mult=2, drop_p=0.0, max_seq_len=8192, grad_checkpoint=False)
    else:
        cfg = ModelConfig(**load_model_config(name="medium"))
    cfg.set_vocab_size(tok.vocab_size)
    model = TransformerLM(cfg)
    if not dry:
        model.load_state_dict(load_file(ckpt), strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.to(dev)
    model.model.freqs_cis = None

    def base_sha():
        h = hashlib.sha256()
        for name, p in sorted(model.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    sha_before = base_sha()
    torch.manual_seed(SEED)
    adapter = torch.nn.Linear(COND_DIM, cfg.d_model, bias=False).to(dev)
    torch.nn.init.zeros_(adapter.weight)
    state = {"cond": None}

    def pre_hook(_mod, args, kwargs):
        if state["cond"] is None:
            return None
        return (args[0] + adapter(state["cond"]),) + args[1:], kwargs

    model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)

    def logits_of(ids, rows):
        state["cond"] = None if rows is None else torch.tensor([rows], dtype=torch.float32, device=dev)
        out = model(torch.tensor([ids], dtype=torch.long, device=dev))
        state["cond"] = None
        return out

    def window_nll(ex, rows):
        with torch.no_grad():
            lg = logits_of(ex["ids"][:-1], rows)
            f = ex["first_label"]
            nll = torch.nn.functional.cross_entropy(lg[0, f - 1:].float(), torch.tensor(ex["ids"][f:], device=dev), reduction="none")
        return nll.cpu().numpy()

    def evaluate():
        out = {"by_song": {}, "pooled": {}}
        pooled = collections.defaultdict(lambda: collections.defaultdict(list))
        for song, exs in dev_ex.items():
            per = collections.defaultdict(lambda: collections.defaultdict(list))
            for ex in exs:
                w = wrong_payload(ex["payload"])
                stratum = "with_wrong" if w else ("cold" if not ex["payload"][1] else "warm_empty")
                arms = {"a_off": None, "b_real": ex["rows"]}
                if w:
                    arms["c_wrong"] = with_current_condition(ex, w)
                for arm, rows in arms.items():
                    nll = window_nll(ex, rows)
                    for kind, v in zip(ex["label_kinds"], nll):
                        for key in (f"{stratum}/{arm}/all", f"{stratum}/{arm}/{kind}"):
                            per[key]["v"].append(float(v))
                            pooled[key]["v"].append(float(v))
            out["by_song"][song] = {k: {"tokens": len(d["v"]), "mean_nll": round(sum(d["v"]) / len(d["v"]), 5)} for k, d in per.items()}
        out["pooled"] = {k: {"tokens": len(d["v"]), "mean_nll": round(sum(d["v"]) / len(d["v"]), 5)} for k, d in pooled.items()}
        return out

    report = {"issue": 1760, "step": "train", "quality_claimed": False, "dry_run": dry,
              "params": {"steps": STEPS, "lr": LR, "seed": SEED, "eval_per_song": EVAL_PER_SONG, "gen_primers": GEN_PRIMERS,
                         "gen_seeds": GEN_SEEDS, "gen_tokens": GEN_TOKENS, "dtype": "float32", "batch": 1},
              "counts": counts, "train_examples": [{"song": e["song"], "k": e["k"], "tokens": len(e["ids"])} for e in train_pick]}
    report["eval_step0"] = evaluate()
    p0 = report["eval_step0"]["pooled"]
    report["checks"] = {"step0_arms_identical": all(p0[k]["mean_nll"] == p0[k.replace("b_real", arm)]["mean_nll"]
                                                    for k in p0 if "/b_real/" in k for arm in ("a_off", "c_wrong")
                                                    if k.replace("b_real", arm) in p0)}
    if not report["checks"]["step0_arms_identical"]:
        stop(report, report_path, "step 0 arms differ (adapter is not a no-op before training)")
    opt = torch.optim.Adam([adapter.weight], lr=LR)
    losses = []
    for ex in train_pick:
        state["cond"] = torch.tensor([ex["rows"]], dtype=torch.float32, device=dev)
        lg = model(torch.tensor([ex["ids"][:-1]], dtype=torch.long, device=dev))
        state["cond"] = None
        f = ex["first_label"]
        loss = torch.nn.functional.cross_entropy(lg[0, f - 1:].float(), torch.tensor(ex["ids"][f:], device=dev))
        losses.append(round(float(loss.item()), 5))
        reason = step_guard(float(loss.item()), time.time() - t_start)
        if reason:
            report["train_losses"] = losses
            stop(report, report_path, reason)
        opt.zero_grad()
        loss.backward()
        opt.step()
    report["train_losses"] = losses
    os.makedirs(out_dir, exist_ok=True)
    w = adapter.weight.detach().cpu().numpy().astype(np.float32)
    np.save(os.path.join(out_dir, f"adapter_step{len(losses)}.npy"), w)
    report["adapter"] = {"path": os.path.join(out_dir, f"adapter_step{len(losses)}.npy"), "sha256": sha256_bytes(w.tobytes()),
                         "shape": list(w.shape), "fro_norm": round(float(np.linalg.norm(w)), 6)}
    report["eval_final"] = evaluate()

    gen = []
    vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
    for song_pos, ex_pos in GEN_PRIMERS:
        song = split["dev"][song_pos]["idx"]
        ex = dev_ex[song][ex_pos]
        prime = ex["ids"][:ex["first_label"]]
        prime_rows = ex["rows"][:ex["first_label"] - 1]
        w_pay = wrong_payload(ex["payload"])
        for seed in GEN_SEEDS:
            for arm, pay in (("a_off", None), ("b_real", ex["payload"]), ("c_wrong", w_pay)):
                if arm == "c_wrong" and pay is None:
                    continue
                g = torch.Generator(device="cpu").manual_seed(seed)
                seq, rows = list(prime), list(prime_rows)
                st = GrammarState()
                for t in (vocab[i] for i in prime):
                    try:
                        st = st.advance(t)
                    except ValueError:
                        st = GrammarState()
                violations, eos = 0, 0
                for _ in range(GEN_TOKENS):
                    rows_now = None if pay is None else rows + [cond_vector(pay)]
                    with torch.no_grad():
                        lg = logits_of(seq, rows_now)[0, -1].float().cpu()
                    nxt = int(torch.multinomial(torch.softmax(lg, -1), 1, generator=g).item())
                    t = vocab[nxt]
                    try:
                        st = st.advance(t)
                    except ValueError:
                        violations += 1
                    eos += t == "<E>"
                    seq.append(nxt)
                    rows.append(cond_vector(pay) if pay is not None else [0.0] * COND_DIM)
                    if t == "<E>":
                        break
                gen.append({"song": song, "k": ex["k"], "seed": seed, "arm": arm, "tokens": len(seq) - len(prime),
                            "grammar_violations": violations, "eos": eos})
    report["generation"] = gen
    report["checks"]["base_unchanged"] = base_sha() == sha_before
    report["checks"]["only_adapter_in_optimizer"] = all(p is adapter.weight for g in opt.param_groups for p in g["params"])
    report["checks"]["steps_done"] = len(losses) == STEPS
    report["run"] = {"git_head": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                     "torch": torch.__version__, "seconds": round(time.time() - t_start, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({"checks": report["checks"], "seconds": report["run"]["seconds"]}))
    if not all(report["checks"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
