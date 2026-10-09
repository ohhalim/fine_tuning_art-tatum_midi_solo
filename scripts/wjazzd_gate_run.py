#!/usr/bin/env python3
"""WJazzD gate pilot: the same 128-update pilot with the condition injected only at pitch positions
(docs/experiments/WJAZZD_GATE_PILOT.md).

Identical to wjazzd_pilot_run.py in data, order, optimizer, loss and injection point; the only change
is that condition rows are multiplied by aria_pitch_gate.gates(tokens), computed from the prefix only.
A new zero-initialised adapter is trained; the pilot adapter is not reused. Decision on validation;
test is reported as an exploratory set. --precheck runs no optimizer.
Run with the Aria venv: wjazzd_gate_run.py <aria repo> <checkpoint> <wjazzd.db> <data json> <out dir> <pilot result json> --precheck|--train
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import signal
import sqlite3
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_contract_v2 import features  # noqa: E402
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402
from aria_pitch_gate import gate_step, gates  # noqa: E402
from aria_token_validity import check as validity  # noqa: E402
from wjazzd_pilot_data import plan_for, row  # noqa: E402
from wjazzd_pilot_run import DIM, GEN_SEEDS, GEN_TOKENS, LR, SEED, UPDATES, WALL_LIMIT_S, wrong_rows  # noqa: E402

GEN_SONGS = 4
DECISION_SPLIT = "validation"


def gated(rows, toks):
    return [[x * g for x in r] for r, g in zip(rows, gates(toks))]


def main() -> None:
    aria_repo, ckpt, db, data_path, out_dir, pilot_result, mode = sys.argv[1:8]
    assert mode in ("--precheck", "--train")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"mode": mode, "phases": {}, "checks": {}, "train_loss": [], "eval": {}, "generation": [], "result": None, "stopped_at": None}

    def phase(name, t0, **extra):
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(), **extra,
               "mps_current_allocated_bytes": torch.mps.current_allocated_memory()}
        log["phases"][name] = rec
        if rec["vm_pressure_level"] >= 4:
            raise Stop(f"memory pressure critical after {name}")

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    def on_alarm(*_):
        raise Stop(f"wall limit {WALL_LIMIT_S} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL_LIMIT_S)
    try:
        import torch
        from safetensors.torch import load_file
        from ariautils.tokenizer import AbsTokenizer
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        torch.manual_seed(SEED)
        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        dev = torch.device("mps")
        with open(data_path) as f:
            data = json.load(f)
        by_split = collections.defaultdict(list)
        for ch in data["chunks"]:
            by_split[ch["split"]].append(ch)
        t0 = time.perf_counter()
        ckpt_sha = file_sha(ckpt)
        tok = AbsTokenizer()
        vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
        cfg = ModelConfig(**load_model_config(name="medium"))
        cfg.set_vocab_size(tok.vocab_size)
        model = TransformerLM(cfg)
        model.load_state_dict(load_file(ckpt), strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(dev)
        model.model.freqs_cis = None
        phase("load", t0)

        def base_param_sha():
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        params_sha = base_param_sha()
        adapter = torch.nn.Linear(DIM, cfg.d_model, bias=False).to(dev)
        torch.nn.init.zeros_(adapter.weight)
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

        toks_of = lambda ch: [vocab[i] for i in ch["token_ids"]]
        gated_rows = {id(ch): gated(ch["cond"], toks_of(ch)) for ch in data["chunks"]}
        pitch_ids = collections.defaultdict(list)
        for i, t in enumerate(vocab):
            if isinstance(t, tuple) and t[0] == "piano":
                pitch_ids[t[1]].append(i)
        pitch_index = {p: torch.tensor(v, device=dev) for p, v in pitch_ids.items()}
        ids_of = lambda ch: torch.tensor([ch["token_ids"]], device=dev)

        # ---- prechecks (no optimizer)
        mism = collections.Counter()
        on_total = 0
        for ch in data["chunks"]:
            tk = toks_of(ch)
            g = gates(tk)
            state_, step = "header", []
            for t in tk:
                state_, gg = gate_step(state_, t)
                step.append(gg)
            if step != g:
                mism["stepwise_vs_full"] += 1
            for i in range(len(tk) - 1):
                nxt = tk[i + 1]
                is_start = (isinstance(nxt, tuple) and nxt[0] == "piano") or nxt in ("<T>", "<D>", "<E>")
                if g[i]:
                    on_total += 1
                    mism["on_but_next_not_note_start"] += not is_start
                elif i >= 1 and is_start:
                    mism["off_but_next_note_start"] += 1
            for cut in (len(tk) // 3, len(tk) // 2):
                alt = tk[: cut + 1] + [("piano", 60, 80), ("onset", 0), ("dur", 10), "<E>"]
                if gates(alt)[: cut + 1] != g[: cut + 1]:
                    mism["prefix_invariance"] += 1
        check("gate_consistency_on_data", not any(mism.values()), counts=dict(mism), gate_on_positions=on_total,
              positions=sum(len(ch["token_ids"]) for ch in data["chunks"]))
        ch0 = by_split["train"][0]
        with torch.no_grad():
            a = run(ids_of(ch0)).float()
            b = run(ids_of(ch0), gated_rows[id(ch0)], adapter).float()
        check("zero_adapter_equals_base", (a - b).abs().max().item() == 0.0)
        longest = max(by_split["train"], key=lambda ch: len(ch["token_ids"]))
        t0 = time.perf_counter()
        logits = run(ids_of(longest), gated_rows[id(longest)], adapter)
        ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids_of(longest)[0, 1:], reduction="none")
        mask = torch.tensor(longest["mask"], device=dev, dtype=torch.float32)
        loss = (ce * mask).sum() / mask.sum()
        loss.backward()
        gr = adapter.weight.grad
        phase("forward_backward_longest_chunk", t0, tokens=len(longest["token_ids"]))
        check("cost_probe_grad", torch.isfinite(loss).item() and gr is not None and torch.isfinite(gr).all().item() and gr.abs().sum().item() > 0)
        check("base_has_no_grad", all(p.grad is None for p in model.parameters()))
        adapter.weight.grad = None

        if mode == "--train":
            opt = torch.optim.Adam([adapter.weight], lr=LR)
            base_ids = {id(p) for p in model.parameters()}
            check("optimizer_has_no_base_param", sum(id(p) in base_ids for g_ in opt.param_groups for p in g_["params"]) == 0)
            songs = sorted({ch["melid"] for ch in by_split["train"]})
            per_song = {m: [ch for ch in by_split["train"] if ch["melid"] == m] for m in songs}
            pick = random.Random(SEED)
            t0 = time.perf_counter()
            for step in range(1, UPDATES + 1):
                m = pick.choice(songs)
                ch = pick.choice(per_song[m])
                logits = run(ids_of(ch), gated_rows[id(ch)], adapter)
                ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids_of(ch)[0, 1:], reduction="none")
                mask = torch.tensor(ch["mask"], device=dev, dtype=torch.float32)
                loss = (ce * mask).sum() / mask.sum()
                if not torch.isfinite(loss).item():
                    raise Stop(f"non-finite loss at update {step}")
                opt.zero_grad()
                loss.backward()
                if not torch.isfinite(adapter.weight.grad).all().item():
                    raise Stop(f"non-finite gradient at update {step}")
                opt.step()
                log["train_loss"].append({"update": step, "melid": m, "loss": loss.item()})
            phase("train_128_updates", t0)
            path = os.path.join(out_dir, "adapter_wjazzd_gate_diagnostic_only.pt")
            torch.save({"weight": adapter.weight.detach().cpu(), "note": "gate pilot diagnostic; not a product model"}, path)
            log["adapter_sha256"] = file_sha(path)

            def evaluate(split):
                res = collections.defaultdict(lambda: collections.defaultdict(list))
                strata = collections.defaultdict(lambda: collections.defaultdict(list))
                with torch.no_grad():
                    for ch in by_split[split]:
                        ids, tk = ids_of(ch), toks_of(ch)
                        conds = {"base": (None, None), "correct": (gated_rows[id(ch)], adapter),
                                 "wrong": (gated(wrong_rows(ch["cond"]), tk), adapter),
                                 "inactive": ([[0.0] * DIM] * len(ch["cond"]), adapter)}
                        mask = torch.tensor(ch["mask"], device=dev, dtype=torch.float32)
                        for name, (rows, ad) in conds.items():
                            lp = torch.log_softmax(run(ids, rows, ad)[0].float(), -1)
                            cont = -(lp[:-1].gather(1, ids[0, 1:, None])[:, 0] * mask).sum().item() / mask.sum().item()
                            pn = []
                            for t in ch["pitch_targets"]:
                                i = t["pos"]
                                marg = -torch.logsumexp(lp[i - 1, pitch_index[vocab[ch["token_ids"][i]][1]]], 0).item()
                                pn.append(marg)
                                strata[name][t["stratum"]].append(marg)
                                if t["anticipation"]:
                                    strata[name]["anticipation"].append(marg)
                            res[name][ch["melid"]].append({"pitch": sum(pn) / len(pn), "continuation": cont})
                out = {}
                for name, sg in res.items():
                    song = {m: {k: sum(c[k] for c in cs) / len(cs) for k in ("pitch", "continuation")} for m, cs in sg.items()}
                    out[name] = {"song_mean": {k: sum(v[k] for v in song.values()) / len(song) for k in ("pitch", "continuation")},
                                 "per_song": song,
                                 "strata_pitch_nll_token_pooled": {s: {"n": len(v), "mean": sum(v) / len(v)} for s, v in strata[name].items()}}
                ps, pw, pb = out["correct"]["per_song"], out["wrong"]["per_song"], out["base"]["per_song"]
                out["counts"] = {"correct_lt_wrong": sum(ps[m]["pitch"] < pw[m]["pitch"] for m in ps),
                                 "correct_lt_base": sum(ps[m]["pitch"] < pb[m]["pitch"] for m in ps), "songs": len(ps)}
                return out

            t0 = time.perf_counter()
            log["eval"][DECISION_SPLIT] = evaluate(DECISION_SPLIT)
            log["eval"]["test_exploratory"] = evaluate("test")
            phase("evaluation", t0)

            t0 = time.perf_counter()
            con = sqlite3.connect(db)
            dec = by_split[DECISION_SPLIT]
            songs_v = []
            for ch in dec:
                if ch["melid"] not in songs_v:
                    songs_v.append(ch["melid"])
            for m in songs_v[:GEN_SONGS]:
                ch = next(c for c in dec if c["melid"] == m)
                plan = plan_for(con, m, ch["t0"])
                for seed in GEN_SEEDS:
                    for name in ("base", "correct", "wrong"):
                        g_ = torch.Generator().manual_seed(seed)
                        prefix_ids = ch["token_ids"][: ch["first_target"]]
                        ids = torch.tensor([prefix_ids], device=dev)
                        tk = [vocab[i] for i in prefix_ids]
                        new_ids = []
                        for _ in range(GEN_TOKENS):
                            if name == "base":
                                rows, ad = None, None
                            else:
                                rows = [row(f) for f in features(tk, plan)]
                                rows = wrong_rows(rows) if name == "wrong" else rows
                                rows, ad = gated(rows, tk), adapter
                            with torch.no_grad():
                                lg = run(ids, rows, ad)[0, -1].float().cpu()
                            nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=g_).item()
                            new_ids.append(nxt)
                            tk.append(vocab[nxt])
                            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
                            if vocab[nxt] == "<E>":
                                break
                        v = validity([vocab[i] for i in prefix_ids], [vocab[i] for i in new_ids])
                        log["generation"].append({"melid": m, "seed": seed, "plan": name, "prefix_ids": prefix_ids,
                                                  "generated_ids": new_ids, "chunk_t0": ch["t0"], **v})
            phase("generation", t0)
            check("base_params_unchanged", base_param_sha() == params_sha)
            check("checkpoint_file_unchanged", file_sha(ckpt) == ckpt_sha)
            with open(pilot_result) as f:
                pilot = json.load(f)
            pv = pilot["eval"]["validation_info"]
            log["reference_ungated_pilot_validation"] = {k: pv[k]["song_mean"] for k in ("base", "correct", "wrong")}
            ev = log["eval"][DECISION_SPLIT]
            valid = collections.Counter(g["plan"] for g in log["generation"] if g["valid"])
            n_each = len(songs_v[:GEN_SONGS]) * len(GEN_SEEDS)
            log["decision"] = {"net_effect_correct_lt_base_song_mean": ev["correct"]["song_mean"]["pitch"] < ev["base"]["song_mean"]["pitch"],
                               "correct_lt_base_songs": ev["counts"]["correct_lt_base"],
                               "condition_use_correct_lt_wrong_songs": ev["counts"]["correct_lt_wrong"],
                               "condition_use_observed": ev["counts"]["correct_lt_wrong"] >= 7 and ev["correct"]["song_mean"]["pitch"] < ev["wrong"]["song_mean"]["pitch"],
                               "valid_generations": {k: valid.get(k, 0) for k in ("base", "correct", "wrong")}, "of_each": n_each,
                               "generation_not_worse_than_base": valid.get("correct", 0) >= valid.get("base", 0)}
        else:
            check("base_params_unchanged", base_param_sha() == params_sha)
        log["result"] = "completed"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except Exception as e:
        log["result"], log["stopped_at"] = "stopped", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        name = "gate_precheck.json" if mode == "--precheck" else "gate_result.json"
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v.get("ok") for k, v in log["checks"].items()}, "decision": log.get("decision")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
