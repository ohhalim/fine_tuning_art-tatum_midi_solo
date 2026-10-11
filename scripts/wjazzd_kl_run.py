#!/usr/bin/env python3
"""WJazzD KL comparison: the gate pilot with an optional KL anchor to the frozen base
(docs/experiments/ARIA_COND_CONTRACT_V2.md, "가능한 첫 실험").

Same data, order, optimizer, gate, injection point and 128 updates as wjazzd_gate_run.py. The loss is
CE_mean + lambda * mean KL(p_adapter || p_base) at loss-mask positions whose gate is off (onset or
dur prediction); lambda = 0 adds no term and must reproduce the gate run. Reads the saved data.json
only (sha256 checked, no re-tokenisation). Every invocation starts a fresh zero adapter, optimizer
and Random(0) order.
  --train        128 updates, validation decision, test exploratory, validation generations
  --sealed-eval  one evaluation of a trained adapter on the sealed held-out set; run only after all
                 development criteria passed
Run with the Aria venv:
  wjazzd_kl_run.py <aria repo> <ckpt> <wjazzd.db> <data json> <out dir> <gate result json> <lambda> --train
  wjazzd_kl_run.py <aria repo> <ckpt> <wjazzd.db> <sealed json> <out dir> <gate result json> <lambda> --sealed-eval <adapter>
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
from aria_pitch_gate import gates  # noqa: E402
from aria_token_validity import check as validity  # noqa: E402
from kl_anchor import kl_adapter_base, kl_positions, kl_term  # noqa: E402
from wjazzd_gate_run import gated  # noqa: E402
from wjazzd_pilot_data import plan_for, row  # noqa: E402
from wjazzd_pilot_run import DIM, GEN_SEEDS, GEN_TOKENS, LR, SEED, UPDATES, WALL_LIMIT_S, wrong_rows  # noqa: E402

DATA_SHA = "fc4e8ea2e91c1098a8caa09fcf5306fe22f454bb94ed6860ddfb129811e6157c"
SEALED_SHA = "ffadc24c8fb52353"
GATE_ADAPTER_SHA = "53a7183f527ad9c4d1ea54b314a98a2966ea8ea1f858950d7a62ec33908875cd"
REPRO_W, REPRO_NLL = 1e-6, 1e-4
DEV_GEN_SONGS, SEALED_GEN_SONGS = 4, 8


def main() -> None:
    aria_repo, ckpt, db, data_path, out_dir, gate_result, lam, mode = sys.argv[1:9]
    lam = float(lam)
    assert mode in ("--train", "--sealed-eval")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    tag = f"lambda{lam:g}"
    log = {"mode": mode, "lambda": lam, "phases": {}, "checks": {}, "train_loss": [], "eval": {}, "generation": [],
           "result": None, "stopped_at": None}

    def phase(name, t0):
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(),
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
        data_sha = file_sha(data_path)
        check("data_sha256", data_sha == DATA_SHA if mode == "--train" else data_sha.startswith(SEALED_SHA), sha=data_sha)
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
            out = model(ids)
            state["adapter"], state["cond"] = None, None          # hook state restored after every call
            return out

        toks_of = lambda ch: [vocab[i] for i in ch["token_ids"]]
        gate_of = {id(ch): gates(toks_of(ch)) for ch in data["chunks"]}
        gated_rows = {id(ch): gated(ch["cond"], toks_of(ch)) for ch in data["chunks"]}
        kl_pos = {id(ch): kl_positions(ch["mask"], gate_of[id(ch)][:-1]) for ch in data["chunks"]}
        pitch_ids = collections.defaultdict(list)
        for i, t in enumerate(vocab):
            if isinstance(t, tuple) and t[0] == "piano":
                pitch_ids[t[1]].append(i)
        pitch_index = {p: torch.tensor(v, device=dev) for p, v in pitch_ids.items()}
        ids_of = lambda ch: torch.tensor([ch["token_ids"]], device=dev)
        bad_pos = sum(1 for ch in data["chunks"] for k in kl_pos[id(ch)]
                      if not (isinstance(toks_of(ch)[k + 1], tuple) and toks_of(ch)[k + 1][0] in ("onset", "dur")))
        check("kl_positions_predict_onset_or_dur", bad_pos == 0, positions=sum(len(v) for v in kl_pos.values()), bad=bad_pos)

        def evaluate(chunks, kl_diag=False):
            res = collections.defaultdict(lambda: collections.defaultdict(list))
            strata = collections.defaultdict(lambda: collections.defaultdict(list))
            kl_rev, kl_fwd = [], []
            with torch.no_grad():
                for ch in chunks:
                    ids, tk = ids_of(ch), toks_of(ch)
                    conds = {"base": (None, None), "correct": (gated_rows[id(ch)], adapter),
                             "wrong": (gated(wrong_rows(ch["cond"]), tk), adapter),
                             "inactive": ([[0.0] * DIM] * len(ch["cond"]), adapter)}
                    mask = torch.tensor(ch["mask"], device=dev, dtype=torch.float32)
                    lp_cache = {}
                    for name, (rows, ad) in conds.items():
                        logits = run(ids, rows, ad)[0].float()
                        lp = torch.log_softmax(logits, -1)
                        lp_cache[name] = logits
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
                    if kl_diag and kl_pos[id(ch)]:
                        idx = torch.tensor(kl_pos[id(ch)], device=dev)
                        kl_rev.append(kl_adapter_base(lp_cache["correct"][idx], lp_cache["base"][idx]).mean().item())
                        kl_fwd.append(kl_adapter_base(lp_cache["base"][idx], lp_cache["correct"][idx]).mean().item())
            out = {}
            for name, sg in res.items():
                song = {m: {k: sum(c[k] for c in cs) / len(cs) for k in ("pitch", "continuation")} for m, cs in sg.items()}
                out[name] = {"song_mean": {k: sum(v[k] for v in song.values()) / len(song) for k in ("pitch", "continuation")},
                             "per_song": song,
                             "strata_pitch_nll_token_pooled": {s: {"n": len(v), "mean": sum(v) / len(v)} for s, v in strata[name].items()}}
            ps, pw, pb = out["correct"]["per_song"], out["wrong"]["per_song"], out["base"]["per_song"]
            out["counts"] = {"correct_lt_wrong": sum(ps[m]["pitch"] < pw[m]["pitch"] for m in ps),
                             "correct_lt_base": sum(ps[m]["pitch"] < pb[m]["pitch"] for m in ps), "songs": len(ps)}
            if kl_diag:
                out["kl_diagnostic_correct_plan"] = {"reverse_kl_adapter_base_chunk_mean": sum(kl_rev) / len(kl_rev) if kl_rev else None,
                                                     "forward_kl_base_adapter_chunk_mean": sum(kl_fwd) / len(kl_fwd) if kl_fwd else None}
            return out

        def generate(chunks, n_songs):
            con = sqlite3.connect(db)
            songs = []
            for ch in chunks:
                if ch["melid"] not in songs:
                    songs.append(ch["melid"])
            outg = []
            for m in songs[:n_songs]:
                ch = next(c for c in chunks if c["melid"] == m)
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
                        outg.append({"melid": m, "seed": seed, "plan": name, "prefix_ids": prefix_ids,
                                     "generated_ids": new_ids, "chunk_t0": ch["t0"], **v})
            return outg, len(songs[:n_songs]) * len(GEN_SEEDS)

        def judge(ev, gen, n_each, need_cond):
            valid = collections.Counter(g["plan"] for g in gen if g["valid"])
            return {"net_effect_correct_lt_base_song_mean": ev["correct"]["song_mean"]["pitch"] < ev["base"]["song_mean"]["pitch"],
                    "correct_lt_base_songs": ev["counts"]["correct_lt_base"],
                    "condition_correct_lt_wrong_songs": ev["counts"]["correct_lt_wrong"], "of_songs": ev["counts"]["songs"],
                    "condition_song_mean_correct_lt_wrong": ev["correct"]["song_mean"]["pitch"] < ev["wrong"]["song_mean"]["pitch"],
                    "condition_use_observed": ev["counts"]["correct_lt_wrong"] >= need_cond and ev["correct"]["song_mean"]["pitch"] < ev["wrong"]["song_mean"]["pitch"],
                    "valid_generations": {k: valid.get(k, 0) for k in ("base", "correct", "wrong")}, "of_each": n_each,
                    "generation_correct_ge_base": valid.get("correct", 0) >= valid.get("base", 0)}

        if mode == "--train":
            opt = torch.optim.Adam([adapter.weight], lr=LR)
            base_ids = {id(p) for p in model.parameters()}
            check("optimizer_has_no_base_param", sum(id(p) in base_ids for g_ in opt.param_groups for p in g_["params"]) == 0)
            ch_fixed = by_split["validation"][0]
            with torch.no_grad():
                teacher_before = run(ids_of(ch_fixed)).float().cpu()
            songs = sorted({ch["melid"] for ch in by_split["train"]})
            per_song = {m: [ch for ch in by_split["train"] if ch["melid"] == m] for m in songs}
            pick = random.Random(SEED)
            t0 = time.perf_counter()
            for step in range(1, UPDATES + 1):
                m = pick.choice(songs)
                ch = pick.choice(per_song[m])
                ids = ids_of(ch)
                if lam > 0:
                    with torch.no_grad():
                        base_logits = run(ids)[0].detach()
                logits = run(ids, gated_rows[id(ch)], adapter)
                ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids[0, 1:], reduction="none")
                mask = torch.tensor(ch["mask"], device=dev, dtype=torch.float32)
                ce_mean = (ce * mask).sum() / mask.sum()
                loss = ce_mean
                kl_val, n_kl = 0.0, len(kl_pos[id(ch)])
                if lam > 0:
                    kl, n_kl = kl_term(logits[0], base_logits, kl_pos[id(ch)])
                    loss = ce_mean + lam * kl
                    kl_val = kl.item()
                if not torch.isfinite(loss).item():
                    raise Stop(f"non-finite loss at update {step}")
                opt.zero_grad()
                loss.backward()
                if not torch.isfinite(adapter.weight.grad).all().item():
                    raise Stop(f"non-finite gradient at update {step}")
                opt.step()
                log["train_loss"].append({"update": step, "melid": m, "ce": ce_mean.item(), "kl": kl_val, "kl_positions": n_kl})
            phase("train_128_updates", t0)
            with torch.no_grad():
                teacher_after = run(ids_of(ch_fixed)).float().cpu()
            check("teacher_unchanged", torch.equal(teacher_before, teacher_after))
            path = os.path.join(out_dir, f"adapter_wjazzd_kl_{tag}_diagnostic_only.pt")
            torch.save({"weight": adapter.weight.detach().cpu(), "note": f"KL comparison {tag}; diagnostic only"}, path)
            log["adapter_sha256"] = file_sha(path)
            re = torch.nn.Linear(DIM, cfg.d_model, bias=False).to(dev)
            re.weight.data.copy_(torch.load(path)["weight"].to(dev))
            with torch.no_grad():
                a1 = run(ids_of(ch_fixed), gated_rows[id(ch_fixed)], adapter).float()
                b1 = run(ids_of(ch_fixed), gated_rows[id(ch_fixed)], re).float()
            check("save_reload_logits", (a1 - b1).abs().max().item() == 0.0)
            t0 = time.perf_counter()
            log["eval"]["validation"] = evaluate(by_split["validation"], kl_diag=True)
            log["eval"]["test_exploratory"] = evaluate(by_split["test"], kl_diag=True)
            phase("evaluation", t0)
            with open(gate_result) as f:
                gate = json.load(f)
            if lam == 0:
                ref_w = torch.load(os.path.join(os.path.dirname(gate_result), "adapter_wjazzd_gate_diagnostic_only.pt"))["weight"]
                check("gate_adapter_file_sha", file_sha(os.path.join(os.path.dirname(gate_result), "adapter_wjazzd_gate_diagnostic_only.pt")) == GATE_ADAPTER_SHA)
                wdiff = (adapter.weight.detach().cpu() - ref_w).abs().max().item()
                ndiff = max(abs(log["eval"]["validation"][k]["song_mean"]["pitch"] - gate["eval"]["validation"][k]["song_mean"]["pitch"])
                            for k in ("base", "correct", "wrong"))
                log["repro_lambda0"] = {"max_abs_weight_diff": wdiff, "max_abs_song_mean_pitch_nll_diff": ndiff,
                                        "tolerances": [REPRO_W, REPRO_NLL]}
                check("lambda0_reproduces_gate_run", wdiff <= REPRO_W and ndiff <= REPRO_NLL, max_abs_weight_diff=wdiff, max_abs_nll_diff=ndiff)
            t0 = time.perf_counter()
            log["generation"], n_each = generate(by_split["validation"], DEV_GEN_SONGS)
            phase("generation", t0)
            log["decision_development"] = judge(log["eval"]["validation"], log["generation"], n_each, need_cond=7)
            d = log["decision_development"]
            log["development_all_passed"] = bool(d["net_effect_correct_lt_base_song_mean"] and d["condition_use_observed"] and d["generation_correct_ge_base"])
        else:
            adapter_path = sys.argv[9]
            adapter.weight.data.copy_(torch.load(adapter_path)["weight"].to(dev))
            log["adapter_sha256"] = file_sha(adapter_path)
            t0 = time.perf_counter()
            log["eval"]["sealed"] = evaluate(by_split["sealed_heldout"])
            log["generation"], n_each = generate(by_split["sealed_heldout"], SEALED_GEN_SONGS)
            phase("sealed_evaluation", t0)
            log["decision_sealed"] = judge(log["eval"]["sealed"], log["generation"], n_each, need_cond=13)
        check("base_params_unchanged", base_param_sha() == params_sha)
        check("checkpoint_file_unchanged", file_sha(ckpt) == ckpt_sha)
        log["result"] = "completed"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except Exception as e:
        log["result"], log["stopped_at"] = "stopped", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        name = f"kl_{tag}_{'train' if mode == '--train' else 'sealed'}.json"
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v.get("ok") for k, v in log["checks"].items()},
                          "repro": log.get("repro_lambda0"), "dev": log.get("decision_development"),
                          "dev_all": log.get("development_all_passed"), "sealed": log.get("decision_sealed")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
