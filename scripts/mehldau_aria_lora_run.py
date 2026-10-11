#!/usr/bin/env python3
"""Mehldau Aria LoRA pilot runner (issue #1726, docs/experiments/MEHLDAU_ARIA_LORA.md).

Modes (one per process):
  --precheck  data/manifest hashes, window structure, loss mask, zero LoRA == base, eval and
              generation pipelines on a few items, one train-mode forward/backward on the longest
              train window (no optimizer step), time projections. No training.
  --train     base eval (val, probe) -> exactly 128 Adam updates -> adapter eval (val, probe) ->
              paired generations -> development verdict. Run only after the plan is reviewed.
  --test      once, only if the --train report says the development verdict passed and no test
              report exists yet: base vs adapter on the test split.

Fixed (issue #1726): frozen model-gen F32 on MPS, LoRA r16 alpha16 on mixed_qkv + att_proj_linear of
16 blocks, A seed 0, B zero, gradient checkpoint on, batch 1, Adam lr 1e-4, 128 updates, sampling
random.Random(0): recording uniform, then window uniform. Loss targets: tokens after <S>.
Evaluation: first <= 8 windows per recording, continuation CE per recording (token-weighted inside
a recording), primary summary = mean over recordings. Generation: 4 val prompts x seeds 1, 2 x
{base, adapter}, 512 new tokens, temperature 1, no min_p, stop at <E>.
Development verdict: val dCE < 0, probe dCE <= +0.02 nat/token, adapter valid samples >= base.

  mehldau_aria_lora_run.py <aria repo> <checkpoint> <data json> <manifest json> <out dir> --precheck|--train|--test
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402
from aria_lora import AriaLoRA, grad_problems, input_mismatches, named_lora, unchanged  # noqa: E402
from aria_token_validity import check as token_check  # noqa: E402

SEED, RANK, ALPHA, LR, UPDATES = 0, 16, 16, 1e-4, 128
GEN_SEEDS, GEN_TOKENS, TEMP = (1, 2), 512, 1.0
VAL_MAX_DCE, PROBE_MAX_DCE = 0.0, 0.02
RELOAD_TOLERANCE = 0.0      # fixed before the run
WALL = {"--precheck": 900, "--train": 3600, "--test": 1200}
NGRAM = 16


def longest_repeat(pitches):
    best = 0
    for per in range(1, 9):
        for s0 in range(len(pitches)):
            k = 0
            while s0 + per + k < len(pitches) and pitches[s0 + k] == pitches[s0 + per + k]:
                k += 1
            best = max(best, k + per if k >= per else 0)
    return best


def note_events(tokens):
    """(abs onset ms, end ms, pitch) for complete notes; <T> advances 5 s."""
    seg, out, cur = 0, [], {}
    for t in tokens:
        if t == "<T>":
            seg += 1
        elif isinstance(t, tuple) and t[0] == "piano":
            cur = {"p": t[1]}
        elif isinstance(t, tuple) and t[0] == "onset" and "p" in cur:
            cur["on"] = seg * 5000 + t[1]
        elif isinstance(t, tuple) and t[0] == "dur" and "on" in cur:
            out.append((cur["on"], cur["on"] + t[1], cur["p"]))
            cur = {}
    return out


def max_polyphony(ev):
    pts = sorted([(s, 1) for s, _, _ in ev] + [(e, -1) for _, e, _ in ev])
    cur = best = 0
    for _, d in pts:
        cur += d
        best = max(best, cur)
    return best


def main() -> None:
    aria_repo, ckpt, data_path, manifest_path, out_dir, mode = sys.argv[1:7]
    assert mode in WALL
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    report_path = os.path.join(out_dir, {"--precheck": "precheck.json", "--train": "train_report.json", "--test": "test_report.json"}[mode])
    if mode == "--test":
        dev = json.load(open(os.path.join(out_dir, "train_report.json")))
        assert dev.get("development", {}).get("pass") is True, "development verdict did not pass: test stays unopened"
        assert not os.path.exists(report_path), "test report exists: the test split is evaluated once"
        cur = {"checkpoint_sha256": file_sha(ckpt), "data_sha256": file_sha(data_path),
               "manifest_sha256": file_sha(manifest_path), "lora_file_sha256": file_sha(os.path.join(out_dir, "lora_u128.pt"))}
        rec = {**{k: dev["provenance"].get(k) for k in ("checkpoint_sha256", "data_sha256", "manifest_sha256")},
               "lora_file_sha256": dev.get("lora_file_sha256")}
        bad = input_mismatches(rec, cur)
        assert not bad, f"inputs differ from the train report: {bad}"
    log = {"mode": mode, "phases": {}, "checks": {}, "result": None, "stopped_at": None}
    import torch

    peak = {"v": 0, "run": False}

    def poll():
        while peak["run"]:
            peak["v"] = max(peak["v"], torch.mps.driver_allocated_memory())
            time.sleep(0.005)

    def phase(name, t0, **extra):
        torch.mps.synchronize()
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(), **extra,
               "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()}
        log["phases"][name] = rec
        if rec["vm_pressure_level"] >= 4:
            raise Stop(f"memory pressure critical after {name}")

    def check(name, ok, **detail):
        log["checks"][name] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {name}")

    def on_alarm(*_):
        raise Stop(f"wall limit {WALL[mode]} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL[mode])
    try:
        from safetensors.torch import load_file
        from ariautils.tokenizer import AbsTokenizer
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        dev_ = torch.device("mps")
        t0 = time.perf_counter()
        manifest = json.load(open(manifest_path))
        check("data_sha_matches_manifest", file_sha(data_path) == manifest["data_sha256"])
        data = json.load(open(data_path))
        meta = {r["path"]: r for r in manifest["recordings"]}
        recs = collections.defaultdict(list)
        for r in data["recordings"]:
            recs[r["split"]].append(r)
        tok = AbsTokenizer()
        cfg_dict = load_model_config(name="medium")
        cfg = ModelConfig(**cfg_dict)
        cfg.set_vocab_size(tok.vocab_size)
        cfg.grad_checkpoint = True
        model = TransformerLM(cfg)
        model.load_state_dict(load_file(ckpt), strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(dev_)
        model.model.freqs_cis = None
        lora = AriaLoRA(model, rank=RANK, alpha=ALPHA, seed=SEED, device=dev_)

        def base_sha():
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        base_sha0 = base_sha()
        log["provenance"] = {"checkpoint_sha256": file_sha(ckpt), "data_sha256": manifest["data_sha256"],
                             "manifest_sha256": file_sha(manifest_path), "torch": torch.__version__,
                             "tokenizer": "ariautils AbsTokenizer", "config": cfg_dict, "lora_params": lora.n_params()}
        vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
        cat = []
        pitch_of = torch.full((tok.vocab_size,), -1, dtype=torch.long)
        for i, t in enumerate(vocab):
            if isinstance(t, tuple) and t[0] == "piano":
                cat.append("pitch")
                pitch_of[i] = t[1]
            elif isinstance(t, tuple) and t[0] in ("onset", "dur"):
                cat.append(t[0])
            elif t == "<T>":
                cat.append("time")
            else:
                cat.append("other")
        cat_names = ["pitch", "onset", "dur", "time", "other"]
        cat_id = torch.tensor([cat_names.index(c) for c in cat], device=dev_)
        pmat = torch.zeros(tok.vocab_size, 128, device=dev_)
        for i in range(tok.vocab_size):
            if pitch_of[i] >= 0:
                pmat[i, pitch_of[i]] = 1.0
        pitch_of = pitch_of.to(dev_)
        eos_id, dim_id, s_id = tok.encode(["<E>"])[0], tok.encode(["<D>"])[0], tok.encode(["<S>"])[0]
        phase("load", t0)

        def window_ce(ids, s_index):
            """Per-target-token CE for targets after <S>, plus pitch-marginal CE for note tokens."""
            x = torch.tensor([ids], device=dev_)
            with torch.no_grad():
                lp = torch.log_softmax(model(x).float()[0, :-1], -1)
            tgt = x[0, 1:]
            pos = torch.arange(len(ids) - 1, device=dev_) >= s_index
            lp, tgt = lp[pos], tgt[pos]
            ce = -lp.gather(1, tgt[:, None])[:, 0]
            is_p = pitch_of[tgt] >= 0
            pm = -torch.log((lp.exp()[is_p] @ pmat).gather(1, pitch_of[tgt][is_p][:, None])[:, 0])
            return ce, cat_id[tgt], pm, ce[is_p]

        def evaluate(split, on):
            model.eval()
            lora.on = on
            out = []
            for r in recs[split]:
                n = meta[r["path"]]["eval_windows"]
                acc = {"tokens": 0, "ce": 0.0, "cat_ce": [0.0] * 5, "cat_n": [0] * 5, "pitch_marg": 0.0, "pitch_tok": 0.0}
                for w in r["windows"][:n]:
                    ce, cid, pm, ce_p = window_ce(w["ids"], w["s_index"])
                    acc["tokens"] += ce.numel()
                    acc["ce"] += ce.sum().item()
                    for k in range(5):
                        sel = cid == k
                        acc["cat_ce"][k] += ce[sel].sum().item()
                        acc["cat_n"][k] += int(sel.sum().item())
                    acc["pitch_marg"] += pm.sum().item()
                    acc["pitch_tok"] += ce_p.sum().item()
                out.append({"path": r["path"], "group": meta[r["path"]]["group"], "pianist": meta[r["path"]]["pianist"],
                            "windows": n, **acc, "mean_ce": acc["ce"] / acc["tokens"]})
            lora.on = True
            return out

        def summary(rows):
            by_group = collections.defaultdict(list)
            for r in rows:
                by_group[r["group"] or r["path"]].append(r["mean_ce"])
            tot = sum(r["tokens"] for r in rows)
            cat_n = [sum(r["cat_n"][k] for r in rows) for k in range(5)]
            return {"recordings": len(rows), "tokens": tot,
                    "recording_mean_ce": sum(r["mean_ce"] for r in rows) / len(rows),
                    "group_mean_ce": sum(sum(v) / len(v) for v in by_group.values()) / len(by_group),
                    "token_weighted_ce": sum(r["ce"] for r in rows) / tot,
                    "by_category": {cat_names[k]: (sum(r["cat_ce"][k] for r in rows) / cat_n[k] if cat_n[k] else None) for k in range(5)},
                    "category_tokens": dict(zip(cat_names, cat_n)),
                    "pitch_marginal_ce": sum(r["pitch_marg"] for r in rows) / cat_n[0],
                    "velocity_given_pitch_ce": (sum(r["pitch_tok"] for r in rows) - sum(r["pitch_marg"] for r in rows)) / cat_n[0]}

        def compare(base_rows, ad_rows):
            d = [a["mean_ce"] - b["mean_ce"] for a, b in zip(ad_rows, base_rows)]
            sb, sa = summary(base_rows), summary(ad_rows)
            return {"base": sb, "adapter": sa, "delta_recording_mean": sum(d) / len(d),
                    "delta_group_mean": sa["group_mean_ce"] - sb["group_mean_ce"],
                    "delta_token_weighted": sa["token_weighted_ce"] - sb["token_weighted_ce"],
                    "recordings_improved": sum(1 for x in d if x < 0), "per_recording_delta": [
                        {"path": a["path"], "delta": a["mean_ce"] - b["mean_ce"], "base": b["mean_ce"]} for a, b in zip(ad_rows, base_rows)],
                    "delta_by_category": {k: (sa["by_category"][k] - sb["by_category"][k]) if sb["by_category"][k] is not None else None for k in cat_names},
                    "delta_pitch_marginal": sa["pitch_marginal_ce"] - sb["pitch_marginal_ce"],
                    "delta_velocity_given_pitch": sa["velocity_given_pitch_ce"] - sb["velocity_given_pitch_ce"]}

        train_grams = set()
        for r in recs["train"]:
            for w in r["windows"]:
                ids = w["ids"]
                train_grams.update(tuple(ids[i:i + NGRAM]) for i in range(len(ids) - NGRAM + 1))

        def generate(prompt_ids, seed, on, max_new):
            model.eval()
            lora.on = on
            g = torch.Generator().manual_seed(seed)
            seq = list(prompt_ids)
            new = []
            with torch.no_grad():
                for _ in range(max_new):
                    lg = model(torch.tensor([seq], device=dev_))[0, -1].float() / TEMP
                    nxt = torch.multinomial(torch.softmax(lg, -1).cpu(), 1, generator=g).item()
                    seq.append(nxt)
                    new.append(nxt)
                    if nxt == eos_id:
                        break
            lora.on = True
            pt, nt = tok.decode(list(prompt_ids)), tok.decode(new)
            v = token_check(pt, nt)
            ev = note_events(pt + nt)
            new_ev = note_events(nt)
            ps = [t[1] for t in nt if isinstance(t, tuple) and t[0] == "piano"]
            grams = [tuple(new[i:i + NGRAM]) for i in range(len(new) - NGRAM + 1)]
            return {"seed": seed, "adapter": on, "new_tokens": len(new), "valid": v["valid"], "validity": v,
                    "ended_by_eos": new[-1] == eos_id if new else False, "notes": len(new_ev),
                    "pitch_range": [min(ps), max(ps)] if ps else None, "longest_repeat": longest_repeat(ps),
                    "max_polyphony": max_polyphony(new_ev) if new_ev else 0,
                    "music_seconds": round((max(e for _, e, _ in ev) - max((e for _, e, _ in note_events(pt)), default=0)) / 1000, 3) if new_ev else 0.0,
                    "train_16gram_hits": sum(1 for x in grams if x in train_grams), "16grams": len(grams), "ids": new}

        if mode == "--precheck":
            # structure of every window
            bad = collections.Counter()
            for split, rs in recs.items():
                for r in rs:
                    for w in r["windows"]:
                        ids = w["ids"]
                        bad["over_1024"] += len(ids) > 1024
                        bad["s_index_not_S"] += ids[w["s_index"]] != s_id
                        bad["eos_or_dim_inside"] += (eos_id in ids) or (dim_id in ids)
                        bad["prefix_not_instrument"] += vocab[ids[0]] != ("prefix", "instrument", "piano")
            check("window_structure", sum(bad.values()) == 0, **bad)
            w0 = recs["val"][0]["windows"][0]
            n_mask = sum(1 for j in range(len(w0["ids"]) - 1) if j >= w0["s_index"])
            check("loss_mask_targets_after_S", n_mask == len(w0["ids"]) - w0["s_index"] - 1 and w0["s_index"] == 1,
                  targets=n_mask, s_index=w0["s_index"], first_target=str(vocab[w0["ids"][w0["s_index"] + 1]]))
            x = torch.tensor([w0["ids"]], device=dev_)
            model.eval()
            with torch.no_grad():
                lora.on = False
                a = model(x).float()
                lora.on = True
                b = model(x).float()
            check("zero_lora_equals_base", (a - b).abs().max().item() == 0.0, max_abs_diff=(a - b).abs().max().item())
            t = time.perf_counter()
            ce, cid, pm, ce_p = window_ce(w0["ids"], w0["s_index"])
            torch.mps.synchronize()
            t_eval = time.perf_counter() - t
            check("eval_finite_and_consistent", bool(torch.isfinite(ce).all()) and int(cid.numel()) == ce.numel()
                  and bool((pm <= ce_p + 1e-5).all()), window_tokens=len(w0["ids"]), mean_ce=ce.mean().item(),
                  eval_s=round(t_eval, 3))
            check("prompts_available", all(p["available"] for p in manifest["prompts"]) and len(data["prompts"]) == 4,
                  prompt_tokens=[p["prompt_tokens"] for p in manifest["prompts"]])
            t = time.perf_counter()
            gsmoke = generate(data["prompts"][0]["ids"], 99, False, 16)
            t_gen = (time.perf_counter() - t) / max(1, gsmoke["new_tokens"])
            check("generation_pipeline", gsmoke["new_tokens"] > 0, new_tokens=gsmoke["new_tokens"], s_per_token=round(t_gen, 4))
            longest = max((w for r in recs["train"] for w in r["windows"]), key=lambda w: len(w["ids"]))
            model.train()
            peak["v"], peak["run"] = torch.mps.driver_allocated_memory(), True
            th = threading.Thread(target=poll, daemon=True)
            th.start()
            t = time.perf_counter()
            x = torch.tensor([longest["ids"]], device=dev_)
            out = model(x[:, :-1]).float()[0]
            pos = torch.arange(x.shape[1] - 1, device=dev_) >= longest["s_index"]
            loss = torch.nn.functional.cross_entropy(out[pos], x[0, 1:][pos])
            loss.backward()
            torch.mps.synchronize()
            t_step = time.perf_counter() - t
            peak["run"] = False
            th.join()
            n_none = sum(1 for p in lora.parameters() if p.grad is None)
            check("train_step_grads", n_none == 0 and all(p.grad is None for p in model.parameters()),
                  lora_grads_none=n_none, loss=loss.item(), step_s=round(t_step, 3), observed_max_driver_bytes=peak["v"])
            n_eval = sum(meta[r["path"]]["eval_windows"] for s in ("val", "probe") for r in recs[s])
            log["projection_s"] = {"train_128": round(UPDATES * t_step * 1.1, 1), "eval_val_probe_x2": round(2 * n_eval * t_eval, 1),
                                   "generation_16x512": round(16 * GEN_TOKENS * t_gen * 1.6, 1),
                                   "note": "eval/step from one window; generation from 16 tokens with a short context, scaled 1.6x for longer context"}
            log["result"] = "precheck_pass"

        elif mode == "--train":
            t = time.perf_counter()
            base_val, base_probe = evaluate("val", False), evaluate("probe", False)
            phase("base_eval", t)
            torch.manual_seed(SEED)
            rng = random.Random(SEED)
            init_sd = {n: p.detach().cpu().clone() for n, p in named_lora(lora)}
            opt = torch.optim.Adam(lora.parameters(), lr=LR)
            check("optimizer_only_lora", {id(p) for g in opt.param_groups for p in g["params"]} == {id(p) for p in lora.parameters()})
            model.train()
            peak["v"], peak["run"] = torch.mps.driver_allocated_memory(), True
            th = threading.Thread(target=poll, daemon=True)
            th.start()
            t = time.perf_counter()
            steps = []
            for u in range(UPDATES):
                r = rng.choice(recs["train"])
                wi = rng.randrange(len(r["windows"]))
                w = r["windows"][wi]
                x = torch.tensor([w["ids"]], device=dev_)
                out = model(x[:, :-1]).float()[0]
                pos = torch.arange(x.shape[1] - 1, device=dev_) >= w["s_index"]
                loss = torch.nn.functional.cross_entropy(out[pos], x[0, 1:][pos])
                if not torch.isfinite(loss):
                    raise Stop(f"non-finite loss at update {u}")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                bad = grad_problems(named_lora(lora))
                if bad:
                    raise Stop(f"LoRA grad problem before step at update {u}: {bad[:4]}")
                if any(p.grad is not None for p in model.parameters()):
                    raise Stop(f"base parameter received a grad at update {u}")
                opt.step()
                if not all(bool(torch.isfinite(p).all()) for p in lora.parameters()):
                    raise Stop(f"non-finite LoRA parameter after step at update {u}")
                steps.append({"update": u, "path": r["path"], "window": wi, "loss": loss.item()})
            torch.mps.synchronize()
            peak["run"] = False
            th.join()
            phase("train", t, observed_max_driver_bytes=peak["v"])
            log["train_steps"] = steps
            check("updates_done", len(steps) == UPDATES, updates=len(steps))
            same = unchanged(lora, init_sd)
            check("every_lora_tensor_changed", not same, unchanged=same)
            check("base_grads_none_after_training", all(p.grad is None for p in model.parameters()))
            check("base_sha_unchanged_after_training", base_sha() == base_sha0, sha=base_sha0[:16])
            # reload from the saved file only, compared on a fixed train window
            probe_w = recs["train"][0]["windows"][0]
            xw = torch.tensor([probe_w["ids"]], device=dev_)
            model.eval()
            with torch.no_grad():
                before = model(xw).float()
            sd = lora.state_dict()
            lora_path = os.path.join(out_dir, "lora_u128.pt")
            torch.save(sd, lora_path)
            log["lora_file_sha256"] = file_sha(lora_path)
            with torch.no_grad():
                for p in lora.parameters():
                    p.zero_()
                zeroed = model(xw).float()
            lora.load_state_dict(torch.load(lora_path))
            with torch.no_grad():
                after = model(xw).float()
            d_reload, d_zero = (after - before).abs().max().item(), (zeroed - before).abs().max().item()
            check("reload_from_file_equals_trained", d_reload <= RELOAD_TOLERANCE and d_zero > 0,
                  max_abs_diff=d_reload, tolerance=RELOAD_TOLERANCE, zeroed_differs_by=d_zero,
                  window=f"{recs['train'][0]['path']}#0", file_sha256=log["lora_file_sha256"])
            check("checkpoint_sha_unchanged", file_sha(ckpt) == log["provenance"]["checkpoint_sha256"])
            t = time.perf_counter()
            ad_val, ad_probe = evaluate("val", True), evaluate("probe", True)
            phase("adapter_eval", t)
            log["val"], log["probe"] = compare(base_val, ad_val), compare(base_probe, ad_probe)
            t = time.perf_counter()
            gens = []
            for p in data["prompts"]:
                for s in GEN_SEEDS:
                    for on in (False, True):
                        gens.append({"prompt": p["path"], **generate(p["ids"], s, on, GEN_TOKENS)})
            phase("generation", t)
            with open(os.path.join(out_dir, "generations.json"), "w") as f:
                json.dump(gens, f)
            log["generation"] = [{k: v for k, v in g.items() if k != "ids"} for g in gens]
            vb = sum(g["valid"] for g in gens if not g["adapter"])
            va = sum(g["valid"] for g in gens if g["adapter"])
            crit = {"val_delta_lt_0": log["val"]["delta_recording_mean"] < VAL_MAX_DCE,
                    "probe_delta_le_0.02": log["probe"]["delta_recording_mean"] <= PROBE_MAX_DCE,
                    "adapter_valid_ge_base": va >= vb}
            log["development"] = {"criteria": crit, "valid_base": vb, "valid_adapter": va, "pass": all(crit.values())}
            log["result"] = "train_done"

        else:
            lora.load_state_dict(torch.load(os.path.join(out_dir, "lora_u128.pt")))
            t = time.perf_counter()
            log["test"] = compare(evaluate("test", False), evaluate("test", True))
            phase("test_eval", t)
            log["result"] = "test_done"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except RuntimeError as e:
        log["result"], log["stopped_at"] = "error", f"{type(e).__name__}: {e}"[:500]
    finally:
        signal.alarm(0)
        peak["run"] = False
        with open(report_path, "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(log["result"], log["stopped_at"], flush=True)


if __name__ == "__main__":
    main()
