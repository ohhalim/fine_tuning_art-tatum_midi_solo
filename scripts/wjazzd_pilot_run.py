#!/usr/bin/env python3
"""WJazzD harmonic-condition pilot: prechecks, 128 updates, held-out evaluation (docs/experiments/WJAZZD_PILOT.md).

--precheck runs everything that does not step an optimizer: zero adapter equals base, mask
positions, condition time axis against the database, adapter save/reload, and one forward/backward
on the longest train chunk for cost. --train additionally runs exactly 128 Adam updates, the test
evaluation and the fixed free generations; it is run only after the user approved the plan.
Run with the Aria venv: wjazzd_pilot_run.py <aria repo> <checkpoint> <wjazzd.db> <data json> <out dir> --precheck|--train
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
from aria_cond_loss_arms import grammar_check  # noqa: E402
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402
from wjazzd_pilot_data import chroma, key, plan_for, row, status  # noqa: E402

UPDATES, LR, SEED, WALL_LIMIT_S, GEN_TOKENS, GEN_SEEDS, GEN_SONGS = 128, 1e-3, 0, 900, 96, (1, 2), 4
DIM = 29


def wrong_rows(rows):
    """Fixed wrong plan: current and next chroma moved up one semitone, flags and delta kept."""
    out = []
    for r in rows:
        out.append(r[11:12] + r[0:11] + r[23:24] + r[12:23] + r[24:])
    return out


def main() -> None:
    aria_repo, ckpt, db, data_path, out_dir, mode = sys.argv[1:7]
    assert mode in ("--precheck", "--train")
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"mode": mode, "phases": {}, "checks": {}, "train_loss": [], "eval": {}, "generation": [], "result": None, "stopped_at": None}

    def phase(name, t0, **extra):
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(), **extra}
        rec["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
        rec["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
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
        chunks = data["chunks"]
        by_split = collections.defaultdict(list)
        for ch in chunks:
            by_split[ch["split"]].append(ch)
        t0 = time.perf_counter()
        ckpt_sha = file_sha(ckpt)
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

        vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
        pitch_ids = collections.defaultdict(list)
        for i, t in enumerate(vocab):
            if isinstance(t, tuple) and t[0] == "piano":
                pitch_ids[t[1]].append(i)
        pitch_index = {p: torch.tensor(v, device=dev) for p, v in pitch_ids.items()}

        def ids_of(ch):
            return torch.tensor([ch["token_ids"]], device=dev)

        # ---- prechecks (no optimizer)
        ch0 = by_split["train"][0]
        with torch.no_grad():
            a = run(ids_of(ch0)).float()
            b = run(ids_of(ch0), ch0["cond"], adapter).float()
        check("zero_adapter_equals_base", (a - b).abs().max().item() == 0.0, max_abs_diff=(a - b).abs().max().item())
        bad_mask = []
        for ch in chunks:
            toks = [vocab[i] for i in ch["token_ids"]]
            m = ch["mask"]
            if len(m) != len(toks) - 1 or not m[ch["first_target"] - 1] or any(m[: ch["first_target"] - 1]) \
                    or any(m[k] for k in range(len(m)) if toks[k + 1] == "<E>") \
                    or not (isinstance(toks[ch["first_target"]], tuple) and toks[ch["first_target"]][0] == "piano") \
                    or len(ch["cond"]) != len(toks) or any(len(r) != DIM for r in ch["cond"]):
                bad_mask.append(ch["melid"])
        check("masks_and_shapes", not bad_mask, chunks=len(chunks), bad=bad_mask[:5])
        con = sqlite3.connect(db)
        rng = random.Random(1)
        axis = []
        for ch in rng.sample(chunks, 3):
            toks = [vocab[i] for i in ch["token_ids"]]
            plan = plan_for(con, ch["melid"], ch["t0"])
            feats = features(toks, plan)
            for i in rng.sample(range(len(toks)), 5):
                t_s = ch["t0"] + feats[i]["t_ms"] / 1000.0
                db_chord = con.execute("select chord from beats where melid=? and chord!='' and onset<=? order by onset desc limit 1",
                                       (ch["melid"], t_s + 1e-9)).fetchone()
                expect = chroma((key(db_chord[0]), status(db_chord[0]))) if db_chord else [0.0] * 12
                axis.append(expect == ch["cond"][i][:12])
        check("condition_time_axis", all(axis), samples=len(axis), matched=sum(axis))
        same = sum(1 for ch in chunks for r, w in zip(ch["cond"], wrong_rows(ch["cond"])) if r[:24] == w[:24] and any(r[:24]))
        log["checks"]["wrong_plan_same_chroma_positions"] = {"ok": True, "info_only": True, "count": same}
        path = os.path.join(out_dir, "adapter_zero_precheck.pt")
        torch.save({"weight": adapter.weight.detach().cpu()}, path)
        re = torch.nn.Linear(DIM, cfg.d_model, bias=False).to(dev)
        re.weight.data.copy_(torch.load(path)["weight"].to(dev))
        with torch.no_grad():
            c1 = run(ids_of(ch0), ch0["cond"], re).float()
        check("save_reload", (c1 - b).abs().max().item() == 0.0)
        longest = max(by_split["train"], key=lambda ch: len(ch["token_ids"]))
        t0 = time.perf_counter()
        logits = run(ids_of(longest), longest["cond"], adapter)
        ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), ids_of(longest)[0, 1:], reduction="none")
        mask = torch.tensor(longest["mask"], device=dev, dtype=torch.float32)
        loss = (ce * mask).sum() / mask.sum()
        loss.backward()
        g = adapter.weight.grad
        phase("forward_backward_longest_chunk", t0, tokens=len(longest["token_ids"]))
        check("cost_probe_grad", torch.isfinite(loss).item() and g is not None and torch.isfinite(g).all().item()
              and g.abs().sum().item() > 0, loss=loss.item(), grad_norm=g.norm().item())
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
                logits = run(ids_of(ch), ch["cond"], adapter)
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
                log["train_loss"].append({"update": step, "melid": m, "loss": loss.item(), "targets": int(mask.sum().item())})
            phase("train_128_updates", t0)
            check("base_has_no_grad_after_training", all(p.grad is None for p in model.parameters()))
            path = os.path.join(out_dir, "adapter_wjazzd_pilot_diagnostic_only.pt")
            torch.save({"weight": adapter.weight.detach().cpu(), "note": "WJazzD pilot diagnostic; not a product model"}, path)
            log["adapter_sha256"] = file_sha(path)

            def evaluate(split):
                res = collections.defaultdict(lambda: collections.defaultdict(list))
                strata_res = collections.defaultdict(lambda: collections.defaultdict(list))
                with torch.no_grad():
                    for ch in by_split[split]:
                        ids = ids_of(ch)
                        conds = {"base": (None, None), "correct": (ch["cond"], adapter),
                                 "wrong": (wrong_rows(ch["cond"]), adapter), "inactive": ([[0.0] * DIM] * len(ch["cond"]), adapter)}
                        mask = torch.tensor(ch["mask"], device=dev, dtype=torch.float32)
                        for name, (rows, ad) in conds.items():
                            lp = torch.log_softmax(run(ids, rows, ad)[0].float(), -1)
                            cont = -(lp[:-1].gather(1, ids[0, 1:, None])[:, 0] * mask).sum().item() / mask.sum().item()
                            pn, jn = [], []
                            for t in ch["pitch_targets"]:
                                i = t["pos"]
                                p = vocab[ch["token_ids"][i]][1]
                                marg = -torch.logsumexp(lp[i - 1, pitch_index[p]], 0).item()
                                joint = -lp[i - 1, ch["token_ids"][i]].item()
                                pn.append(marg)
                                jn.append(joint)
                                strata_res[name][t["stratum"]].append(marg)
                                if t["anticipation"]:
                                    strata_res[name]["anticipation"].append(marg)
                            res[name][ch["melid"]].append({"pitch": sum(pn) / len(pn), "joint": sum(jn) / len(jn), "continuation": cont})
                out = {}
                for name, songs_ in res.items():
                    song_avg = {m: {k: sum(c[k] for c in cs) / len(cs) for k in ("pitch", "joint", "continuation")} for m, cs in songs_.items()}
                    out[name] = {"song_mean": {k: sum(v[k] for v in song_avg.values()) / len(song_avg) for k in ("pitch", "joint", "continuation")},
                                 "per_song": song_avg,
                                 "strata_pitch_nll": {s: {"n": len(v), "mean": sum(v) / len(v)} for s, v in strata_res[name].items()}}
                ps = out["correct"]["per_song"]
                pw = out["wrong"]["per_song"]
                better = sum(1 for m in ps if ps[m]["pitch"] < pw[m]["pitch"])
                n = len(ps)
                out["decision"] = {"correct_lt_wrong_songs": better, "of": n,
                                   "song_mean_correct_lt_wrong": out["correct"]["song_mean"]["pitch"] < out["wrong"]["song_mean"]["pitch"],
                                   "result": ("condition_use_observed" if better >= 7 and out["correct"]["song_mean"]["pitch"] < out["wrong"]["song_mean"]["pitch"]
                                              else "partial" if better >= 5 else "not_observed") if n == 8 else "n_differs"}
                return out

            t0 = time.perf_counter()
            log["eval"]["test"] = evaluate("test")
            log["eval"]["validation_info"] = evaluate("validation")
            phase("evaluation", t0)

            t0 = time.perf_counter()
            test_songs = []
            for ch in by_split["test"]:
                if ch["melid"] not in test_songs:
                    test_songs.append(ch["melid"])
            for m in test_songs[:GEN_SONGS]:
                ch = next(c for c in by_split["test"] if c["melid"] == m)
                plan = plan_for(con, m, ch["t0"])
                for seed in GEN_SEEDS:
                    for name in ("correct", "wrong"):
                        g_ = torch.Generator().manual_seed(seed)
                        ids = torch.tensor([ch["token_ids"][: ch["first_target"]]], device=dev)
                        toks = [vocab[i] for i in ch["token_ids"][: ch["first_target"]]]
                        new = []
                        for _ in range(GEN_TOKENS):
                            rows = [row(f) for f in features(toks, plan)]
                            rows = wrong_rows(rows) if name == "wrong" else rows
                            with torch.no_grad():
                                lg = run(ids, rows, adapter)[0, -1].float().cpu()
                            nxt = torch.multinomial(torch.softmax(lg, -1), 1, generator=g_).item()
                            t = vocab[nxt]
                            new.append(t)
                            toks.append(t)
                            ids = torch.cat([ids, torch.tensor([[nxt]], device=dev)], dim=1)
                            if t == "<E>":
                                break
                        feats = features(toks, plan)
                        pitches, tones = [], []
                        base_len = ch["first_target"]
                        for i in range(base_len, len(toks) - 1):
                            if isinstance(toks[i], tuple) and toks[i][0] == "piano" and isinstance(toks[i + 1], tuple) and toks[i + 1][0] == "onset":
                                t_ms = feats[i + 1]["t_ms"]
                                seg = next((p_ for p_ in plan if p_[0] <= t_ms < p_[1]), None)
                                pc = toks[i][1] % 12
                                pitches.append(toks[i][1])
                                if seg is not None:
                                    tones.append(chroma(seg[2])[pc] == 1.0)
                        longest = 0
                        for per in range(1, 9):
                            for s0 in range(len(pitches)):
                                k = 0
                                while s0 + per + k < len(pitches) and pitches[s0 + k] == pitches[s0 + per + k]:
                                    k += 1
                                longest = max(longest, k + per if k >= per else 0)
                        log["generation"].append({"melid": m, "seed": seed, "plan": name, "tokens": len(new),
                                                  "eos": new[-1] == "<E>", "notes": len(pitches),
                                                  "grammar_errors": len(grammar_check(toks[:base_len], new)),
                                                  "longest_repeat_notes": longest,
                                                  "chord_tone_ratio_descriptive": (sum(tones) / len(tones)) if tones else None})
            phase("generation", t0)
            check("base_params_unchanged", base_param_sha() == params_sha)
            check("checkpoint_file_unchanged", file_sha(ckpt) == ckpt_sha)
        else:
            check("base_params_unchanged", base_param_sha() == params_sha)
        log["result"] = "completed"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except Exception as e:
        log["result"], log["stopped_at"] = "stopped", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        name = "precheck.json" if mode == "--precheck" else "result.json"
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v.get("ok") for k, v in log["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
