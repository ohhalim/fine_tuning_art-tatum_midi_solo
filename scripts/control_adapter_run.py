#!/usr/bin/env python3
"""Same-update, approximate-token-budget control adapter vs the Mehldau adapter (issue #1733,
docs/experiments/CONTROL_ADAPTER_DESIGN.md).

Arms: M = Mehldau train split of #1726, C = C-pool (control_pool_data.py). Training seeds {0, 1, 2}: the seed sets
the LoRA A init and the random.Random(seed) recording-then-window schedule, exactly as the #1726 runner. M0 is the
existing #1726 adapter (not retrained); M1, M2, C0, C1, C2 are new. All are reported; no seed is picked.
Evaluation: the #1726 val (12 recordings, 10 families) and probe (20) windows, per-recording continuation CE.

Modes (one per process):
  --schedule        no model. Window schedules and loss-target counts for every arm/seed, M0 against the #1726
                    train_steps, C/M token ratio per seed within +-5 % (else stop: redesign before running),
                    val families, probe duplicate check (titles + interval 16-gram containment >= 0.2).
  --base            base and M0 on val/probe; per-recording CE must equal the #1726 report (|diff| <= 1e-5).
  --train ARM SEED  one adapter: checked training on the fixed schedule, reload-from-file check, val/probe eval.
  --analyze         family-weighted D_val, D_probe, S per seed, registered operating criteria, full tables.

  control_adapter_run.py <aria repo> <checkpoint> <mehldau data> <mehldau manifest> <control data> <control manifest>
                         <1726 run dir> <out dir> <report dir> --schedule|--base|--train ARM SEED|--analyze
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import random
import signal
import statistics
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402
from aria_lora import AriaLoRA, grad_problems, input_mismatches, named_lora, unchanged  # noqa: E402

RANK, ALPHA, LR, UPDATES, SEEDS = 16, 16, 1e-4, 128, (0, 1, 2)
TOKEN_TOLERANCE, BASE_TOLERANCE, RELOAD_TOLERANCE = 0.05, 1e-5, 0.0
WALL = {"--schedule": 600, "--base": 900, "--train": 1200, "--analyze": 300}
# registered operating criteria (Astra review of #1734); not a significance test
CRIT = {"mean_D_val_max": -0.03, "mean_S_max": -0.02, "abs_mean_D_probe_max": 0.02, "families_M_better_min": 7}
CAT_NAMES = ["pitch", "onset", "dur", "time", "other"]


def schedule(train_recs, seed):
    rng = random.Random(seed)
    out = []
    for _ in range(UPDATES):
        r = rng.choice(train_recs)
        wi = rng.randrange(len(r["windows"]))
        w = r["windows"][wi]
        out.append({"path": r["path"], "window": wi, "loss_targets": len(w["ids"]) - 1 - w["s_index"]})
    return out


def sha_json(x):
    return hashlib.sha256(json.dumps(x, sort_keys=True).encode()).hexdigest()


def main() -> None:
    (aria_repo, ckpt, m_data_path, m_manifest_path, c_data_path, c_manifest_path,
     run1726, out_dir, report_dir, mode) = sys.argv[1:11]
    assert mode in WALL
    arm = seed = None
    if mode == "--train":
        arm, seed = sys.argv[11], int(sys.argv[12])
        assert arm in ("M", "C") and seed in SEEDS and not (arm == "M" and seed == 0), "M0 is the existing #1726 adapter"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    name = {"--schedule": "schedule.json", "--base": "eval_base_M0.json", "--analyze": "analysis.json"}.get(mode) or f"eval_{arm}{seed}.json"
    report_path = os.path.join(report_dir, name)
    log = {"mode": mode, "arm": arm, "seed": seed, "checks": {}, "phases": {}, "result": None, "stopped_at": None}

    def check(nm, ok, **detail):
        log["checks"][nm] = {"ok": bool(ok), **detail}
        if not ok:
            raise Stop(f"check failed: {nm}")

    def on_alarm(*_):
        raise Stop(f"wall limit {WALL[mode]} s")

    signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(WALL[mode])
    try:
        t0 = time.perf_counter()
        r1726 = json.load(open(os.path.join(run1726, "train_report.json")))
        mm, cm = json.load(open(m_manifest_path)), json.load(open(c_manifest_path))
        inputs = {"checkpoint_sha256": file_sha(ckpt), "mehldau_data_sha256": file_sha(m_data_path),
                  "mehldau_manifest_sha256": file_sha(m_manifest_path), "control_data_sha256": file_sha(c_data_path),
                  "control_manifest_sha256": file_sha(c_manifest_path),
                  "m0_lora_sha256": file_sha(os.path.join(run1726, "lora_u128.pt"))}
        rec = {"checkpoint_sha256": r1726["provenance"]["checkpoint_sha256"], "mehldau_data_sha256": r1726["provenance"]["data_sha256"],
               "mehldau_manifest_sha256": r1726["provenance"]["manifest_sha256"], "control_data_sha256": cm["data_sha256"],
               "control_manifest_sha256": inputs["control_manifest_sha256"], "m0_lora_sha256": r1726["lora_file_sha256"]}
        check("inputs_match_1726_and_control_manifest", not input_mismatches(rec, inputs), mismatches=input_mismatches(rec, inputs))
        log["inputs"] = inputs
        meta = {r["path"]: r for r in mm["recordings"]}
        m_data, c_data = json.load(open(m_data_path)), json.load(open(c_data_path))
        recs = collections.defaultdict(list)
        for r in m_data["recordings"]:
            recs[r["split"]].append(r)
        train = {"M": recs["train"], "C": c_data["recordings"]}

        if mode == "--schedule":
            sched = {f"{a}{s}": schedule(train[a], s) for a in ("M", "C") for s in SEEDS}
            check("m0_schedule_equals_1726_run", [(x["path"], x["window"]) for x in sched["M0"]] ==
                  [(x["path"], x["window"]) for x in r1726["train_steps"]])
            tokens = {k: sum(x["loss_targets"] for x in v) for k, v in sched.items()}
            ratios = {s: tokens[f"C{s}"] / tokens[f"M{s}"] for s in SEEDS}
            log["schedules"] = {k: {"sha256": sha_json(v), "loss_targets": tokens[k], "distinct_recordings": len({x["path"] for x in v}),
                                    "steps": v} for k, v in sched.items()}
            log["token_ratio_C_over_M"] = ratios
            # families
            val_fam = collections.defaultdict(list)
            for r in recs["val"]:
                val_fam[meta[r["path"]]["group"]].append(r["path"])
            log["val_families"] = {g: ps for g, ps in val_fam.items()}
            from ariautils.midi import MidiDict
            from mehldau_aria_data import MERGE_AT, PIJAMA, containment, interval_ngrams
            probe = [meta[r["path"]] for r in recs["probe"]]
            grams = {p["path"]: interval_ngrams(MidiDict.from_midi(str(PIJAMA / "midi" / p["path"])).note_msgs) for p in probe}
            dup_titles = [t for t, n in collections.Counter(p["title"] for p in probe).items() if n > 1]
            pairs = [(a["path"], b["path"], round(containment(grams[a["path"]], grams[b["path"]]), 4))
                     for i, a in enumerate(probe) for b in probe[i + 1:]]
            close = [x for x in pairs if x[2] >= MERGE_AT]
            log["probe_family_check"] = {"recordings": len(probe), "duplicate_titles": dup_titles,
                                         "max_pair_containment": max(x[2] for x in pairs), "pairs_at_or_above_threshold": close}
            check("probe_has_no_duplicate_family", not dup_titles and not close)
            check("val_has_10_families_12_recordings", len(val_fam) == 10 and sum(len(v) for v in val_fam.values()) == 12)
            check("token_ratio_within_tolerance", all(abs(x - 1) <= TOKEN_TOLERANCE for x in ratios.values()),
                  ratios=ratios, tolerance=TOKEN_TOLERANCE)
            log["result"] = "schedule_done"
            return

        import torch
        from safetensors.torch import load_file
        from ariautils.tokenizer import AbsTokenizer
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        sched_rep = json.load(open(os.path.join(report_dir, "schedule.json")))
        if mode != "--analyze":
            check("schedule_done_on_same_inputs", sched_rep.get("result") == "schedule_done" and sched_rep["inputs"] == inputs)

        if mode == "--analyze":
            ev = {}
            for k in ["base_M0"] + [f"{a}{s}" for a in ("M", "C") for s in SEEDS if (a, s) != ("M", 0)]:
                e = json.load(open(os.path.join(report_dir, f"eval_{k}.json")))
                check(f"eval_{k}_done", e.get("result") in ("base_done", "train_done") and e["inputs"] == inputs)
                ev[k] = e
            rows = {"base": ev["base_M0"]["eval"]["base"], "M0": ev["base_M0"]["eval"]["M0"]}
            for a in ("M", "C"):
                for s in SEEDS:
                    if (a, s) != ("M", 0):
                        rows[f"{a}{s}"] = ev[f"{a}{s}"]["eval"]
            fam = {split: collections.defaultdict(list) for split in ("val", "probe")}
            for split in ("val", "probe"):
                for r in rows["base"][split]:
                    fam[split][meta[r["path"]]["group"] if split == "val" else r["path"]].append(r["path"])

            def per_rec(k, split):
                return {r["path"]: r["mean_ce"] for r in rows[k][split]}

            base_ce = {sp: per_rec("base", sp) for sp in ("val", "probe")}

            def delta(k, split):
                ce = per_rec(k, split)
                return {p: ce[p] - base_ce[split][p] for p in ce}

            def fam_mean(d, split):
                return statistics.mean(statistics.mean(d[p] for p in ps) for ps in fam[split].values())

            per_seed, fam_diff = {}, collections.defaultdict(list)
            for s in SEEDS:
                out = {}
                for split in ("val", "probe"):
                    dm, dc = delta(f"M{s}", split), delta(f"C{s}", split)
                    out[split] = {"dM_family": fam_mean(dm, split), "dC_family": fam_mean(dc, split),
                                  "dM_recording": statistics.mean(dm.values()), "dC_recording": statistics.mean(dc.values()),
                                  "recordings_M_better": sum(dm[p] < dc[p] for p in dm), "recordings": len(dm)}
                    out[split]["D_family"] = out[split]["dM_family"] - out[split]["dC_family"]
                    out[split]["D_recording"] = out[split]["dM_recording"] - out[split]["dC_recording"]
                    if split == "val":
                        for g, ps in fam["val"].items():
                            fam_diff[g].append(statistics.mean(dm[p] - dc[p] for p in ps))
                out["S_family"] = out["val"]["D_family"] - out["probe"]["D_family"]
                out["S_recording"] = out["val"]["D_recording"] - out["probe"]["D_recording"]
                per_seed[s] = out
            mean_d_val = statistics.mean(per_seed[s]["val"]["D_family"] for s in SEEDS)
            mean_d_probe = statistics.mean(per_seed[s]["probe"]["D_family"] for s in SEEDS)
            mean_s = statistics.mean(per_seed[s]["S_family"] for s in SEEDS)
            fam_seed_mean = {g: statistics.mean(v) for g, v in fam_diff.items()}
            fam_better = sum(v < 0 for v in fam_seed_mean.values())
            crit = {"mean_D_val": mean_d_val <= CRIT["mean_D_val_max"], "mean_S": mean_s <= CRIT["mean_S_max"],
                    "abs_mean_D_probe": abs(mean_d_probe) <= CRIT["abs_mean_D_probe_max"],
                    "each_seed_D_val_negative": all(per_seed[s]["val"]["D_family"] < 0 for s in SEEDS),
                    "each_seed_S_negative": all(per_seed[s]["S_family"] < 0 for s in SEEDS),
                    "families_M_better": fam_better >= CRIT["families_M_better_min"]}
            # explanatory only: token-weighted category deltas
            cats = {}
            for k in [f"{a}{s}" for a in ("M", "C") for s in SEEDS]:
                cats[k] = {}
                for split in ("val", "probe"):
                    def pooled(rs):
                        n = [sum(r["cat_n"][i] for r in rs) for i in range(5)]
                        return {CAT_NAMES[i]: sum(r["cat_ce"][i] for r in rs) / n[i] for i in range(5) if n[i]} | \
                               {"pitch_marginal": sum(r["pitch_marg"] for r in rs) / n[0],
                                "velocity_given_pitch": (sum(r["pitch_tok"] for r in rs) - sum(r["pitch_marg"] for r in rs)) / n[0]}
                    a_, b_ = pooled(rows[k][split]), pooled(rows["base"][split])
                    cats[k][split] = {c: a_[c] - b_[c] for c in a_}
            log["per_seed"] = per_seed
            log["family_seed_mean_M_minus_C"] = fam_seed_mean
            log["summary"] = {"mean_D_val_family": mean_d_val, "mean_D_probe_family": mean_d_probe, "mean_S_family": mean_s,
                              "families_M_better": fam_better, "families": len(fam_seed_mean),
                              "D_val_range": [min(per_seed[s]["val"]["D_family"] for s in SEEDS), max(per_seed[s]["val"]["D_family"] for s in SEEDS)],
                              "S_range": [min(per_seed[s]["S_family"] for s in SEEDS), max(per_seed[s]["S_family"] for s in SEEDS)]}
            log["criteria"] = crit
            log["verdict"] = "registered_specificity_signal_met" if all(crit.values()) else "not_met_or_unclear"
            log["category_deltas_explanatory"] = cats
            log["token_budget"] = {"loss_targets": {k: v["loss_targets"] for k, v in sched_rep["schedules"].items()},
                                   "ratio_C_over_M": sched_rep["token_ratio_C_over_M"]}
            log["result"] = "analysis_done"
            return

        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        dev_ = torch.device("mps")
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
        lora = AriaLoRA(model, rank=RANK, alpha=ALPHA, seed=seed if seed is not None else 0, device=dev_)
        vocab = [tok.decode([i])[0] for i in range(tok.vocab_size)]
        cat, pitch_of = [], torch.full((tok.vocab_size,), -1, dtype=torch.long)
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
        cat_id = torch.tensor([CAT_NAMES.index(c) for c in cat], device=dev_)
        pmat = torch.zeros(tok.vocab_size, 128, device=dev_)
        for i in range(tok.vocab_size):
            if pitch_of[i] >= 0:
                pmat[i, pitch_of[i]] = 1.0
        pitch_of = pitch_of.to(dev_)

        def base_sha():
            h = hashlib.sha256()
            for nm, p in sorted(model.named_parameters()):
                h.update(nm.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        base_sha0 = base_sha()
        log["phases"]["load"] = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure()}

        def window_ce(ids, s_index):      # identical to the #1726 runner
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

        def evaluate(split, on):          # identical to the #1726 runner
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
                out.append({"path": r["path"], "windows": n, **acc, "mean_ce": acc["ce"] / acc["tokens"]})
            lora.on = True
            return out

        if mode == "--base":
            t = time.perf_counter()
            base = {"val": evaluate("val", False), "probe": evaluate("probe", False)}
            lora.load_state_dict(torch.load(os.path.join(run1726, "lora_u128.pt")))
            m0 = {"val": evaluate("val", True), "probe": evaluate("probe", True)}
            log["phases"]["eval"] = {"wall_s": round(time.perf_counter() - t, 3), **rss(), **pressure()}
            diffs = []
            for split in ("val", "probe"):
                ref = {x["path"]: x for x in r1726[split]["per_recording_delta"]}
                for b, m in zip(base[split], m0[split]):
                    diffs.append(abs(b["mean_ce"] - ref[b["path"]]["base"]))
                    diffs.append(abs(m["mean_ce"] - (ref[m["path"]]["base"] + ref[m["path"]]["delta"])))
            check("base_and_m0_reproduce_1726", max(diffs) <= BASE_TOLERANCE, max_abs_diff=max(diffs), tolerance=BASE_TOLERANCE)
            log["eval"] = {"base": base, "M0": m0}
            log["result"] = "base_done"
            return

        # --train ARM SEED
        steps_fixed = sched_rep["schedules"][f"{arm}{seed}"]["steps"]
        peak = {"v": 0, "run": False}

        def poll():
            while peak["run"]:
                peak["v"] = max(peak["v"], torch.mps.driver_allocated_memory())
                time.sleep(0.005)

        torch.manual_seed(seed)
        rng = random.Random(seed)
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
            r = rng.choice(train[arm])
            wi = rng.randrange(len(r["windows"]))
            if (r["path"], wi) != (steps_fixed[u]["path"], steps_fixed[u]["window"]):
                raise Stop(f"schedule mismatch at update {u}")
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
        log["phases"]["train"] = {"wall_s": round(time.perf_counter() - t, 3), "observed_max_driver_bytes": peak["v"], **rss(), **pressure()}
        log["train_steps"] = steps
        check("updates_done_on_fixed_schedule", len(steps) == UPDATES)
        same = unchanged(lora, init_sd)
        check("every_lora_tensor_changed", not same, unchanged=same)
        check("base_grads_none_after_training", all(p.grad is None for p in model.parameters()))
        check("base_sha_unchanged_after_training", base_sha() == base_sha0, sha=base_sha0[:16])
        probe_w = train[arm][0]["windows"][0]
        xw = torch.tensor([probe_w["ids"]], device=dev_)
        model.eval()
        with torch.no_grad():
            before = model(xw).float()
        lora_path = os.path.join(out_dir, f"lora_{arm}{seed}.pt")
        torch.save(lora.state_dict(), lora_path)
        log["lora_file_sha256"] = file_sha(lora_path)
        with torch.no_grad():
            for p in lora.parameters():
                p.zero_()
            zeroed = model(xw).float()
        lora.load_state_dict(torch.load(lora_path))
        with torch.no_grad():
            after = model(xw).float()
        d_reload, d_zero = (after - before).abs().max().item(), (zeroed - before).abs().max().item()
        check("reload_from_file_equals_trained", d_reload <= RELOAD_TOLERANCE and d_zero > 0, max_abs_diff=d_reload, zeroed_differs_by=d_zero)
        check("checkpoint_sha_unchanged", file_sha(ckpt) == inputs["checkpoint_sha256"])
        t = time.perf_counter()
        log["eval"] = {"val": evaluate("val", True), "probe": evaluate("probe", True)}
        log["phases"]["eval"] = {"wall_s": round(time.perf_counter() - t, 3), **rss(), **pressure()}
        log["result"] = "train_done"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except RuntimeError as e:
        log["result"], log["stopped_at"] = "error", f"{type(e).__name__}: {e}"[:500]
    finally:
        signal.alarm(0)
        with open(report_path, "w") as f:
            json.dump(log, f, ensure_ascii=False, indent=1)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v["ok"] for k, v in log["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
