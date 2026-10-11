#!/usr/bin/env python3
"""Aria LoRA one-step resource smoke (issue #1724). No training result, no quality claim.

One arm per process: --ckpt on|off. Both arms run the model in train mode (dropout is 0 by
structure, recorded) and differ only in the gradient-checkpoint flag. Both arms keep the same hook
that marks the embedding output as requiring grad, so the reentrant checkpoint can route gradients
to LoRA while the embedding weights themselves stay frozen.

Fixed before the run (issue #1724): Tatum train98 song sorted first, AbsTokenizer unchanged, global
random seeded before tokenising, first 1024 ids, batch 1, F32, MPS. LoRA r16, alpha 16 on
mixed_qkv and att_proj_linear of all 16 blocks (32 pairs), A kaiming-uniform (seed 0), B zeros,
Adam lr 1e-4, exactly one step. Logit checks run in eval/no_grad on the same device and dtype,
tolerance 0.

  aria_lora_smoke.py <aria repo> <checkpoint> <midi file> <out dir> --ckpt on|off
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import signal
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_cond_smoke import Stop, file_sha, pressure, rss  # noqa: E402

SEQ, RANK, ALPHA, LR, SEED, WALL_LIMIT_S = 1024, 16, 16, 1e-4, 0, 900
TARGETS = ("mixed_qkv", "att_proj_linear")
TOLERANCE = 0.0


def main() -> None:
    aria_repo, ckpt, midi_path, out_dir = sys.argv[1:5]
    assert sys.argv[5] == "--ckpt" and sys.argv[6] in ("on", "off")
    use_ckpt = sys.argv[6] == "on"
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"arm": f"ckpt_{sys.argv[6]}", "fixed": {"seq": SEQ, "rank": RANK, "alpha": ALPHA, "lr": LR, "seed": SEED,
                                                   "targets": TARGETS, "tolerance": TOLERANCE, "dtype": "float32",
                                                   "mode": "train (step), eval/no_grad (logit checks)"},
           "phases": {}, "checks": {}, "layers": [], "result": None, "stopped_at": None}
    import torch

    peak = {"v": 0, "run": False}

    def poll():
        while peak["run"]:
            peak["v"] = max(peak["v"], torch.mps.driver_allocated_memory())
            time.sleep(0.002)

    def sync():
        torch.mps.synchronize()

    def phase(name, t0, **extra):
        sync()
        rec = {"wall_s": round(time.perf_counter() - t0, 4), **rss(), **pressure(), **extra,
               "mps_current_allocated_bytes": torch.mps.current_allocated_memory(),
               "mps_driver_allocated_bytes": torch.mps.driver_allocated_memory()}
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
        from safetensors.torch import load_file
        from ariautils.midi import MidiDict
        from ariautils.tokenizer import AbsTokenizer
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        dev = torch.device("mps")
        t0 = time.perf_counter()
        tok = AbsTokenizer()
        random.seed(SEED)                                   # AbsTokenizer inserts <D> through global random
        ids = tok.encode(tok.tokenize(MidiDict.from_midi(midi_path)))
        check("window_full_length", len(ids) >= SEQ, ids_total=len(ids))
        ids = torch.tensor([ids[:SEQ]], device=dev)
        log["input"] = {"midi": midi_path, "midi_sha256": file_sha(midi_path), "ids_sha256":
                        hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest(), "tokens": SEQ}
        cfg_dict = load_model_config(name="medium")
        cfg = ModelConfig(**cfg_dict)
        cfg.set_vocab_size(tok.vocab_size)
        cfg.grad_checkpoint = use_ckpt
        log["provenance"] = {"checkpoint": ckpt, "checkpoint_sha256": file_sha(ckpt), "config": cfg_dict,
                             "vocab_size": tok.vocab_size, "tokenizer": "ariautils AbsTokenizer",
                             "torch": torch.__version__}
        model = TransformerLM(cfg)
        model.load_state_dict(load_file(ckpt), strict=True)
        for p in model.parameters():
            p.requires_grad_(False)
        model.to(dev)
        model.model.freqs_cis = None
        drop = sorted({b.resid_dropout for b in model.model.encode_layers})
        log["dropout"] = {"resid_dropout_values": drop, "attention_dropout": "none (SDPA called without dropout_p)"}
        check("dropout_zero_by_structure", drop == [0.0])
        phase("load", t0)

        def base_sha():
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        sha_before = base_sha()
        # LoRA lives outside the model; forward hooks add the low-rank path to each target Linear.
        gen = torch.Generator().manual_seed(SEED)
        lora = torch.nn.ModuleDict()
        for i, block in enumerate(model.model.encode_layers):
            for name in TARGETS:
                lin = getattr(block, name)
                a = torch.empty(RANK, lin.in_features)
                bound = 1 / math.sqrt(lin.in_features)      # kaiming_uniform(a=sqrt(5)) bound for fan_in
                a.uniform_(-bound, bound, generator=gen)
                pair = torch.nn.ParameterDict({"A": torch.nn.Parameter(a), "B": torch.nn.Parameter(torch.zeros(lin.out_features, RANK))})
                lora[f"{i}_{name}"] = pair
        lora.to(dev)
        state = {"on": True}

        def make_hook(key):
            def hook(_mod, inp, out):
                if not state["on"]:
                    return out
                pr = lora[key]
                return out + (inp[0] @ pr["A"].t() @ pr["B"].t()) * (ALPHA / RANK)
            return hook

        for i, block in enumerate(model.model.encode_layers):
            for name in TARGETS:
                getattr(block, name).register_forward_hook(make_hook(f"{i}_{name}"))
        model.model.tok_embeddings.register_forward_hook(lambda _m, _i, out: out.requires_grad_(True))
        n_lora = sum(p.numel() for p in lora.parameters())
        check("lora_param_count", n_lora == 2_359_296 and len(lora) == 32, params=n_lora, pairs=len(lora))
        init_state = {k: v.detach().cpu().clone() for k, v in lora.state_dict().items()}
        log["lora_init_sha256"] = hashlib.sha256(b"".join(init_state[k].numpy().tobytes() for k in sorted(init_state))).hexdigest()

        def logits(on):
            model.eval()
            state["on"] = on
            with torch.no_grad():
                out = model(ids).float()
            state["on"] = True
            return out

        base_logits = logits(False)
        d = (logits(True) - base_logits).abs().max().item()
        check("init_equals_base", d <= TOLERANCE, max_abs_diff=d)

        opt = torch.optim.Adam(lora.parameters(), lr=LR)
        opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
        check("optimizer_only_lora", opt_ids == {id(p) for p in lora.parameters()}
              and not any(id(p) in opt_ids for p in model.parameters()), optimizer_params=len(opt_ids))

        # ---- the one measured step, train mode
        model.train()
        torch.manual_seed(SEED)
        sync()
        peak["v"], peak["run"] = torch.mps.driver_allocated_memory(), True
        th = threading.Thread(target=poll, daemon=True)
        th.start()
        t = time.perf_counter()
        out = model(ids[:, :-1])
        loss = torch.nn.functional.cross_entropy(out.float().reshape(-1, out.shape[-1]), ids[:, 1:].reshape(-1))
        sync()
        t_fwd = time.perf_counter() - t
        t = time.perf_counter()
        loss.backward()
        sync()
        t_bwd = time.perf_counter() - t
        peak_fb = peak["v"]
        grads = {k: {"A": lora[k]["A"].grad, "B": lora[k]["B"].grad} for k in lora}
        for k in lora:
            ga, gb = grads[k]["A"], grads[k]["B"]
            rec = {"pair": k,
                   "B_grad_none": gb is None, "A_grad_none": ga is None,
                   "B_grad_norm": None if gb is None else gb.norm().item(),
                   "A_grad_norm": None if ga is None else ga.norm().item(),
                   "B_grad_finite": None if gb is None else bool(torch.isfinite(gb).all().item())}
            log["layers"].append(rec)
        check("every_B_grad_finite_nonzero", all(r["B_grad_finite"] and r["B_grad_norm"] > 0 for r in log["layers"]),
              failing=[r["pair"] for r in log["layers"] if not (r["B_grad_finite"] and (r["B_grad_norm"] or 0) > 0)])
        check("every_A_grad_zero_at_B0", all(r["A_grad_norm"] == 0.0 for r in log["layers"]))
        check("base_grads_none", all(p.grad is None for p in model.parameters()))
        t = time.perf_counter()
        opt.step()
        sync()
        t_opt = time.perf_counter() - t
        peak["run"] = False
        th.join()
        log["step"] = {"loss": loss.item(), "forward_s": round(t_fwd, 4), "backward_s": round(t_bwd, 4),
                       "optimizer_s": round(t_opt, 4), "observed_max_driver_bytes_forward_backward": peak_fb,
                       "observed_max_driver_bytes_step": peak["v"],
                       "memory_note": "polling every 2 ms of driver_allocated_memory; observed maximum, not a true peak"}
        phase("after_step", t)
        changed = [k for k in lora if not torch.equal(lora[k]["B"].detach().cpu(), init_state[f"{k}.B"])]
        check("lora_B_changed_every_pair", len(changed) == 32, changed=len(changed))
        check("base_hash_unchanged", base_sha() == sha_before, sha=sha_before[:16])

        # ---- save LoRA, rebuild it from the file alone on the same base, compare logits
        after = logits(True)
        d_base = (logits(False) - base_logits).abs().max().item()
        check("base_logits_unchanged", d_base <= TOLERANCE, max_abs_diff=d_base)
        path = os.path.join(out_dir, "lora_step1_smoke_only.pt")
        torch.save({k: v.detach().cpu() for k, v in lora.state_dict().items()}, path)
        with torch.no_grad():
            for p in lora.parameters():
                p.zero_()
        lora.load_state_dict(torch.load(path))
        d_reload = (logits(True) - after).abs().max().item()
        check("reload_equals_saved", d_reload <= TOLERANCE, max_abs_diff=d_reload, file_sha256=file_sha(path))
        log["result"] = "pass"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except RuntimeError as e:
        log["result"], log["stopped_at"] = "error", f"{type(e).__name__}: {e}"[:500]
    finally:
        signal.alarm(0)
        peak["run"] = False
        with open(os.path.join(out_dir, "report.json"), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(log["result"], log["stopped_at"], json.dumps(log.get("step")), flush=True)


if __name__ == "__main__":
    main()
