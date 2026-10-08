#!/usr/bin/env python3
"""Aria chord-condition adapter, single-batch smoke (docs/experiments/ARIA_COND_SMOKE.md).

Frozen local Aria checkpoint on MPS, a zero-initialised Linear(12 -> d_model) whose output is
added to the input of the last transformer block. Real score data: alignment and forward only.
Synthetic fixture: forward, backward, exactly one Adam step, save and reload. Stops at the first
failed check. Run with the Aria venv:
aria_cond_smoke.py <aria repo> <checkpoint> <score pair dir> <out dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
from aria_cond_contract import chroma_per_position  # noqa: E402
from aria_t4b_sim import write_notes  # noqa: E402
from inference.control.chord_label import QUALITIES  # noqa: E402

NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
MAX_TOKENS, WALL_LIMIT_S, LR, SEED = 128, 900, 1e-3, 0


class Stop(Exception):
    pass


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def pressure():
    free = [ln for ln in sh(["memory_pressure", "-Q"]).splitlines() if "free percentage" in ln]
    return {"vm_pressure_level": int(sh(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"]) or -1),
            "free_percent_line": free[0] if free else None}


def rss():
    cur = int(sh(["ps", "-o", "rss=", "-p", str(os.getpid())]) or 0) * 1024
    return {"rss_now_bytes": cur, "rss_max_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def file_sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def chord_pcs(root, quality):
    r = NAMES.index(root)
    return {(r + i) % 12 for i in QUALITIES[quality][0]}


def main() -> None:
    aria_repo, ckpt, score_dir, out_dir = sys.argv[1:5]
    os.makedirs(out_dir, exist_ok=True)
    sys.path.insert(0, aria_repo)
    log = {"phases": {}, "checks": {}, "env": {}, "result": None, "stopped_at": None}

    def phase(name, t0, **extra):
        rec = {"wall_s": round(time.perf_counter() - t0, 3), **rss(), **pressure(), **extra}
        if torch.backends.mps.is_available():
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
        from ariautils.midi import MidiDict
        from ariautils.tokenizer import AbsTokenizer
        from aria.config import load_model_config
        from aria.model import ModelConfig, TransformerLM

        torch.manual_seed(SEED)
        log["env"] = {"torch": torch.__version__, "mps": torch.backends.mps.is_available(),
                      "mps_fallback_env": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
                      "start": pressure(), "checkpoint": ckpt}
        check("mps_fallback_off", not os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"))
        check("mps_available", torch.backends.mps.is_available())
        dev = torch.device("mps")

        t0 = time.perf_counter()
        ckpt_sha_before = file_sha(ckpt)
        phase("checkpoint_hash", t0, sha256=ckpt_sha_before)

        t0 = time.perf_counter()
        tok = AbsTokenizer()
        cfg = ModelConfig(**load_model_config(name="medium"))
        cfg.set_vocab_size(tok.vocab_size)
        model = TransformerLM(cfg)
        model.load_state_dict(load_file(ckpt), strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        n_params = sum(p.numel() for p in model.parameters())
        dtypes = sorted({str(p.dtype) for p in model.parameters()})
        phase("load_cpu", t0, params=n_params, dtypes=dtypes)

        # MPS support: 16-token forward on CPU, then the same model moved to MPS
        probe = torch.tensor([tok.encode([("prefix", "instrument", "piano"), "<S>"] +
                                         [x for p in range(60, 64) for x in (("piano", p, 80), ("onset", 0), ("dur", 500))] +
                                         ["<T>", "<T>"])], dtype=torch.long)
        t0 = time.perf_counter()
        with torch.no_grad():
            cpu_logits = model(probe).float()
        phase("forward16_cpu", t0, tokens=probe.shape[1])
        t0 = time.perf_counter()
        model.to(dev)
        model.model.freqs_cis = None    # rotary table is a cached attribute, not a buffer: rebuilt on MPS (run 1 stopped here)
        phase("move_to_mps", t0)
        t0 = time.perf_counter()
        with torch.no_grad():
            mps_logits = model(probe.to(dev)).float().cpu()
        diff = (mps_logits - cpu_logits).abs().max().item()
        phase("forward16_mps", t0)
        check("mps_forward16", torch.isfinite(mps_logits).all().item(), max_abs_diff_vs_cpu=diff)
        del cpu_logits

        def base_param_sha():
            h = hashlib.sha256()
            for name, p in sorted(model.named_parameters()):
                h.update(name.encode())
                h.update(p.detach().cpu().contiguous().numpy().tobytes())
            return h.hexdigest()

        t0 = time.perf_counter()
        params_sha_before = base_param_sha()
        phase("param_hash_before", t0)

        adapter = torch.nn.Linear(12, cfg.d_model, bias=False).to(dev)
        torch.nn.init.zeros_(adapter.weight)
        state = {"cond": None, "on": False}

        def pre_hook(_mod, args, kwargs):
            if not state["on"]:
                return None
            return (args[0] + adapter(state["cond"]),) + args[1:], kwargs

        model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)

        def run(ids, cond=None):
            state["on"], state["cond"] = cond is not None, cond
            return model(ids)

        def tokens_and_cond(notes, plan_ms, shift_ms=0):
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "x.mid")
                write_notes(notes, path)
                toks = tok.tokenize(MidiDict.from_midi(path))[:MAX_TOKENS]
            plan = [(on - shift_ms, end - shift_ms, pcs) for on, end, pcs in plan_ms]
            cond = torch.tensor([chroma_per_position(toks, plan)], dtype=torch.float32, device=dev)
            return toks, torch.tensor([tok.encode(toks)], dtype=torch.long, device=dev), cond

        # real score data: alignment and forward only
        conv = json.load(open(os.path.join(score_dir, "conversion.json")))
        bpm = conv["tempos"][0]["bpm"]
        check("score_single_tempo", len(conv["tempos"]) == 1, bpm=bpm)
        ms = lambda beat: round(beat * 60000 / bpm)
        notes = [(n["pitch"], ms(n["onset_beat"]) / 1000, ms(n["end_beat"]) / 1000, 80) for n in conv["notes"]]
        plan = [(ms(c["onset"]), ms(c["end"]), chord_pcs(c["root"], c["quality"]))
                for c in conv["chords"] if c["label"] == "chord"]
        shift = round(min(n[1] for n in notes) * 1000)
        t0 = time.perf_counter()
        toks, ids, cond = tokens_and_cond(notes, plan, shift)
        first_onset = next(x for x in toks if isinstance(x, tuple) and x[0] == "onset")
        check("score_alignment", first_onset[1] == 0 and cond.shape[1] == ids.shape[1],
              tokens=ids.shape[1], leading_silence_ms=shift, positions_with_chord=int((cond.sum(-1) > 0).sum()))
        with torch.no_grad():
            base = run(ids).float()
            hooked = run(ids, cond).float()
        phase("score_forward_dry_run", t0, tokens=ids.shape[1])
        check("score_zero_adapter_identical", (hooked - base).abs().max().item() == 0.0,
              max_abs_diff=(hooked - base).abs().max().item())
        del base, hooked

        # synthetic fixture: Cmaj7 -> Cm7 -> Cmaj7, eighth notes at 120 BPM, one <T> boundary
        cmaj7, cm7 = [60, 64, 67, 71, 64, 67, 71, 72], [60, 63, 67, 70, 63, 67, 70, 72]
        fx_notes = [(p, 0.25 * i + 2.0 * k, 0.25 * i + 2.0 * k + 0.2, 80)
                    for k, arp in enumerate([cmaj7, cm7, cmaj7]) for i, p in enumerate(arp)]
        fx_plan = [(0, 2000, chord_pcs("C", "maj7")), (2000, 4000, chord_pcs("C", "m7")), (4000, 6000, chord_pcs("C", "maj7"))]
        toks, ids, cond = tokens_and_cond(fx_notes, fx_plan)
        check("fixture_shape", ids.shape[1] <= MAX_TOKENS and "<T>" in toks, tokens=ids.shape[1])

        t0 = time.perf_counter()
        with torch.no_grad():
            base = run(ids).float()
            hooked_ng = run(ids, cond).float()
        # run 2 compared a grad-enabled hooked pass with a no_grad base; compare in one autograd mode
        check("fixture_zero_adapter_identical", (hooked_ng - base).abs().max().item() == 0.0,
              max_abs_diff=(hooked_ng - base).abs().max().item(), mode="both no_grad")
        hooked = run(ids, cond)
        log["checks"]["grad_mode_vs_no_grad"] = {"ok": True, "info_only": True,
                                                 "max_abs_logit_diff_zero_adapter": (hooked.float() - hooked_ng).abs().max().item()}
        loss = torch.nn.functional.cross_entropy(hooked[0, :-1].float(), ids[0, 1:])
        phase("fixture_forward", t0, tokens=ids.shape[1])
        check("loss_finite", torch.isfinite(loss).item(), loss=loss.item())

        t0 = time.perf_counter()
        loss.backward()
        phase("fixture_backward", t0)
        g = adapter.weight.grad
        check("adapter_grad", g is not None and torch.isfinite(g).all().item() and g.abs().sum().item() > 0,
              grad_norm=None if g is None else g.norm().item())
        check("base_has_no_grad", all(p.grad is None for p in model.parameters()))

        opt = torch.optim.Adam([adapter.weight], lr=LR)
        base_ids = {id(p) for p in model.parameters()}
        in_opt = [p for grp in opt.param_groups for p in grp["params"]]
        check("optimizer_has_no_base_param", sum(id(p) in base_ids for p in in_opt) == 0 and len(in_opt) == 1,
              optimizer_params=len(in_opt))
        w_before = adapter.weight.detach().clone()
        t0 = time.perf_counter()
        opt.step()
        phase("optimizer_step", t0)
        check("adapter_changed", (adapter.weight.detach() - w_before).abs().max().item() > 0,
              max_abs_change=(adapter.weight.detach() - w_before).abs().max().item())

        # wiring only: after the update a different chord plan changes the output
        swapped = [(on, end, chord_pcs("C", "m7") if pcs == chord_pcs("C", "maj7") else chord_pcs("C", "maj7"))
                   for on, end, pcs in fx_plan]
        _, _, cond_swapped = tokens_and_cond(fx_notes, swapped)
        with torch.no_grad():
            a = run(ids, cond).float()
            b = run(ids, cond_swapped).float()
            loss_after = torch.nn.functional.cross_entropy(a[0, :-1], ids[0, 1:]).item()
        log["checks"]["wiring_condition_changes_output"] = {"ok": True, "info_only": True,
                                                             "max_abs_logit_diff": (a - b).abs().max().item(),
                                                             "loss_before": loss.item(), "loss_after": loss_after}

        t0 = time.perf_counter()
        path = os.path.join(out_dir, "adapter_smoke_only.pt")
        torch.save({"weight": adapter.weight.detach().cpu(), "note": "smoke artifact; do not reuse or deploy"}, path)
        reloaded = torch.nn.Linear(12, cfg.d_model, bias=False).to(dev)
        reloaded.weight.data.copy_(torch.load(path)["weight"].to(dev))
        original = adapter
        adapter = reloaded
        with torch.no_grad():
            c = run(ids, cond).float()
        adapter = original
        phase("save_reload", t0)
        check("reload_logits", (c - a).abs().max().item() <= 1e-6, max_abs_diff=(c - a).abs().max().item(),
              adapter_sha256=file_sha(path))

        t0 = time.perf_counter()
        check("base_params_unchanged", base_param_sha() == params_sha_before, sha256=params_sha_before[:16])
        check("checkpoint_file_unchanged", file_sha(ckpt) == ckpt_sha_before)
        phase("param_hash_after", t0)
        log["result"] = "possible"
    except Stop as e:
        log["result"], log["stopped_at"] = "stopped", str(e)
    except Exception as e:  # OOM, unsupported op, anything else: record and stop
        log["result"], log["stopped_at"] = "stopped", f"{type(e).__name__}: {e}"
    finally:
        signal.alarm(0)
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(log, f, indent=1, ensure_ascii=False, default=str)
        print(json.dumps({"result": log["result"], "stopped_at": log["stopped_at"],
                          "checks": {k: v["ok"] for k, v in log["checks"].items()}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
