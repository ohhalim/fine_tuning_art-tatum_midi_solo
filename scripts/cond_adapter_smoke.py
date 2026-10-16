#!/usr/bin/env python3
"""Memory and time smoke for the minimal condition adapter (issue #1760 step E, docs/experiments/COND_ADAPTER_MIN.md).

No learning result is read. Among the requests of the 12 fixed songs (cond_adapter_split.py; production crop settings,
windows over CAP left out and counted), the longest input (session context + window) gets one batch-1 fp32 forward,
window cross-entropy, backward into the adapter only and one Adam step on MPS. Base parameters are frozen and their
hash is checked before and after. Run with the Aria venv:

  cond_adapter_smoke.py <aria repo> <checkpoint> <#1756 export dir> <split json> <report json>
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import resource
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from aria_mlx_cond import COND_DIM, cond_rows  # noqa: E402
from block_contract import block_start  # noqa: E402
from block_request import payload  # noqa: E402
from crop_contract import CAP, HEADER, session_position_requests, session_requests  # noqa: E402
from lh_role_proxy import Note  # noqa: E402

LR = 1e-3                          # the existing adapter recipe (aria_cond_32step.py), fixed before any run
WALL_LIMIT_S = 900


def pressure():
    out = subprocess.run(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], capture_output=True, text=True).stdout.strip()
    thermal = subprocess.run(["notifyutil", "-g", "com.apple.system.thermalpressurelevel"], capture_output=True, text=True).stdout.split()
    return {"vm_pressure_level": int(out) if out.isdigit() else None, "thermal": int(thermal[-1]) if thermal else None}


def longest_request(export_dir, song_idxs):
    """(song idx, request row, session ids, cond rows, window label range) of the longest allowed input."""
    best, skipped = None, 0
    for idx in song_idxs:
        d = json.load(gzip.open(os.path.join(export_dir, "requests", f"{idx:03d}.json.gz")))
        nj = json.load(open(os.path.join(export_dir, "songs", f"{idx:03d}.notes.json")))
        tokens = [tuple(t) if isinstance(t, list) else t for t in d["tokens"]]
        hum = [Note(nid, p, on, off) for nid, p, on, off in nj["human"]]
        srows = session_requests(tokens, d["t0"])
        for r in srows:
            win = r["win_end"] - r["ctx_end"]
            if win == 0:
                continue
            if win + (r["cont_end"] - r["win_end"]) > CAP:
                skipped += 1
                continue
            n = r["ctx_len"] + win
            if best is None or n > best[0]:
                best = (n, idx, r, tokens, d["token_ids"], hum, d["t0"], srows)
    n, idx, r, tokens, ids, hum, t0, srows = best
    sess = list(ids[:r["win_end"]]) if r["keep_from"] is None else list(ids[:len(HEADER)]) + list(ids[r["keep_from"]:r["win_end"]])
    tags = session_position_requests(tokens, srows, r["k"])[:len(sess) - 1]     # one per input position (the last token is only a label)
    pays = {j: payload(hum, block_start(j, t0), j) for j in set(tags) if j >= 0}
    return {"song": idx, "k": r["k"], "tokens": n, "ctx_len": r["ctx_len"], "window": r["win_end"] - r["ctx_end"],
            "ids": sess, "rows": cond_rows(tags, pays), "skipped_over_cap": skipped}


def main() -> None:
    aria_repo, ckpt, export_dir, split_path, report_path = sys.argv[1:6]
    sys.path.insert(0, aria_repo)
    import torch
    from safetensors.torch import load_file
    from aria.config import load_model_config
    from aria.model import ModelConfig, TransformerLM
    from ariautils.tokenizer import AbsTokenizer

    split = json.load(open(split_path))
    songs = [s["idx"] for s in split["train"] + split["dev"]]
    ex = longest_request(export_dir, songs)
    report = {"issue": 1760, "step": "E", "training_result_read": False, "params": {"lr": LR, "dtype": "float32", "batch": 1},
              "example": {k: ex[k] for k in ("song", "k", "tokens", "ctx_len", "window", "skipped_over_cap")}, "phases": {}}
    dev = torch.device("mps")
    t0 = time.perf_counter()
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

    def base_sha():
        h = hashlib.sha256()
        for name, p in sorted(model.named_parameters()):
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    sha_before = base_sha()
    report["phases"]["load"] = {"s": round(time.perf_counter() - t0, 2), **pressure()}
    adapter = torch.nn.Linear(COND_DIM, cfg.d_model, bias=False).to(dev)
    torch.nn.init.zeros_(adapter.weight)
    cond = torch.tensor([ex["rows"]], dtype=torch.float32, device=dev)
    assert cond.shape[1] == len(ex["ids"]) - 1, (cond.shape, len(ex["ids"]))

    def pre_hook(_mod, args, kwargs):
        return (args[0] + adapter(cond),) + args[1:], kwargs

    model.model.encode_layers[-1].register_forward_pre_hook(pre_hook, with_kwargs=True)
    opt = torch.optim.Adam([adapter.weight], lr=LR)
    ids = torch.tensor([ex["ids"]], dtype=torch.long, device=dev)
    first = ex["ctx_len"]                              # labels: the window tokens ids[first:]
    t1 = time.perf_counter()
    logits = model(ids[:, :-1])
    loss = torch.nn.functional.cross_entropy(logits[0, first - 1:].float(), ids[0, first:])
    loss.backward()
    opt.step()
    torch.mps.synchronize()
    report["phases"]["forward_backward_step"] = {
        "s": round(time.perf_counter() - t1, 2), "mps_current_allocated_gib": round(torch.mps.current_allocated_memory() / 2 ** 30, 2),
        "mps_driver_allocated_gib": round(torch.mps.driver_allocated_memory() / 2 ** 30, 2),
        "rss_max_gib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 30, 2), **pressure()}
    report["checks"] = {"loss_finite": bool(torch.isfinite(loss).item()), "adapter_grad_nonzero": bool(adapter.weight.abs().sum().item() > 0),
                        "base_unchanged": base_sha() == sha_before,
                        "only_adapter_in_optimizer": all(p is adapter.weight for g in opt.param_groups for p in g["params"])}
    report["pass"] = all(report["checks"].values()) and (time.perf_counter() - t0) < WALL_LIMIT_S
    report["run"] = {"git_head": subprocess.run(["git", "-C", HERE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
                     "torch": torch.__version__, "seconds": round(time.perf_counter() - t0, 1)}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print(json.dumps({k: report[k] for k in ("example", "phases", "checks", "pass")}))


if __name__ == "__main__":
    main()
