#!/usr/bin/env python3
"""Merge a LoRA-only snapshot into its base checkpoint as a runtime full checkpoint.

The runtime and ``load_model_with_lora`` want a full ``checkpoint_*.pt``. This
writes base weights + the snapshot's LoRA tensors in train_qlora's payload
format, then reloads it through ``load_model_with_lora`` and checks the logits
match the in-memory model exactly. Refuses to overwrite.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "scripts"))


def sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--snapshot", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--note", default="")
    ap.add_argument("--lora-targets", default=None,
                    help="optional check; targets are inferred from the snapshot")
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error(f"refusing to overwrite {args.output}")

    import torch
    from utilities.device import use_cuda
    use_cuda(False)
    from scripts.generate import load_model_with_lora
    from scripts.train_qlora import (build_lora_model_from_state, checkpoint_model_config,
                                     checkpoint_payload_state_dict, load_lora_snapshot,
                                     lora_targets_in_state_dict)

    payload = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = checkpoint_model_config(payload)
    lora = torch.load(args.snapshot, map_location="cpu")
    if not lora or not all("lora_" in k for k in lora):
        ap.error("snapshot must contain only lora_ tensors")
    snapshot_targets = lora_targets_in_state_dict(lora)
    if args.lora_targets is not None:
        declared = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        if sorted(declared) != sorted(snapshot_targets):
            ap.error(f"--lora-targets {declared} does not match snapshot targets {snapshot_targets}")
    model, targets = build_lora_model_from_state(
        cfg, checkpoint_payload_state_dict(payload), extra_targets=snapshot_targets)
    try:
        load_lora_snapshot(model, lora)
    except ValueError as exc:
        ap.error(str(exc))
    cfg = {**cfg, "lora_targets": targets}
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_config": cfg,
        "training_config": {"training_mode": "adapter_snapshot_export",
                            "base_checkpoint": str(args.base), "base_sha1": sha1(args.base),
                            "snapshot": str(args.snapshot), "snapshot_sha1": sha1(args.snapshot),
                            "note": args.note},
        "model_state_dict": model.state_dict(),
    }, args.output)

    x = torch.randint(0, 388, (1, min(256, int(cfg["max_sequence"]))))
    with torch.no_grad():
        expected = model(x)
        loaded = load_model_with_lora(lora_path=str(args.output.parent),
                                      checkpoint_path=str(args.output),
                                      prefer_full_checkpoint=True)
        actual = loaded(x)
    identical = bool(torch.equal(expected, actual))
    print(json.dumps({"output": str(args.output), "reload_logits_identical": identical,
                      "lora_tensors": len(lora)}))
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
