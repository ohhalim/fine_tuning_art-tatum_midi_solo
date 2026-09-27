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
    ap.add_argument("--lora-targets", default="out_proj", help="must match the snapshot")
    args = ap.parse_args(argv)
    if args.output.exists():
        ap.error(f"refusing to overwrite {args.output}")

    import torch
    from utilities.device import use_cuda
    use_cuda(False)
    from model.music_transformer import MusicTransformer
    from scripts.checkpoint_utils import load_state_dict_with_token_resize
    from scripts.generate import load_model_with_lora
    from scripts.train_qlora import (add_lora_targets, add_lora_to_model, checkpoint_model_config,
                                     checkpoint_payload_state_dict)

    payload = torch.load(args.base, map_location="cpu", weights_only=False)
    cfg = checkpoint_model_config(payload)
    model = MusicTransformer(n_layers=cfg["n_layers"], num_heads=cfg["num_heads"],
                             d_model=cfg["d_model"], dim_feedforward=cfg["dim_feedforward"],
                             max_sequence=cfg["max_sequence"], rpr=cfg["rpr"])
    model, _ = add_lora_to_model(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"],
                                 dropout=cfg["lora_dropout"])
    load_state_dict_with_token_resize(model, checkpoint_payload_state_dict(payload), strict=True)
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    extra = [t for t in targets if t != "out_proj"]
    if extra:
        add_lora_targets(model, extra, r=cfg["lora_r"], alpha=cfg["lora_alpha"],
                         dropout=cfg["lora_dropout"])
    cfg = {**cfg, "lora_targets": targets}
    lora = torch.load(args.snapshot, map_location="cpu")
    if not lora or not all("lora_" in k for k in lora):
        ap.error("snapshot must contain only lora_ tensors")
    missing = set(lora) - set(model.state_dict())
    if missing:
        ap.error(f"snapshot keys not in model: {sorted(missing)[:3]}")
    model.load_state_dict(lora, strict=False)
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
