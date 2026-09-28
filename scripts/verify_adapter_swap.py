#!/usr/bin/env python3
"""Check AdapterBank on real checkpoints (docs/experiments/ADAPTER_SWAP.md criteria 1-2).

1. After every swap the bank model's logits equal a separately loaded, merged
   model of that adapter (max |diff| < 1e-5), over a swap sequence A,B,A,B.
2. A pair trained on different bases is refused.
Also reports the swap time and how many tensors a swap copies.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--adapter", action="append", required=True, metavar="NAME=CHECKPOINT",
                    help="two or more adapters that should share a base")
    ap.add_argument("--foreign", metavar="NAME=CHECKPOINT",
                    help="an adapter on a different base; the bank must refuse it")
    ap.add_argument("--allow-different-bases", action="store_true",
                    help="verify a whole-model swap between adapters on different bases")
    ap.add_argument("--swaps", type=int, default=20)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    from utilities.device import use_cuda
    use_cuda(False)
    from scripts.adapter_bank import AdapterBank
    from scripts.generate import load_model_with_lora
    from scripts.train_qlora import merge_lora_for_inference

    def merged(path):
        m = load_model_with_lora(lora_path=str(Path(path).parent), checkpoint_path=str(path),
                                 prefer_full_checkpoint=True, max_sequence=1024)
        merge_lora_for_inference(m)
        return m.eval()

    specs = dict(s.split("=", 1) for s in args.adapter)
    bank = AdapterBank({n: merged(p) for n, p in specs.items()},
                       allow_different_bases=args.allow_different_bases)
    refs = {n: merged(p) for n, p in specs.items()}
    torch.manual_seed(0)
    x = torch.randint(0, 388, (1, 256))
    names = list(specs)
    order = [names[i % len(names)] for i in range(1, args.swaps + 1)]
    diffs = []
    with torch.no_grad():
        want = {n: refs[n](x) for n in names}
        for n in order:
            bank.select(n)
            diffs.append((n, float((bank.model(x) - want[n]).abs().max())))
    out = {"schema": "adapter_swap_verify_v1", "adapters": specs, "shared_base": bank.shared_base,
           "swap_keys": len(bank.swap_keys), "swap_key_examples": bank.swap_keys[:4],
           "swaps": len(bank.swap_ms),
           "swap_ms_p50": statistics.median(bank.swap_ms), "swap_ms_max": max(bank.swap_ms),
           "logits_max_abs_diff": max(d for _, d in diffs), "logits_ok": max(d for _, d in diffs) < 1e-5}
    if args.foreign:
        fname, fpath = args.foreign.split("=", 1)
        try:
            AdapterBank({names[0]: merged(specs[names[0]]), fname: merged(fpath)})
            out["foreign_refused"] = False
        except ValueError as exc:
            out["foreign_refused"] = True
            out["foreign_error"] = str(exc)[:200]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
