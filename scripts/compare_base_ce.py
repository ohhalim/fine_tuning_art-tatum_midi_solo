#!/usr/bin/env python3
"""Token CE of full checkpoints on the same crops (C0 gate in MEHLDAU_CLEAN_BASE.md)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--target-dir", type=Path, required=True, help="train/val token dirs")
    ap.add_argument("--generic-list", type=Path, required=True)
    ap.add_argument("--jazz-dir", type=Path,
                    default=Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/data/jazz_full/train"))
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from utilities.device import get_device, use_cuda
    use_cuda(args.device == "mps")
    device = get_device()
    from utilities.constants import TOKEN_PAD
    from scripts.generate import load_model_with_lora
    from scripts.run_mehldau_update_budget_diag import fixed_crops
    from scripts.validate_style_distance import load, middle_chunk

    def songs(split):
        return [load(f) for f in sorted((args.target_dir / split).glob("*.npy"))]

    names = json.loads(args.generic_list.read_text())["generic_probe_files"]
    sets = {"target_train": fixed_crops(songs("train"), 512, 2),
            "target_val": fixed_crops(songs("val"), 512),
            "generic_probe": [middle_chunk(load(args.jazz_dir / n), 1024) for n in names]}
    out = {"schema": "base_ce_compare_v1", "target_dir": str(args.target_dir), "models": {}}
    for spec in args.checkpoint:
        name, path = spec.split("=", 1)
        model = load_model_with_lora(lora_path=str(Path(path).parent), checkpoint_path=path,
                                     prefer_full_checkpoint=True).to(device)
        model.eval()
        row = {}
        with torch.no_grad():
            for key, crops in sets.items():
                total = n = 0
                for c in crops:
                    c = np.asarray(c)
                    x = torch.tensor(c[:-1]).unsqueeze(0).to(device)
                    y = torch.tensor(c[1:]).to(device)
                    keep = y != TOKEN_PAD
                    total += F.cross_entropy(model(x)[0][keep], y[keep], reduction="sum").item()
                    n += int(keep.sum())
                row[key] = total / n
        out["models"][name] = {"checkpoint": path, **row}
        print(name, {k: round(v, 4) for k, v in row.items()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
