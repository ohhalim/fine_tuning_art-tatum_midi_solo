#!/usr/bin/env python3
"""Compare one artist's CV fold adapters trained on two different bases.

ΔCE is relative to each arm's own base, so it cannot say which base+adapter
predicts the artist's unseen songs better. This script scores the same
held-out songs with the same crops under both arms and reports absolute CE:
each song is scored by the fold adapter that held it out. Generic jazz CE is
averaged over each arm's fold adapters. The difference (arm 2 - arm 1) gets a
paired song-bootstrap CI conditional on the fixed fold models.
Pre-registered in docs/experiments/SHARED_BASE.md. Likelihood only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "music_transformer" / "third_party", ROOT / "scripts"):
    sys.path.insert(0, str(p))

import numpy as np

MAIN = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def paired_diff_ci(a, b, iters: int = 2000, seed: int = 0) -> list[float]:
    """95% CI of mean(b - a) under paired song resampling."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(iters, len(d)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=BASE_CKPT,FOLD_PREFIX",
                    help="exactly two; the first is the reference")
    ap.add_argument("--folds-json", type=Path, required=True)
    ap.add_argument("--train-dir", type=Path, required=True)
    ap.add_argument("--update", type=int, default=128)
    ap.add_argument("--generic-list", type=Path, required=True)
    ap.add_argument("--jazz-dir", type=Path, default=MAIN / "data/jazz_full/train")
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    if len(args.arm) != 2:
        ap.error("need exactly two --arm")

    import torch
    import torch.nn.functional as F
    from utilities.device import get_device, use_cuda
    use_cuda(args.device == "mps")
    device = get_device()
    from utilities.constants import TOKEN_PAD
    from scripts.eval_cross_artist import song_crops
    from scripts.eval_cv_cross import check_held_out_once
    from scripts.generate import load_model_with_lora
    from scripts.train_qlora import load_lora_snapshot
    from scripts.validate_style_distance import load

    held_out = {int(k): set(v) for k, v in json.loads(args.folds_json.read_text())["held_out"].items()}
    files = sorted(args.train_dir.glob("*.npy"))
    check_held_out_once(held_out, [f.name for f in files])
    songs = [(f.name, song_crops(load(f), False)) for f in files]
    generic = [(n, song_crops(load(args.jazz_dir / n), True))
               for n in json.loads(args.generic_list.read_text())["generic_probe_files"]]

    arms = {}
    for spec in args.arm:
        name, rest = spec.split("=", 1)
        base_ckpt, prefix = rest.split(",")
        model = load_model_with_lora(lora_path=str(Path(base_ckpt).parent), checkpoint_path=base_ckpt,
                                     prefer_full_checkpoint=True).to(device)
        model.eval()
        zero = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

        def ce(crops):
            total = n = 0
            with torch.no_grad():
                for c in crops:
                    c = np.asarray(c)
                    x = torch.tensor(c[:-1]).unsqueeze(0).to(device)
                    y = torch.tensor(c[1:]).to(device)
                    keep = y != TOKEN_PAD
                    total += F.cross_entropy(model(x)[0][keep], y[keep], reduction="sum").item()
                    n += int(keep.sum())
            return total / n

        load_lora_snapshot(model, zero)
        base_own = {n: ce(c) for n, c in songs}
        base_gen = {n: ce(c) for n, c in generic}
        own, gen = {}, {n: [] for n, _ in generic}
        for k, held in sorted(held_out.items()):
            snap = Path(f"{prefix}{k}") / f"lora_update{args.update:03d}.pt"
            load_lora_snapshot(model, torch.load(snap, map_location="cpu"))
            for n, c in songs:
                if n in held:
                    own[n] = ce(c)
            for n, c in generic:
                gen[n].append(ce(c))
            print(f"{name} fold {k} scored", flush=True)
        names = [n for n, _ in songs]
        gnames = [n for n, _ in generic]
        arms[name] = {"base_checkpoint": base_ckpt, "fold_prefix": prefix,
                      "own_abs": [own[n] for n in names], "own_base": [base_own[n] for n in names],
                      "gen_abs": [float(np.mean(gen[n])) for n in gnames],
                      "gen_base": [base_gen[n] for n in gnames]}
        del model

    (ref, a), (alt, b) = arms.items()
    summary = {}
    for name, r in arms.items():
        summary[name] = {"own_abs_macro": float(np.mean(r["own_abs"])),
                         "own_base_macro": float(np.mean(r["own_base"])),
                         "own_d_macro": float(np.mean(np.subtract(r["own_abs"], r["own_base"]))),
                         "gen_abs_mean": float(np.mean(r["gen_abs"])),
                         "gen_base_mean": float(np.mean(r["gen_base"])),
                         "gen_d_mean": float(np.mean(np.subtract(r["gen_abs"], r["gen_base"])))}
        summary[name]["specialisation"] = summary[name]["own_d_macro"] - summary[name]["gen_d_mean"]
    diff = {"own_abs": summary[alt]["own_abs_macro"] - summary[ref]["own_abs_macro"],
            "own_abs_ci95": paired_diff_ci(a["own_abs"], b["own_abs"]),
            "gen_abs": summary[alt]["gen_abs_mean"] - summary[ref]["gen_abs_mean"],
            "gen_abs_ci95": paired_diff_ci(a["gen_abs"], b["gen_abs"]),
            "own_songs_alt_better": int(sum(y < x for x, y in zip(a["own_abs"], b["own_abs"])))}
    out = {"schema": "base_cv_compare_v1", "update": args.update, "reference": ref, "alternative": alt,
           "songs": [n for n, _ in songs], "musical_quality_verified": False, "style_verified": False,
           "arms": arms, "summary": summary, "difference_alt_minus_ref": diff}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for name, s in summary.items():
        print(f"{name}: own abs {s['own_abs_macro']:.4f} (base {s['own_base_macro']:.4f}, d {s['own_d_macro']:+.4f}) "
              f"generic abs {s['gen_abs_mean']:.4f} (d {s['gen_d_mean']:+.4f}) spec {s['specialisation']:+.4f}")
    print(f"{alt} - {ref}: own abs {diff['own_abs']:+.4f} {[round(x, 4) for x in diff['own_abs_ci95']]} "
          f"({diff['own_songs_alt_better']}/{len(songs)} songs better) | generic abs {diff['gen_abs']:+.4f} "
          f"{[round(x, 4) for x in diff['gen_abs_ci95']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
