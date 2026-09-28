#!/usr/bin/env python3
"""3x3 cross evaluation: models x (Tatum holdout, Mehldau holdout, generic jazz).

docs/experiments/TATUM_VS_MEHLDAU.md §4. Every column uses the same songs and
crops for every model: holdout songs -> every non-overlapping 512-token crop;
generic songs -> the middle 1024 tokens. Reports token-weighted and song-macro
CE, dCE vs the first model (the base), and a song-level bootstrap 95% CI of the
macro dCE. Raw CE is not comparable across columns (song difficulty differs).
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


def song_crops(tokens, generic: bool):
    from scripts.run_mehldau_update_budget_diag import fixed_crops
    from scripts.validate_style_distance import middle_chunk

    return [middle_chunk(tokens, 1024)] if generic else fixed_crops([tokens], 512)


def bootstrap_mean_ci(values, iters: int = 2000, seed: int = 0) -> list[float]:
    v = np.asarray(values, dtype=float)
    if len(v) == 1:
        return [float(v[0]), float(v[0])]
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(iters, len(v)))].mean(1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def column_stats(per_song: list[tuple[float, int]]) -> dict:
    total = sum(s for s, _ in per_song)
    n = sum(c for _, c in per_song)
    return {"token_ce": total / n, "macro_ce": float(np.mean([s / c for s, c in per_song])),
            "songs": len(per_song), "tokens": int(n)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", action="append", required=True, metavar="NAME=CHECKPOINT",
                    help="first model is the reference for dCE")
    ap.add_argument("--column", action="append", required=True, metavar="NAME=DIR",
                    help="holdout dir of .npy songs (searched recursively)")
    ap.add_argument("--generic-list", type=Path, required=True)
    ap.add_argument("--jazz-dir", type=Path, default=MAIN / "data/jazz_full/train")
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F
    from utilities.device import get_device, use_cuda
    use_cuda(args.device == "mps")
    device = get_device()
    from utilities.constants import TOKEN_PAD
    from scripts.generate import load_model_with_lora
    from scripts.validate_style_distance import load

    columns = {}
    for spec in args.column:
        name, d = spec.split("=", 1)
        files = sorted(Path(d).rglob("*.npy"))
        columns[name] = [(f.name, song_crops(load(f), False)) for f in files]
    names = json.loads(args.generic_list.read_text())["generic_probe_files"]
    columns["generic"] = [(n, song_crops(load(args.jazz_dir / n), True)) for n in names]

    results = {}
    for spec in args.model:
        mname, path = spec.split("=", 1)
        model = load_model_with_lora(lora_path=str(Path(path).parent), checkpoint_path=path,
                                     prefer_full_checkpoint=True).to(device)
        model.eval()
        results[mname] = {"checkpoint": path, "columns": {}}
        with torch.no_grad():
            for cname, songs in columns.items():
                per_song = []
                for _, crops in songs:
                    total = n = 0
                    for c in crops:
                        c = np.asarray(c)
                        x = torch.tensor(c[:-1]).unsqueeze(0).to(device)
                        y = torch.tensor(c[1:]).to(device)
                        keep = y != TOKEN_PAD
                        total += F.cross_entropy(model(x)[0][keep], y[keep], reduction="sum").item()
                        n += int(keep.sum())
                    per_song.append((total, n))
                results[mname]["columns"][cname] = {**column_stats(per_song),
                                                    "per_song_ce": [s / c for s, c in per_song],
                                                    "songs_list": [s for s, _ in songs]}
        print(mname, {c: round(v["token_ce"], 4) for c, v in results[mname]["columns"].items()}, flush=True)

    ref = next(iter(results))
    for mname, r in results.items():
        for cname, col in r["columns"].items():
            base_col = results[ref]["columns"][cname]
            diffs = [a - b for a, b in zip(col["per_song_ce"], base_col["per_song_ce"])]
            col["d_token_ce"] = col["token_ce"] - base_col["token_ce"]
            col["d_macro_ce"] = col["macro_ce"] - base_col["macro_ce"]
            col["d_macro_ci95"] = bootstrap_mean_ci(diffs)
            col["per_song_d_ce"] = diffs
    out = {"schema": "cross_artist_v1", "reference_model": ref,
           "musical_quality_verified": False, "style_verified": False, "models": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print("\nmodel / column: dCE token (macro) [CI]")
    for mname, r in results.items():
        cells = [f"{c}: {v['d_token_ce']:+.4f} ({v['d_macro_ce']:+.4f}) "
                 f"[{v['d_macro_ci95'][0]:+.4f},{v['d_macro_ci95'][1]:+.4f}]"
                 for c, v in r["columns"].items()]
        print(mname, " | ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
