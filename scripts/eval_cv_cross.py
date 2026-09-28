#!/usr/bin/env python3
"""Cross-evaluate CV fold adapters on own held-out, other-artist and generic songs.

Pre-registered in docs/experiments/TATUM_VS_MEHLDAU_CV_CROSS.md. Per artist:
own dCE for each of its 16 songs comes from the fold adapter that held the song
out; other-artist and generic dCE are averaged over that artist's 4 fold
adapters. Post-selection exploratory analysis: the CIs are song-resampling
intervals conditional on the fixed fold models and the selected budget, and no
confirmatory ranking is declared.
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


def average_ranks(x) -> np.ndarray:
    """Ranks with ties given the mean of their positions (1-based)."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    return float(np.corrcoef(average_ranks(a), average_ranks(b))[0, 1])


def check_held_out_once(held_out: dict[int, set], songs: list[str]) -> None:
    """Every song must be held out by exactly one fold (and no unknown songs)."""
    counts = {n: 0 for n in songs}
    for names in held_out.values():
        for n in names:
            if n not in counts:
                raise ValueError(f"held-out song not in the training set: {n}")
            counts[n] += 1
    bad = {n: c for n, c in counts.items() if c != 1}
    if bad:
        raise ValueError(f"songs not held out exactly once: {bad}")


def indices(own: np.ndarray, other: np.ndarray, generic: np.ndarray) -> dict:
    return {"self_gain": float(own.mean()), "self_gain_negative": bool(own.mean() < 0),
            "specialisation": float(own.mean() - generic.mean()),
            "specificity": float(own.mean() - other.mean()),
            "other_gain": float(other.mean()), "generic_gain": float(generic.mean())}


def bootstrap_rank(t: dict, m: dict, iters: int = 2000, seed: int = 0) -> dict:
    """CIs for D = index_tatum - index_mehldau, conditional on the fixed fold models.

    Paired song resampling (review of #1501): one index draw per song set is
    applied to every role that uses those songs. Tatum songs are the Tatum
    adapter's "own" set and the Mehldau adapter's "other" set, and vice versa;
    the generic songs are shared. Retraining variance is not included.
    """
    rng = np.random.default_rng(seed)
    n_t, n_m, n_g = len(t["own"]), len(m["own"]), len(t["generic"])
    if len(m["other"]) != n_t or len(t["other"]) != n_m:
        raise ValueError("other-artist sets must be the same songs as the own sets")
    rows = []
    for _ in range(iters):
        it, im, ig = rng.integers(0, n_t, n_t), rng.integers(0, n_m, n_m), rng.integers(0, n_g, n_g)
        t_own, t_oth, t_gen = t["own"][it].mean(), t["other"][im].mean(), t["generic"][ig].mean()
        m_own, m_oth, m_gen = m["own"][im].mean(), m["other"][it].mean(), m["generic"][ig].mean()
        rows.append((t_own - t_gen, m_own - m_gen, t_own - t_oth, m_own - m_oth))
    a = np.array(rows)

    def ci(x):
        return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]

    def direction(c):
        # Exploratory only: a CI excluding 0 is a direction consistent with the
        # data given these models, not a confirmed ranking.
        if c[1] < 0:
            return "tatum_larger_exploratory"
        if c[0] > 0:
            return "mehldau_larger_exploratory"
        return "difference_not_confirmed"

    out = {"resampling": "paired song indices per artist across roles; generic paired; "
                         "conditional on fixed fold models and selected budget",
           "tatum_spec_ci95": ci(a[:, 0]), "mehldau_spec_ci95": ci(a[:, 1]),
           "tatum_specif_ci95": ci(a[:, 2]), "mehldau_specif_ci95": ci(a[:, 3]),
           "D_spec_ci95": ci(a[:, 0] - a[:, 1]), "D_specif_ci95": ci(a[:, 2] - a[:, 3])}
    out["D_spec_direction"] = direction(out["D_spec_ci95"])
    out["D_specif_direction"] = direction(out["D_specif_ci95"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--artist", action="append", required=True,
                    metavar="NAME=FOLD_PREFIX,FOLDS_JSON,TRAIN_DIR",
                    help="exactly two: tatum and mehldau")
    ap.add_argument("--update", type=int, default=128)
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
    from scripts.eval_cross_artist import song_crops
    from scripts.generate import load_model_with_lora
    from scripts.train_qlora import load_lora_snapshot
    from scripts.validate_style_distance import load

    artists = {}
    for spec in args.artist:
        name, rest = spec.split("=", 1)
        prefix, folds_json, train_dir = rest.split(",")
        folds = json.loads(Path(folds_json).read_text())["held_out"]
        songs = sorted(Path(train_dir).glob("*.npy"))
        held_out = {int(k): set(v) for k, v in folds.items()}
        check_held_out_once(held_out, [f.name for f in songs])
        artists[name] = {"prefix": prefix, "held_out": held_out,
                         "songs": [(f.name, song_crops(load(f), False)) for f in songs]}
    if set(artists) != {"tatum", "mehldau"}:
        ap.error("need --artist tatum=... and --artist mehldau=...")
    generic = [(n, song_crops(load(args.jazz_dir / n), True))
               for n in json.loads(args.generic_list.read_text())["generic_probe_files"]]

    model = load_model_with_lora(lora_path=str(args.base.parent), checkpoint_path=str(args.base),
                                 prefer_full_checkpoint=True).to(device)
    model.eval()
    base_lora = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}

    def song_ce(crops):
        total = n = 0
        with torch.no_grad():
            for c in crops:
                c = np.asarray(c)
                x = torch.tensor(c[:-1]).unsqueeze(0).to(device)
                y = torch.tensor(c[1:]).to(device)
                keep = y != TOKEN_PAD
                total += F.cross_entropy(model(x)[0][keep], y[keep], reduction="sum").item()
                n += int(keep.sum())
        return total, n

    def score(song_list):
        return {name: song_ce(crops) for name, crops in song_list}

    load_lora_snapshot(model, base_lora)
    base = {a: score(v["songs"]) for a, v in artists.items()}
    base["generic"] = score(generic)
    print("base scored", flush=True)

    per, fold_rows = {}, {}
    for a, info in artists.items():
        other = "mehldau" if a == "tatum" else "tatum"
        own_d, own_tok = {}, {}
        other_d = {n: [] for n, _ in artists[other]["songs"]}
        gen_d = {n: [] for n, _ in generic}
        for k, held in sorted(info["held_out"].items()):
            snap = Path(f"{info['prefix']}{k}") / f"lora_update{args.update:03d}.pt"
            load_lora_snapshot(model, torch.load(snap, map_location="cpu"))
            own = score([(n, c) for n, c in info["songs"] if n in held])
            for n, (s, c) in own.items():
                bs, bc = base[a][n]
                own_d[n] = s / c - bs / bc
                own_tok[n] = (s, c)
            for n, (s, c) in score(artists[other]["songs"]).items():
                bs, bc = base[other][n]
                other_d[n].append(s / c - bs / bc)
            for n, (s, c) in score(generic).items():
                bs, bc = base["generic"][n]
                gen_d[n].append(s / c - bs / bc)
            fold_own = np.array([own_d[n] for n in own])
            fold_oth = np.array([other_d[n][-1] for n, _ in artists[other]["songs"]])
            fold_gen = np.array([gen_d[n][-1] for n, _ in generic])
            fold_rows.setdefault(a, []).append({"fold": k, **indices(fold_own, fold_oth, fold_gen)})
            print(f"{a} fold {k} scored", flush=True)
        if len(own_d) != len(info["songs"]):
            raise RuntimeError(f"{a}: {len(own_d)} songs held out, expected {len(info['songs'])}")
        own_tok_ce = (sum(s for s, _ in own_tok.values()) / sum(c for _, c in own_tok.values())
                      - sum(base[a][n][0] for n in own_tok) / sum(base[a][n][1] for n in own_tok))
        per[a] = {"own": np.array([own_d[n] for n, _ in info["songs"]]),
                  "other": np.array([np.mean(other_d[n]) for n, _ in artists[other]["songs"]]),
                  "generic": np.array([np.mean(gen_d[n]) for n, _ in generic]),
                  "own_token_weighted": own_tok_ce,
                  "own_by_song": own_d,
                  "own_tokens": [base[a][n][1] for n, _ in info["songs"]]}
    out = {"schema": "cv_cross_v2", "update": args.update,
           "analysis": "post-selection exploratory (u128 was chosen from these held-out scores)",
           "confirmatory_ranking": False,
           "musical_quality_verified": False, "style_verified": False, "artists": {}}
    for a, v in per.items():
        out["artists"][a] = {**indices(v["own"], v["other"], v["generic"]),
                             "self_gain_token_weighted": v["own_token_weighted"],
                             "own_per_song": v["own_by_song"],
                             "other_per_song": v["other"].tolist(), "n_own": len(v["own"]),
                             "n_other": len(v["other"]), "n_generic": len(v["generic"]),
                             "folds": fold_rows[a],
                             # exploratory: does the per-song gain depend on song length?
                             "own_gain_vs_tokens_spearman": spearman(v["own"], v["own_tokens"])}
    out["ranking"] = bootstrap_rank(per["tatum"], per["mehldau"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    for a in ("tatum", "mehldau"):
        r = out["artists"][a]
        print(f"{a}: self {r['self_gain']:+.4f} (token {r['self_gain_token_weighted']:+.4f}) "
              f"other {r['other_gain']:+.4f} generic {r['generic_gain']:+.4f} "
              f"spec {r['specialisation']:+.4f} specif {r['specificity']:+.4f}")
    rk = out["ranking"]
    print("D_spec", [round(x, 4) for x in rk["D_spec_ci95"]], rk["D_spec_direction"],
          "| D_specif", [round(x, 4) for x in rk["D_specif_ci95"]], rk["D_specif_direction"])
    for a in ("tatum", "mehldau"):
        for f in out["artists"][a]["folds"]:
            print(f"  {a} fold {f['fold']}: self {f['self_gain']:+.4f} spec {f['specialisation']:+.4f} "
                  f"specif {f['specificity']:+.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
