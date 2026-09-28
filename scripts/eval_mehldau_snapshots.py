#!/usr/bin/env python3
"""Score LoRA snapshots from one run (V2 Mehldau, T1 Tatum). Likelihood specialisation first.

Pre-registered in docs/experiments/MEHLDAU_STYLE_SHIFT.md.

Primary: token CE (no label smoothing, eval mode) on Mehldau train crops and on
generic jazz probe chunks; specialisation = dCE_target - dCE_generic vs
update 0.

Exploratory only (both descriptor metrics failed their validity bar): JS shift
and classifier P(Mehldau) of long generations from a neutral primer, and exact
n-gram copy rate against the Mehldau train songs.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

MAIN_REPO = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def ngram_set(tokens, n):
    seq = [int(t) for t in tokens]
    return {tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1))}


def copy_rate(tokens, reference: set, n: int):
    grams = ngram_set(tokens, n)
    return None if not grams else sum(g in reference for g in grams) / len(grams)


def free_generation_validity(tokens) -> dict:
    """Grammar validity of an untimed generation.

    The bar validator also requires the exact target duration and every note
    closed at the end, which a fixed-length free generation cannot meet by
    construction (it is cut mid-phrase). Those two checks are dropped here;
    per-bar validity is covered by the runtime, which validates every block.
    """
    from scripts.run_resident_model_probe import validate_generated_token_block

    v = validate_generated_token_block(tokens, lookahead_ms=1.0)
    valid = (v["decode_error"] is None and v["decoded_note_count"] > 0
             and v["orphan_note_off_count"] == 0 and v["duplicate_note_on_count"] == 0
             and v["silent_note_count"] == 0)
    return {"grammar_valid": bool(valid), "decoded_notes": v["decoded_note_count"],
            "orphan_note_off": v["orphan_note_off_count"],
            "duplicate_note_on": v["duplicate_note_on_count"],
            "silent_notes": v["silent_note_count"], "open_at_end": v["stuck_note_count"]}


def bootstrap_diff(a, b, iters=2000, seed=0):
    """95% CI of mean(b) - mean(a) resampling seeds (paired by seed index)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    idx = rng.integers(0, len(a), size=(iters, len(a)))
    d = (b[idx] - a[idx]).mean(1)
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path,
                    default=MAIN_REPO / "outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt")
    ap.add_argument("--snapshot-dir", type=Path, required=True)
    ap.add_argument("--updates", default=None, help="comma list; default all lora_update*.pt")
    ap.add_argument("--data-dir", type=Path, default=MAIN_REPO / "data/mehldau_full")
    ap.add_argument("--jazz-dir", type=Path, default=MAIN_REPO / "data/jazz_full/train")
    ap.add_argument("--validity-json", type=Path, required=True, help="V1 output (file lists)")
    ap.add_argument("--classifier", type=Path, default=None,
                    help="optional V1b-style model json (exploratory P(target))")
    ap.add_argument("--target-name", default="mehldau")
    ap.add_argument("--lora-targets", default=None,
                    help="optional check; targets are inferred from the snapshots and a "
                         "mismatch with this list is an error")
    ap.add_argument("--no-generate", action="store_true", help="likelihood only")
    ap.add_argument("--primer", type=Path, required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5,6,7,8")
    ap.add_argument("--gen-tokens", type=int, default=768)
    ap.add_argument("--device", choices=["cpu", "mps"], default="mps")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"output dir not empty: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from utilities.device import get_device, use_cuda
    use_cuda(args.device == "mps")
    device = get_device()
    from utilities.constants import TOKEN_PAD
    from midi_processor.processor import decode_midi
    from scripts.generate import build_primer, generate_once
    from scripts.run_mehldau_update_budget_diag import fixed_crops
    from scripts.style_distance import (LogisticModel, distance, feature_counts, feature_vector,
                                        pool, tokens_to_notes)
    from scripts.train_qlora import (build_lora_model_from_state, checkpoint_model_config,
                                     checkpoint_payload_state_dict, load_lora_snapshot,
                                     lora_targets_in_state_dict)
    from scripts.validate_style_distance import load, middle_chunk

    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint_model_config(payload)
    files = sorted(args.snapshot_dir.glob("lora_update*.pt"),
                   key=lambda f: int(re.search(r"(\d+)", f.stem).group(1)))
    if args.updates:
        wanted = {int(u) for u in args.updates.split(",")}
        files = [f for f in files if int(re.search(r"(\d+)", f.stem).group(1)) in wanted]
    if not files:
        ap.error(f"no lora_update*.pt snapshots in {args.snapshot_dir}")
    snapshot_targets = lora_targets_in_state_dict(torch.load(files[0], map_location="cpu"))
    if args.lora_targets is not None:
        declared = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        if sorted(declared) != sorted(snapshot_targets):
            ap.error(f"--lora-targets {declared} does not match snapshot targets {snapshot_targets}")
    model, lora_targets = build_lora_model_from_state(
        cfg, checkpoint_payload_state_dict(payload), extra_targets=snapshot_targets)
    model = model.to(device)
    max_seq = int(cfg["max_sequence"])

    train_songs = [load(f) for f in sorted((args.data_dir / "train").glob("*.npy"))]
    target_train_crops = fixed_crops(train_songs, 512, 2)
    val_songs = [load(f) for f in sorted((args.data_dir / "val").glob("*.npy"))]
    target_val_crops = fixed_crops(val_songs, 512)
    validity = json.loads(args.validity_json.read_text())
    generic_crops = [middle_chunk(load(args.jazz_dir / n), max_seq)
                     for n in validity["generic_probe_files"]]
    ref_m = pool(feature_counts(tokens_to_notes(t)) for t in train_songs)
    ref_g = pool(feature_counts(tokens_to_notes(load(args.jazz_dir / n)))
                 for n in validity["generic_reference_files"])
    clf = (LogisticModel.from_dict(json.loads(args.classifier.read_text()))
           if args.classifier else None)
    train_grams = {n: set().union(*(ngram_set(t, n) for t in train_songs)) for n in (8, 16)}

    def ce(crops):
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for c in crops:
                c = np.asarray(c)
                x = torch.tensor(c[:-1]).unsqueeze(0).to(device)
                y = torch.tensor(c[1:]).to(device)
                logits = model(x)[0]
                keep = y != TOKEN_PAD
                total += F.cross_entropy(logits[keep], y[keep], reduction="sum").item()
                count += int(keep.sum())
        return total / count

    primer = build_primer(conditioning_midi=str(args.primer), primer_max_tokens=32,
                          append_sep_token=True, control_format="control_v1",
                          role="lead", tempo_bpm=128)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    rows, generations = [], {}
    for f in files:
        update = int(re.search(r"(\d+)", f.stem).group(1))
        load_lora_snapshot(model, torch.load(f, map_location="cpu"))
        t0 = time.perf_counter()
        row = {"update": update, "ce_target_train": ce(target_train_crops),
               "ce_target_val": ce(target_val_crops),
               "ce_generic_probe": ce(generic_crops)}
        gens, per_seed = [], []
        for seed in ([] if args.no_generate else seeds):
            torch.manual_seed(seed)
            tokens, _ = generate_once(model=model, primer=primer,
                                      target_length=min(max_seq, len(primer) + args.gen_tokens),
                                      strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                                      grammar_mask=True, return_metadata=True)
            tokens = [int(t) for t in tokens]
            gens.append(tokens)
            counts = feature_counts(tokens_to_notes(tokens))
            per_seed.append({
                "seed": seed, "tokens": len(tokens),
                "notes": int(counts["register"].sum()),
                "js_shift": distance(counts, ref_g) - distance(counts, ref_m),
                "p_target": (float(clf.predict_proba(feature_vector(counts)[None])[0])
                             if clf else None),
                "copy8": copy_rate(tokens, train_grams[8], 8),
                "copy16": copy_rate(tokens, train_grams[16], 16),
                **free_generation_validity(tokens)})
            decode_midi(tokens, file_path=str(args.output_dir / f"gen_u{update:03d}_s{seed}.mid"))
        generations[update] = gens
        row["per_seed"] = per_seed
        for k in ("js_shift", "p_target", "copy8", "copy16", "notes", "grammar_valid"):
            vals = [s[k] for s in per_seed if s[k] is not None]
            row[f"mean_{k}"] = float(np.mean(vals)) if vals else None
        row["wall_s"] = round(time.perf_counter() - t0, 1)
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "per_seed"}), flush=True)

    # Deltas need update 0; without it only absolute values are reported.
    base = next((r for r in rows if r["update"] == 0), None)
    for r in ([] if base is None else rows):
        r["d_ce_target_train"] = r["ce_target_train"] - base["ce_target_train"]
        r["d_ce_target_val"] = r["ce_target_val"] - base["ce_target_val"]
        r["d_ce_generic"] = r["ce_generic_probe"] - base["ce_generic_probe"]
        r["specialisation_train"] = r["d_ce_target_train"] - r["d_ce_generic"]
        r["specialisation_val"] = r["d_ce_target_val"] - r["d_ce_generic"]
        for split in ("train", "val"):
            r[f"specialised_{split}_without_big_generic_loss"] = (
                r[f"specialisation_{split}"] <= -0.02 and r["d_ce_generic"] <= 0.05)
        for k in ("js_shift", "p_target"):
            if r["per_seed"] and all(s[k] is not None for s in r["per_seed"] + base["per_seed"]):
                r[f"{k}_diff_ci95_vs_u0"] = bootstrap_diff([s[k] for s in base["per_seed"]],
                                                           [s[k] for s in r["per_seed"]])
    report = {"schema": "snapshot_eval_v2", "checkpoint": str(args.checkpoint),
              "snapshot_dir": str(args.snapshot_dir), "primer": str(args.primer),
              "seeds": seeds, "gen_tokens": args.gen_tokens, "device": str(device),
              "target_name": args.target_name, "data_dir": str(args.data_dir),
              "lora_targets": lora_targets,
              "validity_json": str(args.validity_json),
              "n_target_train_crops": len(target_train_crops), "n_target_val_crops": len(target_val_crops),
              "n_generic_crops": len(generic_crops),
              "descriptor_metrics_validity_passed": False,
              "style_verified": False, "musical_quality_verified": False,
              "rows": rows}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "generated_tokens.json").write_text(json.dumps(
        {str(k): v for k, v in generations.items()}) + "\n")
    print("\nupdate  dCE_tr   dCE_val  dCE_gen  spec_val js_shift copy16")
    for r in ([] if base is None else rows):
        js = r["mean_js_shift"] if r["mean_js_shift"] is not None else float("nan")
        c16 = r["mean_copy16"] if r["mean_copy16"] is not None else float("nan")
        print(f"{r['update']:>5} {r['d_ce_target_train']:+.4f} {r['d_ce_target_val']:+.4f} "
              f"{r['d_ce_generic']:+.4f} {r['specialisation_val']:+.4f} {js:+.4f} {c16:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
