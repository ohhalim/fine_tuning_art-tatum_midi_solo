#!/usr/bin/env python3
"""Does the Mehldau adapter move more when it gets more optimizer updates?

One continuous LoRA run from the generic base, with the training loop of
``train_qlora.py`` (same dataset class, loss, LoRA wrapper, gradient
accumulation 4, batch 4, lr 3e-4). The only thing compared is the number of
optimizer updates: snapshots are taken inside one run at update 0, 8 and later
updates until the CPU wall-clock budget runs out. Effective batch, crop order,
seed and lr schedule are shared by every snapshot.

This is NOT a reproduction of ``outputs/mehldau_lora/from_base``. The lr
schedule is planned for this run's own epoch count, so update 8 here sees a
slightly different lr than update 8 there.

Evaluation never uses label smoothing or dropout: token cross-entropy on fixed,
deterministic crops of the train songs and, separately, of the val songs.
Both val songs are in the base pretrain set, so val says nothing about
generalisation.

Generation: fixed primer, same seeds, one take per (seed, bar) for each
snapshot. Token identity between snapshots tells whether the update changed the
sampled output at all. Nothing here verifies an audible Mehldau style.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

MAIN_REPO = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")


def fixed_crops(sequences, length: int, max_per_song: int | None = None):
    """Deterministic non-overlapping crops of ``length + 1`` tokens."""
    out = []
    for tokens in sequences:
        starts = list(range(0, max(1, len(tokens) - length), length))
        if max_per_song is not None:
            starts = starts[:max_per_song]
        for s in starts:
            crop = tokens[s : s + length + 1]
            if len(crop) >= 2:
                out.append(np.asarray(crop, dtype=np.int64))
    return out


def token_identity(a, b) -> float:
    """Fraction of positions where two token sequences agree, over the longer length."""
    n = max(len(a), len(b))
    if n == 0:
        return 1.0
    same = sum(1 for x, y in zip(a, b) if int(x) == int(y))
    return same / n


def lora_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items() if "lora_" in k}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path,
                   default=MAIN_REPO / "outputs/d0_experiment/armB_full2777/ckpt/checkpoint_epoch8.pt")
    p.add_argument("--data-dir", type=Path, default=MAIN_REPO / "data/mehldau_full")
    p.add_argument("--primer", type=Path, default=None,
                   help="required for --mode budget")
    p.add_argument("--mode", choices=["budget", "overfit"], default="budget",
                   help="overfit: one fixed batch, accumulation 1, path check only")
    p.add_argument("--overfit-updates", type=int, default=24)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--budget-seconds", type=float, default=300.0,
                   help="wall-clock cap for the training loop only")
    p.add_argument("--planned-epochs", type=int, default=64,
                   help="sets the cosine schedule length; training stops earlier on budget")
    p.add_argument("--snapshot-updates", default="0,8,16,24,32,48,64")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--gradient-accumulation", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-length", type=int, default=512)
    p.add_argument("--eval-crops-per-train-song", type=int, default=2)
    p.add_argument("--gen-seeds", default="42,100,200")
    p.add_argument("--gen-bars", type=int, default=4)
    p.add_argument("--bpm", type=int, default=128)
    p.add_argument("--generation-tokens", type=int, default=96)
    args = p.parse_args(argv)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        p.error(f"output dir not empty, refusing to overwrite: {args.output_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import torch
    import torch.nn.functional as F
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader

    from utilities.device import use_cuda
    use_cuda(False)
    from model.loss import SmoothCrossEntropyLoss
    from model.music_transformer import MusicTransformer
    from utilities.constants import TOKEN_PAD, VOCAB_SIZE
    from scripts.checkpoint_utils import load_state_dict_with_token_resize
    from scripts.train_qlora import (MidiDataset, add_lora_to_model,
                                     checkpoint_model_config,
                                     checkpoint_payload_state_dict, set_seed)

    set_seed(args.seed)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_payload_state_dict(payload)
    cfg = checkpoint_model_config(payload)
    model = MusicTransformer(n_layers=cfg["n_layers"], num_heads=cfg["num_heads"],
                             d_model=cfg["d_model"], dim_feedforward=cfg["dim_feedforward"],
                             max_sequence=cfg["max_sequence"], rpr=cfg["rpr"])
    model, _ = add_lora_to_model(model, r=cfg["lora_r"], alpha=cfg["lora_alpha"],
                                 dropout=cfg["lora_dropout"])
    load_state_dict_with_token_resize(model, state, strict=True)
    max_seq = int(cfg["max_sequence"])

    train_ds = MidiDataset(str(args.data_dir), max_seq=max_seq, split="train")
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    trainable = [q for q in model.parameters() if q.requires_grad]
    optimizer = AdamW(trainable, lr=args.lr, weight_decay=0.01)
    batches_per_epoch = len(loader)
    updates_per_epoch = math.ceil(batches_per_epoch / args.gradient_accumulation)
    scheduler = CosineAnnealingLR(optimizer, T_max=updates_per_epoch * args.planned_epochs,
                                  eta_min=1e-6)
    loss_fn = SmoothCrossEntropyLoss(args.label_smoothing, VOCAB_SIZE, ignore_index=TOKEN_PAD)

    def load_split(split):
        return [np.load(f, allow_pickle=True).ravel().astype(np.int64)
                for f in sorted((args.data_dir / split).glob("*.npy"))]
    train_eval = fixed_crops(load_split("train"), args.eval_length, args.eval_crops_per_train_song)
    val_eval = fixed_crops(load_split("val"), args.eval_length)

    def eval_ce(crops):
        model.eval()
        total, n = 0.0, 0
        with torch.no_grad():
            for c in crops:
                x = torch.tensor(c[:-1]).unsqueeze(0)
                y = torch.tensor(c[1:])
                logits = model(x)[0]
                keep = y != TOKEN_PAD
                total += F.cross_entropy(logits[keep], y[keep], reduction="sum").item()
                n += int(keep.sum())
        return total / n, n

    if args.mode == "overfit":
        # Path check only: can this loop drive loss down on data it sees every
        # update? Says nothing about style or generalisation.
        songs = load_split("train")[: args.batch_size]
        batch = torch.full((len(songs), max_seq), TOKEN_PAD, dtype=torch.long)
        for i, t in enumerate(songs):
            t = torch.from_numpy(t[:max_seq]).long()
            batch[i, : len(t)] = t
        x, y = batch[:, :-1], batch[:, 1:]
        fixed = [np.concatenate([x[i].numpy(), y[i, -1:].numpy()]) for i in range(len(songs))]
        opt = AdamW(trainable, lr=args.lr, weight_decay=0.01)
        rows, t0 = [], time.perf_counter()
        ce0, n_tok = eval_ce(fixed)
        rows.append({"update": 0, "fixed_batch_ce": ce0, "wall_s": 0.0})
        for u in range(1, args.overfit_updates + 1):
            model.train()
            out = model(x)
            loss_fn(out.view(-1, out.size(-1)), y.reshape(-1)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if u % 4 == 0 or u == args.overfit_updates:
                ce, _ = eval_ce(fixed)
                rows.append({"update": u, "fixed_batch_ce": ce,
                             "wall_s": round(time.perf_counter() - t0, 1)})
                print(json.dumps(rows[-1]), flush=True)
            if time.perf_counter() - t0 >= args.budget_seconds:
                break
        va, _ = eval_ce(val_eval)
        (args.output_dir / "report.json").write_text(json.dumps({
            "schema": "mehldau_fixed_batch_overfit_v1",
            "interpretation": "path check only; not a style or generalisation result",
            "checkpoint": str(args.checkpoint), "batch_songs": len(songs),
            "tokens_per_eval": n_tok, "lr": args.lr, "accumulation": 1,
            "label_smoothing_train": args.label_smoothing, "curve": rows,
            "val_ce_after": va,
            "adam_state_steps": sorted({int(v["step"]) for v in opt.state_dict()["state"].values()}),
        }, indent=2) + "\n")
        return 0
    if args.primer is None:
        p.error("--primer is required for --mode budget")

    snapshots_wanted = sorted({int(s) for s in args.snapshot_updates.split(",") if s.strip()})
    snapshots, curve = {}, []

    def take_snapshot(update, lr, wall, tokens_seen, last_train_loss):
        tr, tr_n = eval_ce(train_eval)
        va, va_n = eval_ce(val_eval)
        snapshots[update] = lora_state(model)
        torch.save(snapshots[update], args.output_dir / f"lora_update{update:03d}.pt")
        row = {"update": update, "lr": lr, "train_wall_s": round(wall, 1),
               "train_tokens_seen": tokens_seen,
               "train_loss_ls_dropout_last_epoch": last_train_loss,
               "eval_train_ce": tr, "eval_train_tokens": tr_n,
               "eval_val_ce": va, "eval_val_tokens": va_n}
        curve.append(row)
        print(json.dumps(row), flush=True)

    updates, tokens_seen, train_wall = 0, 0, 0.0
    take_snapshot(0, optimizer.param_groups[0]["lr"], 0.0, 0, None)
    stopped = "planned_epochs"
    for epoch in range(1, args.planned_epochs + 1):
        model.train()
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss = 0.0
        for bi, (x, y) in enumerate(loader):
            out = model(x)
            raw = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            (raw / args.gradient_accumulation).backward()
            tokens_seen += int((y != TOKEN_PAD).sum())
            epoch_loss += raw.item()
            if (bi + 1) % args.gradient_accumulation == 0 or (bi + 1) == batches_per_epoch:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                updates += 1
        train_wall += time.perf_counter() - t0
        if updates in snapshots_wanted:
            take_snapshot(updates, optimizer.param_groups[0]["lr"], train_wall, tokens_seen,
                          epoch_loss / batches_per_epoch)
        if train_wall >= args.budget_seconds:
            stopped = "budget"
            break
    if updates not in snapshots:
        take_snapshot(updates, optimizer.param_groups[0]["lr"], train_wall, tokens_seen,
                      epoch_loss / batches_per_epoch)
    adam_steps = {int(v["step"]) for v in optimizer.state_dict()["state"].values()}

    # ---- generation from a fixed primer, same seeds, per snapshot ----
    import pretty_midi
    from midi_processor.processor import decode_midi
    from scripts.generate import build_primer, generate_once
    from scripts.run_jazz_mvp import fit_window

    primer = build_primer(conditioning_midi=str(args.primer), primer_max_tokens=32,
                          append_sep_token=True, control_format="control_v1",
                          role="lead", tempo_bpm=args.bpm)
    gen_seeds = [int(s) for s in args.gen_seeds.split(",") if s.strip()]
    bar_seconds = 240.0 / args.bpm
    compare = sorted({0, 8, updates} & set(snapshots))
    generated = {}
    for update in compare:
        model.load_state_dict(snapshots[update], strict=False)
        model.eval()
        takes, midi_out = {}, pretty_midi.PrettyMIDI(initial_tempo=float(args.bpm))
        lead = pretty_midi.Instrument(program=0, name=f"update{update}")
        for seed in gen_seeds:
            for bar in range(args.gen_bars):
                torch.manual_seed(seed + bar)
                tokens, _ = generate_once(
                    model=model, primer=primer,
                    target_length=min(max_seq, len(primer) + args.generation_tokens),
                    strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                    grammar_mask=True, target_duration_seconds=bar_seconds,
                    return_metadata=True)
                takes[f"{seed}:{bar}"] = [int(t) for t in tokens]
                if seed == gen_seeds[0]:
                    for n in fit_window(decode_midi(tokens), bar_seconds).instruments:
                        for note in n.notes:
                            lead.notes.append(pretty_midi.Note(
                                note.velocity, note.pitch,
                                note.start + bar * bar_seconds, note.end + bar * bar_seconds))
        midi_out.instruments = [lead]
        midi_out.write(str(args.output_dir / f"gen_update{update:03d}_seed{gen_seeds[0]}.mid"))
        generated[update] = takes

    identity = {}
    for update in compare:
        if update == 0:
            continue
        per = [token_identity(generated[0][k], generated[update][k]) for k in generated[0]]
        exact = sum(1 for k in generated[0] if generated[0][k] == generated[update][k])
        identity[f"0_vs_{update}"] = {
            "mean_token_identity": sum(per) / len(per),
            "exact_take_matches": exact, "takes": len(per),
            "mean_tokens_per_take": sum(len(t) for t in generated[update].values()) / len(per)}

    report = {
        "schema": "mehldau_update_budget_diag_v1",
        "question": "does the adapter's effect grow with optimizer updates inside one run?",
        "checkpoint": str(args.checkpoint), "data_dir": str(args.data_dir),
        "primer": str(args.primer),
        "primer_sha1": hashlib.sha1(args.primer.read_bytes()).hexdigest(),
        "model_max_sequence": max_seq,
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "batches_per_epoch": batches_per_epoch, "updates_per_epoch": updates_per_epoch,
        "optimizer_updates": updates, "adam_state_steps": sorted(adam_steps),
        "train_wall_s": round(train_wall, 1), "stopped_by": stopped,
        "train_tokens_seen": tokens_seen,
        "curve": curve, "generation_identity": identity,
        "val_songs_in_base_pretrain": True,
        "mehldau_style_verified": False, "musical_quality_verified": False,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.output_dir / "generated_tokens.json").write_text(json.dumps(
        {str(k): v for k, v in generated.items()}) + "\n")
    print(json.dumps({k: report[k] for k in ("optimizer_updates", "adam_state_steps",
                                             "train_wall_s", "stopped_by",
                                             "generation_identity")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
