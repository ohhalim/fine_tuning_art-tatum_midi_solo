#!/usr/bin/env python3
"""Did the Mehldau adapter change anything? Loss and generation, side by side.

Three checkpoints on identical inputs: the generic jazz base, the Tatum-adapted
LoRA, and the Mehldau LoRA. Same held-out primer, same seeds, same budget.

The leakage that shapes every number here
-----------------------------------------
All 18 Mehldau files are already inside ``jazz_full``, the base pretrain set -
verified by exact token-sequence match, including both files the Mehldau split
calls validation (they sit in ``jazz_full/train``). So:

* validation loss is optimistically biased for every arm, base included
* nothing here can speak to generalisation on unseen Mehldau material
* an adapter that "fits Mehldau" is fitting songs the base already fit

The honest question left is narrower: does the adapter *shift generation*
relative to the base on the same input? That is measurable without held-out
data, because it compares two models on one input rather than a model against
unseen truth.

Copy risk is measured, not assumed: generated n-grams are checked against the
training songs, because 16 songs and a small adapter is exactly the setup where
memorisation looks like style.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer"))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

import numpy as np

NGRAM = 8


def load_split(data_dir: Path, split: str) -> list[np.ndarray]:
    return [np.load(p, allow_pickle=True).ravel().astype(np.int64)
            for p in sorted((data_dir / split).glob("*.npy"))]


def evaluate_loss(model, sequences, *, max_sequence: int) -> float:
    """Mean token cross-entropy over fixed-length crops. No training."""
    import torch
    import torch.nn.functional as F

    from utilities.constants import TOKEN_PAD

    model.eval()
    losses = []
    with torch.no_grad():
        for tokens in sequences:
            for start in range(0, max(1, len(tokens) - max_sequence), max_sequence):
                crop = tokens[start : start + max_sequence + 1]
                if len(crop) < 2:
                    continue
                x = torch.tensor(crop[:-1], dtype=torch.long).unsqueeze(0)
                y = torch.tensor(crop[1:], dtype=torch.long).unsqueeze(0)
                logits = model.forward(x)
                losses.append(F.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
                    ignore_index=TOKEN_PAD).item())
    return statistics.mean(losses) if losses else float("nan")


def ngram_set(tokens, n: int = NGRAM) -> set[tuple]:
    seq = [int(t) for t in tokens]
    return {tuple(seq[i : i + n]) for i in range(max(0, len(seq) - n + 1))}


def copy_risk(generated, training_ngrams: set[tuple]) -> dict[str, float]:
    """Fraction of generated n-grams that appear verbatim in the training songs."""
    grams = ngram_set(generated)
    if not grams:
        return {"ngram_count": 0, "copied_fraction": None}
    copied = sum(1 for g in grams if g in training_ngrams)
    return {"ngram_count": len(grams), "copied_fraction": copied / len(grams)}


def describe(takes) -> dict[str, object]:
    """Reference descriptors over a list of per-bar note sets.

    Timing has to be computed inside a bar and only then pooled. Concatenating
    every bar's notes first collapses the inter-onset intervals, because bars
    from different seeds all start at the same bar-relative time and the pooled
    median becomes 0. Pitch range has the same problem in reverse: the union of
    many takes spans the whole register no matter what any single take did.

    None of these is a quality judgement.
    """
    flat = [note for take in takes for note in take]
    if not flat:
        return {"note_count": 0}
    iois, ranges, classes = [], [], []
    for take in takes:
        if len(take) < 2:
            continue
        starts = sorted(n.start for n in take)
        iois.extend(round((b - a) * 1000) for a, b in zip(starts, starts[1:]))
        pitches = [n.pitch for n in take]
        ranges.append(max(pitches) - min(pitches))
        classes.append(len({p % 12 for p in pitches}))
    return {
        "note_count": len(flat),
        "take_count": len(takes),
        "notes_per_take": round(len(flat) / len(takes), 2),
        "median_pitch_range_per_take": statistics.median(ranges) if ranges else None,
        "median_pitch_classes_per_take": statistics.median(classes) if classes else None,
        "median_ioi_ms": statistics.median(iois) if iois else None,
        "median_velocity": statistics.median(n.velocity for n in flat),
        "median_duration_ms": statistics.median(
            round((n.end - n.start) * 1000) for n in flat),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", type=Path, required=True,
                        help="generic jazz base checkpoint (armB)")
    parser.add_argument("--tatum", type=Path, required=True,
                        help="Tatum-adapted checkpoint (armD)")
    parser.add_argument("--mehldau", type=Path, required=True, action="append",
                        help="Mehldau full checkpoint (repeatable, named by its parent "
                             "directory). Must be a checkpoint_epoch*.pt, not "
                             "lora_weights.pt: loading LoRA weights alone leaves the "
                             "base transformer randomly initialised and the numbers "
                             "are meaningless")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo/data/mehldau_full"))
    parser.add_argument("--primer", type=Path, required=True)
    parser.add_argument("--seeds", default="42,100,200")
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=int, default=128)
    parser.add_argument("--generation-tokens", type=int, default=96)
    parser.add_argument("--max-sequence", type=int, default=512)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mehldau_eval"))
    args = parser.parse_args(argv)

    import pretty_midi
    import torch

    from midi_processor.processor import decode_midi
    from scripts.generate import build_primer, generate_once, load_model_with_lora
    from scripts.run_jazz_mvp import fit_window

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    val = load_split(args.data_dir, "val")
    train_seqs = load_split(args.data_dir, "train")
    training_ngrams: set[tuple] = set()
    for seq in train_seqs:
        training_ngrams |= ngram_set(seq)
    print(f"val {len(val)} songs, train {len(train_seqs)} songs, "
          f"{len(training_ngrams):,} distinct training {NGRAM}-grams\n")

    arms = [("base", args.base), ("tatum", args.tatum)]
    for path in args.mehldau:
        if "lora_weights" in path.name:
            parser.error("pass a checkpoint_epoch*.pt, not lora_weights.pt")
        arms.append((f"mehldau_{path.parent.name}", path))
    bar_seconds = 240.0 / args.bpm
    results = []
    for name, checkpoint in arms:
        # Always a full checkpoint. prefer_full_checkpoint=False would skip the
        # base weights entirely and evaluate LoRA on a random transformer.
        model = load_model_with_lora(
            lora_path=str(checkpoint.parent), checkpoint_path=str(checkpoint),
            prefer_full_checkpoint=True, max_sequence=args.max_sequence,
        )
        val_loss = evaluate_loss(model, val, max_sequence=256)

        primer = build_primer(conditioning_midi=str(args.primer), primer_max_tokens=32,
                              append_sep_token=True, control_format="control_v1",
                              role="lead", tempo_bpm=args.bpm)
        phrase = pretty_midi.PrettyMIDI(initial_tempo=float(args.bpm))
        lead = pretty_midi.Instrument(program=0, name=f"lead_{name}")
        copy_scores, emitted, takes = [], [], []
        for seed in seeds:
            for bar in range(args.bars):
                torch.manual_seed(seed + bar)
                tokens, _meta = generate_once(
                    model=model, primer=primer,
                    target_length=min(args.max_sequence,
                                      len(primer) + args.generation_tokens),
                    strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
                    grammar_mask=True, target_duration_seconds=bar_seconds,
                    return_metadata=True)
                emitted.extend(int(t) for t in tokens)
                copy_scores.append(copy_risk(tokens, training_ngrams))
                midi = fit_window(decode_midi(tokens), bar_seconds)
                notes = [n for i in midi.instruments for n in i.notes]
                # Descriptors pool every seed: one seed gives ~30 notes, which is
                # too few for a median to mean anything. Only the first seed is
                # written to the listening MIDI so the file stays one 8-bar take.
                takes.append(notes)
                if seed != seeds[0]:
                    continue
                for note in notes:
                    lead.notes.append(pretty_midi.Note(
                        note.velocity, note.pitch,
                        note.start + bar * bar_seconds,
                        note.end + bar * bar_seconds))
        phrase.instruments = [lead]
        path = args.output_dir / f"{name}.mid"
        phrase.write(str(path))

        fractions = [c["copied_fraction"] for c in copy_scores
                     if c["copied_fraction"] is not None]
        results.append({
            "arm": name, "val_loss": val_loss,
            "copied_ngram_fraction_mean": statistics.mean(fractions) if fractions else None,
            "copied_ngram_fraction_max": max(fractions) if fractions else None,
            "emitted_token_count": len(emitted),
            "descriptors": describe(takes),
            "descriptor_note_count": sum(len(t) for t in takes),
            "listening_take_note_count": len(lead.notes),
            "midi": str(path),
        })
        print(f"{name:<9} val_loss {val_loss:.4f}  "
              f"copied {NGRAM}-grams mean "
              f"{(f'{statistics.mean(fractions):.3f}' if fractions else '-')}  "
              f"max {(f'{max(fractions):.3f}' if fractions else '-')}  "
              f"descriptor notes {sum(len(t) for t in takes)} over {len(takes)} takes")

    print(f"\n{'arm':<20}{'val_loss':>10}{'notes':>7}{'n/take':>8}{'range':>7}{'pc':>5}"
          f"{'IOI ms':>8}{'vel':>6}{'dur ms':>8}")
    for row in results:
        d = row["descriptors"]
        print(f"{row['arm']:<20}{row['val_loss']:>10.4f}{d.get('note_count', 0):>7}"
              f"{d.get('notes_per_take', 0):>8}"
              f"{(d.get('median_pitch_range_per_take') or 0):>7}"
              f"{(d.get('median_pitch_classes_per_take') or 0):>5}"
              f"{(d.get('median_ioi_ms') or 0):>8}{(d.get('median_velocity') or 0):>6}"
              f"{(d.get('median_duration_ms') or 0):>8}")
    print("range/pc/IOI are medians over per-bar takes, not over the pooled notes.")

    base_loss = next(r["val_loss"] for r in results if r["arm"] == "base")
    print()
    for row in results:
        if row["arm"].startswith("mehldau"):
            print(f"val_loss base {base_loss:.4f} -> {row['arm']} {row['val_loss']:.4f} "
                  f"(delta {row['val_loss'] - base_loss:+.4f})")
    print("Both validation songs are inside the base pretrain set, so this delta "
          "says nothing about\ngeneralisation. It only says whether the adapter "
          "moved the model on material the base\nhad already seen.")
    print("Descriptors are reference values, not quality. Nobody has listened yet.")

    (args.output_dir / "report.json").write_text(json.dumps({
        "schema": "mehldau_adapter_eval_v1",
        "seeds": seeds, "bars": args.bars, "bpm": args.bpm,
        "ngram": NGRAM,
        "val_songs_in_base_pretrain": True,
        "mehldau_files_in_base_pretrain": 18,
        "checkpoints": {name: str(path) for name, path in arms},
        "mehldau_style_verified": False,
        "musical_quality_verified": False,
        "arms": results,
    }, indent=2) + "\n")
    print(f"\nreport: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
