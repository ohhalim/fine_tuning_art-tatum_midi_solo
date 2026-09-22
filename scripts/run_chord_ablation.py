#!/usr/bin/env python3
"""Separate two confounds the first chord A/B left tangled.

The earlier experiment compared a chord primer against a bare ``[60]`` primer
and against the default path at the same time, so three things moved together:
whether harmony was stated, whether the control prefix was present, and how
long the primer was. Nothing could be attributed.

Two experiments here, both small.

1. ``--ablation`` — a 2x2 over (chord primer or the default conditioning MIDI)
   x (control prefix present or absent). Generation budget and seeds are equal
   across all four arms. Primer length is not, and cannot be here: the chord
   guide encodes to ~11 tokens while the conditioning MIDI fills the 48-token
   budget, and padding a primer would change what the model reads. So the
   prefix comparison *within* one kind is clean, and the chord-versus-default
   comparison across kinds still carries a length and content difference. The
   bare ``[60]`` arm is gone either way: the no-chord reference is the real
   default path, not a degenerate primer.

2. ``--following`` — does the solo track the progression bar by bar, or does it
   only land in the right key? Generate against a progression, then score each
   bar twice: against the chord that was actually there, and against the chord
   a permutation of the *same multiset* puts there. Whole-piece key fit is
   identical between the two yardsticks by construction, so only bar-to-chord
   alignment differs. Paired, per bar.

Why the control prefix is a variable at all
-------------------------------------------
This checkpoint appears never to have been trained on one:

* armB trained from scratch on ``jazz_full``, which contains 0 control tokens
* ``crop_control_v1_sequence`` reads the prefix out of the data, it never
  synthesises one, so no control token could reach the loss
* the LoRA stage left ``embedding.weight`` bit-identical to armB (max abs
  delta 0.0), so those rows are still at armB's initialisation

That is an argument about weights. Whether it changes the output is what
``--ablation`` measures, and the runtime default is not touched either way
until there is a reason in the numbers.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pretty_midi

from inference.control.chord_primer import chord_guide_notes, chord_tone_pitch_classes

# One key throughout, so key fit cannot explain a difference between orders.
PROGRESSION = ["Dm7", "G7", "Cmaj7", "Am7"]
PERMUTATION = ["Cmaj7", "Am7", "Dm7", "G7"]

PRIMER_MAX_TOKENS = 48

# Below this a per-bar chord-tone ratio is mostly sampling noise: one note gives
# 0.0 or 1.0 and nothing in between.
MIN_NOTES_FOR_RATIO = 8


def _prefix_tokens(bpm: int) -> list[int]:
    from scripts.control_tokens import control_prefix_tokens
    from utilities.constants import TOKEN_COND_SEP

    return control_prefix_tokens(role="lead", tempo_bpm=bpm) + [TOKEN_COND_SEP]


def build_primer(kind: str, chord: str, *, bpm: int, conditioning_midi: Path | None,
                 with_prefix: bool):
    """One primer for one arm.

    ``kind`` is ``"chord"`` (harmony stated as notes) or ``"default"`` (the
    fixed conditioning MIDI the runtime uses today).

    The prefix is paid for out of the body budget, so turning it on cannot push
    the primer past ``PRIMER_MAX_TOKENS``. Length is therefore comparable
    between the two prefix arms of one kind, but not between kinds - the chord
    guide is just shorter than the conditioning MIDI.
    """
    import torch

    from scripts.generate import (
        encode_midi_simple,
        encode_notes_simple,
        truncate_tokens_preserving_velocity,
    )

    if kind == "chord":
        tokens = encode_notes_simple(chord_guide_notes([chord], bpm=bpm, bars=1))
    else:
        if conditioning_midi is None:
            raise ValueError("default arm needs --conditioning-midi")
        tokens = encode_midi_simple(str(conditioning_midi))

    prefix = _prefix_tokens(bpm) if with_prefix else []
    # The prefix costs slots from the body rather than extending the primer, so
    # the two prefix arms of one kind stay the same length.
    body = truncate_tokens_preserving_velocity(tokens, max(1, PRIMER_MAX_TOKENS - len(prefix)))
    return torch.tensor(prefix + body, dtype=torch.long)


def generate_bar(model, primer, *, bpm, seed, generation_tokens, max_sequence):
    import torch

    from midi_processor.processor import decode_midi
    from scripts.generate import generate_once
    from scripts.run_jazz_mvp import fit_window
    from scripts.run_resident_model_probe import validate_generated_token_block

    bar_seconds = 240.0 / bpm
    torch.manual_seed(seed)
    started = time.perf_counter_ns()
    tokens, _meta = generate_once(
        model=model, primer=primer,
        target_length=min(max_sequence, len(primer) + generation_tokens),
        strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
        grammar_mask=True, target_duration_seconds=bar_seconds, return_metadata=True,
    )
    elapsed_ms = (time.perf_counter_ns() - started) / 1e6
    valid = validate_generated_token_block(tokens, lookahead_ms=bar_seconds * 1000,
                                           allow_rest_bar=True)
    if not valid["valid"]:
        return [], elapsed_ms, False
    midi = fit_window(decode_midi(tokens), bar_seconds)
    return ([n for i in midi.instruments for n in i.notes], elapsed_ms, True)


def fit(notes, chord: str) -> float | None:
    """Chord-tone fraction for one bar. Scoring only; nothing is filtered."""
    if not notes:
        return None
    tones = chord_tone_pitch_classes(chord)
    return sum(1 for n in notes if n.pitch % 12 in tones) / len(notes)


def run_ablation(model, *, bars, bpm, seeds, generation_tokens, max_sequence,
                 conditioning_midi, output_dir):
    """2x2: chord primer or default conditioning, prefix present or absent."""
    arms = [(kind, prefix) for kind in ("default", "chord") for prefix in (True, False)]
    results = []
    for kind, with_prefix in arms:
        name = f"{kind}_{'prefix' if with_prefix else 'noprefix'}"
        per_seed = []
        for seed in seeds:
            notes_all, elapsed, valid = [], [], 0
            for bar in range(bars):
                chord = PROGRESSION[bar % len(PROGRESSION)]
                primer = build_primer(kind, chord, bpm=bpm,
                                      conditioning_midi=conditioning_midi,
                                      with_prefix=with_prefix)
                notes, ms, ok = generate_bar(model, primer, bpm=bpm, seed=seed + bar,
                                             generation_tokens=generation_tokens,
                                             max_sequence=max_sequence)
                elapsed.append(ms)
                valid += int(ok)
                for note in notes:
                    notes_all.append((bar, note))
            own = [f for f in (fit([n for b, n in notes_all if b == bar],
                                   PROGRESSION[bar % len(PROGRESSION)])
                               for bar in range(bars)) if f is not None]
            per_seed.append({
                "seed": seed, "valid_bars": valid,
                "note_count": len(notes_all),
                "chord_tone_own": statistics.mean(own) if own else None,
                "primer_tokens": len(primer),
                "generation_ms_p50": statistics.median(elapsed) if elapsed else None,
            })
        results.append({"arm": name, "kind": kind, "with_prefix": with_prefix,
                        "seeds": per_seed})

    print(f"{'arm':<20}{'primer':<9}{'valid':<9}{'notes':<8}{'own':<8}{'gen p50':<10}")
    for row in results:
        valid = sum(s["valid_bars"] for s in row["seeds"])
        total = bars * len(seeds)
        notes = sum(s["note_count"] for s in row["seeds"])
        owns = [s["chord_tone_own"] for s in row["seeds"] if s["chord_tone_own"] is not None]
        gens = [s["generation_ms_p50"] for s in row["seeds"] if s["generation_ms_p50"]]
        print(f"{row['arm']:<20}{row['seeds'][0]['primer_tokens']:<9}{f'{valid}/{total}':<9}"
              f"{notes:<8}{(f'{statistics.mean(owns):.3f}' if owns else '-'):<8}"
              f"{(f'{statistics.mean(gens):.0f}ms' if gens else '-'):<10}")
    lengths = {row["arm"]: row["seeds"][0]["primer_tokens"] for row in results}
    print(f"\ngeneration budget and seeds are equal across arms. Primer length is NOT:\n"
          f"  {lengths}")
    print("The chord guide is simply shorter than the conditioning MIDI, and the budget\n"
          "only truncates, it never pads. So comparing a chord row against a default row\n"
          "still mixes harmony with primer length and content. Comparing the two rows\n"
          "*within* one kind isolates the prefix, which is the question this answers.")
    return results


def run_following(model, *, bars, bpm, seeds, generation_tokens, max_sequence, output_dir):
    """Per-bar paired test: the true chord versus a permutation of the same multiset."""
    rows = []
    for seed in seeds:
        for bar in range(bars):
            true_chord = PROGRESSION[bar % len(PROGRESSION)]
            other_chord = PERMUTATION[bar % len(PERMUTATION)]
            primer = build_primer("chord", true_chord, bpm=bpm,
                                  conditioning_midi=None, with_prefix=False)
            notes, _ms, ok = generate_bar(model, primer, bpm=bpm, seed=seed + bar,
                                          generation_tokens=generation_tokens,
                                          max_sequence=max_sequence)
            if not ok or not notes:
                continue
            true_fit, other_fit = fit(notes, true_chord), fit(notes, other_chord)
            shared = chord_tone_pitch_classes(true_chord) & chord_tone_pitch_classes(other_chord)
            rows.append({"seed": seed, "bar": bar, "true_chord": true_chord,
                         "permuted_chord": other_chord, "note_count": len(notes),
                         "fit_true": true_fit, "fit_permuted": other_fit,
                         "paired_delta": true_fit - other_fit,
                         "shared_chord_tones": len(shared),
                         "identical_chord": true_chord == other_chord})

    scored = [r for r in rows if not r["identical_chord"]]
    print(f"\n{'seed':<7}{'bar':<5}{'true':<8}{'permuted':<10}{'n':<5}"
          f"{'fit_true':<10}{'fit_perm':<10}{'delta':<9}{'shared':<7}")
    for r in scored:
        print(f"{r['seed']:<7}{r['bar']:<5}{r['true_chord']:<8}{r['permuted_chord']:<10}"
              f"{r['note_count']:<5}{r['fit_true']:<10.3f}{r['fit_permuted']:<10.3f}"
              f"{r['paired_delta']:<+9.3f}{r['shared_chord_tones']:<7}")

    if scored:
        def summarise(sel, label):
            if not sel:
                return
            deltas = [r["paired_delta"] for r in sel]
            notes = sum(r["note_count"] for r in sel)
            # Note-weighted, because a one-note bar yields a ratio of 0 or 1 and
            # would otherwise carry the same weight as a twenty-note bar.
            weighted = sum(r["paired_delta"] * r["note_count"] for r in sel) / notes
            true_w = sum(r["fit_true"] * r["note_count"] for r in sel) / notes
            perm_w = sum(r["fit_permuted"] * r["note_count"] for r in sel) / notes
            print(f"  {label:<18} bars {len(sel):>2}  notes {notes:>3}  "
                  f"mean {statistics.mean(deltas):+.3f}  weighted {weighted:+.3f}  "
                  f"(true {true_w:.3f} vs permuted {perm_w:.3f})  "
                  f"positive {sum(1 for d in deltas if d > 0)}/{len(deltas)}")

        print("\npaired delta (true chord minus permuted chord):")
        summarise(scored, "all bars")
        summarise([r for r in scored if r["note_count"] >= MIN_NOTES_FOR_RATIO],
                  f"notes >= {MIN_NOTES_FOR_RATIO}")
        summarise([r for r in scored if r["note_count"] < MIN_NOTES_FOR_RATIO],
                  f"notes < {MIN_NOTES_FOR_RATIO}")
        print(f"  note counts: {sorted(r['note_count'] for r in scored)}")
        print(f"The notes >= {MIN_NOTES_FOR_RATIO} row is the one to read. A bar with three\n"
              "notes gives a ratio that is almost all sampling noise, and those bars pull\n"
              "the unweighted mean around.")
        print("Bars inside one run share a primer policy and a model state, so these are\n"
              "not independent samples; the count is descriptive, not a significance test.")
        shared = {r["shared_chord_tones"] for r in scored}
        print(f"shared chord tones per pair: {sorted(shared)} of 4 - the more they share,\n"
              "the less this test can separate, and diatonic sevenths share a lot.")
        identical = len(rows) - len(scored)
        if identical:
            print(f"{identical} bars dropped: permutation put the same chord there.")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--conditioning-midi", type=Path)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=int, default=128)
    parser.add_argument("--seeds", default="42,100,200")
    parser.add_argument("--generation-tokens", type=int, default=96)
    parser.add_argument("--max-sequence", type=int, default=192)
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--following", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chord_ablation"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if args.dry_run:
        print(f"progression  {PROGRESSION}")
        print(f"permutation  {PERMUTATION}  (same multiset, one key)")
        for bar in range(4):
            true_chord, other = PROGRESSION[bar], PERMUTATION[bar]
            shared = chord_tone_pitch_classes(true_chord) & chord_tone_pitch_classes(other)
            print(f"  bar {bar}: {true_chord:<8} vs {other:<8} shared tones {len(shared)}/4")
        return 0

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --dry-run")
    if not (args.ablation or args.following):
        parser.error("pick --ablation, --following, or both")

    from scripts.generate import load_model_with_lora

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_with_lora(lora_path=str(args.checkpoint.parent),
                                checkpoint_path=str(args.checkpoint),
                                prefer_full_checkpoint=True, max_sequence=args.max_sequence)

    report = {"schema": "chord_ablation_v1", "bpm": args.bpm, "bars": args.bars,
              "seeds": seeds, "progression": PROGRESSION, "permutation": PERMUTATION,
              "primer_max_tokens": PRIMER_MAX_TOKENS,
              "learned_chord_conditioning": False,
              "chord_following_verified": False,
              "musical_quality_verified": False}
    if args.ablation:
        report["ablation"] = run_ablation(
            model, bars=args.bars, bpm=args.bpm, seeds=seeds,
            generation_tokens=args.generation_tokens, max_sequence=args.max_sequence,
            conditioning_midi=args.conditioning_midi, output_dir=args.output_dir)
    if args.following:
        report["following"] = run_following(
            model, bars=args.bars, bpm=args.bpm, seeds=seeds,
            generation_tokens=args.generation_tokens, max_sequence=args.max_sequence,
            output_dir=args.output_dir)

    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport: {args.output_dir / 'report.json'}")
    print("Key fit, bar-by-bar following and human musical judgement are three "
          "different claims. Only the first two are measured here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
