#!/usr/bin/env python3
"""Does the generation follow the chords it was primed with?

Same checkpoint, same seed, same bar count. Only the chord progression behind
the primer changes. If the model is deaf to harmony the arms come out the same.

Arms
----
``none``      no chord guide at all - the current default path, for reference
``<name>``    one arm per progression given with --progression
``shuffled``  the first progression's chords in a fixed shuffled order

The shuffled arm is the one that makes the numbers mean anything. A raw
chord-tone ratio can rise simply because the model likes the notes the chords
happen to contain, so the test is not "is the ratio high" but "is a phrase
scored against its own progression better than the same phrase scored against a
different one". That comparison holds the generated notes fixed and moves only
the yardstick.

Why the 1/3 null is not used as a verdict
-----------------------------------------
Four chord tones out of twelve pitch classes gives 1/3 only if pitch classes
were independent and uniform. Real jazz solos are neither - they sit in a key,
repeat pitches, and lean on scale degrees. Exceeding 1/3 is therefore not
evidence of harmonic response, and falling below it is not evidence of failure.
The report prints it as a reference line only.

    python scripts/run_chord_primer_ab.py --dry-run
    FORCE_CPU=1 uv run python scripts/run_chord_primer_ab.py \
        --checkpoint <ckpt> --output-dir outputs/chord_ab
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pretty_midi

from inference.control.chord_primer import build_chord_primer, chord_tone_pitch_classes

DEFAULT_PROGRESSIONS = {
    "ii_V_I": ["Dm7", "G7", "Cmaj7", "Cmaj7"],
    "minor": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
    "blues": ["C7", "F7", "C7", "G7"],
}


def score_against(notes, progression, *, bars: int, bar_seconds: float,
                  transpose: int = 0) -> float | None:
    """Fraction of notes whose pitch class belongs to the chord sounding then.

    Scoring only. Nothing here filters or rewrites a note.

    ``transpose`` shifts the yardstick, not the music. At 6 semitones it gives
    the clean control: the same progression shape with a pitch-class set that
    barely overlaps the original, so a phrase that genuinely tracks its harmony
    should score clearly worse against it.
    """
    if not notes:
        return None
    hits = 0
    for note in notes:
        bar = min(bars - 1, max(0, int(note.start / bar_seconds)))
        chord = progression[bar % len(progression)]
        tones = {(pc + transpose) % 12 for pc in chord_tone_pitch_classes(chord)}
        if note.pitch % 12 in tones:
            hits += 1
    return hits / len(notes)


def diversity(notes) -> dict[str, float | int]:
    pitches = [n.pitch for n in notes]
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    return {
        "note_count": len(notes),
        "unique_pitch_classes": len({p % 12 for p in pitches}),
        "pitch_range": (max(pitches) - min(pitches)) if pitches else 0,
        "median_interval": statistics.median(intervals) if intervals else 0,
    }


def run_arm(*, name, progression, model, bars, bpm, seed, generation_tokens,
            max_sequence, output_dir):
    """Generate one phrase for one progression and return its measurements."""
    import torch

    from midi_processor.processor import decode_midi
    from scripts.generate import generate_once
    from scripts.run_jazz_mvp import fit_window
    from scripts.run_resident_model_probe import validate_generated_token_block

    bar_seconds = 240.0 / bpm
    phrase = pretty_midi.PrettyMIDI(initial_tempo=float(bpm))
    lead = pretty_midi.Instrument(program=0, name=f"lead_{name}")
    rows, elapsed_ms = [], []

    for bar in range(bars):
        if progression is None:
            tokens_primer, used_chord = [], False
        else:
            # One bar of harmony, so the primer states the chord sounding now.
            tokens_primer, used_chord = build_chord_primer(
                [progression[bar % len(progression)]], bpm=bpm, bars=1
            )
        primer = torch.tensor(tokens_primer or [60], dtype=torch.long)

        torch.manual_seed(seed + bar)
        started = time.perf_counter_ns()
        tokens, _meta = generate_once(
            model=model, primer=primer,
            target_length=min(max_sequence, len(primer) + generation_tokens),
            strip_primer=True, temperature=1.0, top_k=32, top_p=0.95,
            grammar_mask=True, target_duration_seconds=bar_seconds, return_metadata=True,
        )
        elapsed_ms.append((time.perf_counter_ns() - started) / 1e6)

        valid = validate_generated_token_block(tokens, lookahead_ms=bar_seconds * 1000,
                                               allow_rest_bar=True)
        rows.append({"bar": bar, "valid": bool(valid["valid"]), "used_chord": used_chord,
                     "primer_tokens": len(tokens_primer)})
        if not valid["valid"]:
            continue
        bar_midi = fit_window(decode_midi(tokens), bar_seconds)
        for instrument in bar_midi.instruments:
            for note in instrument.notes:
                lead.notes.append(pretty_midi.Note(
                    note.velocity, note.pitch,
                    note.start + bar * bar_seconds, note.end + bar * bar_seconds))

    phrase.instruments = [lead]
    path = Path(output_dir) / f"{name}.mid"
    phrase.write(str(path))
    return {
        "arm": name,
        "progression": progression,
        "valid_bars": sum(1 for r in rows if r["valid"]),
        "bars": bars,
        "generation_ms_p50": statistics.median(elapsed_ms) if elapsed_ms else None,
        "generation_ms_max": max(elapsed_ms) if elapsed_ms else None,
        "diversity": diversity(lead.notes),
        "midi": str(path),
        "_notes": lead.notes,
        "bars_detail": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--bars", type=int, default=8)
    parser.add_argument("--bpm", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generation-tokens", type=int, default=96)
    parser.add_argument("--max-sequence", type=int, default=192)
    parser.add_argument("--progression", action="append",
                        help="name=C7,F7,... (repeatable; defaults to three built-ins)")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/chord_ab"))
    parser.add_argument("--dry-run", action="store_true",
                        help="build primers and print them; loads no model")
    args = parser.parse_args(argv)

    progressions = dict(DEFAULT_PROGRESSIONS)
    if args.progression:
        progressions = {}
        for item in args.progression:
            name, _, chords = item.partition("=")
            progressions[name] = [c.strip() for c in chords.split(",") if c.strip()]

    first = next(iter(progressions.values()))
    shuffled = list(first)
    random.Random(args.seed).shuffle(shuffled)
    if shuffled == first and len(set(first)) > 1:
        shuffled = first[1:] + first[:1]

    if args.dry_run:
        for name, chords in list(progressions.items()) + [("shuffled", shuffled)]:
            tokens, used = build_chord_primer([chords[0]], bpm=args.bpm, bars=1)
            print(f"{name:<10} {chords}  primer {len(tokens)} tokens "
                  f"(max id {max(tokens) if tokens else '-'}, used_chord={used})")
        return 0

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --dry-run")

    from scripts.generate import load_model_with_lora

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model_with_lora(lora_path=str(args.checkpoint.parent),
                                checkpoint_path=str(args.checkpoint),
                                prefer_full_checkpoint=True, max_sequence=args.max_sequence)

    arms = [("none", None)] + list(progressions.items()) + [("shuffled", shuffled)]
    results = []
    for name, chords in arms:
        results.append(run_arm(name=name, progression=chords, model=model, bars=args.bars,
                               bpm=args.bpm, seed=args.seed,
                               generation_tokens=args.generation_tokens,
                               max_sequence=args.max_sequence, output_dir=args.output_dir))

    bar_seconds = 240.0 / args.bpm
    named = [r for r in results if r["progression"] is not None]
    print(f"{'arm':<10}{'valid':<8}{'notes':<7}{'own':<8}{'cross':<8}{'delta':<9}"
          f"{'tri':<8}{'triΔ':<9}{'pc':<5}{'gen p50':<10}")
    for row in results:
        own = cross = delta = None
        if row["progression"] is not None:
            own = score_against(row["_notes"], row["progression"],
                                bars=args.bars, bar_seconds=bar_seconds)
            others = [score_against(row["_notes"], other["progression"],
                                    bars=args.bars, bar_seconds=bar_seconds)
                      for other in named if other["arm"] != row["arm"]]
            others = [o for o in others if o is not None]
            cross = statistics.mean(others) if others else None
            delta = (own - cross) if (own is not None and cross is not None) else None
        tritone = None
        if row["progression"] is not None:
            tritone = score_against(row["_notes"], row["progression"], bars=args.bars,
                                    bar_seconds=bar_seconds, transpose=6)
        row["chord_tone_own"] = own
        row["chord_tone_cross_mean"] = cross
        row["chord_tone_delta"] = delta
        row["chord_tone_tritone"] = tritone
        row["chord_tone_tritone_delta"] = (own - tritone) if (own is not None and tritone is not None) else None
        tri_text = "-" if row["chord_tone_tritone"] is None else f"{row['chord_tone_tritone']:.3f}"
        tri_delta_text = ("-" if row["chord_tone_tritone_delta"] is None
                          else f"{row['chord_tone_tritone_delta']:+.3f}")
        gen_text = "-" if not row["generation_ms_p50"] else f"{row['generation_ms_p50']:.0f}ms"
        own_text = "-" if own is None else f"{own:.3f}"
        cross_text = "-" if cross is None else f"{cross:.3f}"
        delta_text = "-" if delta is None else f"{delta:+.3f}"
        print(f"{row['arm']:<10}{row['valid_bars']}/{row['bars']:<6}"
              f"{row['diversity']['note_count']:<7}{own_text:<8}{cross_text:<8}{delta_text:<9}"
              f"{tri_text:<8}{tri_delta_text:<9}"
              f"{row['diversity']['unique_pitch_classes']:<5}{gen_text:<10}")
    print(f"delta > 0 means the phrase fits the chords it was primed with better "
          f"than chords it never saw.")
    if deltas:
        print(f"mean delta {statistics.mean(deltas):+.3f} over {len(deltas)} arms")
    print("1/3 reference line assumes independent uniform pitch classes, which a "
          "jazz solo is not. It is not a pass mark.")

    for row in results:
        row.pop("_notes", None)
    report = {"schema": "chord_primer_ab_v1", "bpm": args.bpm, "bars": args.bars,
              "seed": args.seed, "conditioning": "note_vocabulary_chord_primer",
              "learned_chord_conditioning": False, "rule_based_note_filtering": False,
              "musical_quality_verified": False, "arms": results}
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nreport: {args.output_dir / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
