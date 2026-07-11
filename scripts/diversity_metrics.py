"""
Diversity metrics for Music-Transformer symbolic output.

WHY: cross-entropy / perplexity cannot detect the failure modes you observed
("voicing-diversity collapse", "harmonic-progression repetition", "long-range
structure breakdown"). A model can lower loss while producing a narrower, more
repetitive distribution of music. These metrics measure that distribution
directly, so D0-(a) 16-file probe vs D0-(b) full-corpus can be compared on the
axis that actually matters.

WORKS ON: legacy Music-Transformer ("performance") tokenization — the scheme
your training shards actually use (verified: tokens in 0..388).
  note_on   0..127     pitch onset
  note_off  128..255   pitch = token-128
  time_shift 256..355  time += (token-256+1)/100 sec   (10ms .. 1000ms/step)
  velocity  356..387   vel = (token-356)*4

USAGE
  # a directory of .npy token shards (training data or decoded generations)
  ./.venv/bin/python scripts/diversity_metrics.py --npy_dir data/roles/lead/tokenized/train

  # compare two sets side by side (e.g. probe vs full corpus, or data vs samples)
  ./.venv/bin/python scripts/diversity_metrics.py \
      --npy_dir data/roles/lead/tokenized/train \
      --compare_dir data/jazz_processed/train \
      --labels probe full --json_out outputs/diversity_probe_vs_full.json

  # a single sample as a python API
  from scripts.diversity_metrics import metrics_for_tokens
  m = metrics_for_tokens(np.load("sample.npy"))

Each metric is a scalar (or short vector) per sequence; the CLI reports the
mean/std across sequences in a set, plus corpus-level pooled metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np

# ---- authoritative token layout (from third_party/midi_processor/processor.py)
NOTE_ON_START, NOTE_ON_END = 0, 127
NOTE_OFF_START, NOTE_OFF_END = 128, 255
TIME_SHIFT_START, TIME_SHIFT_END = 256, 355
VELOCITY_START, VELOCITY_END = 356, 387
TIME_STEP_SEC = 0.01  # each time-shift step = (value+1)/100 s


# =============================================================================
# Decode: token stream -> note events
# =============================================================================
def decode_notes(tokens: Sequence[int]):
    """Return list of (onset_sec, pitch, duration_sec, velocity).

    Mirrors the processor's event semantics: a NOTE_ON opens a note at the
    current timeline; the matching NOTE_OFF closes it. Unmatched ONs are closed
    at end-of-sequence. TIME_SHIFT advances the timeline.
    """
    t = 0.0
    vel = 0
    open_notes = {}  # pitch -> (onset, velocity)
    notes = []
    for tok in tokens:
        tok = int(tok)
        if NOTE_ON_START <= tok <= NOTE_ON_END:
            pitch = tok - NOTE_ON_START
            open_notes[pitch] = (t, vel)
        elif NOTE_OFF_START <= tok <= NOTE_OFF_END:
            pitch = tok - NOTE_OFF_START
            if pitch in open_notes:
                onset, v = open_notes.pop(pitch)
                notes.append((onset, pitch, max(t - onset, 0.0), v))
        elif TIME_SHIFT_START <= tok <= TIME_SHIFT_END:
            t += (tok - TIME_SHIFT_START + 1) * TIME_STEP_SEC
        elif VELOCITY_START <= tok <= VELOCITY_END:
            vel = (tok - VELOCITY_START) * 4
        # control/pad tokens (>=389) and anything else: ignored
    for pitch, (onset, v) in open_notes.items():
        notes.append((onset, pitch, 0.0, v))
    notes.sort(key=lambda n: (n[0], n[1]))
    return notes


# =============================================================================
# Helpers
# =============================================================================
def _entropy(counts) -> float:
    """Shannon entropy (bits) of a Counter / iterable of counts."""
    vals = np.asarray(list(counts.values()) if isinstance(counts, dict) else list(counts), float)
    tot = vals.sum()
    if tot <= 0:
        return 0.0
    p = vals[vals > 0] / tot
    return float(-(p * np.log2(p)).sum())


def group_voicings(notes, window_sec: float = 0.05):
    """Group near-simultaneous note onsets into chords (pitch-class sets).

    Returns list of frozenset(pitch_class). A window of 50 ms clusters notes a
    human hears as one voicing (rolled chords included).
    """
    if not notes:
        return []
    onsets = np.array([n[0] for n in notes])
    order = np.argsort(onsets)
    voicings = []
    cur_pcs, cur_t0 = set(), None
    for i in order:
        onset, pitch = notes[i][0], notes[i][1]
        if cur_t0 is None or onset - cur_t0 <= window_sec:
            if cur_t0 is None:
                cur_t0 = onset
            cur_pcs.add(pitch % 12)
        else:
            voicings.append(frozenset(cur_pcs))
            cur_pcs, cur_t0 = {pitch % 12}, onset
    if cur_pcs:
        voicings.append(frozenset(cur_pcs))
    return voicings


def ngram_repetition(seq, n: int = 4) -> float:
    """Fraction of n-grams that are repeats (1 - unique/total). 0 = all unique."""
    if len(seq) < n:
        return 0.0
    grams = [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


# =============================================================================
# Per-sequence metrics
# =============================================================================
def notes_from_midi(path):
    """Read a .mid/.midi into the same (onset, pitch, dur, vel) note list.

    Uses pretty_midi directly (not a token round-trip) so metrics on generated
    MIDI are independent of encode/decode quantization.
    """
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(path))
    notes = []
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            notes.append((float(n.start), int(n.pitch), float(n.end - n.start), int(n.velocity)))
    notes.sort(key=lambda n: (n[0], n[1]))
    return notes


def metrics_for_midi(path) -> dict:
    return _metrics_from_notes(notes_from_midi(path), token_seq=None)


def metrics_for_tokens(tokens) -> dict:
    tokens = np.asarray(tokens).ravel()
    return _metrics_from_notes(decode_notes(tokens), token_seq=tokens)


def _metrics_from_notes(notes, token_seq=None) -> dict:
    n_notes = len(notes)

    pitches = [p for _, p, _, _ in notes]
    pcs = [p % 12 for p in pitches]
    voicings = group_voicings(notes)
    voicing_sizes = [len(v) for v in voicings]
    span = (notes[-1][0] - notes[0][0]) if n_notes > 1 else 0.0

    # inter-onset intervals (rhythm)
    onsets = sorted(n[0] for n in notes)
    iois = np.diff(onsets) if len(onsets) > 1 else np.array([])

    return {
        "n_notes": n_notes,
        "n_voicings": len(voicings),
        "duration_sec": round(span, 2),
        "note_density_per_sec": round(n_notes / span, 3) if span > 0 else 0.0,
        # --- pitch / harmony diversity ---
        "pitch_class_entropy_bits": round(_entropy(Counter(pcs)), 3),   # max 3.585 (=log2 12)
        "pitch_range_semitones": (max(pitches) - min(pitches)) if pitches else 0,
        "unique_voicings": len(set(voicings)),
        "voicing_entropy_bits": round(_entropy(Counter(voicings)), 3),
        "distinct_voicing_ratio": round(len(set(voicings)) / len(voicings), 3) if voicings else 0.0,
        "mean_voicing_size": round(float(np.mean(voicing_sizes)), 3) if voicing_sizes else 0.0,
        # --- repetition (progression / motif) ---
        "voicing_4gram_repeat": round(ngram_repetition(voicings, 4), 3),
        "pitch_4gram_repeat": round(ngram_repetition(pitches, 4), 3),
        "token_8gram_repeat": round(ngram_repetition(token_seq.tolist(), 8), 3) if token_seq is not None else -1.0,
        # --- rhythm ---
        "ioi_cv": round(float(np.std(iois) / np.mean(iois)), 3) if iois.size and np.mean(iois) > 0 else 0.0,
    }


# =============================================================================
# Set-level aggregation
# =============================================================================
def _load_dir(d: Path):
    """Load a set as list of (name, notes, token_seq|None).

    Accepts .npy token shards (token_seq preserved) OR .mid/.midi (token_seq None).
    """
    npy = sorted(d.glob("*.npy"))
    mids = sorted(list(d.glob("*.mid")) + list(d.glob("*.midi")))
    if npy:
        out = []
        for f in npy:
            toks = np.load(f).ravel()
            out.append((f.name, decode_notes(toks), toks))
        return out
    if mids:
        return [(f.name, notes_from_midi(f), None) for f in mids]
    return []


def summarize_set(data_dir: Path) -> dict:
    items = _load_dir(data_dir)
    if not items:
        raise SystemExit(f"No .npy or .mid files in {data_dir}")
    per_seq = [_metrics_from_notes(notes, token_seq=toks) for _, notes, toks in items]

    keys = [k for k in per_seq[0] if isinstance(per_seq[0][k], (int, float))]
    agg = {}
    for k in keys:
        vals = np.array([m[k] for m in per_seq], float)
        vals = vals[vals >= 0]  # drop sentinel -1 (token metric absent for MIDI)
        if vals.size == 0:
            continue
        agg[k] = {"mean": round(float(vals.mean()), 3), "std": round(float(vals.std()), 3)}

    # corpus-level pooled diversity: unique voicings across the WHOLE set
    all_voicings = []
    for _, notes, _toks in items:
        all_voicings.extend(group_voicings(notes))
    pooled = {
        "n_sequences": len(items),
        "pooled_unique_voicings": len(set(all_voicings)),
        "pooled_total_voicings": len(all_voicings),
        "pooled_voicing_entropy_bits": round(_entropy(Counter(all_voicings)), 3),
        "pooled_distinct_voicing_ratio": round(len(set(all_voicings)) / len(all_voicings), 3) if all_voicings else 0.0,
    }
    return {"per_sequence_mean_std": agg, "pooled": pooled}


def main() -> int:
    ap = argparse.ArgumentParser(description="Diversity metrics for symbolic music tokens")
    ap.add_argument("--npy_dir", required=True, help="Directory of .npy token shards")
    ap.add_argument("--compare_dir", default=None, help="Second set to compare against")
    ap.add_argument("--labels", nargs=2, default=["A", "B"], help="Labels for the two sets")
    ap.add_argument("--json_out", default=None, help="Write full results to this JSON path")
    args = ap.parse_args()

    results = {args.labels[0]: summarize_set(Path(args.npy_dir))}
    if args.compare_dir:
        results[args.labels[1]] = summarize_set(Path(args.compare_dir))

    # console: the diversity-critical rows
    keyrows = [
        ("pitch_class_entropy_bits", "PC entropy (max 3.58)"),
        ("voicing_entropy_bits", "voicing entropy"),
        ("distinct_voicing_ratio", "distinct-voicing ratio"),
        ("voicing_4gram_repeat", "voicing 4gram repeat (lower=better)"),
        ("token_8gram_repeat", "token 8gram repeat (lower=better)"),
        ("note_density_per_sec", "note density /s"),
    ]
    labels = list(results.keys())
    print("\n" + "=" * 72)
    print(f"{'metric':<38}" + "".join(f"{l:>16}" for l in labels))
    print("-" * 72)
    for k, desc in keyrows:
        row = f"{desc:<38}"
        for l in labels:
            m = results[l]["per_sequence_mean_std"].get(k)
            if m is None:
                row += f"{'n/a':>16}"
            else:
                row += f"{m['mean']:>10.3f}±{m['std']:<5.3f}"
        print(row)
    print("-" * 72)
    prow = f"{'pooled unique voicings':<38}"
    erow = f"{'pooled voicing entropy':<38}"
    for l in labels:
        prow += f"{results[l]['pooled']['pooled_unique_voicings']:>16}"
        erow += f"{results[l]['pooled']['pooled_voicing_entropy_bits']:>16.3f}"
    print(prow)
    print(erow)
    print("=" * 72)
    print("Collapse signature: LOW pitch/voicing entropy + LOW distinct ratio")
    print("                    + HIGH n-gram repeat. Compare probe vs full corpus.")

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(results, indent=2))
        print(f"\nFull results -> {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
