#!/usr/bin/env python3
"""T1: generate with the public WJazzD CMT checkpoint over our progressions (docs/experiments/T1_PUBLIC_CHECKPOINTS.md).

The CMT code (ckycky3/CMT-pytorch, no license stated) is imported from ``--cmt-dir``
outside this repository and not copied. Input per the checkpoint's hparams: 16 frames
per bar, a 12-d chord chroma per frame (index k = frame k, one extra frame after the
last), a one-bar ``tonic_held`` prime (the HSE pipeline default). Writes the raw
output (CMT's own 120 BPM grid, velocity 100) and the pitch/rhythm index sequences.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PROGRESSIONS = {"P": ["Gm7", "C7", "Fmaj7", "Fmaj7"] * 2, "Q": ["Gm7b5", "C7", "Fm7", "Fm7"] * 2}
BASIS_NOTE = 60                 # CMT pitch index 0 = C4 (utils.pitch_to_midi default)
SUSTAIN, REST = 48, 49          # pitch_range, pitch_range + 1 (preprocess.py)
RHYTHM_ONSET, RHYTHM_HOLD = 2, 1
CMT_SECONDS_PER_FRAME = 1 / 8   # (frame_per_bar / 4) * 2 frames per second in pitch_to_midi
# Post-hoc prime (T1 result): bar 1 of the 8th-note head BebopNet received (ours_iiVI_F.xml), over Gm7.
# The preregistered tonic_held prime (one whole note) gave whole/half-note output.
HEAD_BAR1 = [67, 69, 70, 72, 74, 72, 70, 69]


def chroma(chords, frames_per_bar: int):
    """[(len(chords) * frames_per_bar) + 1, 12]; the extra frame continues the progression."""
    import numpy as np
    from inference.control.harmony_contract import chord_pcs

    rows = []
    for k in range(len(chords) * frames_per_bar + 1):
        _, pcs = chord_pcs(chords[(k // frames_per_bar) % len(chords)])
        row = np.zeros(12, dtype=np.float32)
        row[list(pcs)] = 1.0
        rows.append(row)
    return np.stack(rows)


def read_hparams(text: str) -> dict:
    """The two-level ``section: / key: scalar`` part of CMT's hparams.yaml (no PyYAML in this venv)."""
    out, section = {}, None
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(("#", "-")):
            continue
        indent = len(line) - len(line.lstrip())
        key, _, value = line.strip().partition(":")
        value = value.strip().strip("'\"")
        if indent == 0:
            section = key
            out[section] = {}
        elif indent == 2 and section is not None and value:
            try:
                out[section][key] = int(value)
            except ValueError:
                try:
                    out[section][key] = float(value)
                except ValueError:
                    out[section][key] = {"True": True, "False": False}.get(value, value)
    return out


def to_notes(pitch_idx):
    """[(pitch, start, end)] on CMT's grid, as utils.pitch_to_midi reads the indices."""
    out, on = [], None
    for t, idx in enumerate(pitch_idx):
        if idx < SUSTAIN:
            if on is not None:
                out.append((on[0], on[1], t * CMT_SECONDS_PER_FRAME))
            on = (BASIS_NOTE + int(idx), t * CMT_SECONDS_PER_FRAME)
        elif idx == REST and on is not None:
            out.append((on[0], on[1], t * CMT_SECONDS_PER_FRAME))
            on = None
    if on is not None:
        out.append((on[0], on[1], len(pitch_idx) * CMT_SECONDS_PER_FRAME))
    return out


def main(argv=None) -> int:
    import numpy as np
    import pretty_midi
    import torch

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cmt-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", default="ckpt/best_jazz_model_8bars.pth.tar")
    ap.add_argument("--hparams", default="ckpt/hparams.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--prime", choices=["tonic_held", "head"], default="tonic_held")
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(args.cmt_dir))
    from model import ChordConditionedMelodyTransformer

    hp = read_hparams((args.cmt_dir / args.hparams).read_text())
    m, ex = hp["model"], hp["experiment"]
    model = ChordConditionedMelodyTransformer(**m)
    state = torch.load(args.cmt_dir / args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    fpb, bars, prime_len, topk = m["frame_per_bar"], m["num_bars"], ex["num_prime"], ex["topk"]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"checkpoint": args.checkpoint, "epoch": state.get("epoch"), "hparams_model": m, "topk": topk,
              "prime": args.prime, "runs": []}
    for name, chords in PROGRESSIONS.items():
        assert len(chords) == bars, (len(chords), bars)
        chord = torch.tensor(chroma(chords, fpb)).unsqueeze(0)
        prime_pitch = torch.full((1, prime_len), SUSTAIN, dtype=torch.long)
        prime_rhythm = torch.full((1, prime_len), RHYTHM_HOLD, dtype=torch.long)
        if args.prime == "tonic_held":                       # G4 (index 7) held for the whole bar
            onsets = [(0, 67)]
        else:                                                # eighth notes: an onset every second frame
            onsets = [(2 * k, p) for k, p in enumerate(HEAD_BAR1)]
        for frame, pitch in onsets:
            prime_pitch[0, frame] = pitch - BASIS_NOTE
            prime_rhythm[0, frame] = RHYTHM_ONSET
        for seed in args.seeds:
            torch.manual_seed(seed)
            np.random.seed(seed)
            t_gen = time.time()
            with torch.no_grad():
                res = model.sampling(prime_rhythm, prime_pitch, chord, topk=topk)
            gen_s = time.time() - t_gen
            pitch_idx = res["pitch"][0].tolist()
            rhythm_idx = res["rhythm"][0].tolist()
            notes = to_notes(pitch_idx)
            pm = pretty_midi.PrettyMIDI(initial_tempo=120)
            inst = pretty_midi.Instrument(program=0, name="melody")
            inst.notes = [pretty_midi.Note(velocity=100, pitch=p, start=s, end=e) for p, s, e in notes]
            pm.instruments.append(inst)
            stem = f"cmt_{args.prime}_{name}_s{seed}"
            pm.write(str(args.output_dir / f"{stem}_raw.mid"))
            report["runs"].append({"progression": name, "chords": chords, "seed": seed, "notes": len(notes),
                                   "pitch_range": [min((p for p, _, _ in notes), default=None),
                                                   max((p for p, _, _ in notes), default=None)],
                                   "pitch_idx": pitch_idx, "rhythm_idx": rhythm_idx, "file": f"{stem}_raw.mid",
                                   "gen_seconds": round(gen_s, 3)})
            print(stem, "notes", len(notes), "range", report["runs"][-1]["pitch_range"])
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
