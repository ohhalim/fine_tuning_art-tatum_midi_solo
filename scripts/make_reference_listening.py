#!/usr/bin/env python3
"""Listening v2: reference X from a real Mehldau song, then A/B, "which is closer to X?".

Pre-registered in docs/experiments/MEHLDAU_STYLE_SHIFT.md. Every clip is a
fixed-length window so length and density are comparable:

* X: middle ``clip_seconds`` of Mehldau train song i
* A/B: first ``clip_seconds`` of the same-seed generation from two snapshots

A/B order is random; the key goes to ``key.json``.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "music_transformer" / "third_party"))

import numpy as np

MAIN_REPO = Path("/Users/ohhalim/git_box/fine_tuning_art-tatum_midi_solo")
DEFAULT_SF2 = Path.home() / ".local/share/soundfonts/generaluser-gs/v1.471.sf2"


def window(notes, start: float, length: float):
    """Notes overlapping [start, start+length), shifted to 0 and clipped at the end."""
    import pretty_midi

    out = []
    end = start + length
    for n in notes:
        if n.start >= end or n.end <= start or n.start < start:
            continue
        out.append(pretty_midi.Note(n.velocity, n.pitch, n.start - start, min(n.end, end) - start))
    return out


def write_clip(notes, path: Path) -> None:
    import pretty_midi

    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)
    inst.notes = notes
    pm.instruments.append(inst)
    pm.write(str(path))


def render(midi: Path, wav: Path, sf2: Path) -> None:
    subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(wav), "-r", "44100", str(sf2), str(midi)],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--update-a", type=int, required=True)
    ap.add_argument("--update-b", type=int, required=True)
    ap.add_argument("--seeds", default="1,2,3,4,5,6")
    ap.add_argument("--data-dir", type=Path, default=MAIN_REPO / "data/mehldau_full")
    ap.add_argument("--clip-seconds", type=float, default=15.0)
    ap.add_argument("--soundfont", type=Path, default=DEFAULT_SF2)
    ap.add_argument("--shuffle-seed", type=int, default=2027)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"output dir not empty: {args.output_dir}")
    if shutil.which("fluidsynth") is None:
        ap.error("fluidsynth not found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work = args.output_dir / "midi"
    work.mkdir()

    import pretty_midi
    from scripts.style_distance import tokens_to_notes

    songs = sorted((args.data_dir / "train").glob("*.npy"))
    rng = random.Random(args.shuffle_seed)
    key = {"update_a": args.update_a, "update_b": args.update_b, "clip_seconds": args.clip_seconds,
           "eval_dir": str(args.eval_dir), "pairs": []}
    for i, seed in enumerate(int(s) for s in args.seeds.split(",")):
        ref_notes = tokens_to_notes(np.load(songs[i]).ravel())
        total = max(n.end for n in ref_notes)
        x = window(ref_notes, max(0.0, total / 2 - args.clip_seconds / 2), args.clip_seconds)
        write_clip(x, work / f"pair{i + 1}_X.mid")
        render(work / f"pair{i + 1}_X.mid", args.output_dir / f"pair{i + 1}_X.wav", args.soundfont)

        sources = [args.update_a, args.update_b]
        rng.shuffle(sources)
        entry = {"pair": i + 1, "seed": seed, "reference_song": songs[i].name,
                 "reference_notes": len(x)}
        for label, update in zip("AB", sources):
            gen = pretty_midi.PrettyMIDI(str(args.eval_dir / f"gen_u{update:03d}_s{seed}.mid"))
            notes = sorted((n for inst in gen.instruments for n in inst.notes), key=lambda n: n.start)
            first = notes[0].start if notes else 0.0
            clip = window(notes, first, args.clip_seconds)
            write_clip(clip, work / f"pair{i + 1}_{label}.mid")
            render(work / f"pair{i + 1}_{label}.mid", args.output_dir / f"pair{i + 1}_{label}.wav",
                   args.soundfont)
            entry[label] = update
            entry[f"{label}_notes"] = len(clip)
        key["pairs"].append(entry)
    (args.output_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")

    lines = ["# 청취 v2 — 답안지", "",
             "쌍마다 **X(실제 멜다우 발췌)를 먼저** 듣고, A와 B 중 **X에 더 가까운 쪽**을 고르세요.",
             "음악적 선호가 아니라 X와의 유사성입니다. 모르면 '모름'. `key.json`은 다 쓴 뒤에 여세요.", "",
             "| 쌍 | X에 더 가까운 쪽 (A/B/모름) | 메모 |", "|---|---|---|"]
    lines += [f"| {p['pair']} | | |" for p in key["pairs"]]
    (args.output_dir / "ANSWER_SHEET.md").write_text("\n".join(lines) + "\n")
    print(f"{len(key['pairs'])} pairs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
