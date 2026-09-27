#!/usr/bin/env python3
"""Build a blind A/B listening set from snapshot_eval generations.

Pairs the same seed from two snapshots, renders both with fluidsynth, and
assigns A/B at random. The key goes to ``key.json``; the listener should open
only ``ANSWER_SHEET.md`` and the wav files. Seeds are fixed up front, not
chosen by any score.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
from pathlib import Path

DEFAULT_SF2 = Path.home() / ".local/share/soundfonts/generaluser-gs/v1.471.sf2"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-dir", type=Path, required=True)
    ap.add_argument("--update-a", type=int, required=True)
    ap.add_argument("--update-b", type=int, required=True)
    ap.add_argument("--seeds", default="1,2,3,4")
    ap.add_argument("--soundfont", type=Path, default=DEFAULT_SF2)
    ap.add_argument("--shuffle-seed", type=int, default=2026)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        ap.error(f"output dir not empty: {args.output_dir}")
    if shutil.which("fluidsynth") is None:
        ap.error("fluidsynth not found")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.shuffle_seed)
    key = {"update_a": args.update_a, "update_b": args.update_b,
           "eval_dir": str(args.eval_dir), "soundfont": str(args.soundfont), "pairs": []}
    for i, seed in enumerate(int(s) for s in args.seeds.split(",")):
        sources = [(args.update_a, args.eval_dir / f"gen_u{args.update_a:03d}_s{seed}.mid"),
                   (args.update_b, args.eval_dir / f"gen_u{args.update_b:03d}_s{seed}.mid")]
        rng.shuffle(sources)
        entry = {"pair": i + 1, "seed": seed}
        for label, (update, midi) in zip("AB", sources):
            wav = args.output_dir / f"pair{i + 1}_{label}.wav"
            subprocess.run(["fluidsynth", "-ni", "-q", "-F", str(wav), "-r", "44100",
                            str(args.soundfont), str(midi)], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            entry[label] = update
        key["pairs"].append(entry)
    (args.output_dir / "key.json").write_text(json.dumps(key, indent=2) + "\n")

    lines = ["# 블라인드 청취 — 답안지", "",
             "`key.json`은 답을 다 쓴 뒤에 여세요. 각 쌍의 A/B는 같은 primer·seed에서 나온 두 모델의 생성입니다.", "",
             "| 쌍 | 더 멜다우 같은 쪽 (A/B/모름) | 음악적으로 더 나은 쪽 (A/B/같음) | 메모 |",
             "|---|---|---|---|"]
    lines += [f"| {p['pair']} | | | |" for p in key["pairs"]]
    (args.output_dir / "ANSWER_SHEET.md").write_text("\n".join(lines) + "\n")
    print(f"{len(key['pairs'])} pairs -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
