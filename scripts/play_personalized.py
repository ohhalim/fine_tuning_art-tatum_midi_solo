#!/usr/bin/env python3
"""Play the personalised solo models without typing checkpoint paths.

Presets (docs/PERSONALIZATION_STATUS.md):
  tatum    Tatum final adapter (#1509) on the common base
  mehldau  Mehldau final adapter (#1497) on the Mehldau-free base
  swap     both in one session; Tatum and Mehldau alternate every 4 bars (#1519),
           or follow the keyboard's Program Change with --live-select (#1525)
  base     the common base with no artist adapter, for comparison

Everything else is the normal runtime (``run_continuous_jazz.py``) with its
current defaults (KV cache, merged LoRA, late fetch, start budget) plus the
chord primer with two sub-blocks per bar as half-bar scheduler blocks. Extra runtime flags can follow ``--``. No quality or style
claim: nobody has listened.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = {
    "tatum": "outputs/final_tatum/export/checkpoint_update518.pt",
    "mehldau": "outputs/clean_base/c2_export/checkpoint_update128.pt",
    "base": "outputs/tvm/common_base/checkpoint_epoch8.pt",
}
PRIMER = "outputs/chord_ab/ii_V_I.mid"


def build_command(preset: str, *, bars: int, bpm: int, chords: str, seed: int, output_dir: Path,
                  input_port: str | None = None, live_select: bool = False,
                  swap_bars: int = 4, capture: bool = False, python: str = sys.executable,
                  root: Path = ROOT, extra=()) -> list[str]:
    if preset not in ("tatum", "mehldau", "swap", "base"):
        raise ValueError(f"unknown preset {preset!r}")
    if live_select and preset != "swap":
        raise ValueError("--live-select applies to the swap preset")
    if live_select and not input_port:
        raise ValueError("--live-select needs --input-port (the keyboard sends Program Change)")
    ckpt = lambda name: str(root / CHECKPOINTS[name])
    cmd = [python, str(root / "scripts" / "run_continuous_jazz.py"),
           "--checkpoint", ckpt("tatum" if preset == "swap" else preset),
           "--conditioning-midi", str(root / PRIMER),
           "--chord-primer", "--chord-blocks-per-bar", "2",
           # Half-bar scheduler blocks: input and adapter choice land about
           # 0.9 s after they arrive instead of 1.8 s (docs/experiments/HALF_BAR_BLOCKS.md).
           "--half-bar-blocks",
           "--chords", chords, "--bars", str(bars), "--bpm", str(bpm), "--seed", str(seed),
           "--output-dir", str(output_dir)]
    if preset == "swap":
        # The two final adapters sit on different bases (#1517): whole-model swap.
        cmd += ["--adapter-name", "tatum", "--swap-adapter", f"mehldau={ckpt('mehldau')}",
                "--allow-different-bases"]
        cmd += (["--adapter-control", "program"] if live_select
                else ["--adapter-schedule", f"tatum:{swap_bars},mehldau:{swap_bars}"])
    if input_port:
        cmd += ["--input-port", input_port]
    if capture:
        cmd += ["--capture"]
    return cmd + [a for a in extra if a != "--"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, choices=["tatum", "mehldau", "swap", "base"])
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--bpm", type=int, default=128)
    ap.add_argument("--chords", default="Dm7,G7,Cmaj7,A7")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--input-port", help="keyboard MIDI input name (optional)")
    ap.add_argument("--live-select", action="store_true",
                    help="swap preset: Program Change 0 = Tatum, 1 = Mehldau, from --input-port")
    ap.add_argument("--swap-bars", type=int, default=4)
    ap.add_argument("--capture", action="store_true", help="record what was sent (see the runtime)")
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the runtime command and exit")
    ap.add_argument("extra", nargs=argparse.REMAINDER, help="after --: more run_continuous_jazz flags")
    args = ap.parse_args(argv)
    out = args.output_dir or ROOT / "outputs" / "play" / f"{args.preset}_seed{args.seed}"
    try:
        cmd = build_command(args.preset, bars=args.bars, bpm=args.bpm, chords=args.chords,
                            seed=args.seed, output_dir=out, input_port=args.input_port,
                            live_select=args.live_select, swap_bars=args.swap_bars,
                            capture=args.capture, extra=args.extra)
    except ValueError as exc:
        ap.error(str(exc))
    if args.dry_run:
        print(" ".join(cmd))
        return 0
    missing = [p for p in cmd if p.endswith(".pt") and not Path(p.split("=", 1)[-1]).exists()]
    if missing:
        ap.error(f"checkpoint not found (run from the main checkout?): {missing}")
    return subprocess.call(cmd, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
