#!/usr/bin/env python3
"""Start the runtime between FL Studio and the IAC buses (docs/phase1/TYPING_KEYBOARD_FL.md).

FL project: iCloud ``mvp/mvp.flp`` (MIDI Out channel on port 5, Serum #2 input
port 7). macOS IAC Driver: port "AI In" (FL -> runtime) and a second bus
(runtime -> FL). Port names are found here because the Korean IAC device name
reaches Python garbled. Ctrl-C stops; all notes are then switched off.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def find_ports(inputs, outputs, *, source_hint: str = "AI In"):
    src = next((n for n in inputs if source_hint in n), None)
    dst = next((n for n in outputs if source_hint not in n and "IAC" in n), None)
    if src is None or dst is None:
        raise SystemExit(f"IAC ports not found (inputs {inputs}, outputs {outputs}); "
                         "turn on IAC Driver with ports 'AI In' and a second bus")
    return src, dst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="tatum", choices=["tatum", "mehldau", "base"])
    ap.add_argument("--bars", type=int, default=64)
    ap.add_argument("--bpm", type=int, default=128)
    ap.add_argument("--chords", default="Cmaj7")
    ap.add_argument("--follow", action=argparse.BooleanOptionalAction, default=True,
                    help="use the held chord as the next block's chord (else keep --chords)")
    ap.add_argument("--solo", action=argparse.BooleanOptionalAction, default=True,
                    help="send only the top line (no chords) of what the model plays")
    ap.add_argument("--comp", action=argparse.BooleanOptionalAction, default=True,
                    help="with --solo: soft chord voicing once a bar so the progression is audible")
    args = ap.parse_args(argv)
    import mido

    src, dst = find_ports(mido.get_input_names(), mido.get_output_names())
    cmd = [sys.executable, str(ROOT / "scripts" / "play_personalized.py"), "--preset", args.preset,
           "--bars", str(args.bars), "--bpm", str(args.bpm), "--chords", args.chords,
           "--input-port", src, "--output-dir", str(ROOT / "outputs" / "fl_live" / args.preset), "--",
           "--port", dst, "--live-chords", "follow" if args.follow else "observe",
           "--chord-split", "128", "--ignore-echo-ms", "30", "--live-metrics",
           *(["--solo-line"] if args.solo else []), *(["--comp"] if args.solo and args.comp else [])]
    try:
        return subprocess.call(cmd, cwd=ROOT, env={**os.environ, "FORCE_CPU": "1"})
    except KeyboardInterrupt:
        return 130
    finally:
        with mido.open_output(dst) as out:
            for ch in range(16):
                out.send(mido.Message("control_change", channel=ch, control=123, value=0))


if __name__ == "__main__":
    raise SystemExit(main())
