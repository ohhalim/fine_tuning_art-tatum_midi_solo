#!/usr/bin/env python3
"""Hold chords on a virtual port and see whether the runtime follows (#1576).

docs/experiments/LIVE_CHORDS.md. Launches a preset with ``--live-chords
observe|follow``, waits for the first adopted block, then holds each chord of
``--plan`` for ``--hold-s`` seconds. Times are taken from the runtime's own
report (chord onsets and block starts on the session clock), so the probe's
send timing does not need to be exact. Virtual port only: no claim about a
real keyboard, FL Studio, style or quality.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "music_transformer", ROOT / "scripts"):
    sys.path.insert(0, str(p))

from inference.app.fallback import parse_chord  # noqa: E402

PLAN = "Dbmaj7,Gb7,Bm7,Bbm7b5,E7,Abmaj7"


def pcs(chord: str) -> frozenset:
    root, iv = parse_chord(chord)
    return frozenset((root + i) % 12 for i in iv)


def voicing(chord: str) -> list[int]:
    """Root position, root in 36-47: stays below the default split (60)."""
    root, iv = parse_chord(chord)
    return [36 + root + i for i in iv]


def recognition_matches(changes, plan) -> bool:
    """Same chords in the same order, compared by pitch-class content (enharmonics agree)."""
    return [pcs(c["chord"]) for c in changes] == [pcs(c) for c in plan]


def block_notes(report: dict, block: int) -> list[int]:
    """Pitches played in half-bar block ``block`` (played_bars are whole bars)."""
    half = 60.0 / report["bpm"] * 2
    bar = report["played_bars"][block // 2]["notes"] if block // 2 < len(report["played_bars"]) else []
    lo = (block % 2) * half
    return [n[0] for n in bar if lo <= n[1] < lo + half]


def follow_metrics(report: dict) -> dict:
    """Chord-tone ratio of adopted blocks against the chord held at each block's start."""
    lc = report["live_chords"]
    block_ms = 60.0 / report["bpm"] * 2 * 1000
    changes = sorted(lc["changes"], key=lambda c: c["onset_ms"])
    adopted = set(report.get("adopted_blocks", []))
    hits = total = blocks = 0
    per_block = []
    for b in range(report.get("blocks", len(report["played_bars"]) * 2)):
        start = b * block_ms
        held = [c for c in changes if c["onset_ms"] <= start]
        if not held or b not in adopted:
            continue
        target = pcs(held[-1]["chord"])
        notes = block_notes(report, b)
        h = sum(1 for p in notes if p % 12 in target)
        hits += h
        total += len(notes)
        blocks += 1
        per_block.append({"block": b, "target": held[-1]["chord"], "notes": len(notes), "chord_tones": h})
    latencies = [c["latency_ms"] for c in changes if c.get("latency_ms") is not None]
    return {"chord_tone_ratio": hits / total if total else None, "notes": total, "blocks": blocks,
            "latencies_ms": latencies, "seen_latencies_ms": [c["seen_latency_ms"] for c in changes],
            "per_block": per_block}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["observe", "follow"], required=True)
    ap.add_argument("--preset", default="tatum")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bars", type=int, default=16)
    ap.add_argument("--bpm", type=int, default=128)
    ap.add_argument("--static-chords", default="Cmaj7")
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--hold-s", type=float, default=3.75)
    ap.add_argument("--start-after-s", type=float, default=1.0)
    ap.add_argument("--port-name", default="LiveChordProbe")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("runtime_args", nargs=argparse.REMAINDER,
                    help="after --: more run_continuous_jazz flags (e.g. --generation-tokens 192)")
    args = ap.parse_args(argv)
    import mido
    from scripts.play_personalized import build_command

    plan = [c for c in args.plan.split(",") if c]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = build_command(args.preset, bars=args.bars, bpm=args.bpm, chords=args.static_chords, seed=args.seed,
                        output_dir=args.output_dir, input_port=args.port_name,
                        extra=["--live-chords", args.mode, "--live-metrics",
                               *[a for a in args.runtime_args if a != "--"]])
    first_block = threading.Event()
    log = (args.output_dir / "runtime.log").open("w")
    sent = []
    with mido.open_output(args.port_name, virtual=True) as port:
        time.sleep(0.5)   # let CoreMIDI publish the port before the child looks for it
        child = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=ROOT, text=True)

        def pump():
            for line in child.stdout:
                log.write(line)
                if line.startswith("block "):
                    first_block.set()
        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        while not first_block.wait(0.1):
            if child.poll() is not None:
                break
        if first_block.is_set():
            t0 = time.monotonic() + args.start_after_s
            held: list[int] = []
            for i, name in enumerate(plan):
                while time.monotonic() < t0 + i * args.hold_s:
                    time.sleep(0.002)
                for p in held:
                    port.send(mido.Message("note_off", note=p, velocity=0))
                held = voicing(name)
                for p in held:
                    port.send(mido.Message("note_on", note=p, velocity=80))
                sent.append({"chord": name, "at_s": round(time.monotonic() - t0, 3), "notes": held})
            while time.monotonic() < t0 + len(plan) * args.hold_s:
                time.sleep(0.01)
            for p in held:
                port.send(mido.Message("note_off", note=p, velocity=0))
        code = child.wait()
        reader.join(timeout=5)
    log.close()
    out = {"schema": "live_chord_probe_v1", "mode": args.mode, "seed": args.seed, "plan": plan,
           "command": cmd, "exit_code": code, "sent": sent}
    report_path = args.output_dir / "continuous_report.json"
    if code == 0 and report_path.exists():
        report = json.loads(report_path.read_text())
        lc = report.get("live_chords", {})
        out["recognized"] = [c["chord"] for c in lc.get("changes", [])]
        out["recognition_matches"] = recognition_matches(lc.get("changes", []), plan)
        out["follow"] = follow_metrics(report)
        out["fallback_blocks"] = report["production"]["fallback_bar_count"]
        out["deadline_misses"] = report["scheduler_dispatch_deadline_miss_count"]
    (args.output_dir / "probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k not in ("command", "sent")}
                     | {"follow": {k: v for k, v in out.get("follow", {}).items() if k != "per_block"}}, indent=1))
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
