#!/usr/bin/env python3
"""Drive a continuous session and switch adapters live from a MIDI port.

Opens a virtual MIDI output (the "controller"), starts
``run_continuous_jazz.py --input-port <it> --adapter-control ...`` as a child
process, sends Program Change (or CC) messages at fixed times after launch,
then reads the report: for every selector message, the bar it arrived in and
the first bar played with the adapter it chose. docs/experiments/ADAPTER_LIVE_SELECT.md.
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def switch_latencies(report: dict) -> list[dict]:
    """Bars from arrival to the first bar that plays the chosen adapter."""
    swap = report["adapter_swap"]
    per_bar = swap["per_bar"]
    bar_ms = 240_000.0 / report["bpm"]
    rows = []
    for e in swap.get("control_events", []):
        arrived = math.floor(e["received_ms_from_start"] / bar_ms)
        applied = None
        if e["selected"] is not None:
            applied = next((b for b in range(max(arrived, 0), len(per_bar))
                            if per_bar[b] == e["selected"]), None)
        rows.append({**e, "arrived_bar": arrived, "applied_bar": applied,
                     "latency_bars": None if applied is None else applied - arrived,
                     # arrival -> downbeat of the first bar that plays the choice
                     "latency_ms": None if applied is None
                     else applied * bar_ms - e["received_ms_from_start"]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port-name", default="AdapterSelectProbe")
    ap.add_argument("--sends", required=True,
                    help='"SECONDS:VALUE,..." after launch; VALUE is a program (or CC value)')
    ap.add_argument("--cc", type=int, default=None, help="send CC N instead of Program Change")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("runtime_args", nargs=argparse.REMAINDER,
                    help="after --: arguments for run_continuous_jazz.py (without --output-dir, --input-port)")
    args = ap.parse_args(argv)
    import mido

    sends = []
    for part in args.sends.split(","):
        if part.strip():
            t, v = part.split(":")
            sends.append((float(t), int(v)))
    extra = [a for a in args.runtime_args if a != "--"]
    control = f"cc:{args.cc}" if args.cc is not None else "program"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [args.python, str(ROOT / "scripts" / "run_continuous_jazz.py"), *extra,
           "--input-port", args.port_name, "--adapter-control", control,
           "--output-dir", str(args.output_dir)]
    log = (args.output_dir / "runtime.log").open("w")
    with mido.open_output(args.port_name, virtual=True) as port:
        time.sleep(0.5)   # let CoreMIDI publish the port before the child looks for it
        t0 = time.monotonic()
        child = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        sent = []
        for at, value in sorted(sends):
            while time.monotonic() - t0 < at:
                if child.poll() is not None:
                    break
                time.sleep(0.005)
            if child.poll() is not None:
                break
            msg = (mido.Message("control_change", control=args.cc, value=value) if args.cc is not None
                   else mido.Message("program_change", program=value))
            port.send(msg)
            sent.append({"at_s": round(time.monotonic() - t0, 3), "value": value})
        code = child.wait()
    log.close()
    report_path = args.output_dir / "continuous_report.json"
    out = {"schema": "live_select_probe_v1", "command": cmd, "exit_code": code, "sent": sent}
    if code == 0 and report_path.exists():
        report = json.loads(report_path.read_text())
        out["per_bar"] = report["adapter_swap"]["per_bar"]
        out["switches"] = switch_latencies(report)
        out["fallback_bars"] = report["production"]["fallback_bar_count"]
        out["deadline_misses"] = report["scheduler_dispatch_deadline_miss_count"]
    (args.output_dir / "probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "command"}, indent=2))
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
