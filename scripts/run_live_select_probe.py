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


def adopted_blocks(report: dict) -> tuple[set[int], str]:
    """Blocks whose model block the scheduler actually got (Astra M3/M2).

    New reports record ``adopted_blocks`` from the producer. For older reports:
    ``get()`` marks every bar that was not ready as a fallback, so in a
    completed run whose every record is a model block with no fallback, get()
    returned the model block for each bar. Anything else is reported as partial.
    """
    if "adopted_blocks" in report:
        return set(report["adopted_blocks"]), "recorded"
    detail = report["bars_detail"]
    model = {b["bar_index"] for b in detail if b["source"] == "model" and not b["used_fallback"]}
    if report.get("run_completed") and len(model) == len(detail):
        return model, "reconstructed_all_model"
    return model, "reconstructed_partial"


def switch_latencies(report: dict) -> list[dict]:
    """Per selector message: the block that consumed it, whether that block was
    adopted, and arrival -> downbeat of the first adopted block playing the choice.

    ``noop`` (asked for the active adapter), ``superseded`` (overridden before
    any block used it) and ``ignored`` (program out of range) messages get no
    latency. A "block" is half a bar with --half-bar-blocks.
    """
    swap = report["adapter_swap"]
    per_bar = swap["per_bar"]
    bar_ms = 60_000.0 / report["bpm"] * report.get("block_beats", 4)
    adopted, how = adopted_blocks(report)
    current = next(iter(swap["adapters"]))            # --adapter-name comes first
    rows = []
    for e in swap.get("control_events", []):
        arrived = math.floor(e["received_ms_from_start"] / bar_ms)
        legacy = "consumed_block" not in e
        noop = e.get("noop", e["selected"] is not None and e["selected"] == current)
        superseded = e.get("superseded", False)
        consumed = e.get("consumed_block")
        if legacy and e["selected"] is not None and not noop:
            # Old reports: with no noop, the first block at/after arrival that
            # plays the choice is the block whose generation consumed it.
            consumed = next((b for b in range(max(arrived, 0), len(per_bar))
                             if per_bar[b] == e["selected"]), None)
        if e["selected"] is None:
            status, effective = "ignored", None
        elif noop:
            status, effective = "noop", None
        elif superseded:
            status, effective = "superseded", None
        elif consumed is None:
            status, effective = "not_consumed", None
        else:
            effective = next((b for b in range(consumed, len(per_bar))
                              if b in adopted and per_bar[b] == e["selected"]), None)
            status = ("applied" if effective == consumed
                      else "applied_after_fallback" if effective is not None else "not_adopted")
        if e["selected"] is not None:
            current = e["selected"]
        rows.append({**e, "arrived_bar": arrived, "consumed_block": consumed, "status": status,
                     "adoption_evidence": how, "mapping": "legacy_reconstructed" if legacy else "recorded",
                     "applied_bar": effective,
                     "latency_bars": None if effective is None else effective - arrived,
                     "latency_ms": None if effective is None
                     else effective * bar_ms - e["received_ms_from_start"]})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port-name", default="AdapterSelectProbe")
    ap.add_argument("--sends", default="",
                    help='"SECONDS:VALUE,..." after launch; VALUE is a program (or CC value)')
    ap.add_argument("--notes", default="",
                    help='"SECONDS:PITCH,..." after launch: note_on (velocity 90) then note_off '
                         '0.25 s later, to check that played input reaches the primer')
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
            msg = (mido.Message("control_change", control=args.cc, value=int(v)) if args.cc is not None
                   else mido.Message("program_change", program=int(v)))
            sends.append((float(t), msg))
    for part in args.notes.split(","):
        if part.strip():
            t, pitch = part.split(":")
            sends.append((float(t), mido.Message("note_on", note=int(pitch), velocity=90)))
            sends.append((float(t) + 0.25, mido.Message("note_off", note=int(pitch), velocity=0)))
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
        for at, msg in sorted(sends, key=lambda x: x[0]):
            while time.monotonic() - t0 < at:
                if child.poll() is not None:
                    break
                time.sleep(0.005)
            if child.poll() is not None:
                break
            port.send(msg)
            sent.append({"at_s": round(time.monotonic() - t0, 3), "message": str(msg)})
        code = child.wait()
    log.close()
    report_path = args.output_dir / "continuous_report.json"
    out = {"schema": "live_select_probe_v1", "command": cmd, "exit_code": code, "sent": sent}
    if code == 0 and report_path.exists():
        report = json.loads(report_path.read_text())
        out["per_bar"] = report["adapter_swap"]["per_bar"]
        out["switches"] = switch_latencies(report)
        out["input_blocks"] = [
            {"block": b["bar_index"], "input_event_count": b["input_event_count"],
             "input_to_block_start_ms": b["input_to_bar_start_ms"]}
            for b in report["bars_detail"] if b["input_event_count"]]
        out["fallback_bars"] = report["production"]["fallback_bar_count"]
        out["deadline_misses"] = report["scheduler_dispatch_deadline_miss_count"]
    (args.output_dir / "probe.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "command"}, indent=2))
    return 0 if code == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
