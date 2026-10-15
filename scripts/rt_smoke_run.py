#!/usr/bin/env python3
"""Run the real-clock MIDI smoke (issue #1747, docs/experiments/REALTIME_MIDI_SMOKE.md): producer (MLX venv) ->
sender (this venv, virtual MIDI output) -> independent capture (this venv). Scenario A: normal run. Scenario B: the
producer is SIGKILLed after it reports block KILL_AT. Writes one report with the comparison and timing.

  rt_smoke_run.py <t3_aria python> <aria repo> <checkpoint> <data json> <out dir> <report json>
"""
from __future__ import annotations

import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCENARIOS = {"A_normal": {"blocks": 32, "kill_at": None}, "B_producer_killed": {"blocks": 32, "kill_at": 16}}


def nearest_rank(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(q * len(s)) - 1))]


def dist(xs):
    return {"n": len(xs), "median": round(statistics.median(xs), 3), "p99_nearest_rank": round(nearest_rank(xs, 0.99), 3),
            "max": round(max(xs), 3), "min": round(min(xs), 3)} if xs else None


def parse_capture(rows):
    out = []
    for r in rows:
        b = r["bytes"]
        status = b[0] & 0xF0
        if status == 0x90 and b[2] > 0:
            out.append(("on", b[1], b[2], r["capture_ns"]))
        elif status in (0x80, 0x90):
            out.append(("off", b[1], 0, r["capture_ns"]))
    return out


def alternates(events):
    state = {}
    for kind, p in events:
        if (kind == "on") == bool(state.get(p)):
            return False
        state[p] = kind == "on"
    return not any(state.values())


def run_scenario(name, cfg, t3_py, aria_repo, ckpt, data, out_dir):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    f = {k: os.path.join(d, v) for k, v in {"ready": "capture.ready", "stop": "capture.stop", "cap": "capture.json",
                                             "snd": "sender.json", "prog": "producer_progress.jsonl", "prod": "producer.json",
                                             "perr": "producer.stderr", "serr": "sender.stderr"}.items()}
    for k in ("ready", "stop"):
        if os.path.exists(f[k]):
            os.remove(f[k])
    port = f"rt_smoke_{name}_{os.getpid()}"
    sender = subprocess.Popen([sys.executable, os.path.join(HERE, "rt_sender.py"), port, f["ready"], f["snd"]],
                              stdin=subprocess.PIPE, stderr=open(f["serr"], "w"))
    capture = subprocess.Popen([sys.executable, os.path.join(HERE, "rt_capture.py"), port, f["ready"], f["stop"], f["cap"], "180"])
    producer = subprocess.Popen([t3_py, os.path.join(HERE, "rt_producer.py"), aria_repo, ckpt, data, str(cfg["blocks"]), f["prog"], f["prod"]],
                                stdout=sender.stdin, stderr=open(f["perr"], "w"))
    sender.stdin.close()
    killed_at = None
    if cfg["kill_at"] is not None:
        while producer.poll() is None:
            try:
                rows = [json.loads(x) for x in open(f["prog"]) if x.strip()]
            except FileNotFoundError:
                rows = []
            if rows and rows[-1]["block"] >= cfg["kill_at"]:
                producer.send_signal(signal.SIGKILL)
                killed_at = rows[-1]["block"]
                break
            time.sleep(0.05)
    producer.wait(timeout=300)
    sender.wait(timeout=120)
    open(f["stop"], "w").close()
    capture.wait(timeout=30)
    with open(os.path.join(d, "run_meta.json"), "w") as m:
        json.dump({"port": port, "producer_returncode": producer.returncode, "sender_returncode": sender.returncode,
                   "capture_returncode": capture.returncode, "killed_at_block": killed_at}, m)
    return analyze(name, d)


BLOCK_MS, BEAT_MS = 937.5, 468.75


def analyze(name, d):
    """Everything below reads only the saved files, so a report can be rebuilt without running again.
    Three sequences are compared separately (Astra review of #1747, M1): the scheduler's whole expected MIDI stream
    (saved since the fix; older runs lack it), what the sender sent successfully, and what the capture received."""
    f = {k: os.path.join(d, v) for k, v in {"cap": "capture.json", "snd": "sender.json", "prog": "producer_progress.jsonl",
                                             "prod": "producer.json", "meta": "run_meta.json"}.items()}
    meta = json.load(open(f["meta"])) if os.path.exists(f["meta"]) else None
    snd, cap = json.load(open(f["snd"])), json.load(open(f["cap"]))
    prod = json.load(open(f["prod"])) if os.path.exists(f["prod"]) else None
    prog = [json.loads(x) for x in open(f["prog"]) if x.strip()]
    send_key = "send_return_ns" if snd["sent"] and "send_return_ns" in snd["sent"][0] else "send_ns"
    sent = [(s["kind"], s["pitch"], s["velocity"] if s["kind"] == "on" else 0) for s in snd["sent"]]
    got = parse_capture(cap["messages"])
    got3 = [g[:3] for g in got]
    exp = [(k, p, v if k == "on" else 0) for k, p, v, _ in snd["expected_midi"]] if "expected_midi" in snd else None

    def cmp(a, b):
        return {"ok": a == b, "left": len(a), "right": len(b),
                "first_difference": next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None if len(a) == len(b) else min(len(a), len(b)))}

    timing = {}
    if sent == got3:
        timing = {"send_return_minus_scheduled_ms": dist([(s[send_key] - s["scheduled_ns"]) / 1e6 for s in snd["sent"]]),
                  "capture_minus_send_return_ms": dist([(g[3] - s[send_key]) / 1e6 for s, g in zip(snd["sent"], got)]),
                  "capture_minus_scheduled_ms": dist([(g[3] - s["scheduled_ns"]) / 1e6 for s, g in zip(snd["sent"], got)])}
    ons = [x for x in snd["logical_sent"] if x[1] == "on"]
    offs = [x for x in snd["logical_sent"] if x[1] == "off"]
    midi_ons = sum(1 for x in sent if x[0] == "on")
    # sender-side scheduling deadline (M3): the block's notes must be in the scheduler before W + k*B + beat
    origin = snd.get("origin") or ([prod["t0_ms"], prod["wall_origin_ns"]] if prod else None)
    sender_margin, block0, margins = None, None, []
    if origin is not None:
        W = origin[1]
        rb = snd["received_blocks"]
        margins = [(W + (b["block"] * BLOCK_MS + BEAT_MS) * 1e6 - b["recv_ns"]) / 1e6 for b in rb if b["block"] > 0]
        sender_margin = dist(margins)
        b0 = [b for b in rb if b["block"] == 0]
        block0 = round((W - b0[0]["recv_ns"]) / 1e6, 3) if b0 else None
    scheduled = snd.get("scheduled_notes")
    warm = [b for b in prog if b["block"] > 0 and not b.get("chord_input")]
    chord = [b for b in prog if b.get("chord_input")]
    exit_ok = snd["error"] is None and snd.get("port_closed") is True and not snd.get("send_failures")
    return {
        "run_meta": meta, "producer_finished_report": prod is not None,
        "producer_last_progress_block": prog[-1]["block"] if prog else None,
        "scheduler_expected_saved": exp is not None,
        "sender": {"end": snd["end"], "error": snd["error"], "port_closed": snd.get("port_closed"), "pending_at_stop": snd["pending_at_stop"],
                   "sounding_at_stop": snd["sounding_at_stop"], "send_failures": len(snd.get("send_failures", [])),
                   "clean_stop": snd.get("clean_stop"), "dropped_velocity0": snd["dropped_velocity0"],
                   "extended_zero_length": snd["extended_zero_length"], "blocks_received": len(snd["received_blocks"])},
        "checks": {
            "A1a_scheduler_expected_equals_sent": cmp(exp, sent) if exp is not None else {"ok": None, "reason": "expected MIDI stream not saved in this run"},
            "A1b_sent_equals_capture": cmp(sent, got3),
            "A2_capture_alternates_per_pitch": {"ok": alternates([g[:2] for g in got])},
            "A3_logical_on_off_pairs": {"ok": len({x[2] for x in ons}) == len(ons) and sorted(x[2] for x in ons) == sorted(x[2] for x in offs),
                                        "on": len(ons), "off": len(offs)},
            "A3_midi_on_equals_logical_on": {"ok": midi_ons == len(ons), "midi_on": midi_ons, "logical_on": len(ons)},
            "A3_scheduled_notes_accounted": {"ok": None if scheduled is None else scheduled == len(ons) + snd.get("pending_on_at_stop", 0),
                                             "scheduled": scheduled, "delivered_on": len(ons), "cancelled_on_at_stop": snd.get("pending_on_at_stop")},
            "A3_producer_contract": {"ok": None if prod is None else all(prod["checks"].values()),
                                     "detail": None if prod is None else prod["checks"]},
            "sender_exit": {"ok": (exit_ok and meta["sender_returncode"] == 0) if meta else None, "recorded_ok": exit_ok,
                            "process_exit_verified": meta is not None},
        },
        "timing": timing,
        "sender_schedule": {"margin_to_deadline_ms": sender_margin, "block0_ms_before_W": block0,
                            "deadline_misses": None if origin is None else sum(1 for m in margins if m < 0)},
        "producer": {"cold_prebuffer_s": prog[0]["cold_s"] if prog else None,
                     "warm_ready_margin_ms": dist([b["ready_margin_ms"] for b in warm]),
                     "chord_ready_margin_ms": dist([b["ready_margin_ms"] for b in chord]),
                     "warm_ready_s": dist([b["ready_s"] for b in warm]), "chord_ready_s": dist([b["ready_s"] for b in chord]),
                     "late_block_start_ms": dist([b["late_start_ms"] for b in prog if b["block"] > 0]),
                     "deadline_misses": sum(1 for b in prog if b["block"] > 0 and b["ready_margin_ms"] < 0),
                     "blocks_reported": len(prog), "committed_token_sha256": None if prod is None else prod["committed_token_sha256"]},
    }


def main() -> None:
    t3_py, aria_repo, ckpt, data, out_dir, report_path = sys.argv[1:7]
    analyze_only = len(sys.argv) > 7 and sys.argv[7] == "--analyze-only"
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    report = {"analyze_only": analyze_only, "scenarios": {}, "note": "time.monotonic_ns in all three processes (system-wide on macOS). p99 by nearest rank; "
                                       "with few events it equals one of the largest. Virtual MIDI only, no synth or audio."}
    for name, cfg in SCENARIOS.items():
        report["scenarios"][name] = (analyze(name, os.path.join(out_dir, name)) if analyze_only
                                     else run_scenario(name, cfg, t3_py, aria_repo, ckpt, data, out_dir))
        print(name, json.dumps({k: v["ok"] for k, v in report["scenarios"][name]["checks"].items()}), flush=True)
    with open(report_path, "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
