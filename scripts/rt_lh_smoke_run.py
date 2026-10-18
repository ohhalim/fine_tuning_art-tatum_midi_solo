#!/usr/bin/env python3
"""Run the opt-in left-hand input smoke (issue #1797, docs/experiments/LH_INPUT_SMOKE.md): feeder (virtual LH port)
-> producer (MLX venv, LH input thread) -> sender (virtual model port) -> independent capture. Scenarios A normal,
B forced crop, C slow producer. Each scenario is analysed from its saved files.

  rt_lh_smoke_run.py <t3_aria python> <aria repo> <ckpt> <data json> <lh notes json> <out dir> <report json> [--analyze-only] [--rerun]
      [--precision fp32|bf16|q8|q4]   (issue #1813; passed to every producer, default fp32)
      [--note-cap K] [--seed N] [--scenarios A_normal,B_crop_early]   (issue #1819)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rt_smoke_run import analyze  # noqa: E402

BLOCKS = 17                                      # cold + 16 blocks = 8 bars at 128 BPM
SCENARIOS = {
    "A_normal": [],
    "B_forced_crop": ["--capacity", "400", "--keep-target", "200"],
    "C_slow_producer": ["--slow-block", "6", "--slow-ms", "600"],
}
CHECK_BLOCKS = "4,8,12"
# issue #1807 (docs/experiments/LH_INPUT_SMOKE_RERUN.md): the crop and the slow block are placed before block 4, where
# every #1797 run stopped. Capacity 260 / keep 120 crops at block 2 (offline estimate from the #1797 record: merged
# request 148 > 260 - reserve 131, cut at the prompt's <T>, 93 kept); the slow block is block 2.
SCENARIOS_RERUN = {
    "A_normal": [],
    "B_crop_early": ["--capacity", "260", "--keep-target", "120"],
    "C_slow_block2": ["--slow-block", "2", "--slow-ms", "600"],
}
RERUN_CHECK_BLOCKS = "2,3"


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def code_provenance():
    """Commit and uncommitted changes of the scripts that ran, so a report names the code it came from (#1807 deviation 4)."""
    def git(*a):
        return subprocess.run(["git", "-C", HERE, *a], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_scripts": git("status", "--porcelain", "--", ".").splitlines()}


def run_scenario(name, extra, t3_py, aria_repo, ckpt, data, lh_notes, out_dir, check_blocks=CHECK_BLOCKS):
    d = os.path.join(out_dir, name)
    os.makedirs(d, exist_ok=True)
    f = {k: os.path.join(d, v) for k, v in {"ready": "capture.ready", "stop": "capture.stop", "cap": "capture.json", "snd": "sender.json",
                                             "prog": "producer_progress.jsonl", "prod": "producer.json", "perr": "producer.stderr",
                                             "serr": "sender.stderr", "origin": "origin.json", "fready": "feeder.ready", "feed": "feeder.json"}.items()}
    for k in ("ready", "stop", "origin", "fready"):
        if os.path.exists(f[k]):
            os.remove(f[k])
    out_port, lh_port = f"lh_smoke_out_{name}_{os.getpid()}", f"lh_smoke_in_{name}_{os.getpid()}"
    sender = subprocess.Popen([sys.executable, os.path.join(HERE, "rt_sender.py"), out_port, f["ready"], f["snd"]],
                              stdin=subprocess.PIPE, stderr=open(f["serr"], "w"))
    capture = subprocess.Popen([sys.executable, os.path.join(HERE, "rt_capture.py"), out_port, f["ready"], f["stop"], f["cap"], "240"])
    feeder = subprocess.Popen([sys.executable, os.path.join(HERE, "rt_lh_feeder.py"), lh_port, lh_notes, f["origin"], f["fready"], f["feed"]])
    t = time.monotonic()
    while not os.path.exists(f["fready"]) and time.monotonic() - t < 30:
        time.sleep(0.02)
    args = [t3_py, os.path.join(HERE, "rt_lh_producer.py"), aria_repo, ckpt, data, str(BLOCKS), lh_port, f["origin"], f["prog"], f["prod"],
            "--check-blocks", check_blocks if name == "A_normal" else ""] + extra
    producer = subprocess.Popen(args, stdout=sender.stdin, stderr=open(f["perr"], "w"))
    sender.stdin.close()
    producer.wait(timeout=600)
    sender.wait(timeout=120)
    feeder.wait(timeout=120)
    open(f["stop"], "w").close()
    capture.wait(timeout=30)
    with open(os.path.join(d, "run_meta.json"), "w") as m:
        json.dump({"ports": [out_port, lh_port], "args": args[1:], "producer_returncode": producer.returncode,
                   "sender_returncode": sender.returncode, "capture_returncode": capture.returncode,
                   "feeder_returncode": feeder.returncode}, m)
    return analyze_lh(name, d)


B1_GUARD_NS = 2_000_000        # a message sent within 2 ms of the final drain may still be in flight


def b1(got, sent, sent_rows, final_drain_ns):
    """Received = the in-order prefix of what the feeder sent, and nothing sent before the producer's final drain
    (minus the in-flight guard) is missing (doc 9, deviation 2). Messages sent later are counted, not required."""
    due = sum(1 for r in sent_rows if final_drain_ns is not None and r["send_ns"] < final_drain_ns - B1_GUARD_NS)
    return {"ok": got == sent[:len(got)] and len(got) >= due, "received": len(got), "sent": len(sent),
            "sent_before_final_drain": due, "sent_after_final_drain": len(sent) - due}


def before_horizon(d):
    """Post-hoc (not preregistered): notes a block delivered with an onset before the previous block's commit horizon
    h_{k-1} = s_k + beat (the time already declared final), and how late the sender put each note on the wire."""
    snd = json.load(open(os.path.join(d, "sender.json")))
    t0, W = snd["origin"]
    ends, c = [], 0
    for b in snd["received_blocks"]:
        c += b["notes"]
        ends.append((b["block"], c))
    block_of = lambda i: next(b for b, e in ends if i < e)
    # logical_sent holds the delivery time; the planned onset is the sender row's due_ms (matched by kind, pitch, delivery)
    due_of = {(r["kind"], r["pitch"], r["tick_ms"]): (r["due_ms"], r["send_return_ns"] - r["scheduled_ns"]) for r in snd["sent"]}
    if snd["sent"] and "block" in snd["sent"][0]:      # since #1798 the sender records note_id and block per row
        block_of = {r["note_id"]: r["block"] for r in snd["sent"] if r["kind"] == "on"}.get
    rows, unmatched = [], 0
    for tick, kind, nid, pitch in snd["logical_sent"]:
        if kind != "on":
            continue
        if ("on", pitch, tick) not in due_of:
            unmatched += 1
            continue
        due, late_ns = due_of[("on", pitch, tick)]
        k, h_prev = block_of(nid), t0 + block_of(nid) * 937.5 + 468.75
        if k > 0 and due < h_prev:
            rows.append({"block": k, "note_id": nid, "pitch": pitch, "planned_on_ms": due, "delivered_ms": round(tick, 3),
                         "before_horizon_ms": round(h_prev - due, 3), "send_late_ms": round(late_ns / 1e6, 3)})
    late = [(r["send_return_ns"] - r["scheduled_ns"]) / 1e6 for r in snd["sent"]]
    return {"notes_before_previous_horizon": rows, "unmatched_on": unmatched, "rejected_late_on": snd.get("rejected_late_on"), "notes_on": sum(1 for x in snd["logical_sent"] if x[1] == "on"),
            "send_late_max_ms": round(max(late), 3) if late else None, "sent_over_10ms_late": sum(1 for x in late if x > 10)}


REQUIRED_A_CHECKS = ("A1a_scheduler_expected_equals_sent", "A1b_sent_equals_capture", "A2_capture_alternates_per_pitch",
                     "A3_logical_on_off_pairs", "A3_midi_on_equals_logical_on", "A3_scheduled_notes_accounted",
                     "A3_producer_contract", "sender_exit")
STOP_REASONS = ("deadline_miss", "horizon_underfill", "capacity_failure")


def rerun_checks(out, prod, capacity, snd):
    """Issue #1807 criteria, fixed before the run. Fail closed: a missing key, a missing check or a mismatch is False.
    A normal end needs the sender's own end record and every warm block; an abort needs the sender's abort record to
    name the producer's reason and block (Astra review M1)."""
    from crop_contract import reserve
    from aria_mlx_stream_loop import CAP
    ph = out["posthoc_before_horizon"]
    stop = prod["abort"]
    forwarded = [b for b in prod["blocks"] if b.get("len_N") is not None]
    if stop is None:
        stop_ok = snd.get("end") == "end" and [b["block"] for b in prod["blocks"]] == list(range(1, BLOCKS))
    else:
        stop_ok = (stop.get("reason") in STOP_REASONS and snd.get("end") == "abort"
                   and snd.get("abort") == {"reason": stop.get("reason"), "block": stop.get("block")})
    checks = out["a_checks"].get("checks") or {}
    safe = out["c2_safe_stop"]
    return {
        "no_note_before_previous_horizon": ph.get("notes_before_previous_horizon") == [] and ph.get("unmatched_on") == 0,
        "no_late_intake_rejection": ph.get("rejected_late_on") == [],
        "forwarded_blocks_fit_capacity": bool(forwarded) and all(b["len_N"] + reserve(CAP) <= capacity for b in forwarded),
        "stop_is_recorded_abort_or_end": stop_ok,
        "safe_stop": safe.get("pending_on_at_stop") == 0 and safe.get("port_closed") is True and safe.get("stop_offs_failed") == 0,
        "existing_checks": all((checks.get(k) or {}).get("ok") is True for k in REQUIRED_A_CHECKS)
                           and out["b1_input_integrity"].get("ok") is True and out["b2_causal"] is True,
    }


def loop_speed(prod):
    """Milliseconds per sample from suffix ready to horizon covered, per warm block that sampled before its horizon (#1813).
    Includes the Python sampling and grammar mask; the first sample of a block reuses the prefill logits."""
    rows = [{"block": b["block"], "samples": b["samples_to_ready"], "ms_per_sample": round(b["ready_s"] * 1000 / b["samples_to_ready"], 3)}
            for b in prod["blocks"] if b.get("samples_to_ready") and b.get("ready_s") is not None]
    xs = sorted(r["ms_per_sample"] for r in rows)
    return {"blocks": rows, "median_ms_per_sample": xs[len(xs) // 2] if xs else None}


def density(prod):
    """Per warm block: notes committed for its window, samples before the horizon, of those under the note cap (#1819)."""
    rows = [{"block": b["block"], "notes": b.get("notes"), "samples": b.get("samples_to_ready"), "capped": b.get("capped_to_ready"),
             "margin_ms": b.get("ready_margin_ms")} for b in prod["blocks"] if b.get("samples_to_ready") is not None]
    smp = [r["samples"] for r in rows]
    return {"blocks": rows, "max_samples": max(smp) if smp else None,
            "blocks_capped_before_ready": sum(1 for r in rows if r["capped"]),
            "max_notes": max((r["notes"] for r in rows if r["notes"] is not None), default=None)}


def analyze_lh(name, d):
    base = analyze(name, d)
    prod_p = os.path.join(d, "producer.json")
    prod = json.load(open(prod_p)) if os.path.exists(prod_p) else None
    feed = json.load(open(os.path.join(d, "feeder.json")))
    snd = json.load(open(os.path.join(d, "sender.json")))
    out = {"a_checks": base, "posthoc_before_horizon": before_horizon(d)}
    if prod is None:
        out["producer_report_missing"] = True
        return out
    got = [(r["type"], r.get("pitch")) for r in prod["received"] if r["type"] in ("note_on", "note_off") and not r.get("before_origin")]
    sent = [("note_on" if s["kind"] == "note_on" else "note_off", s["pitch"]) for s in feed["sent"]]
    blocks = prod["blocks"]
    s_k = {b["block"]: prod["t0_ms"] + b["block"] * 937.5 for b in blocks}
    causal = all((b["lh_max_on"] is None or b["lh_max_on"] < s_k[b["block"]]) and
                 (b["lh_max_end"] is None or b["lh_max_end"] <= s_k[b["block"]]) for b in blocks)
    margins = [b["ready_margin_ms"] for b in blocks]
    out.update({
        "b1_input_integrity": b1(got, sent, feed["sent"], prod.get("final_drain_ns")),
        "b2_causal": causal,
        "b3_equivalence_sample": prod["equivalence_sample"],
        "c1_deadline": {"blocks": len(blocks), "late": sum(1 for x in margins if x < 0), "margin_min_ms": min(margins) if margins else None,
                        "margin_median_ms": sorted(margins)[len(margins) // 2] if margins else None},
        "abort": prod["abort"], "crops": prod["crops"], "lh_counts": prod["lh_counts"],
        "c2_safe_stop": {"sender_end": snd.get("end"), "pending_on_at_stop": snd.get("pending_on_at_stop"),
                         "sounding_at_stop": snd.get("sounding_at_stop"), "port_closed": snd.get("port_closed"),
                         "stop_offs_failed": snd.get("stop_offs_failed")},
        "reuse": [{k: b[k] for k in ("block", "crop", "P_reuse", "V", "S", "len_N", "lh_notes", "lh_held", "suffix_ms", "ready_margin_ms", "tokens")}
                  for b in blocks]})
    return out


def main() -> None:
    t3_py, aria_repo, ckpt, data, lh_notes, out_dir, report_path = sys.argv[1:8]
    analyze_only = "--analyze-only" in sys.argv[8:]          # rebuild the report from the saved run files, no model run
    rerun = "--rerun" in sys.argv[8:]                        # issue #1807 scenario set and criteria
    scenarios, check_blocks = (SCENARIOS_RERUN, RERUN_CHECK_BLOCKS) if rerun else (SCENARIOS, CHECK_BLOCKS)
    tail = sys.argv[8:]
    flag = lambda name: tail[tail.index(name) + 1] if name in tail else None  # noqa: E731
    precision, note_cap, seed, only = flag("--precision"), flag("--note-cap"), flag("--seed"), flag("--scenarios")
    passed = [x for f in ("--precision", "--note-cap", "--seed") if flag(f) for x in (f, flag(f))]
    if only:                                                   # issue #1819: a subset of the scenario set
        scenarios = {k: v for k, v in scenarios.items() if k in only.split(",")}
    os.makedirs(out_dir, exist_ok=True)
    prev_code = None
    if analyze_only and os.path.exists(report_path):            # keep the code that ran; the analyzer is recorded apart
        prev_code = json.load(open(report_path)).get("code")
    report = {"issue": 1797, "precision": precision or "fp32", "note_cap": int(note_cap or 0), "seed": seed,
              "code": prev_code if analyze_only else code_provenance(), "analyzer_code": code_provenance(),
              "inputs": {"data_sha256": sha(data), "lh_notes_sha256": sha(lh_notes), "checkpoint_sha256": sha(ckpt)},
              "analyze_only": analyze_only, "scenarios": {}}
    for name, extra in scenarios.items():
        extra = extra + passed
        report["scenarios"][name] = (analyze_lh(name, os.path.join(out_dir, name)) if analyze_only
                                     else run_scenario(name, extra, t3_py, aria_repo, ckpt, data, lh_notes, out_dir, check_blocks))
        if rerun and "producer_report_missing" not in report["scenarios"][name]:
            prod = json.load(open(os.path.join(out_dir, name, "producer.json")))
            snd = json.load(open(os.path.join(out_dir, name, "sender.json")))
            sc = report["scenarios"][name]
            sc["rerun_checks"] = rerun_checks(sc, prod, prod["capacity"], snd)
            if name == "B_crop_early":
                sc["rerun_checks"]["crop_happened"] = prod["crops"] >= 1
            if name == "C_slow_block2":
                sc["rerun_checks"]["slow_block_stopped_on_deadline"] = prod["abort"] == {"reason": "deadline_miss", "block": 2}
            sc["precision"] = prod.get("precision", "fp32")
            sc["loop_speed"] = loop_speed(prod)
            sc["density"] = density(prod)
        print(name, json.dumps({k: report["scenarios"][name].get(k) for k in ("abort", "crops", "b1_input_integrity", "b2_causal", "c1_deadline")}),
              flush=True)
    run0 = os.path.join(out_dir, "A_normal_run0_analysis_crash")    # doc 9 deviation 1: first A run, no A3 checks recorded
    if analyze_only and os.path.isdir(run0):
        rows = [json.loads(x) for x in open(os.path.join(run0, "producer_progress.jsonl")) if x.strip()]
        report["run0_supplement"] = {"blocks": [{k: r.get(k) for k in ("block", "S", "len_N", "suffix_ms", "ready_margin_ms", "tokens")} for r in rows],
                                     "abort": json.load(open(os.path.join(run0, "producer.json")))["abort"],
                                     "posthoc_before_horizon": before_horizon(run0)}
    os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
