#!/usr/bin/env python3
"""Aggregate --stall-trace reports (S1, docs/experiments/RUNTIME_STALL_CAUSE.md)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reports", type=Path, required=True, help="dir searched for continuous_report.json")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(argv)
    runs = []
    for path in sorted(args.reports.rglob("continuous_report.json")):
        report = json.loads(path.read_text())
        trace = report.get("stall_trace")
        if trace is None:
            continue
        runs.append({"run": path.parent.name, "misses": report["scheduler_dispatch_deadline_miss_count"],
                     "fallback": report["production"]["fallback_bar_count"],
                     "lateness": report.get("dispatch_attempt_lateness_summary_ms"),
                     **{k: v for k, v in trace.items() if k != "late_events"},
                     "late_events": trace["late_events"]})
    events = [e for r in runs for e in r["late_events"]]
    over20 = [e for e in events if e["lateness_ms"] > 20]

    def share(evts):
        c = Counter(e["category"] for e in evts)
        n = len(evts)
        top = c.most_common(1)[0] if c else (None, 0)
        return {"n": n, "counts": dict(c), "top": top[0], "top_share": (top[1] / n) if n else None,
                "producer_busy_share": (sum(e["producer_busy"] for e in evts) / n) if n else None}

    def mean(key):
        vals = [r["coverage"][key] for r in runs if r.get("coverage") and r["coverage"][key] is not None]
        return sum(vals) / len(vals) if vals else None

    primary = share(events)
    decided = primary["n"] >= 10 and primary["top_share"] is not None and primary["top_share"] >= 2 / 3
    out = {"schema": "stall_trace_summary_v1", "runs": len(runs),
           "runs_with_miss": sum(r["misses"] > 0 for r in runs),
           "fallback_total": sum(r["fallback"] for r in runs),
           "primary_over_10ms": primary, "secondary_over_20ms": share(over20),
           "mean_coverage": {k: mean(k) for k in ("gc_over_1ms", "inproc_gaps", "external_gaps")},
           "gc_max_ms": max((r["gc_max_ms"] or 0) for r in runs) if runs else None,
           "gc_over_1ms_total": sum(r["gc_over_1ms"] for r in runs),
           "verdict": {"rule": ">=10 events and one category >= 2/3", "decided": decided,
                       "cause": primary["top"] if decided else "undetermined"},
           "late_events": [{"run": r["run"], **e} for r in runs for e in r["late_events"]]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "late_events"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
