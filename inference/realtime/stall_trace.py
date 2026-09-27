"""Opt-in tracing to attribute scheduler stalls to GC, the process, or the system.

Three sources, all on ``time.perf_counter_ns`` (mach_absolute_time on macOS,
so a separate process can be compared directly):

* GC collections, via ``gc.callbacks``
* an in-process heartbeat thread that records when it wakes up late; it shares
  the GIL with the scheduler, so a GIL hog or a process-wide stop shows up here
* a heartbeat in a separate process that shares nothing but the OS; a
  system-level stall shows up here too

``classify_late_events`` is pure so the attribution rule can be tested.
"""
from __future__ import annotations

import gc
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HEARTBEAT_PERIOD_S = 0.002
HEARTBEAT_GAP_NS = 5_000_000
GC_MIN_NS = 1_000_000

_EXTERNAL_HEARTBEAT = r"""
import sys, time
period, gap_ns, out = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
with open(out, "w", buffering=1) as f:
    while True:
        t0 = time.perf_counter_ns()
        time.sleep(period)
        t1 = time.perf_counter_ns()
        if t1 - t0 - period * 1e9 > gap_ns:
            f.write(f"{t0} {t1}\n")
"""


@dataclass
class StallTracer:
    gc_intervals: list[tuple[int, int, int]] = field(default_factory=list)
    inproc_gaps: list[tuple[int, int]] = field(default_factory=list)
    external_gaps: list[tuple[int, int]] = field(default_factory=list)
    started_ns: int | None = None
    stopped_ns: int | None = None

    def __post_init__(self) -> None:
        self._gc_start: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc: subprocess.Popen | None = None
        self._ext_file: Path | None = None

    def _on_gc(self, phase: str, info: dict) -> None:
        now = time.perf_counter_ns()
        if phase == "start":
            self._gc_start = now
        elif phase == "stop" and self._gc_start is not None:
            self.gc_intervals.append((self._gc_start, now, int(info.get("generation", -1))))
            self._gc_start = None

    def _heartbeat(self) -> None:
        period_ns = HEARTBEAT_PERIOD_S * 1e9
        while not self._stop.is_set():
            t0 = time.perf_counter_ns()
            time.sleep(HEARTBEAT_PERIOD_S)
            t1 = time.perf_counter_ns()
            if t1 - t0 - period_ns > HEARTBEAT_GAP_NS:
                self.inproc_gaps.append((t0, t1))

    def start(self) -> "StallTracer":
        self.started_ns = time.perf_counter_ns()
        gc.callbacks.append(self._on_gc)
        self._thread = threading.Thread(target=self._heartbeat, name="stall-heartbeat",
                                        daemon=True)
        self._thread.start()
        handle = tempfile.NamedTemporaryFile(prefix="stall_hb_", suffix=".txt", delete=False)
        handle.close()
        self._ext_file = Path(handle.name)
        self._proc = subprocess.Popen(
            [sys.executable, "-c", _EXTERNAL_HEARTBEAT, str(HEARTBEAT_PERIOD_S),
             str(HEARTBEAT_GAP_NS), str(self._ext_file)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return self

    def stop(self) -> None:
        self.stopped_ns = time.perf_counter_ns()
        if self._on_gc in gc.callbacks:
            gc.callbacks.remove(self._on_gc)
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._proc is not None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._proc.kill()
        if self._ext_file is not None and self._ext_file.exists():
            for line in self._ext_file.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    self.external_gaps.append((int(parts[0]), int(parts[1])))
            self._ext_file.unlink()


def _overlaps(a_start: int, a_end: int, intervals) -> bool:
    return any(start <= a_end and end >= a_start for start, end, *_ in intervals)


def classify_late_events(records, *, threshold_ns: int, gc_intervals, inproc_gaps,
                         external_gaps, producer_intervals=()) -> list[dict]:
    """Attribute each dispatch later than ``threshold_ns`` (priority: GC, system, process)."""
    gc_long = [(s, e, g) for s, e, g in gc_intervals if e - s >= GC_MIN_NS]
    out = []
    for r in records:
        lateness = r.dispatch_started_ns - r.target_ns
        if lateness <= threshold_ns:
            continue
        window = (r.target_ns, r.dispatch_started_ns)
        if _overlaps(*window, gc_long):
            category = "gc"
        elif _overlaps(*window, external_gaps):
            category = "system"
        elif _overlaps(*window, inproc_gaps):
            category = "process"
        else:
            category = "unexplained"
        out.append({"bar_index": r.bar_index, "lateness_ms": lateness / 1e6,
                    "target_ns": r.target_ns, "category": category,
                    "producer_busy": _overlaps(*window, producer_intervals)})
    return out


def summarize_trace(tracer: StallTracer, late: list[dict]) -> dict:
    def durations_ms(intervals):
        return sorted(round((e - s) / 1e6, 3) for s, e, *_ in intervals)

    counts: dict[str, int] = {}
    for event in late:
        counts[event["category"]] = counts.get(event["category"], 0) + 1
    gc_ms = durations_ms(tracer.gc_intervals)
    span = ((tracer.stopped_ns or 0) - (tracer.started_ns or 0)) or None

    def coverage(intervals, min_ns=0):
        # Share of the traced time covered: the chance a random instant falls inside.
        if not span:
            return None
        return sum(e - s for s, e, *_ in intervals if e - s >= min_ns) / span

    return {
        "trace_seconds": span / 1e9 if span else None,
        "coverage": {"gc_over_1ms": coverage(tracer.gc_intervals, GC_MIN_NS),
                     "inproc_gaps": coverage(tracer.inproc_gaps),
                     "external_gaps": coverage(tracer.external_gaps)},
        "gc_collections": len(tracer.gc_intervals),
        "gc_max_ms": gc_ms[-1] if gc_ms else None,
        "gc_over_1ms": sum(d >= 1.0 for d in gc_ms),
        "inproc_gap_count": len(tracer.inproc_gaps),
        "inproc_gap_max_ms": durations_ms(tracer.inproc_gaps)[-1] if tracer.inproc_gaps else None,
        "external_gap_count": len(tracer.external_gaps),
        "external_gap_max_ms": (durations_ms(tracer.external_gaps)[-1]
                                if tracer.external_gaps else None),
        "late_event_count": len(late),
        "late_by_category": counts,
        "late_events": late,
    }
