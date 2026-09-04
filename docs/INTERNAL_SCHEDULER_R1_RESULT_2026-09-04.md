# Internal One-Bar Scheduler R1 Result

## Summary

- Issue: #1478
- Scope: internal monotonic clock, deterministic one-bar lookahead scheduler, CoreMIDI
  independent capture, 20ms dispatch/bar-start gate, cumulative timing histogram, watchdog
- Initial v2 result: **R1 transport gate not passed**
- Close-fixed v3 remeasurement: **R1 transport gate not passed on two awake runs**
- Environment validity: 90/120 BPM invalidated by clamshell sleep; 128/160 BPM valid
- CoreMIDI input close boundary: **four-BPM report write and process exit restored**
- Runtime boundary: external MIDI Clock, model inference, FL Studio, Serum, audio output excluded

## Context

- Previous gate: #1476 / PR #1477 direct MIDI R0 transport gate, merged as `565cfd59`
- R0 established: independent CoreMIDI capture, minimum-event gate, capability/quality
  separation, null-not-zero latency, `finally` output reset
- R1 adds: an internal clock and a scheduler between the fixture and the output port

## Initial v2 judgment criteria

Frozen while the initial soak was running and before any result was read. This is not a
pre-run registration. All four BPM runs had to satisfy every line:

- `passed_r1_internal_scheduler_gate == true`
- `bar_start_capture_error_ms.sample_count == expected_bar_count` (90/120/128/160 → 225/300/320/400)
- `scheduler_dispatch_deadline_miss_count == 0`
- v2 `deadline_miss_count == 0` and the six integrity counters all `0`
- `bar_start_capture_error_ms.p99 <= 20.0`
- process exit code `0`

Frozen failure policy: **a failing run is recorded as a failure and the cause is
isolated first. Spin window, fixture and threshold are not retuned to make the gate pass.**

## v3 remeasurement criteria, registered before the run

The v3 code and these criteria are committed before starting the v3 four-BPM soak. The
threshold, spin window and dense fixture are unchanged from the initial run.

- report schema `internal_midi_scheduler_report_v3`
- fixed BPM `90 / 120 / 128 / 160`, requested duration `600s` each
- `run_completed == true` and `wall_clock_soak_completed == true`
- `started_bar_count == completed_bar_count == expected_bar_count`
- sent and captured event counts equal `expected_event_count`
- loss / duplicate / order mismatch / unmatched note-off / stuck note / send failure all `0`
- `scheduler_dispatch_deadline_miss_count == 0`
- `capture_deadline_miss_count == 0`
- `queue_underrun_count == 0`
- `bar_start_capture_error_ms.sample_count == expected_bar_count`
- `bar_start_capture_error_ms.p99 <= 20.0`
- `safe_reset_sent == true`
- process exit code `0`

Failure policy: record the failed run without changing `20ms`, `15ms` spin or the fixture.

## Implementation

- `MonotonicBarClock`: beat and bar targets from an absolute `start_ns`, never accumulated
- `OneBarMidiScheduler`: bar-0 preload, one block enqueued per bar boundary, missing block
  counted as underrun and stopped by the watchdog
- Four separated timestamps: `target_ns`, `dispatch_started_ns`, `send_accepted_ns`,
  independent `capture_ns`
- `bar_start_capture_error` measured at **capture**, not at send
- `capture_deadline_miss_count` from target→capture;
  `scheduler_dispatch_deadline_miss_count` from target→dispatch, kept separate
- incomplete capture reports `capture_deadline_miss_count = null`; gate remains fail-closed
- dispatch misses record their own lateness, event position and note metadata before watchdog stop
- started and completed bar counts kept separate
- Cumulative timing histogram, `spin_window_ms` recorded in every report
- Fault tests: send failure, missing block, watchdog reset

### Fixture change made during this issue

The first implementation emitted one note per beat with a half-beat gate. Its minimum
inter-event gap was 187.5–333.3ms, between 12.5× and 22.2× the 15ms spin window, and it
contained **zero same-target-time events**. That fixture cannot reach the regime where the
busy-spin overlaps a capture callback, so it could not test the spin window it ran under.
R0's mixed fixture had deliberately included simultaneous chord bursts; R1 had dropped them.

`build_deterministic_blocks` now cycles four beat shapes per bar:

| Beat | Voices | Note-off gap | Purpose |
|---|---:|---|---|
| 0 | 1 | half beat | bar-start sample, measures clock accuracy without chord serialisation |
| 1 | 3 | `dense_off_gap_ms` (4ms) | same-target-time chord plus sub-spin-window gap |
| 2 | 1 | `dense_off_gap_ms` (4ms) | sub-spin-window gap without simultaneity |
| 3 | 2 | half beat | simultaneity with a sparse gap |

Properties, asserted by `test_deterministic_blocks_reach_the_spin_contention_regime` so the
sparse regression cannot recur silently:

- 14 events per bar, note-balanced at every BPM (`unmatched 0`, `stuck 0`)
- 36 same-target-time adjacencies per 6 bars, up from `0`
- minimum inter-event gap `4.00ms`, down from `187.5ms`
- 12 of 47 gaps below the 15ms spin window
- exactly one `is_bar_start` event per bar

### Report schema

The dense-fixture change first bumped the report to `v2`. The observability correction bumps
it to `internal_midi_scheduler_report_v3`: dispatch-miss records, started/completed bar counts,
nullable capture deadline count, and fixture metadata change the JSON shape.

`fixture_id = "dense_chord_sub_spin_v1"`, `dense_off_gap_ms = 4.0`, and
`events_per_bar = 14` make the generated load reconstructable from the report instead of the
result document alone.

`block_production_mode = "prebuilt_deterministic_dict"` records that every block exists before
the run starts. **With that mode `queue_underrun_count`, `queue_depth_max` and
`enqueue_lead_time_ms` are fixed by construction, not measurements.** Measured confirmation:
`queue_depth_max` is always `1`, and `enqueue_lead_time_ms` spans p50 1499.990541 to max
1499.995166 at 160 BPM — one bar, with 0.005ms of spread across 7 samples. These three fields
only become informative when a producer supplies blocks during the run, which is R2.

## Spin window measurement

Measured on the dense fixture at 160 BPM, 24s per point, 224 events each. The gate metric is
target→capture.

| spin | target→capture p99 | bar-start p99 | dispatch→capture p99 |
|---:|---:|---:|---:|
| 15ms | **4.558** | **0.770** | 4.477 |
| 8ms | 4.616 | 5.837 | 4.422 |
| 3ms | 8.670 | 9.006 | 0.808 |
| 1ms | 10.504 | 10.062 | 1.710 |

The busy-spin does contend with the capture callback: the transport leg at spin 15 is 5.5×
worse than at spin 3 (4.477 vs 0.808ms p99). That effect is real and it is the smaller one.
Shortening the spin costs more scheduling accuracy than it recovers in transport, so
`spin_window_ms = 15.0` was kept. This reverses the pre-measurement expectation that 15ms was
oversized.

## Validation

```bash
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick
RUN_ID=issue_1478_r1_v3_close_fixed_soak \
  uv run --with-requirements requirements.txt \
  bash scripts/agent_harness.sh internal-scheduler-soak
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo
```

Environment: Python `3.14.7`, `mido` `1.3.3`, `python-rtmidi` `1.5.8`, compiled
MIDI API `CoreMidi`.

- `quick`: 39 tests passed; 36 active Python files compiled; diff whitespace check passed
- `internal-scheduler-soak`: expected non-zero exit because the frozen R1 gate failed; all
  four reports written and the harness reached its final BPM
- `demo`: exit `127` before demo execution because active-path `scripts/run_mvp_demo.sh` is
  absent from `origin/main`; the file exists only under `archive/scripts/`. Unit/compile/diff
  checks inside `demo` passed before the missing-script boundary. No demo-pass claim.

## Initial v2 soak result

Four sequential runs, `spin_window_ms 15`, `deadline_threshold_ms 20`,
`internal_midi_scheduler_report_v2`, artifacts under
`outputs/internal_midi_scheduler/issue_1478_r1_soak_bpm*/`.

| BPM | wall | completed/expected bars | captured | bar-start p50/p99/max | dispatch late p99/max | target→capture p99/max | capture >20ms | stuck |
|---:|---:|---:|---:|---|---|---|---:|---:|
| `90` | `257.0s` | `96/225` | `1344` | `0.498 / 4.720 / 5.649` | `4.316 / 13.858` | `5.683 / 14.271` | `0` | `0` |
| `120` | `576.9s` | `288/300` | `4034` | `0.423 / 4.372 / 10.245` | `4.030 / 15.483` | `6.108 / 15.681` | `0` | `0` |
| `128` | `105.2s` | `55/320` | `782` | `0.444 / 6.464 / 11.046` | `2.223 / 13.116` | `10.861 / 14.824` | `0` | `2` |
| `160` | `176.6s` | `117/400` | `1647` | `0.346 / 4.290 / 6.492` | `10.466 / 16.022` | `10.930 / 25.196` | `1` | `1` |

All four ended with `watchdog_trigger_reason = "dispatch_deadline_miss"`,
`run_completed false`, `passed_r1_internal_scheduler_gate false`, exit code `1`.
No run reached its bar count, so the compound R1 gate failed at every BPM. The bar-start p99,
queue-underrun and reset criteria themselves remained within their limits; they did not all
fail individually.

Common to all four: `queue_underrun_count 0`, `queue_depth_max 1`, `send_failure_count 0`,
`unmatched_note_off_count 0`, `safe_reset_sent true`.

### What actually failed

Aggregated over the four runs: **7,807 events were captured and exactly one exceeded the 20ms
target→capture threshold** — 25.196ms at 160 BPM, still under 50ms. 99.987% of delivered
events landed inside 20ms, and 90/120/128 BPM had no end-to-end breach at all.

Every abort was instead a **dispatch** miss: one event whose `dispatch_started_ns` was more
than 20ms past its `target_ns`, so the watchdog stopped before that event was ever sent. The
large `event_loss_count` and `order_mismatch_count` values are consequences of the abort:
unplayed events are compared with the full expected stream. v2 also represented an incomplete
capture by setting `deadline_miss_count` to the full expected count. v3 replaces that ambiguous
value with `capture_deadline_miss_count = null` while keeping the gate fail-closed.

Miss times were `256.0s`, `576.0s`, `103.1s`, `175.5s` — mean 277.6s, four misses in 1,116s of
running. **A simple rate extrapolation from this censored sample gives eight or nine misses
across 2,400 seconds.** This is a baseline estimate, not a prediction: each run stopped on its
first miss, the events are sparse, and no distributional assumption was validated. The
zero-miss gate was not demonstrated on this machine with this design.

### Two hypotheses that the data refuted

Both were formed before the soak and are recorded because they are wrong:

- *Longer waits at lower BPM overshoot more.* Miss times do not track bar duration at all:
  128 BPM failed first at 103s, 120 BPM last at 576s.
- *The overshoot happens on the bar-boundary wait.* Miss positions inside the bar were the
  1st, 3rd, 13th and 10th event. The hiccup is not tied to a wait length or a position.

What remains is a general OS/runtime scheduling stall that can land on any wait.

## v3 input-close fault isolation

The first v3 90 BPM run reached the input-close boundary after its scheduler loop but did not
write a report or exit. A process sample placed the main thread in
`mido.backends.rtmidi.Input.close` → `rtmidi.MidiIn.close_port` → CoreMIDI
`MIDIPortDispose`; the callback thread was waiting to acquire the Python GIL. The missing 90
BPM report means this interrupted run is not R1 evidence.

The installed Mido `1.3.3` RtMidi input close path assigns `callback = None`. Its callback
setter cancels the native callback and installs Mido's callback wrapper again before closing
the native port. The repository now centralizes input teardown in `close_mido_input`:

- native callback cancellation
- Mido callback reference removal without callback-wrapper reinstallation
- raw RtMidi port close
- Mido wrapper closed-state transition before native handle deletion
- public `close()` fallback for non-RtMidi backends and test doubles
- fail-closed behavior when a Mido RtMidi input does not expose the expected `_rt` handle

All repository Mido input paths use the same helper: the R1 independent capture, the R0 named
echo input, and the R0 virtual sender/capture inputs. `mido==1.3.3` is pinned because the helper
depends on the inspected backend lifecycle.

Close-boundary verification before the second v3 soak:

- R0 CoreMIDI 2-second smoke: sent/captured `100/100`, six integrity counters `0`, reset
  `true`, process exit
- CoreMIDI 20-cycle one-bar close stress at 600 BPM: `20/20` process cycles completed, reset
  `true`, sent/captured counts equal in every cycle
- four-BPM R1 smoke: 90 BPM dispatch failure recorded; 120/128/160 BPM short-run pass; all
  four reports written and harness progression completed

## Close-fixed v3 soak result

Four sequential runs, `spin_window_ms 15`, `deadline_threshold_ms 20`,
`internal_midi_scheduler_report_v3`, artifacts under
`outputs/internal_midi_scheduler/issue_1478_r1_v3_close_fixed_soak_*bpm/`.

| BPM | environment | wall | completed/expected bars | captured | bar-start p50/p99/max | successful dispatch p99/max | target→capture p99/max | missed event lateness | missed event |
|---:|---|---:|---:|---:|---|---|---|---:|---|
| `90` | **invalid: lid wake** | `9.021s` | `3/225` | `42` | `0.564 / 0.851 / 0.857` | `2.865 / 2.910` | `4.590 / 4.712` | `25.594ms` | bar `3`, seq `42`, bar-start `note_on 67` |
| `120` | **invalid: lid wake** | `268.768s` | `134/300` | `1,877` | `0.445 / 3.958 / 5.099` | `8.288 / 15.609` | `11.906 / 18.171` | `116.653ms` | bar `134`, seq `1877`, `note_off 64` |
| `128` | valid: awake | `146.929s` | `78/320` | `1,093` | `0.451 / 10.610 / 14.385` | `9.146 / 14.799` | `10.475 / 14.928` | **`70.101ms`** | bar `78`, seq `1093`, `note_off 64` |
| `160` | valid: awake | `15.217s` | `9/400` | `138` | `0.306 / 3.043 / 3.301` | `9.859 / 11.221` | `12.963 / 13.036` | **`23.717ms`** | bar `9`, seq `138`, `note_off 69` |

Power-log correlation invalidates the 90 and 120 BPM timing results:

- 90 BPM report write `15:23:38 KST`; lid wake `15:23:38 KST`; preceding clamshell sleep
  began `15:12:56 KST`
- 120 BPM report write `15:41:37 KST`; lid wake `15:41:37 KST`; preceding clamshell sleep
  began `15:27:43 KST`
- no sleep transition between the 128 BPM interval `15:41:37–15:44:04 KST` or the 160 BPM
  interval `15:44:04–15:44:20 KST`

The 90/120 reports remain useful only for close/report-write verification after resume. They
are excluded from scheduler timing judgment. `caffeinate` does not prevent clamshell sleep;
future timing runs require an open lid and a post-run power-log validity check.

Common to all four:

- `scheduler_dispatch_deadline_miss_count = 1`
- `watchdog_trigger_reason = "dispatch_deadline_miss"`
- `run_completed = false`, `wall_clock_soak_completed = false`
- `capture_deadline_miss_count = null` because capture is incomplete
- `queue_underrun_count = 0`, `send_failure_count = 0`,
  `unmatched_note_off_count = 0`
- `safe_reset_sent = true`
- report write completed, input close completed, harness proceeded to the next BPM

The v3 fields remove the v2 ambiguity: the missed event itself is now recorded even though it
is not sent, while latency summaries cover successful dispatches only. At 120 BPM, for
example, the successful target→capture maximum was `18.171ms`; the event that stopped the run
was never sent and its dispatch began `116.653ms` late. Successful-event percentiles therefore
cannot be used to override the failed zero-miss gate.

All four v3 runs stop on their first miss, so this result does not estimate a full-run miss
rate or p99.9. The two valid awake runs are sufficient to reject the four-BPM zero-miss gate.
The run also establishes that miss location/lateness and teardown are now observable.

## Decision

- R1 gate is recorded as **not passed**. No retuning of spin window, fixture or threshold was
  performed to change that outcome; the frozen failure policy was followed.
- v3 remeasurement is complete for gate rejection, not for four-BPM tail characterization.
  The two awake runs missed by `70.101ms` and `23.717ms`; neither reached 600 seconds. The
  90/120 timing results are excluded due to clamshell sleep.
- CoreMIDI input close deadlock is treated separately from scheduler readiness. The shared
  close path restored report write and process exit across all four failed soak runs, but it
  does not make the R1 scheduler gate pass.
- `spin_window_ms = 15.0` retained on measurement, not on default.
- The dense fixture is retained. It is the reason the contention regime is reachable at all,
  and the failure it exposes is a property of the runtime, not of the fixture alone.
- Whether the correct response is a different gate, a different watchdog policy, or a
  different runtime is **left open**; it is a design decision and is not made here.

## Remaining Risk

- A first-miss-abort policy cannot measure full-duration miss rate, p99.9 or recovery behavior.
- The report schema does not detect system suspend. This run required external `pmset` log
  correlation to invalidate two BPM results.
- Three v3 aborts occurred on `note_off`, leaving captured `stuck_note_count` 1, 1 and 2 at
  120/128/160 BPM until the out-of-band reset. A live policy must define late note-on and
  note-off handling separately.
- `queue_underrun_count`, `queue_depth_max` and `enqueue_lead_time_ms` are fixed by
  construction under `block_production_mode = "prebuilt_deterministic_dict"` and are not
  evidence of real-time robustness.
- The physical port is cleaned (`safe_reset_sent true`) after every v3 abort because reset
  messages are deliberately excluded from integrity capture, as in R0. This does not prove
  glitch-free recovery during a performance.
- `close_mido_input` depends on Mido `1.3.3` RtMidi private state. The version is pinned and
  unexpected internals fail closed, but a dependency upgrade requires lifecycle revalidation.
- The 20ms scheduler-jitter threshold and jam_bot's wider accompaniment-response budget measure
  different boundaries. The latter is not evidence for relaxing this gate.
- Runs were executed sequentially on a laptop with no other deliberate load, but with no CPU
  isolation, no `gc` tuning and no thread-priority policy.

## Next Gate

Separate capability validation from live recovery policy before producer/model integration:

- record-and-continue diagnostic mode for a complete 600-second miss rate, p99.9 and maximum
- system-suspend detection in the report and environment-validity failure; open-lid run
  requirement because `caffeinate` does not override clamshell sleep
- explicit late-event policy by message type: late note-on drop/dispatch decision, note-off
  safety, active-note ledger, and next-bar resynchronization
- zero stuck notes after in-run recovery, independent of final output reset
- retain the 20ms signal while deciding whether zero misses is a capability gate or an
  availability target
- only after full-run evidence, evaluate mitigations — `gc.freeze()` / `gc.disable()` around
  the scheduler loop, macOS thread priority, or moving the dispatch loop out of Python
