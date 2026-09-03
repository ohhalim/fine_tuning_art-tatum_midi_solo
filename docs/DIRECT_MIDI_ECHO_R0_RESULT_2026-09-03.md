# Direct MIDI Echo R0 Result

## Summary

- Issue: #1476
- Scope: model-free direct MIDI pass-through, independent CoreMIDI capture, R0 gate
- Result: 10-minute CoreMIDI R0 transport gate passed
- Runtime boundary: clock, scheduler, model, FL Studio, Serum, audio output excluded

## Context

- Previous baseline repair: #1474
- Input path before this issue: HTTP request and generated MIDI file
- Missing path: live MIDI callback to MIDI output data plane
- Installed project dependency before this issue: `mido`
- Missing local backend before this issue: `rtmidi`

## Implementation

- `DirectMidiEcho`: callback serialization and one-send-per-input behavior
- `RecordingMidiSink`: in-memory capture
- CoreMIDI virtual sender and echo sources with independent capture input
- CoreMIDI fixture: chord-like burst, same-pitch re-key, CC120/123 with late note-off,
  velocity-zero note-off, control-change, program-change
- 600-second logical fixture: note-on, note-off, velocity-zero note-off, control-change, program-change
- integrity report: loss, duplicate, order mismatch, unmatched note-off, stuck note, crash
- callback processing latency: p50, p95, p99, max
- sender-to-capture latency: independent capture match가 완전할 때만 기록; 미측정은 `null`
- report schema: `direct_midi_echo_report_v2`
- observation: wall-clock soak completion, OS loopback observation, FL Studio audio observation
- quality gate: event integrity와 observation을 분리한 `passed_r0_transport_gate`
- real port command: input/output discovery, named port validation, callback forwarding,
  send-acceptance-only report, output reset
- interrupt/exception boundary: output reset in `finally`

## Validation

Command:

```bash
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh direct-midi-echo
uv run --with-requirements requirements.txt bash scripts/agent_harness.sh direct-midi-coremidi-smoke
uv run --with-requirements requirements.txt python scripts/run_coremidi_virtual_loopback_probe.py \
  --run_id issue_1476_coremidi_v2_final_10min_soak \
  --duration_seconds 600 \
  --rate_hz 50
uv run --with-requirements requirements.txt python scripts/run_direct_midi_echo.py --list_ports
```

Environment:

- Python: `3.14.7`
- `python-rtmidi`: `1.5.8`
- compiled MIDI API: `CoreMidi`

Logical fixture result:

| Metric | Result |
|---|---:|
| logical duration | `600.0s` |
| input events | `2701` |
| output events | `2701` |
| event loss | `0` |
| duplicate output | `0` |
| order mismatch | `0` |
| unmatched note-off | `0` |
| stuck note | `0` |
| crash | `0` |
| callback p50 | `0.001042ms` |
| callback p95 | `0.001125ms` |
| callback p99 | `0.004875ms` |
| callback max | `0.059792ms` |

Local port discovery:

```json
{
  "inputs": [],
  "outputs": []
}
```

The empty list is the persistent-port state before the probe starts. The CoreMIDI smoke creates
two process-lifetime virtual sources, connects separate echo and capture inputs, and removes them
when the process exits.

CoreMIDI virtual loopback smoke:

| Metric | Result |
|---|---:|
| wall-clock duration | `2.7962s` |
| configured event duration / rate | `2.0s / 50Hz` |
| sender events | `100` |
| independent capture events | `100` |
| event loss / duplicate / order mismatch | `0 / 0 / 0` |
| unmatched note-off / stuck note / crash | `0 / 0 / 0` |
| callback p50 / p95 / p99 / max | `0.0169 / 0.0314 / 0.0428 / 0.0669ms` |
| sender-to-capture p50 / p95 / p99 / max | `0.2370 / 0.4617 / 0.8010 / 2.6889ms` |
| safe reset sent | `true` |
| OS MIDI loopback observed | `true` |
| wall-clock soak completed | `false` |
| event integrity passed | `true` |
| R0 transport gate passed | `false` |

CoreMIDI 10-minute R0 gate:

| Metric | Result |
|---|---:|
| wall-clock duration | `601.8317s` |
| configured duration / rate | `600.0s / 50Hz` |
| sender events | `30000` |
| independent capture events | `30000` |
| event loss / duplicate / order mismatch | `0 / 0 / 0` |
| unmatched note-off / stuck note / crash | `0 / 0 / 0` |
| callback p50 / p95 / p99 / max | `0.0208 / 0.0364 / 0.0485 / 0.1465ms` |
| sender-to-capture p50 / p95 / p99 / max | `0.3631 / 0.6908 / 1.3727 / 11.7968ms` |
| safe reset sent | `true` |
| wall-clock soak completed | `true` |
| OS MIDI loopback observed | `true` |
| event integrity passed | `true` |
| R0 transport gate passed | `true` |
| FL Studio audio observed | `false` |

Silent named-port regression:

| Metric | Result |
|---|---:|
| wall-clock duration | `0.2132s` |
| input events | `0` |
| process exit | `1` |
| send acceptance passed | `false` |
| output capture observed | `false` |
| latency percentiles | `null` |
| safe reset sent | `true` |

## Decision

- Direct pass-through core and event-integrity accounting retained
- MCP and LLM excluded from MIDI data plane
- callback processing latency retained as internal overhead only
- sender-to-capture latency reported only from independent CoreMIDI capture
- zero-input runs rejected by the minimum input event gate
- 600-second runs require `passed_r0_transport_gate=true` for exit code `0`
- wall-clock/OS observations retained independently when integrity fails
- no latency samples represented as `null`, not `0.0ms`
- CC120/123 resets separated from late note-off mismatch accounting
- audio latency claim blocked until FL Studio output loopback

## Remaining Risk

- logical fixture executes in approximately `0.0038s`; its latency is not an OS transport result
- named-port runner reports send acceptance only and does not expose integrity fields
- receiver integrity requires the separate CoreMIDI capture probe
- persistent macOS MIDI port count before probe: input `0`, output `0`
- process-lifetime virtual CoreMIDI sources remove the IAC setup requirement for the automated probe
- FL Studio routing may still require a persistent IAC or equivalent port configuration
- R0 fixture is virtual CoreMIDI traffic, not a physical keyboard capture
- reset command send recorded; reset messages intentionally excluded from integrity capture
- sender-to-capture max `11.7968ms`; tail cause and deadline impact not isolated
- MIDI Clock, scheduler, model inference, chord context, adapter switching excluded

## Next Gate

- internal monotonic clock
- deterministic one-bar block scheduler
- normal note-on/off output and independent capture
- deadline miss and queue underrun accounting
- bar-start timing error p50/p95/p99/max
- watchdog reset on stop, underrun, and exception
- fixed BPM `90 / 120 / 128 / 160` validation
