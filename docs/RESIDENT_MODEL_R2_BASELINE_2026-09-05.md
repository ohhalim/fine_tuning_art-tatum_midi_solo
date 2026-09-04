# Resident Model R2 Generation Baseline

## Summary

- Issue: `#1480`
- Scope: resident Stage A Music Transformer generation only
- Checkpoint: D1 Arm D epoch 8, SHA-256
  `5bfef8fcc8fdbbde7e23e5ac55711271194673f2fef66ab3fb3a3e067e7626f8`
- Device: Apple MPS
- Result: checkpoint resident load and repeated generation confirmed
- R2 gate: **not evaluated**
- Next boundary: decoded musical-time target stop/crop/fill contract

## Context

- R0 transport gate: 600-second event-integrity pass
- R1 scheduler: four valid 600-second runs, `17,430/17,430` events, integrity counters `0`
- R1 strict four-BPM timing gate: not passed
- Scheduler-only tuning deferred until model and FL Studio load are present
- Existing `StageAModelRunner`: checkpoint one-time load, per-request generation
- Missing evidence before this probe: actual loaded model shape, resident generation time, and
  output musical-time coverage

## Scope

Included:

- full D1 checkpoint one-time load
- warm-up before measurement
- fixed 32-token conditioning primer
- grammar mask enabled
- total sequence length arms: `48 / 64 / 96 / 128`
- three repeated generations per arm with the same seed set
- generated Stage A time-shift duration measurement
- actual model shape, RPR embedding shape, token-layer resize record

Excluded:

- MIDI decode and repair
- exact one-bar stop/crop/fill
- generated block conversion
- scheduler and CoreMIDI output
- FL Studio and audio output
- p99 gate decision from three samples
- musical quality and performer-style claim

## Review Corrections Before Measurement

### Loaded model shape evidence

The loader now optionally returns the actual loaded model metadata. The report records:

- `model_max_sequence`
- `model_rpr`
- layer, head, model and feed-forward dimensions
- RPR embedding tensor shapes
- checkpoint model-config presence
- resized token-layer keys

The probe no longer relies on checkpoint SHA or loader stdout alone to establish the measured
model shape.

### Token duration source

`TIME_SHIFT_START` and `TIME_SHIFT_END` are derived from `RANGE_NOTE_ON`, `RANGE_NOTE_OFF` and
`RANGE_TIME_SHIFT` in the MIDI processor. The duration parser no longer contains copied
`256..355` vocabulary offsets.

### Fail-closed gate

`all_samples_cover_target_bar` remains a descriptive token-duration observation. It cannot
produce an R2 pass while the exact one-bar stop/crop/fill contract is unvalidated.

The report separates:

- `timing_sample_count_sufficient`
- `one_bar_musical_duration_contract_validated`
- `measurement_sufficient_for_r2_gate`
- `passed_r2_generation_deadline_gate`

The current run uses three repetitions and an unvalidated one-bar contract. Every gate result
therefore remains `null` regardless of raw generation speed.

## Fixed Conditions

- BPM / meter: `128 / 4/4`
- one-bar lookahead: `1,875ms`
- provisional scheduling margin: `20ms`
- provisional generation deadline budget: `1,855ms`
- primer tokens: `32`
- warm-up total tokens: `40`
- temperature / top-k / top-p: `1.0 / 32 / 0.95`
- grammar mask: `true`
- repetitions per arm: `3`
- minimum timing samples for a gate: `20`

The `20ms` margin does not cover the prior R1 maximum scheduler stalls of
`29.519..33.253ms`. It remains a provisional input and is not used for a current pass claim.

## Loaded Model Evidence

| Field | Value |
|---|---:|
| model load | `183.316ms` |
| layers | `6` |
| heads | `8` |
| d_model | `512` |
| feed-forward | `1,024` |
| max sequence | `1,024` |
| RPR | `true` |
| RPR tensors | `6 × [1,024, 64]` |
| checkpoint model config | present |
| resized token layers | none |

The model load is a one-time resident startup cost and is excluded from per-generation timing.

## Result

| Total / generated tokens | Generation p50 / p99 / max | Generated musical duration p50 | All 3 cover one bar | R2 gate |
|---:|---:|---:|---:|---:|
| `48 / 16` | `512.137 / 525.710 / 525.987ms` | `1,990ms` | no | `null` |
| `64 / 32` | `918.646 / 1,112.201 / 1,116.152ms` | `3,870ms` | yes | `null` |
| `96 / 64` | `2,040.589 / 2,076.869 / 2,077.609ms` | `13,200ms` | yes | `null` |
| `128 / 96` | `3,222.321 / 3,328.504 / 3,330.671ms` | `15,300ms` | yes | `null` |

Artifact:

- `outputs/resident_model/issue_1480_shape_audited_v2/report.json`

The generated musical duration has high variance because a token-count target is not a musical
time target. The 48-token arm did not cover one bar in every sample. Longer arms frequently
generated several bars of time and therefore cannot be treated as one-bar output without a
stop/crop/fill contract.

## Decision

- Resident checkpoint load: confirmed
- Grammar-constrained repeated generation: confirmed
- Current fixed-token generation as one-bar block producer: not established
- 20-sample timing campaign: deferred
- Scheduler integration: deferred
- Required next implementation: decoded musical-time target stop, boundary note-off repair,
  overrun crop and underrun/fallback record

Running 20 repetitions before the block contract would only improve the precision of the wrong
unit. The next probe must measure a musical-time-bounded block before a p99 R2 decision.

## Validation

- `uv run --with-requirements requirements.txt python -m unittest tests.test_resident_model_probe tests.test_stage_a_checkpoint_loading`
- focused result: `13 tests`, pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - result before review corrections: `46 tests`, pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - environment/tooling failure: `scripts/run_mvp_demo.sh` absent from the active tree
  - failure predates this change; no inference result produced by this command
- `git diff --check`
