# Resident Model R2 Generation Baseline

## Summary

- Issue: `#1480`
- Scope: resident Stage A Music Transformer generation only
- Checkpoint: D1 Arm D epoch 8, SHA-256
  `5bfef8fcc8fdbbde7e23e5ac55711271194673f2fef66ab3fb3a3e067e7626f8`
- Device: Apple MPS
- Result: checkpoint resident load, repeated generation, and musical-time target stop confirmed
- R2 gate: **not evaluated**
- Next boundary: token-budget underfill handling and decoded one-bar block validation

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
- request BPM, meter, and bar count to generated-duration target conversion
- boundary time-shift crop and generated active-note closure

Excluded:

- MIDI decode and repair
- token-budget underfill fallback
- decoded one-bar duration and note-boundary validation
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

## Duration-bounded v3 Follow-up

### Change

- duration target: `bars × meter quarter-note length × 60 / BPM`
- Stage A time unit: `10ms`
- target-step conversion: ceiling; `1,875ms → 188 steps → 1,880ms`
- final sampled time-shift: crop to the remaining duration-step count
- target reached: autoregressive generation stop
- generated active notes: boundary note-off append
- token ceiling reached first: underfill retained; no fallback

Returned token counts include boundary note-off tokens appended after autoregressive generation.
This can make a returned count exceed `target_total_tokens - primer_tokens` without additional
model forward passes.

### Result

| Total-token ceiling | Returned tokens by seed | Duration min / p50 / max | Cover / underfill | Generation p50 / p99 / max | R2 gate |
|---:|---:|---:|---:|---:|---:|
| `48` | `13 / 20 / 17` | `1,180 / 1,880 / 1,880ms` | `2 / 1` | `438.637 / 461.215 / 461.676ms` | `null` |
| `64` | `13 / 20 / 18` | `1,880 / 1,880 / 1,880ms` | `3 / 0` | `502.300 / 597.903 / 599.854ms` | `null` |
| `96` | `13 / 20 / 18` | `1,880 / 1,880 / 1,880ms` | `3 / 0` | `497.271 / 600.603 / 602.712ms` | `null` |
| `128` | `13 / 20 / 18` | `1,880 / 1,880 / 1,880ms` | `3 / 0` | `508.434 / 597.229 / 599.041ms` | `null` |

Artifact:

- `outputs/resident_model/issue_1480_duration_bounded_v3/report.json`

96-token ceiling p50: `2,040.589ms → 497.271ms`. 128-token ceiling p50:
`3,222.321ms → 508.434ms`. The v3 stop condition removes post-bar token generation in these
three-seed probes. It does not establish a production p99 or musical-quality result.

The 48-token ceiling retains one `1,180ms` underfill. The 64-token ceiling covers the target in
three samples, but the registered minimum is 20 samples. No token ceiling is selected from this
probe.

## Decision

- Resident checkpoint load: confirmed
- Grammar-constrained repeated generation: confirmed
- Musical-time upper boundary: established for sampling generation
- Guaranteed one-bar block production: not established
- 20-sample timing campaign: deferred
- Scheduler integration: deferred
- Required next implementation: token-budget underfill handling, decoded duration validation,
  and boundary note-state validation

Running 20 repetitions before the block contract would only improve the precision of the wrong
unit. The next probe must measure a musical-time-bounded block before a p99 R2 decision.

## Validation

- `uv run --with-requirements requirements.txt python -m unittest tests.test_resident_model_probe tests.test_stage_a_checkpoint_loading`
- focused result: `13 tests`, pass
- `uv run --with-requirements requirements.txt python -m unittest tests.test_music_transformer_duration_limit tests.test_resident_model_probe tests.test_stage_a_checkpoint_loading`
- duration-bounded focused result: `19 tests`, pass
- `bash scripts/agent_harness.sh quick`
  - environment/tooling failure: `python` executable absent outside the project environment
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - duration-bounded result: `55 tests`, compile checks and diff check pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - unit tests, compile checks and diff check pass
  - environment/tooling failure at demo step: `scripts/run_mvp_demo.sh` absent from the active tree
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - result before review corrections: `46 tests`, pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - environment/tooling failure: `scripts/run_mvp_demo.sh` absent from the active tree
  - failure predates this change; no inference result produced by this command
- `git diff --check`
