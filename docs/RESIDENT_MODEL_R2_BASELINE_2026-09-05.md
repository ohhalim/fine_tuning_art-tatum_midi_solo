# Resident Model R2 Generation Baseline

## Summary

- Issue: `#1480`
- Scope: resident Stage A Music Transformer generation, technical validation, and invalid-block fallback
- Checkpoint: D1 Arm D epoch 8, SHA-256
  `5bfef8fcc8fdbbde7e23e5ac55711271194673f2fef66ab3fb3a3e067e7626f8`
- Device: Apple MPS
- Result: model-valid `19/20`, deterministic fallback `1/20`, final-valid `20/20`
- R2 generation deadline gate: **pass**
- R2 operating-headroom gate: **fail**
- R2 decision: **partial success**
- Next boundary: generated/fallback MIDI block conversion and scheduler producer integration

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
- decoded note-boundary validation
- in-memory deterministic fallback for invalid model blocks
- fallback generation and validation timing

Excluded:

- generated block conversion
- scheduler and CoreMIDI output
- FL Studio and audio output
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

## Decode-validated v4 Follow-up

### Fixed Conditions

- total-token ceiling: `64`
- primer tokens: `32`
- seeds: `42..61`
- repetitions: `20`
- duration target: `1,875ms`; quantized target: `1,880ms`
- grammar mask: enabled
- sampling: temperature `1.0`, top-k `32`, top-p `0.95`
- in-memory decode and token note-state validation: included
- file write, scheduler, CoreMIDI, FL Studio: excluded

Technical block validation requires all of the following:

- token duration exactly `1,880ms`
- at least one decoded note
- decoded note end at or before `1,880ms`
- orphan note-off: `0`
- duplicate note-on: `0`
- final active note: `0`
- decode error: none

### Result

| Field | Value |
|---|---:|
| model load | `168.736ms` |
| technically valid blocks | `16/20` |
| reached target duration | `17/20` (includes one empty block) |
| duration underfill | `3/20` |
| empty decoded block | `1/20` |
| decoded target overrun | `0/20` |
| orphan note-off / duplicate note-on / stuck note | `0 / 0 / 0` |
| generation p50 / p95 / p99 / max | `577.140 / 1,180.904 / 1,237.037 / 1,251.070ms` |
| decode+validation p99 | `0.097ms` |
| generation p99 + decode p99 | `1,237.134ms` |
| deadline budget | `1,855ms` |
| block-ready p99 | `1,237.133ms` |
| operating-headroom threshold | `937.5ms` |
| R2 deadline / headroom gate | `null / null` |

Artifact:

- `outputs/resident_model/issue_1480_one_bar_decode_validated_v4/report.json`

The measured generation-plus-decode p99 is below the deadline budget. The contract is not valid
for all samples, so the timing observation cannot produce an R2 pass. The block-ready p99 also
exceeds the separate 50% operating-headroom threshold.

The `16/20` count is a technical block-validity result only. It does not include existing density,
phrase, chord-tone, preference, or performer-style quality gates.

## Token-budget v5 Sweep

### Review Corrections

- returned token count split into model forward steps, sampled output tokens, and appended
  boundary note-offs
- stop reason split into `duration_target`, `token_budget`, and `end_token`
- `samples_covering_target_bar` renamed to `samples_reaching_target_duration`
- separate percentile sum renamed to `generation_p99_plus_decode_validation_p99_ms`
- paired generation+decode measurement retained as `block_ready_time_ms`

`generation_p99_plus_decode_validation_p99_ms` is the sum of two separately calculated p99
values and therefore a conservative deadline input. `block_ready_time_ms.p99` is the p99 of
paired per-sample totals.

### Fixed Conditions

- total-token ceiling arms: `64 / 72 / 80`
- generated model-step ceilings after the 32-token primer: `32 / 40 / 48`
- seeds: `42..61`
- samples per arm: `20`
- checkpoint, primer, sampling parameters, BPM, meter, and duration target unchanged

### Result

| Total-token ceiling | Forward steps max | Stop reason duration / budget / end | Underfill | Empty | Valid | Forward-step cost p50 / p99 | Generation+decode p99 sum | Paired block-ready p99 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `64` | `32` | `17 / 3 / 0` | `3/20` | `1/20` | `16/20` | `22.864 / 25.775ms` | `807.552ms` | `807.542ms` |
| `72` | `40` | `19 / 1 / 0` | `1/20` | `1/20` | `18/20` | `23.163 / 25.303ms` | `947.958ms` | `947.958ms` |
| `80` | `42` | `20 / 0 / 0` | `0/20` | `1/20` | `19/20` | `22.882 / 28.339ms` | `992.553ms` | `992.552ms` |

Artifact:

- `outputs/resident_model/issue_1480_token_budget_sweep_v5/report.json`

Observed underfill stop reasons are exclusively `token_budget`. Raising the ceiling from 64 to
80 removes underfill for the registered seed set. Appended boundary note-offs do not contribute
model forward steps; the 64-token arm's returned maximum of 38 tokens consists of 32 sampled
tokens and 6 boundary note-offs.

The 80-token arm remains below the `1,855ms` deadline input but above the `937.5ms` operating
headroom threshold. Its R2 gates remain `null` because seed 60 reaches the duration target with
zero decoded notes. The next fallback boundary is therefore one technically invalid block, not
the three token-budget underfills removed by the 80-token ceiling.

## Invalid-block Fallback v6

### Implementation Boundary

- existing fallback MIDI construction separated from filesystem write
- resident probe fallback path: in-memory only
- fallback trigger: failed model block technical validation
- fixed fallback context: `Dm7 / G7 / Cmaj7 / A7`, medium density, mid energy
- model-valid blocks retained without fallback
- block-ready time: model generation + model validation + conditional fallback generation and
  fallback validation

The fallback is a deterministic chord-aware rule generator. It is not a phrase-bank retrieval
result and does not establish performer style or musical quality.

### Fallback Integrity Repair

The original fallback note selection allowed a pitch to be selected before its previous note-off.
The same-pitch overlap scan over seeds `0..999` produced:

| Version | Invalid fallback blocks | Failure |
|---|---:|---|
| before active-pitch exclusion | `242/1,000` | same-pitch overlap |
| after active-pitch exclusion | `0/1,000` | none observed |

The repair excludes pitches whose prior note end is later than the candidate start time. If no
chord tone is available, it selects an inactive in-range pitch. Exhausting the full pitch range
raises an explicit fallback failure instead of dropping a note without a counter. Active-pitch
state spans bar boundaries.

Filtering the candidate list changes the seeded pitch sequence for affected inputs. The same
seed remains deterministic with the repaired implementation, but fallback artifacts generated by
the prior implementation are not byte-for-byte reproducible from seed alone.

### Fixed Conditions

- total-token ceiling: `80`
- generated model-step ceiling after primer: `48`
- seeds: `42..61`
- samples: `20`
- checkpoint, 32-token primer, sampling parameters, BPM, meter, and duration target unchanged
- filesystem MIDI write/read: excluded

### Result

| Field | Value |
|---|---:|
| samples / seed range | `20 / 42..61` |
| model-valid / invalid blocks | `19 / 1` |
| fallback used / failed | `1 / 0` |
| final-valid / invalid blocks | `20 / 0` |
| final target-duration contract | model-measured `19`, fallback request-derived `1` |
| generation p99 | `1,017.154ms` |
| model validation p99 | `0.071ms` |
| fallback generation / validation, used sample | `0.295 / 0.016ms` |
| block resolution p99 | `0.289ms` |
| generation p99 + block-resolution p99 | `1,017.443ms` |
| paired block-ready p99 | `1,017.220ms` |
| deadline budget | `1,855ms` |
| operating-headroom threshold | `937.5ms` |
| R2 deadline / headroom gate | `true / false` |

Artifact:

- `outputs/resident_model/issue_1480_invalid_fallback_v6/report.json`

The fallback resolves the seed 60 empty block in `0.311ms` generation plus validation. The
deadline gate passes with `837.557ms` against the provisional budget. The paired p99 exceeds the
operating-headroom threshold by `79.720ms`; the registered R2 decision is
therefore partial success.

Scheduler, CoreMIDI, FL Studio, VST audio, and musical-quality validation remain excluded. The
R2 result does not establish uninterrupted integrated performance. PrettyMIDI-to-scheduled-event
conversion is also excluded from the current block-ready timing and must be included by the next
integration probe.

## Decision

- Resident checkpoint load: confirmed
- Grammar-constrained repeated generation: confirmed
- Musical-time upper boundary: established for sampling generation
- 80-token target-duration production: `20/20`
- technically valid model block production: `19/20`; fallback required for one empty block
- technically valid hybrid block production: `20/20`
- fallback technical scan: `0/1,000` invalid after same-pitch overlap repair
- 20-sample 64-token Arm B campaign: completed; gate withheld
- 20-sample 64/72/80 token-budget sweep: completed; gate withheld
- 20-sample 80-token fallback-inclusive campaign: completed
- R2 generation deadline gate: pass
- R2 operating-headroom gate: fail
- R2 decision: partial success
- Scheduler integration: not measured
- Required next implementation: convert validated model/fallback MIDI to the generated-block
  event contract and connect an asynchronous producer to the scheduler

The next integration must retain fallback provenance and measure queue lead time separately from
the current generation-only deadline result.

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
- `uv run --with-requirements requirements.txt python -m unittest tests.test_resident_model_probe tests.test_music_transformer_duration_limit tests.test_stage_a_checkpoint_loading`
  - decode-validated focused result: `21 tests`, pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - decode-validated result: `57 tests`, compile checks and diff check pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - unit tests, compile checks and diff check pass
  - environment/tooling failure at demo step: `scripts/run_mvp_demo.sh` absent from the active tree
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - result before review corrections: `46 tests`, pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - environment/tooling failure: `scripts/run_mvp_demo.sh` absent from the active tree
  - failure predates this change; no inference result produced by this command
- `uv run --with-requirements requirements.txt python -m unittest tests.test_music_transformer_duration_limit tests.test_resident_model_probe tests.test_stage_a_checkpoint_loading`
  - token-budget metadata focused result: `24 tests`, pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - token-budget v5 result: `60 tests`, compile checks and diff check pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - token-budget v5 unit tests, compile checks and diff check pass
  - environment/tooling failure at demo step: `scripts/run_mvp_demo.sh` absent from the active tree
- `uv run --with-requirements requirements.txt python -m unittest tests.test_resident_model_probe tests.test_music_transformer_duration_limit tests.test_stage_a_checkpoint_loading`
  - invalid-block fallback focused result: `28 tests`, pass
- in-memory fallback scan, seeds `0..999`
  - before active-pitch exclusion: `242/1,000` invalid due to same-pitch overlap
  - after active-pitch exclusion and explicit duration validation: `0/1,000` invalid
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh quick`
  - invalid-block fallback result: `64 tests`, compile checks and diff check pass
- `uv run --with-requirements requirements.txt bash scripts/agent_harness.sh demo`
  - invalid-block fallback unit tests, compile checks and diff check pass
  - environment/tooling failure at demo step: `scripts/run_mvp_demo.sh` absent from the active tree
- `git diff --check`
