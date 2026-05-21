# Current Status and Plan

작성일: 2026-05-21

## Current Focus

현재 이 저장소의 우선순위는 전체 jazz piano MIDI corpus를 audit하고, generic jazz pianist base를 만든 뒤 Brad Mehldau style adaptation으로 좁힐 수 있는지 검증하는 것이다.

현재 브랜치:

- `issue-57-stage-b-8bar-approach-phrase`

현재 범위가 아닌 것:

- Spring Boot backend
- API server MVP
- ERD/PostgreSQL job system
- realtime DAW/plugin integration
- SaaS/UI/product polish

위 문서들은 `docs/archive/`로 이동했다.

## Current Decision

Stage A는 아직 실사용 가능한 jazz solo model이 아니다.

이전에 생성된 MIDI는 `.mid` 파일로는 존재했지만, 실제 piano roll에서는 다음 문제가 있었다.

- note count가 너무 적음
- 긴 sustain block
- chord block처럼 보이는 출력
- solo-line으로 볼 수 없는 구조
- sparse/medium 일부에서 chord-tone 반응이 약함

따라서 지금의 목표는 "그럴듯한 제품 MVP"가 아니라, 전체 dataset 품질과 작은 probe를 통해 model training path를 검증하는 것이다.

## Latest Probe Result

Issue #57은 manual listening/piano-roll review에서 나온 "이전보다 나아졌지만 아직 재즈가 아니라 초등학교 음악/다이아토닉 코드톤 나열처럼 들린다"는 피드백을 8-bar approach phrase probe로 검증한다.

Issue #51에서 이미 adjacent same-note collapse는 아니라는 것을 확인했다. Issue #53에서는 실제 root tone, non-root chord tone, tension 비율을 기록했다. Issue #55는 같은 4-bar 조건에서 `tones_tensions`가 tension을 실제로 늘리고 root bias를 줄이는지 확인했다. Issue #57은 8-bar 길이와 approach/resolution 정책을 추가한다.

Implemented:

- `inference/app/schemas.py` local phrase probe bars limit `8`
- `scripts/run_stage_b_generation_probe.py` `approach_tensions` pitch mode
- `scripts/run_stage_b_generation_probe.py` approach resolution diagnostics
- `scripts/run_stage_b_pitch_mode_compare.py`
- `scripts/run_stage_b_pitch_mode_compare.py` named comparison MIDI export
- `tests/test_request_conditioning.py`
- `tests/test_stage_b_generation_probe.py`
- `tests/test_stage_b_pitch_mode_compare.py`
- `scripts/run_stage_b_sampling_sweep.py` root/tension summary fields
- `scripts/agent_harness.sh stage-b-8bar-approach-phrase`
- output files:
  - `pitch_mode_compare_report.json`
  - `pitch_mode_compare_report.md`
  - mode-specific `review_manifest.json`
  - named comparison MIDI files under `compare_named_midi/`

Result:

- source comparison report: `outputs/stage_b_pitch_mode_compare/harness_stage_b_8bar_approach_phrase/pitch_mode_compare_report.json`
- named review output: `outputs/stage_b_review_candidates/harness_stage_b_8bar_approach_phrase/compare_named_midi/`
- setup: `8` bars, `8` note groups per bar, `64` note groups per sample
- `tones` strict valid samples: `3/3`
- `tones_tensions` strict valid samples: `3/3`
- `approach_tensions` strict valid samples: `3/3`
- `tones` average root tone ratio: around `0.260`
- `tones_tensions` average root tone ratio: around `0.198`
- `approach_tensions` average root tone ratio: `0.000`
- `tones` average tension ratio: `0.000`
- `tones_tensions` average tension ratio: around `0.328`
- `approach_tensions` average tension ratio: around `0.161`
- `approach_tensions` average approach candidate ratio: around `0.500`
- `approach_tensions` average approach resolution ratio: `1.000`
- `tones` average chord-tone ratio: around `0.616`
- `tones_tensions` average chord-tone ratio: around `0.557`
- `approach_tensions` average chord-tone ratio: around `0.419`
- both modes average onset coverage ratio: around `0.500`

Decision:

- 2-bar candidates가 너무 짧았다는 판단은 타당하다.
- 4-bar 후보는 one-note/two-note failure가 아니라 melody-like sketch로 볼 수 있다.
- "근음을 계속 치는 느낌"은 실제 root-only collapse라기보다 chord-tone-only, no-tension 라인의 안전한 inside sound에 가깝다.
- 8-bar `approach_tensions`는 pitch-level resolution을 만들지만, 이것만으로 jazz vocabulary가 생기지는 않는다.
- 현재 단계는 "broken MIDI"에서 "valid but beginner-like melodic exercise"로 올라온 상태다.
- 다음 issue는 pitch filter보다 rhythm/motif/swing-aware phrase grammar 비교가 맞다.

Detail:

- `docs/STAGE_B_CANDIDATE_RANKING_2026-05-20.md`
- `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`
- `docs/STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`
- `docs/STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`
- `docs/STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`
- `docs/STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_PITCH_MODE_COMPARE_2026-05-21.md`
- `docs/STAGE_B_8BAR_APPROACH_PHRASE_2026-05-21.md`

## Active Issue #14

Current task:

- define Stage B duration-explicit tokenization
- keep Stage B separate from Stage A `control_v1` until the tokenizer contract is tested
- add unit tests for token ranges, chord parsing, quantized note encoding, and roundtrip decoding

First implementation target:

- `scripts/stage_b_tokens.py`
- `tests/test_stage_b_tokens.py`
- `docs/STAGE_B_TOKENIZATION_SPEC.md`

Do not start broad training in this issue.

## Active Issue #15

Current task:

- wire `stage_b_v1` into `scripts/prepare_role_dataset.py`
- produce tokenized train/val records from existing role dataset preparation
- keep Stage B tokenized records target-only for the first contract
- do not start model training yet

First implementation target:

- `prepare_role_dataset.py --sequence_format stage_b_v1`
- unit test with explicit train/val manifests
- local Brad 2-file dry run under `outputs/`

Current result:

- `stage_b_v1` prepare path implemented
- unit tests pass
- Brad 2-file dry run produced tokenized train/val records
- detail: `docs/STAGE_B_ROLE_DATASET_PREP_2026-05-19.md`

## Active Issue #16

Current task:

- split Stage B target continuations into short fixed-bar phrase windows
- keep windowed records target-only for the first contract
- prove Brad 2-file window dry run produces many short `.npy` records
- do not start model training yet

First implementation target:

- `prepare_role_dataset.py --stage_b_window_bars`
- `prepare_role_dataset.py --stage_b_window_stride_bars`
- `prepare_role_dataset.py --stage_b_min_window_target_notes`
- unit test with manifest train/val windows
- local Brad 2-file window dry run under `outputs/`

Current result:

- 2-bar Stage B window path implemented
- Brad 2-file dry run produced 137 role samples
- token lengths: min `22`, p50 `77`, max `212`, mean `82.94`
- detail: `docs/STAGE_B_PHRASE_WINDOW_DATASET_2026-05-19.md`

## Active Issue #17

Status:

- completed and merged via PR #17

Completed task:

- connect Stage B phrase/window records to the model training path
- move Stage B token ranges into shared model constants
- ensure model `VOCAB_SIZE` covers Stage B tokens
- add a Stage B window tiny-overfit smoke script
- fail fast if the prepared dataset has no token records or token IDs exceed model vocab

First implementation target:

- `music_transformer/utilities/constants.py`
- `scripts/stage_b_tokens.py`
- `scripts/run_stage_b_window_tiny_overfit.py`
- `scripts/agent_harness.sh stage-b-window-prepare`
- tests for Stage B vocab compatibility and empty dataset rejection

Current result:

- one Brad file prepare-only smoke produced 70 Stage B window records
- max Stage B token id: `544`
- model vocab size: `547`
- one-epoch tiny training smoke completed
- train loss: `6.1135`
- val loss: `5.8195`
- detail: `docs/STAGE_B_WINDOW_TINY_OVERFIT_2026-05-19.md`

## Active Issue #18

Status:

- completed and merged via PR #19

Completed task:

- add a Stage B token generation/decode probe
- make `MusicTransformer.generate()` able to sample Stage B token IDs by using `sample_vocab_size=VOCAB_SIZE`
- decode generated Stage B tokens back to MIDI
- run existing MIDI metrics/gates on decoded output
- report invalid outputs honestly instead of treating MIDI file creation as success

First implementation target:

- `music_transformer/model/music_transformer.py`
- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-generation-probe`
- tests for Stage B primer, full-vocab sampling, and MIDI decode

Current result:

- one Brad file Stage B window prepare succeeded
- one-epoch tiny training succeeded
- generation sampled with `sample_vocab_size=547`
- decoded MIDI file was created
- generated sample failed the review gate with `generated MIDI has no notes`
- `passed_generation_gate=false`
- detail: `docs/STAGE_B_GENERATION_PROBE_2026-05-19.md`

## Active Issue #20

Status:

- completed and merged via PR #21

Completed task:

- add a grammar-constrained Stage B generation mode
- analyze generated tokens for complete `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION` groups
- separate grammar-gate success from full musical review-gate success
- require at least one decoded MIDI note before expanding training scope

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-constrained-probe`
- `tests/test_stage_b_generation_probe.py`
- `docs/STAGE_B_CONSTRAINED_TINY_OVERFIT_2026-05-19.md`

Current result:

- constrained generation produced 8 complete Stage B note groups
- decoded MIDI note count: `8`
- grammar gate passed: `true`
- full review gate passed: `false`
- full gate failure reason: `too many simultaneous notes: 3 > 2`
- decision: Stage B note grammar can now be forced through the model logits, but musical validity still needs overlap/deduplication control

## Completed Issue #22

Status:

- completed and merged via PR #23

Completed task:

- remove duplicate same-onset/same-pitch notes after Stage B decode
- limit excessive overlapping notes before metrics
- keep decoded MIDI note count non-zero
- get constrained Stage B smoke through the full review gate

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-overlap-gate`
- `tests/test_stage_b_generation_probe.py`
- `docs/STAGE_B_OVERLAP_GATE_2026-05-19.md`

Current result:

- constrained generation still produced 8 complete Stage B note groups
- postprocess reduced notes from `8` to `6`
- max simultaneous notes reduced from `3` to `2`
- grammar gate passed: `true`
- full review gate passed: `true`
- detail: `docs/STAGE_B_OVERLAP_GATE_2026-05-19.md`

## Completed Issue #24

Status:

- completed and merged via PR #25

Completed task:

- strengthen the Stage B local generation probe from one sample to multiple samples
- keep the probe honest by reporting sample-level failures
- compare deterministic `top_k=1` collapse against `top_k=2`
- do not claim musical quality from a single passing MIDI file

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/agent_harness.sh stage-b-stronger-probe`
- `tests/test_stage_b_generation_probe.py`
- `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`

Current result:

- `top_k=2`: `1/3` samples passed the full MIDI review gate
- all `3/3` samples passed the Stage B grammar gate
- `top_k=1` negative control collapsed to `0/3` valid samples
- detail: `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`

## Active Issue #29

Current task:

- add collapse diagnostics to Stage B generated token reports
- explain invalid samples with repeated position/pitch metrics
- add a sampling sweep over `top_k` settings
- compare `top_k=1` collapse against `top_k=2`

First implementation target:

- `scripts/run_stage_b_generation_probe.py`
- `scripts/run_stage_b_sampling_sweep.py`
- `scripts/agent_harness.sh stage-b-collapse-sweep`
- `tests/test_stage_b_generation_probe.py`
- `tests/test_stage_b_sampling_sweep.py`
- `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`

Current result:

- `top_k=1`: valid `0/3`, collapse warning `3/3`, avg repeated position/pitch pair ratio `0.875`
- `top_k=2`: valid `1/3`, collapse warning `1/3`, avg repeated position/pitch pair ratio `0.292`
- best config: `top_k=2`, `temperature=0.9`
- decision: grammar is no longer the immediate bottleneck; note distribution collapse is
- detail: `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`

## Dataset Strategy

현재 데이터셋은 Brad Mehldau-only fine-tuning보다 generic jazz pianist base 학습에 더 적합해 보인다.

파일 시스템 기준:

| Split | MIDI files |
|---|---:|
| physical MIDI paths under `midi_dataset` | 5554 |
| active audit tree: `midi_dataset/midi` | 2777 |
| duplicate mirror tree: `midi_dataset/midi_kong` | 2777 |
| active studio | 1994 |
| active live | 783 |
| Brad Mehldau studio | 18 |
| Brad Mehldau live | 54 |
| Brad Mehldau total | 72 |

Decision:

- 전체 dataset은 generic jazz piano prior 후보로 본다.
- Brad Mehldau subset은 style adaptation과 holdout evaluation에 사용한다.
- `midi_dataset/midi_kong`는 `midi_dataset/midi`의 duplicate mirror로 보고 active training tree에서 제외한다.
- 전체 dataset을 바로 train에 넣지 않고 audit 후 candidate manifest를 만든다.
- 자세한 기준은 `docs/DATASET_STRATEGY.md`를 따른다.

## Implemented Foundation

- `control_v1` token format
  - `ROLE_LEAD + TEMPO_* + BAR + conditioning + COND_SEP + target + END`
- role-conditioned dataset preparation
  - `conditioning.mid`
  - `target.mid`
  - tokenized train/val records
- control-aware crop for long training sequences
  - random crop이 `ROLE/TEMPO/BAR/COND_SEP` prompt를 날리지 않도록 수정됨
- full-checkpoint/from-scratch training entrypoint
- adapter training entrypoint
- tiny-overfit smoke harness
- model generation and MIDI validity metrics
- fallback/gate contract for invalid MIDI
- Brad Mehldau dataset audit script
- full jazz piano dataset audit script
- audit-based training manifest split builder
- manifest-based role dataset preparation smoke

## Full Jazz Piano Dataset Audit

Audit command:

```bash
python scripts/audit_jazz_piano_dataset.py
```

Fast smoke:

```bash
python scripts/audit_jazz_piano_dataset.py --max_files 100
```

Generated outputs:

```text
outputs/dataset_audit/jazz_piano_dataset_audit.json
outputs/dataset_audit/jazz_piano_dataset_audit.md
```

These outputs are not committed.

Current full audit result for `midi_dataset/midi`:

| Metric | Value |
|---|---:|
| files | 2777 |
| readable | 2777 |
| candidate | 2775 |
| candidate non-Brad | 2703 |
| candidate Brad | 72 |
| review too long | 1 |
| reject too few notes | 1 |
| exact duplicate hash groups | 0 |

## Brad Mehldau Dataset Audit

Audit command:

```bash
python scripts/audit_brad_mehldau_dataset.py
```

Current result:

| Metric | Value |
|---|---:|
| MIDI files | 18 |
| usable files | 18 |
| unusable files | 0 |
| max_sequence | 512 |
| files exceeding max_sequence | 18 |

Token stats:

| Metric | Min | P50 | P90 | Max | Mean |
|---|---:|---:|---:|---:|---:|
| `control_v1_token_count` | 1136 | 3241 | 5663 | 10653 | 3931.39 |
| `conditioning_token_count` | 468 | 1608 | 2843 | 4550 | 1937.22 |
| `target_token_count` | 419 | 1716 | 2894 | 6098 | 1989.17 |
| `note_count` | 266 | 756 | 1286 | 2636 | 942.33 |

Decision:

- Full-song sequences are too long for plain `max_sequence=512` training.
- Control-aware crop is required.
- If results stay musically invalid, build phrase-window data or duration-explicit tokenization.

## Next Execution Plan

### 0. Completed Issue #13 review point

Brad 2-file `control_v1` probe는 완료됐다.

Current conclusion:

- 더 많은 postprocess로 해결할 단계가 아니다.
- 현 `control_v1` full-song continuation은 solo-line generation representation으로 약하다.
- next issue는 Stage B tokenization spec/tests로 잡는다.

### 0.5. Issue #14 Stage B tokenizer contract

Status:

- completed and merged via PR #14

Goal:

- define `stage_b_v1`
- encode note events as explicit `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION`
- include bar-level chord context with `CHORD_ROOT + CHORD_QUALITY`
- prove encode/decode roundtrip on tiny deterministic notes

Acceptance:

- Stage B token IDs start after existing Stage A control tokens
- chord symbols such as `Cm7`, `F7`, `Bbmaj7`, `F#m7b5` parse into stable tokens
- generated token sequence contains explicit duration tokens
- roundtrip preserves quantized pitch/start/end on a small example
- `bash scripts/agent_harness.sh quick` passes

### 0.6. Issue #15 Stage B role dataset preparation

Status:

- completed and merged via PR #15

Goal:

- let `prepare_role_dataset.py` accept `--sequence_format stage_b_v1`
- encode target MIDI notes with explicit `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION`
- preserve manifest train/val boundaries
- avoid using `COND_SEP` or Stage A NOTE_ON/OFF tokens in Stage B records

Acceptance:

- unit test writes Stage B train/val `.npy` files from tiny MIDI manifests
- a local Brad 2-file dry run writes tokenized Stage B train/val records
- `bash scripts/agent_harness.sh quick` passes

Dry run result:

- output: `outputs/issue15_stage_b_probe2/roles_stage_b_probe2`
- train tokens: `4430`
- val tokens: `6482`

### 0.7. Issue #16 Stage B phrase-window dataset

Status:

- completed and merged via PR #16

Goal:

- split Stage B target MIDI into fixed-bar windows
- normalize note times to the window start
- keep windows only when they have enough target notes
- keep generated token records short enough for tiny-overfit probes

Acceptance:

- unit test writes multiple Stage B windows from one train/val MIDI pair
- local Brad 2-file dry run creates windowed tokenized train/val records
- `bash scripts/agent_harness.sh quick` passes

Dry run result:

- output: `outputs/issue16_stage_b_window_probe2/roles_stage_b_window_probe2`
- role samples: `137`
- tokenized train: `123`
- tokenized val: `14`
- token length p50: `77`
- token length max: `212`

### 0.8. Issue #17 Stage B window tiny-overfit smoke

Status:

- completed and merged via PR #17

Goal:

- prove Stage B phrase/window token records fit the Music Transformer vocabulary
- prepare a small Brad window dataset through the normal dataset entrypoint
- run a minimal full-model tiny training smoke against Stage B records
- avoid treating empty tokenized datasets as successful

Acceptance:

- `STAGE_B_VOCAB_SIZE == VOCAB_SIZE`
- local prepare-only smoke creates non-empty tokenized Stage B windows
- max token id is lower than model `VOCAB_SIZE`
- one-epoch training smoke exits successfully
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-window-prepare` passes

Smoke result:

- prepare-only output: `outputs/stage_b_window_tiny_overfit/harness_stage_b_window_prepare`
- training output: `outputs/stage_b_window_tiny_overfit/harness_stage_b_window_train_e1`
- role samples: `70`
- token length p50: `89`
- token length max: `212`
- max token id: `544`
- vocab size: `547`
- epoch 1 train loss: `6.1135`
- epoch 1 val loss: `5.8195`

### 0.9. Issue #18 Stage B decode/generation probe

Status:

- completed and merged via PR #19

Goal:

- make generation capable of emitting Stage B token IDs above `TOKEN_END`
- decode generated Stage B tokens into MIDI
- apply the same metrics gate used after the Stage A failure
- document whether the first Stage B generation smoke is musically valid

Acceptance:

- `MusicTransformer.generate(..., sample_vocab_size=VOCAB_SIZE)` can sample the full Stage B vocabulary
- generated Stage B tokens can be decoded into a MIDI file
- invalid output is reported as invalid, not as a successful sample
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-generation-probe` passes

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_generation_probe`
- role samples: `70`
- max token id in prepared records: `544`
- sample vocab size: `547`
- epoch 1 train loss: `6.2115`
- epoch 1 val loss: `5.9441`
- generated sample count: `1`
- valid sample count: `0`
- failure reason: `generated MIDI has no notes`
- decision: data/model/decode plumbing works, but generation quality is still not validated

### 0.10. Issue #20 Stage B grammar-constrained tiny-overfit

Status:

- completed and merged via PR #21

Goal:

- constrain generated token families into complete Stage B note groups
- verify decoded MIDI has real notes before broad training
- record grammar success separately from musical review success

Acceptance:

- grammar analyzer counts complete note groups
- constrained generation creates `POSITION + VELOCITY + NOTE_PITCH + NOTE_DURATION` groups
- decoded MIDI has non-zero notes
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-constrained-probe` passes

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_constrained_probe`
- role samples: `70`
- generated sample count: `1`
- complete note groups: `8`
- decoded note count: `8`
- grammar gate sample count: `1`
- valid sample count: `0`
- full gate failure reason: `too many simultaneous notes: 3 > 2`
- decision: next issue should reduce repeated same-position/same-pitch overlaps before broad training

### 0.11. Issue #22 Stage B overlap/dedup gate

Status:

- completed and merged via PR #23

Goal:

- remove duplicate notes at the same onset/pitch
- limit active overlapping notes to `max_simultaneous_notes <= 2`
- verify constrained Stage B decoded MIDI can pass the full review gate

Acceptance:

- overlap/dedup unit tests pass
- constrained smoke keeps decoded note count above zero
- constrained smoke reduces max simultaneous notes to `2`
- constrained smoke passes `validate_metrics`
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-overlap-gate` passes

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_overlap_gate`
- role samples: `70`
- complete note groups: `8`
- before note count: `8`
- after note count: `6`
- before max simultaneous notes: `3`
- after max simultaneous notes: `2`
- valid sample count: `1`
- passed generation gate: `true`
- decision: this is the first Stage B constrained smoke to pass the local review gate, but it is still a constrained/postprocessed diagnostic rather than unconstrained musical generation

### 0.12. Issue #24 Stage B stronger multi-sample probe

Status:

- completed and merged via PR #25

Goal:

- strengthen the single-sample Stage B overlap gate
- record sample-level seeds
- report grammar and full review pass rates
- require all samples to pass grammar gate
- require at least one sample to pass the full MIDI review gate

Acceptance:

- multi-sample summary unit tests pass
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-stronger-probe` passes
- report includes `sample_count`, `valid_sample_rate`, `grammar_gate_sample_rate`, and failure reason counts

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_stronger_probe`
- role samples: `70`
- epoch 3 val loss: `5.0104`
- generated samples: `3`
- grammar gate sample count: `3`
- valid sample count: `1`
- valid sample rate: `0.333`
- grammar gate sample rate: `1.000`
- passed grammar gate: `true`
- passed generation gate: `true`
- negative control: `top_k=1` collapsed to `0/3` valid samples with `note count too low: 2 < 6`
- detail: `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`

### 0.13. Issue #29 Stage B collapse diagnostics and sampling sweep

Status:

- completed and merged via PR #30

Goal:

- detect repeated position/pitch collapse before scaling training
- report collapse diagnostics per generated sample
- compare sampling configs by pass-rate, not by one hand-picked MIDI

Acceptance:

- collapse diagnostics unit tests pass
- sampling sweep summary tests pass
- `bash scripts/agent_harness.sh quick` passes
- `bash scripts/agent_harness.sh stage-b-collapse-sweep` passes
- `report.json` includes collapse warnings and diagnostic failure reasons
- `sweep_report.json` and `sweep_report.md` compare `top_k=1` and `top_k=2`

Smoke result:

- output: `outputs/stage_b_sampling_sweep/harness_stage_b_collapse_sweep`
- `top_k=1`: valid `0/3`, collapse warning `3/3`, avg repeated position/pitch pair ratio `0.875`
- `top_k=2`: valid `1/3`, collapse warning `1/3`, avg repeated position/pitch pair ratio `0.292`
- best config: `top_k=2`, `temperature=0.9`
- decision: grammar is no longer the immediate bottleneck; note distribution collapse is
- detail: `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`

### 0.14. Issue #31 Stage B stricter collapse-aware review gate

Status:

- completed and merged via PR #32

Goal:

- separate basic MIDI validity from stricter collapse-aware sample validity
- require minimum unique pitch, position, and position/pitch pair diversity
- cap collapse warning sample rate during sampling sweep
- prevent one-note, repeated same-position/pitch, or postprocess-heavy outputs from being counted as strong progress

Strict gate defaults:

- minimum unique pitches: `3`
- minimum unique positions: `3`
- minimum unique position/pitch pairs: `4`
- max repeated position/pitch pair ratio: `0.49`
- max postprocess removal ratio: `0.49`
- max collapse warning sample rate: `0.34`

Acceptance:

- strict collapse gate unit tests pass
- sampling sweep summary distinguishes basic and strict gate pass/fail
- `bash scripts/agent_harness.sh stage-b-collapse-sweep` passes
- `bash scripts/agent_harness.sh quick` passes

Smoke result:

- output: `outputs/stage_b_sampling_sweep/harness_stage_b_collapse_sweep`
- `top_k=1`: basic valid `0/3`, strict valid `0/3`, collapse warning `3/3`
- `top_k=2`: basic valid `1/3`, strict valid `1/3`, collapse warning `1/3`
- best config: `top_k=2`, `temperature=0.9`
- passed strict sweep gate: `true`
- decision: strict gate still preserves one valid candidate, so the next issue can move to a Stage B 2-file Brad probe
- detail: `docs/STAGE_B_STRICT_COLLAPSE_GATE_2026-05-20.md`

### 0.15. Issue #33 Stage B 2-file Brad generation probe

Status:

- completed and merged via PR #34

Goal:

- move from one-file tiny smoke to Brad 2-file Stage B generation probe
- verify whether grammar/basic/strict pass-rate survives a larger train/val setup
- decide whether to move toward generic jazz base training or fix another local generation bottleneck

Probe setup:

- input: `./midi_dataset/midi/studio/Brad Mehldau`
- max files: `2`
- generated Stage B windows: `137`
- train samples: `123`
- val samples: `14`
- max token id: `544`
- vocab size: `547`
- training: 3 epochs, full tiny model path, CPU
- best observed val loss: `4.0892`
- generation: constrained, `top_k=2`, temperature `0.9`, 3 samples

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_2file_brad_probe`
- grammar gate: `3/3`
- basic valid: `0/3`
- strict valid: `0/3`
- collapse warning: `0/3`
- avg repeated position/pitch pair ratio: `0.375`
- avg postprocess removal ratio: `0.208`
- avg onset coverage ratio: `0.167`
- avg sustained coverage ratio: `0.417`
- avg position span ratio: `0.740`
- max longest sustained empty run: `11` steps
- failure reason: all samples failed on dead-air ratio near or above `0.800`

Decision:

- Stage B grammar is now stable in this probe.
- The prior collapse issue is not the immediate failure mode in the 2-file probe.
- The current bottleneck is temporal coverage/dead-air, especially sparse onsets and long empty spans inside the 2-bar phrase.
- Do not start generic jazz base training yet.
- Next issue should add position/dead-air coverage diagnostics and a coverage-aware constrained generation probe.

Detail:

- `docs/STAGE_B_2FILE_BRAD_PROBE_2026-05-20.md`

### 0.16. Issue #35 Stage B temporal coverage diagnostics

Status:

- completed and merged via PR #36

Goal:

- explain why the 2-file Brad probe fails dead-air despite passing grammar and collapse checks
- add token-level temporal coverage diagnostics to each sample report
- aggregate coverage summary fields across samples

Implemented diagnostics:

- unique onset position count
- onset coverage ratio
- sustained coverage ratio
- earliest/latest absolute position
- position span ratio
- head/tail empty steps
- longest onset empty run
- longest sustained empty run
- per-bar unique onset positions
- per-bar onset coverage ratio

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_2file_brad_probe`
- grammar gate: `3/3`
- basic valid: `0/3`
- strict valid: `0/3`
- avg onset coverage ratio: `0.167`
- avg sustained coverage ratio: `0.417`
- avg position span ratio: `0.740`
- max longest sustained empty run: `11` steps

Decision:

- dead-air failure is explained by sparse onset coverage and long empty spans.
- MIDI phrase coverage alone is not enough because notes can span much of the phrase while onsets remain sparse.
- Next issue should test coverage-aware constrained generation, not broad training.

Detail:

- `docs/STAGE_B_TEMPORAL_COVERAGE_DIAGNOSTICS_2026-05-20.md`

### 0.17. Issue #37 Stage B coverage-aware constrained generation

Status:

- implemented on `issue-37-stage-b-coverage-aware-generation`

Goal:

- reduce the Stage B 2-file Brad dead-air failure without broad training
- make only constrained `POSITION` selection coverage-aware
- keep pitch, duration, and velocity sampled from model logits
- compare the result against #35 temporal coverage diagnostics

Implementation:

- added coverage-aware position helper for constrained note groups
- added `--coverage_aware_positions`
- added `--coverage_position_window`
- added harness mode `stage-b-coverage-aware-probe`

Smoke result:

- output: `outputs/stage_b_generation_probe/harness_stage_b_coverage_aware_probe`
- grammar gate: `3/3`
- basic valid: `3/3`
- strict valid: `3/3`
- collapse warning: `0/3`
- avg onset coverage ratio: `0.250`
- avg sustained coverage ratio: `0.427`
- avg position span ratio: `0.813`
- max longest sustained empty run: `6` steps
- avg postprocess removal ratio: `0.000`

Decision:

- coverage-aware constrained `POSITION` selection is a useful local generation constraint.
- It fixes the current 2-file Brad probe's dead-air gate failure under this harness.
- It still does not prove unconstrained generation or personalized Brad style.
- Next issue should run an A/B sweep: plain constrained vs coverage-aware constrained, with `note_groups_per_bar` variants.

Detail:

- `docs/STAGE_B_COVERAGE_AWARE_GENERATION_2026-05-20.md`

### 0.18. Issue #39 Stage B coverage-aware A/B sweep

Status:

- implemented on `issue-39-stage-b-coverage-ab-sweep`

Goal:

- compare plain constrained generation with coverage-aware constrained generation
- test note group density values `4`, `6`, and `8`
- record pass-rate and temporal coverage tradeoffs in JSON/Markdown reports

Smoke result:

- output: `outputs/stage_b_coverage_ab_sweep/harness_stage_b_coverage_ab_sweep`
- configs: `6`
- all configs grammar gate: `3/3`
- plain strict valid: `0/3`, `1/3`, `2/3` for groups/bar `4`, `6`, `8`
- coverage strict valid: `3/3`, `3/3`, `3/3` for groups/bar `4`, `6`, `8`
- best config: coverage groups/bar `8`
- best avg onset coverage ratio: `0.500`
- best avg sustained coverage ratio: `0.865`
- best max longest sustained empty run: `1` step

Decision:

- coverage-aware `POSITION` selection is consistently better than plain constrained generation for this 2-file Brad setup.
- More note groups improve temporal coverage, but lower chord-tone ratio can become the next musical-quality issue.
- Next issue should rank candidate samples/configs with multiple metrics rather than only strict pass/fail.

Detail:

- `docs/STAGE_B_COVERAGE_AB_SWEEP_2026-05-20.md`

### 0.19. Issue #41 Stage B candidate ranking report

Status:

- implemented on `issue-41-stage-b-candidate-ranking`

Goal:

- rank generated MIDI candidates from A/B sweep reports
- include strict validity, temporal coverage, dead-air, chord-tone ratio, repetition, pitch diversity, and collapse warning in the score
- produce JSON/Markdown reports for listening/review priority

Smoke result:

- output: `outputs/stage_b_candidate_ranking/harness_stage_b_candidate_ranking`
- top candidate: coverage groups/bar `8`, sample `1`
- score: `91.080`
- strict valid: `true`
- note count: `16`
- onset coverage ratio: `0.500`
- sustained coverage ratio: `0.906`
- dead-air ratio: `0.467`
- chord-tone ratio: `0.313`

Decision:

- Use this report to choose which generated MIDI files to inspect by ear and piano roll.
- The score is a review-prioritization heuristic, not a musical-quality claim.
- Next issue depends on listening review:
  - if rhythm/shape is acceptable but harmony is weak, add chord-aware pitch filtering/ranking
  - if the line is still mechanically patterned, revisit generation constraints before broad training

Detail:

- `docs/STAGE_B_CANDIDATE_RANKING_2026-05-20.md`

### 0.20. Issue #43 Stage B ranking harmonic/repetition gate

Status:

- implemented on `issue-43-stage-b-ranking-harmonic-gate`

Goal:

- stop ranking from promoting MIDI that only looks good by temporal coverage
- read each candidate MIDI directly before scoring
- flag low chord-tone, repeated pitch, and repeated bar-template failures

Review trigger:

- Issue #41 ranked coverage groups/bar `8`, sample `1` as top candidate.
- Piano-roll review showed it was not a usable solo-line candidate.
- The sample had too little pitch variety and too much repeated mechanical structure.

Latest harness result:

- candidate count: `18`
- valid candidates: `12`
- strict candidates: `12`
- viable candidates without review flags: `0`
- flagged candidates: `18`

Decision:

- Do not treat any current Stage B candidate as a good listening sample.
- The ranking report is now honest: strict-valid MIDI can still be musically invalid.
- Next issue should fix generation-side pitch/harmony behavior, not just ranking.

Detail:

- `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`

### 0.21. Issue #45 Stage B chord-aware pitch constrained generation

Status:

- implemented on `issue-45-stage-b-chord-aware-pitch`

Goal:

- fix the generation-side pitch/harmony failure revealed by Issue #43
- constrain `NOTE_PITCH` candidates by current bar chord
- preserve coverage-aware `POSITION` generation
- reduce repeated pitch and low bar-level chord-tone failures

Latest harness result:

- command: `bash scripts/agent_harness.sh stage-b-chord-aware-probe`
- candidate count: `27`
- strict candidates: `21`
- viable candidates without review flags: `9`
- flagged candidates: `18`
- best mode: `coverage_chord`
- best candidate score: `96.6964`
- best candidate reviewable: `true`

Best candidate:

- mode: `coverage_chord`
- groups/bar: `4`
- sample index: `2`
- note count: `8`
- unique pitch count: `6`
- chord-tone ratio: `0.750`
- bar chord-tone ratio: `0.875`
- min bar chord-tone ratio: `0.800`
- dominant pitch ratio: `0.375`
- repeated pitch ratio: `0.250`
- MIDI path: `outputs/stage_b_coverage_ab_sweep/harness_stage_b_chord_aware_probe_ab_sweep_coverage_chord_g4_k2_t0p9/samples/stage_b_sample_2.mid`

Decision:

- This is the first Stage B probe where candidate ranking finds unflagged reviewable MIDI candidates.
- Do not call this a personalized jazz model yet.
- Next step is manual listening/piano-roll review of top `coverage_chord` candidates.

Detail:

- `docs/STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`

### 0.22. Issue #47 Stage B coverage_chord candidate review package

Status:

- implemented on `issue-47-stage-b-review-package`

Goal:

- export top `coverage_chord` ranked candidates for manual listening review
- keep generated MIDI artifacts out of git
- make the next decision about broad training based on actual listening/piano-roll review

Output:

- `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/review_manifest.json`
- `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/review_candidates.md`
- copied MIDI files under `outputs/stage_b_review_candidates/harness_stage_b_chord_aware_probe/midi/`

Selected candidates:

- `rank_01_coverage_chord_g4_s2.mid`
- `rank_02_coverage_chord_g6_s1.mid`
- `rank_03_coverage_chord_g8_s1.mid`
- `rank_04_coverage_chord_g6_s2.mid`
- `rank_05_coverage_chord_g8_s3.mid`
- `rank_06_coverage_chord_g8_s2.mid`

Decision:

- This is now a manual review boundary.
- The next technical issue should wait until these MIDI files are heard or inspected in piano roll.

Detail:

- `docs/STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`

### 0.23. Issue #49 Stage B longer coverage_chord phrase probe

Status:

- implemented on `issue-49-stage-b-longer-phrase-probe`

Review trigger:

- Issue #47 review candidates were valid enough to inspect, but too short.
- Piano-roll review showed fragments that may be melodic, but felt unfinished as phrases.
- The next probe should not ask whether a MIDI file exists; it should ask whether the candidate has enough phrase length to review.

Goal:

- run a `4` bar coverage+chord-aware constrained probe
- increase generated note groups from `8-16` notes to `32` note groups per sample
- export the generated samples directly from the generation probe report
- keep the claim limited to "longer review candidates", not "finished jazz solo model"

Probe setup:

- max files: `2`
- window bars: `4`
- window stride bars: `2`
- minimum window target notes: `8`
- generation bars: `4`
- constrained note groups per bar: `8`
- pitch mode: chord tones
- position mode: coverage-aware
- samples: `3`

Latest local result:

- generated samples: `3`
- strict valid samples: `3`
- grammar valid samples: `3`
- note groups per sample: `32`
- average onset coverage ratio: around `0.500`
- average sustained coverage ratio: around `0.680`
- max longest sustained empty run: `2` steps

Decision:

- This directly addresses the "too short / unfinished word" failure mode.
- The next manual review should open the exported 4-bar MIDI candidates, not the previous 2-bar package.
- If these still feel like fragments, the next issue should move to phrase/motif-level structure rather than simply adding more notes.

Detail:

- `docs/STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`

### 0.24. Issue #51 Stage B phrase contour/repeated-pitch diagnostics

Status:

- implemented on `issue-51-stage-b-phrase-contour-diagnostics`

Review trigger:

- Issue #49 fixed the phrase length problem structurally.
- However, exported 4-bar candidates still had repeated pitch ratio around `0.719`.
- That number alone did not say whether the sample was adjacent same-note collapse, motif reuse, or chord-tone set reuse.

Goal:

- add phrase contour diagnostics to each Stage B generated sample
- expose repeated-pitch risk in review export without dropping the candidate
- make the manual review question more precise

Implemented diagnostics:

- adjacent repeated pitch ratio
- direction change ratio
- longest same pitch run
- unique interval count
- stepwise/leap motion ratio
- contour warning reasons
- review export `risk_flags`

Latest local result:

- repeated pitch ratio: around `0.719`
- adjacent repeated pitch ratio: `0.000`
- average direction change ratio: around `0.689`
- max longest same pitch run: `1`
- risk flags: `high_repeated_pitch_ratio`, plus `high_dominant_pitch_ratio` for one sample

Decision:

- The current 4-bar candidates are not adjacent same-note collapse.
- They still reuse a limited pitch set heavily.
- The next listening review should judge whether that reuse sounds like motif/inside playing or like constrained mechanical pitch cycling.

Detail:

- `docs/STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`

### 0.25. Issue #53 Stage B root bias diagnostics

Status:

- implemented on `issue-53-stage-b-root-bias-diagnostics`

Review trigger:

- Manual review said the 4-bar candidates are melody-like.
- The main concern was that it feels like it keeps hitting root notes.
- Before changing generation, this needs to be measured.

Goal:

- add pitch-role diagnostics to generated sample reports
- expose root tone ratio, non-root chord tone ratio, and tension ratio in review export
- decide whether the perceived root bias is actual root overuse or broader chord-tone-only behavior

Latest local result:

- average root tone ratio: around `0.271`
- top review candidate root tone ratio: around `0.219`
- chord tone ratio: around `0.938-1.000`
- tension ratio: `0.000`
- adjacent repeated pitch ratio: `0.000`
- `tones_tensions` comparison result:
  - root tone ratio: around `0.135`
  - tension ratio: around `0.313`
  - strict valid samples: `3/3`

Decision:

- The current issue is not pure root collapse.
- The stronger diagnosis is "safe chord-tone-only line with no tensions."
- `chord_pitch_mode=tones_tensions` reduces root/no-tension stiffness, but repeated pitch-set behavior remains.
- Next generation comparison should test passing/approach pitch or contour/motif constraints against the current `tones_tensions` baseline.

Detail:

- `docs/STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_PITCH_MODE_COMPARE_2026-05-21.md`

### 1. Run full jazz piano dataset audit

```bash
python scripts/audit_jazz_piano_dataset.py
```

Expected result:

- readable/unreadable counts
- candidate/review/reject counts
- artist/source distribution
- Brad/non-Brad candidate counts
- duplicate exact hash groups
- duration/note-count/piano-program/sustain stats

Status:

- completed for `midi_dataset/midi`
- generated outputs are under `outputs/dataset_audit/`

### 2. Prepare 2-file control_v1 probe dataset

Before broad training, build concrete candidate splits:

```bash
python scripts/build_jazz_training_manifests.py
```

This produces generic non-Brad train/val manifests plus Brad adaptation/holdout manifests under `data/manifests/`.

Then prepare a tokenized generic split without reshuffling train/val:

```bash
python scripts/prepare_role_dataset.py \
  --train_manifest ./data/manifests/generic_jazz_train.txt \
  --val_manifest ./data/manifests/generic_jazz_val.txt \
  --output_dir ./data/roles_generic_jazz \
  --role lead \
  --sequence_format control_v1 \
  --overwrite
```

For a small local contract check before broad training:

```bash
bash scripts/agent_harness.sh manifest-dry-run
```

Current smoke result:

- `audit_max_files`: 100
- generated generic manifest split: train 57, val 10
- smoke prepare subset: train 4, val 2
- tokenized output: train 4, val 2

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles_probe2 \
  --role lead \
  --sequence_format control_v1 \
  --max_files 2 \
  --overwrite
```

Status:

- completed in issue #13 under `outputs/issue13_control_v1_brad_probe2/roles_probe2`

### 3. Train 2-file control_v1 probe

```bash
python scripts/train_stage_a_full.py \
  --data_dir ./data/roles_probe2/lead/tokenized \
  --output_dir ./checkpoints/brad_mehldau_control_v1_probe2 \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --max_sequence 512
```

Status:

- completed in issue #13
- e5 and e100 checkpoints generated locally under `outputs/issue13_control_v1_brad_probe2/`
- generated artifacts are not committed

### 4. Generate and inspect samples

Use the trained checkpoint with `scripts/generate.py` or the inference wrapper.

Status:

- completed in issue #13
- all generated samples failed the review gate

The sample is not valid unless it passes:

- non-zero note count
- enough unique pitches
- phrase coverage
- max note duration ratio
- max simultaneous notes
- no one-note/two-note output
- no long sustain block
- no chord block pretending to be a solo line

### 5. Review point

Review completed after the 2-file probe generated MIDI.

Decision:

- do not continue to `max_files=5` on current `control_v1`
- do not run full 18-file Brad probe on current `control_v1`
- do not start broad generic non-Brad training yet
- move to duration-explicit tokenization
- create phrase/window dataset before the next training run

## Active References

- `docs/BRAD_MEHLDAU_FINETUNING_PLAN.md`
- `docs/DATASET_STRATEGY.md`
- `docs/STAGE_A_TOKEN_FORMAT.md`
- `docs/STAGE_A_TRAINING_MODES.md`
- `docs/STAGE_A_TINY_OVERFIT.md`
- `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
- `docs/STAGE_B_TOKENIZATION_SPEC.md`
- `docs/STAGE_B_ROLE_DATASET_PREP_2026-05-19.md`
- `docs/STAGE_B_PHRASE_WINDOW_DATASET_2026-05-19.md`
- `docs/STAGE_B_WINDOW_TINY_OVERFIT_2026-05-19.md`
- `docs/STAGE_B_GENERATION_PROBE_2026-05-19.md`
- `docs/STAGE_B_CONSTRAINED_TINY_OVERFIT_2026-05-19.md`
- `docs/STAGE_B_OVERLAP_GATE_2026-05-19.md`
- `docs/STAGE_B_STRONGER_MULTISAMPLE_PROBE_2026-05-20.md`
- `docs/STAGE_B_COLLAPSE_SWEEP_2026-05-20.md`
- `docs/STAGE_B_STRICT_COLLAPSE_GATE_2026-05-20.md`
- `docs/STAGE_B_2FILE_BRAD_PROBE_2026-05-20.md`
- `docs/STAGE_B_TEMPORAL_COVERAGE_DIAGNOSTICS_2026-05-20.md`
- `docs/STAGE_B_COVERAGE_AWARE_GENERATION_2026-05-20.md`
- `docs/STAGE_B_COVERAGE_AB_SWEEP_2026-05-20.md`
- `docs/STAGE_B_CANDIDATE_RANKING_2026-05-20.md`
- `docs/STAGE_B_RANKING_HARMONIC_GATE_2026-05-21.md`
- `docs/STAGE_B_CHORD_AWARE_PITCH_2026-05-21.md`
- `docs/STAGE_B_CANDIDATE_REVIEW_EXPORT_2026-05-21.md`
- `docs/STAGE_B_LONGER_PHRASE_PROBE_2026-05-21.md`
- `docs/STAGE_B_PHRASE_CONTOUR_DIAGNOSTICS_2026-05-21.md`
- `docs/STAGE_B_ROOT_BIAS_DIAGNOSTICS_2026-05-21.md`
- `docs/REFERENCES.md`
- `docs/INFERENCE_MODEL_SPEC.md`
- `docs/QA_ACCEPTANCE_PLAN.md`

## Validation

Before committing code or docs:

```bash
bash scripts/agent_harness.sh quick
```

For generation, inference, metrics, or model-loading changes:

```bash
bash scripts/agent_harness.sh demo
```

For training-mode or tiny-overfit changes:

```bash
bash scripts/agent_harness.sh tiny-compare
```

For Stage B window dataset/model-vocab changes:

```bash
bash scripts/agent_harness.sh stage-b-window-prepare
```

For Stage B decode/generation changes:

```bash
bash scripts/agent_harness.sh stage-b-generation-probe
```

For Stage B constrained note-grammar changes:

```bash
bash scripts/agent_harness.sh stage-b-constrained-probe
```

For Stage B overlap/dedup gate changes:

```bash
bash scripts/agent_harness.sh stage-b-overlap-gate
```

For Stage B multi-sample review-gate changes:

```bash
bash scripts/agent_harness.sh stage-b-stronger-probe
```

For Stage B collapse/sampling-sweep changes:

```bash
bash scripts/agent_harness.sh stage-b-collapse-sweep
```

For Stage B 2-file Brad generation changes:

```bash
bash scripts/agent_harness.sh stage-b-2file-brad-probe
```

For Stage B coverage-aware constrained generation changes:

```bash
bash scripts/agent_harness.sh stage-b-coverage-aware-probe
```

For Stage B coverage-aware A/B sweep changes:

```bash
bash scripts/agent_harness.sh stage-b-coverage-ab-sweep
```

For Stage B candidate ranking changes:

```bash
bash scripts/agent_harness.sh stage-b-candidate-ranking
```

For Stage B longer phrase review candidates:

```bash
bash scripts/agent_harness.sh stage-b-longer-phrase-probe
```
