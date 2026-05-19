# Current Status and Plan

작성일: 2026-05-19

## Current Focus

현재 이 저장소의 우선순위는 전체 jazz piano MIDI corpus를 audit하고, generic jazz pianist base를 만든 뒤 Brad Mehldau style adaptation으로 좁힐 수 있는지 검증하는 것이다.

현재 브랜치:

- `issue-16-stage-b-phrase-window-dataset`

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

Issue #13에서 Brad Mehldau real MIDI 2파일로 `control_v1` prepare/train/generate probe를 실행했다.

Result:

- dataset prepare 성공: train 1 sample, val 1 sample
- 5 epoch full-checkpoint training 성공
- 100 epoch full-checkpoint training 성공
- best observed val loss: `4.1306` at epoch `70`
- generation 실패:
  - 5 epoch `top_k=1`: all 0 notes
  - 5 epoch `top_k=32`: 0/1/2-note outputs
  - 100 epoch `top_k=1`: repeated one-note outputs
  - 100 epoch `top_k=32`: long sustain or only 5 notes

Decision:

- Stage A `control_v1`은 runnable pipeline으로는 검증됐다.
- 하지만 reviewable jazz solo MIDI generator로는 실패했다.
- broad generic jazz training으로 바로 넘어가지 않는다.
- 다음은 Stage B duration-explicit, bar-position-aware tokenization과 phrase/window dataset 설계다.

Detail:

- `docs/STAGE_A_BRAD_PROBE2_2026-05-18.md`

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

- implemented on `issue-16-stage-b-phrase-window-dataset`

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
