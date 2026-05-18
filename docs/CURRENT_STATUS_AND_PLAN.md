# Current Status and Plan

작성일: 2026-05-18

## Current Focus

현재 이 저장소의 우선순위는 Brad Mehldau MIDI dataset으로 Stage A symbolic MIDI model을 fine-tuning할 수 있는지 검증하는 것이다.

현재 브랜치:

- `issue-8-brad-mehldau-control-v1-training-probe`

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

따라서 지금의 목표는 "그럴듯한 제품 MVP"가 아니라, 작은 데이터와 명확한 gate로 model training path를 검증하는 것이다.

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

### 1. Prepare 2-file probe dataset

```bash
python scripts/prepare_role_dataset.py \
  --input_dir "./midi_dataset/midi/studio/Brad Mehldau" \
  --output_dir ./data/roles_probe2 \
  --role lead \
  --sequence_format control_v1 \
  --max_files 2 \
  --overwrite
```

Expected result:

- `data/roles_probe2/lead/dataset_summary.json`
- `data/roles_probe2/lead/tokenized/train/*.npy`
- `data/roles_probe2/lead/tokenized/val/*.npy`

### 2. Train 2-file probe

```bash
python scripts/train_stage_a_full.py \
  --data_dir ./data/roles_probe2/lead/tokenized \
  --output_dir ./checkpoints/brad_mehldau_control_v1_probe2 \
  --epochs 1 \
  --batch_size 4 \
  --num_workers 0 \
  --max_sequence 512
```

Goal:

- verify loader/checkpoint path
- verify control-aware crop does not crash
- generate a sample checkpoint
- avoid spending GPU time before the local contract is clear

### 3. Generate and inspect samples

Use the trained checkpoint with `scripts/generate.py` or the inference wrapper.

The sample is not valid unless it passes:

- non-zero note count
- enough unique pitches
- phrase coverage
- max note duration ratio
- max simultaneous notes
- no one-note/two-note output
- no long sustain block
- no chord block pretending to be a solo line

### 4. Review point

Pause for review after the 2-file probe generates MIDI.

At that point decide:

- continue to `max_files=5`
- run full 18-file probe
- change crop/windowing
- move to duration-explicit tokenization

## Active References

- `docs/BRAD_MEHLDAU_FINETUNING_PLAN.md`
- `docs/STAGE_A_TOKEN_FORMAT.md`
- `docs/STAGE_A_TRAINING_MODES.md`
- `docs/STAGE_A_TINY_OVERFIT.md`
- `docs/STAGE_A_CODE_REVIEW_2026-05-18.md`
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
