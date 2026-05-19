# Docs Index

작성일: 2026-05-18

이 디렉터리는 현재 jazz piano MIDI fine-tuning probe를 진행하기 위한 기준 문서만 전면에 둔다. 백엔드/API/ERD/제품 MVP 문서는 `docs/archive/`로 이동했다.

## Active Docs

- `CURRENT_STATUS_AND_PLAN.md`
  - 현재 브랜치 상태, 결정 사항, 다음 실행 순서.
- `BRAD_MEHLDAU_FINETUNING_PLAN.md`
  - Brad Mehldau MIDI dataset audit, training probe order, acceptance criteria.
- `DATASET_STRATEGY.md`
  - 전체 jazz piano corpus audit, generic jazz pianist base, Brad style adaptation 전략.
- `STAGE_A_TOKEN_FORMAT.md`
  - `control_v1` token sequence, legacy format, checkpoint vocab migration 규칙.
- `STAGE_A_TRAINING_MODES.md`
  - full checkpoint/from-scratch training, adapter training, LoRA-only mode 경계.
- `STAGE_A_TINY_OVERFIT.md`
  - tiny-overfit smoke 실행 방법과 통과/실패 판단 기준.
- `STAGE_A_CODE_REVIEW_2026-05-18.md`
  - sustain block/chord block MIDI가 나온 원인과 다음 수정 방향.
- `STAGE_A_BRAD_PROBE2_2026-05-18.md`
  - Brad 2-file `control_v1` training/generation probe 결과와 Stage B 전환 판단.
- `STAGE_B_TOKENIZATION_SPEC.md`
  - duration-explicit, bar-position-aware Stage B tokenization contract.
- `STAGE_B_ROLE_DATASET_PREP_2026-05-19.md`
  - `prepare_role_dataset.py --sequence_format stage_b_v1` 연결 결과와 Brad 2-file dry run.
- `STAGE_B_PHRASE_WINDOW_DATASET_2026-05-19.md`
  - Stage B 2-bar phrase/window dataset extraction and Brad 2-file dry run.
- `STAGE_B_WINDOW_TINY_OVERFIT_2026-05-19.md`
  - Stage B phrase windows가 model vocab/training path에 연결되는지 확인한 tiny-overfit smoke.
- `REFERENCES.md`
  - 2024-2026 symbolic MIDI 연구까지 포함한 fine-tuning/tokenization reference map과 구현 판단 기준.
- `INFERENCE_MODEL_SPEC.md`
  - request-conditioned generation, fallback, metrics, post-processing contract.
- `QA_ACCEPTANCE_PLAN.md`
  - MIDI output을 리뷰 가능한 샘플로 인정하기 위한 gate.
- `RUNPOD_GUIDE.md`
  - GPU training 환경 참고.

## Archived Docs

`docs/archive/`에는 현재 브랜치 범위가 아닌 문서가 있다.

- Spring Boot/API/ERD/backend MVP 문서
- 한 달 포트폴리오 로드맵
- realtime/DAW/plugin/product-planning 문서
- 오래된 리뷰와 외부 리서치 정리

이 문서들은 삭제하지 않고 보관한다. 현재 모델 fine-tuning이 reviewable MIDI를 만들기 전까지는 active plan으로 취급하지 않는다.

## Current Execution Order

1. Full jazz piano corpus audit 결과를 기준으로 generic base 후보 데이터를 확인한다.
2. Brad Mehldau subset은 style adaptation과 holdout evaluation 용도로 분리한다.
3. `max_files=2` Brad `control_v1` probe 결과를 기준으로 Stage A 한계를 문서화한다.
4. broad training 전에 duration-explicit Stage B tokenization과 phrase/window dataset을 설계한다.
5. Stage B phrase/window tiny-overfit와 2-file generation probe를 통과한 뒤 generic jazz base 학습 여부를 다시 결정한다.

핵심 원칙:

> 모델이 valid solo-line MIDI를 만들기 전에는 백엔드, UI, realtime 통합을 확장하지 않는다.
