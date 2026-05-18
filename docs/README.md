# Docs Index

작성일: 2026-05-18

이 디렉터리는 현재 Brad Mehldau MIDI fine-tuning probe를 진행하기 위한 기준 문서만 전면에 둔다. 백엔드/API/ERD/제품 MVP 문서는 `docs/archive/`로 이동했다.

## Active Docs

- `CURRENT_STATUS_AND_PLAN.md`
  - 현재 브랜치 상태, 결정 사항, 다음 실행 순서.
- `BRAD_MEHLDAU_FINETUNING_PLAN.md`
  - Brad Mehldau MIDI dataset audit, training probe order, acceptance criteria.
- `STAGE_A_TOKEN_FORMAT.md`
  - `control_v1` token sequence, legacy format, checkpoint vocab migration 규칙.
- `STAGE_A_TRAINING_MODES.md`
  - full checkpoint/from-scratch training, adapter training, LoRA-only mode 경계.
- `STAGE_A_TINY_OVERFIT.md`
  - tiny-overfit smoke 실행 방법과 통과/실패 판단 기준.
- `STAGE_A_CODE_REVIEW_2026-05-18.md`
  - sustain block/chord block MIDI가 나온 원인과 다음 수정 방향.
- `REFERENCES.md`
  - Music Transformer, REMI, Jazz Transformer, AMT, Aria 등 현재 fine-tuning/tokenization 판단에 필요한 reference map.
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

1. Brad Mehldau dataset audit 결과를 기준으로 데이터 품질을 확인한다.
2. `max_files=2`로 `control_v1` prepare probe를 실행한다.
3. `max_files=2` full-checkpoint training probe를 실행한다.
4. 생성 MIDI를 note count, unique pitch, phrase coverage, max duration, simultaneous notes 기준으로 검증한다.
5. MIDI가 여전히 sustain/chord block이면 duration-explicit tokenization으로 넘어간다.

핵심 원칙:

> 모델이 valid solo-line MIDI를 만들기 전에는 백엔드, UI, realtime 통합을 확장하지 않는다.
