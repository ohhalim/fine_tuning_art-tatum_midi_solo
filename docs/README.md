# Docs Index

작성일: 2026-05-16

이 디렉터리는 `Personalized Live MIDI Improviser` 모델 MVP를 한 달 안에 만들기 위한 기준 문서 모음이다. 구현할 때는 아래 순서로 읽는다.

## 1. 현재 상태

- `CURRENT_STATUS_AND_PLAN.md`
  - 현재 레포 상태, 이미 구현된 Stage A 파이프라인, 다음 우선순위.
- `REVIEW_2026-05-16.md`
  - 현재 모델 MVP의 1차 리뷰, 위험, 결정 사항, 다음 7일 계획.
- `STAGE_A_CODE_REVIEW_2026-05-18.md`
  - Stage A 모델 출력이 sustain block/엉성한 MIDI로 나오는 구조적 원인 분석과 다음 수정 순서.
- `STAGE_A_TINY_OVERFIT.md`
  - 1~3개 MIDI tiny-overfit smoke 실행 방법과 통과/실패 판단 기준.
- `STAGE_A_TRAINING_MODES.md`
  - full checkpoint/from-scratch training과 adapter training을 분리하는 기준.
- `STAGE_A_TOKEN_FORMAT.md`
  - `control_v1` token sequence, legacy format, checkpoint vocab migration 규칙.
- `JAMBOT_MIDI_REFACTOR_PLAN.md`
  - 초기 계획과 2026-02-20 dead-air sweep 기록.

## 2. 이번 MVP 기준 문서

- `MVP_PRD.md`
  - 제품 요구사항, 범위, 성공 기준, 비목표.
- `ONE_MONTH_ROADMAP.md`
  - 4주 로드맵과 주차별 산출물.
- `MVP_IMPLEMENTATION_PLAN.md`
  - 구현 순서, 작업 단위, Definition of Done.
- `SYSTEM_ARCHITECTURE.md`
  - 전체 아키텍처와 컴포넌트 책임.
- `API_SPEC.md`
  - 현재 MVP에서는 Python CLI/FastAPI wrapper 계약. Spring Boot API는 deferred extension.
- `ERD.md`
  - 현재 MVP에서는 파일 기반 result metadata. PostgreSQL ERD는 deferred extension.
- `INFERENCE_MODEL_SPEC.md`
  - MIDI 생성기, fallback, metrics, post-processing 명세.
- `QA_ACCEPTANCE_PLAN.md`
  - 테스트 기준, 수동 QA, MVP accept gate.
- `CODEX_EXECUTION_GUIDE.md`
  - Codex로만 개발할 때의 작업 방식과 프롬프트 규칙.

## 3. 보조 리서치 문서

- `MAGENTA_RT_FINETUNING_GUIDE.md`
  - 오디오 기반 Magenta RT 실험 참고. 이번 MVP 메인 라인이 아니다.
- `RUNPOD_GUIDE.md`
  - RunPod 참고.
- `VELOG_PRIOR_RESEARCH_SYNTHESIS.md`
  - 이전 리서치 정리.

## 4. 구현 순서 요약

1. 기존 `scripts/generate.py`와 `scripts/eval_offline_metrics.py`로 Stage A 생성/평가를 재현한다.
2. 빈 MIDI, density 0, undecodable output을 실패로 처리하는 gate를 추가한다.
3. Python CLI를 MVP 입력 형태로 안정화한다.
4. 모델 output repair와 fallback을 안정화한다.
5. `scripts/run_stage_a_tiny_overfit.py`로 현재 tokenization/training path가 MIDI grammar를 배울 수 있는지 확인한다.
6. tiny-overfit 결과에 따라 duration-explicit tokenization 또는 control-token conditioning으로 넘어간다.

핵심 원칙: backend 확장보다 먼저, 모델 입력에서 valid MIDI와 metrics가 안정적으로 나오는 end-to-end 경로를 완성한다.
