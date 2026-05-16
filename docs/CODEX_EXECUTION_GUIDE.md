# Codex Execution Guide

작성일: 2026-05-16

이 프로젝트는 Codex만으로 구현한다는 전제로 진행한다. 핵심은 Codex에게 한 번에 “전부 만들어줘”라고 하지 않고, 작은 slice 단위로 검증하면서 가는 것이다.

## 1. Codex 기본 규칙

Codex에게 항상 먼저 시킬 일:

1. 현재 레포 구조 확인.
2. 기존 구현 재사용 가능성 확인.
3. 수정할 파일 목록 제안.
4. 최소 구현 단위 제안.
5. 그 다음 코드 수정.

피할 요청:

- “전체 MVP 한 번에 만들어줘.”
- “Spring Boot, FastAPI, 모델, Docker, README 다 만들어줘.”
- “실시간 DAW 연동까지 한 번에 해줘.”

좋은 요청:

- “기존 `scripts/generate.py`가 valid MIDI를 만드는지 확인하고, 실패 조건을 정리해줘.”
- “FastAPI inference server의 최소 파일만 추가해줘.”
- “Spring Boot job entity와 controller skeleton만 만들어줘.”

## 2. 권장 프롬프트 1: 레포 점검

```text
Inspect the repository first. Summarize the existing MIDI generation, training, and evaluation paths. Identify the smallest existing path that can generate a valid MIDI file. Do not write code yet. Return a concrete implementation plan with file paths.
```

## 3. 권장 프롬프트 2: Generation Contract

```text
Implement the smallest Python generation contract for the MVP. Reuse existing scripts where possible. The function should accept bpm, chordProgression, bars, section, energy, density, and output_dir, then produce a valid MIDI file and metrics JSON. Add fallback generation if the model output is empty. Keep changes scoped and list changed files.
```

## 4. 권장 프롬프트 3: FastAPI

```text
Add a minimal FastAPI inference server around the generation contract. Implement GET /health and POST /infer/midi. Use Pydantic schemas, structured error handling, and local output paths. Do not add Spring Boot yet.
```

## 5. 권장 프롬프트 4: Spring Boot

```text
Now add a Spring Boot API server for generation jobs. Implement POST /api/generation-jobs, GET /api/generation-jobs/{jobId}, and GET /api/generation-jobs/{jobId}/download. Store job metadata in PostgreSQL. Call the Python inference server. Keep the first version synchronous if async would add too much complexity.
```

## 6. 권장 프롬프트 5: README

```text
Update README so a new developer can run the MVP locally. Include architecture, setup, example API request, example response, generated output paths, metrics, and known limitations. Do not overstate model quality.
```

## 7. 작업 단위

한 번의 Codex 작업은 이 정도 크기가 적당하다.

- 파일 1~5개 수정.
- 한 feature slice.
- 실행 명령 포함.
- 검증 결과 포함.

너무 큰 작업:

- backend + inference + model + docs 동시 구현.
- DB schema와 Docker와 UI 동시 구현.
- 실시간 MIDI routing과 학습 파이프라인 동시 변경.

## 8. 리뷰 체크리스트

Codex 결과를 받을 때 확인:

- 기존 `scripts/`가 깨졌는가.
- output path가 문서와 맞는가.
- 실패 처리가 있는가.
- README 명령이 실제로 실행 가능한가.
- 빈 MIDI가 성공으로 처리되지 않는가.
- Spring API와 Python API schema가 같은가.

## 9. 커밋 전략

권장 커밋 단위:

1. docs: add MVP product and implementation specs
2. feat: add MIDI generation contract and fallback
3. feat: add FastAPI inference server
4. feat: add Spring generation job API
5. feat: persist generation jobs and metrics
6. docs: add runnable MVP README and demo

원격 push, PR 생성, 배포는 사용자 명시 허락 후에만 진행한다.
