# MVP Implementation Plan

작성일: 2026-05-16

이 문서는 Codex로 구현할 때의 작업 순서다. 한 번에 대량 생성하지 않고, 실행 가능한 slice 단위로 진행한다.

## 1. 구현 원칙

- 기존 코드를 먼저 살린다.
- `scripts/generate.py`를 Stage A generator 기준으로 본다.
- `music_transformer/generate.py`는 legacy로 둔다.
- valid MIDI output을 최우선으로 한다.
- 모델이 실패하면 fallback phrase를 생성한다.
- Spring Boot는 Python path가 안정화된 뒤 붙인다.

## 2. Target Structure

초기에는 기존 구조를 크게 깨지 않는다. 필요한 최소 폴더만 추가한다.

```text
inference/
  app/
    main.py
    schemas.py
    generator.py
    metrics.py
    postprocess.py
    fallback.py
  requirements.txt

backend/
  spring-api/
    src/
    build.gradle

outputs/
  generated/
  metrics/
```

기존 `scripts/`는 학습과 로컬 실험용으로 유지한다.

## 3. Slice 1: Generation Contract 고정

목표:

- Python 함수 하나가 request dict를 받아 MIDI와 metrics를 만든다.

예상 함수:

```python
def generate_midi_phrase(request: GenerationRequest) -> GenerationResult:
    ...
```

해야 할 일:

- MVP request schema 정의.
- output directory 규칙 정의.
- 기존 `scripts/generate.py` 호출 또는 내부 로직 재사용 방식 결정.
- 실패 감지:
  - file not created
  - note count 0
  - unreadable MIDI
  - density gate fail
- fallback phrase 생성.

완료 기준:

- Python에서 함수 호출만으로 MIDI와 metrics가 생성된다.

## 4. Slice 2: CLI 안정화

목표:

- CLI로 MVP 입력을 받아 end-to-end 생성한다.

예상 명령:

```bash
python inference/app/generator.py \
  --bpm 124 \
  --chords Cm7,Fm7,Bb7,Ebmaj7 \
  --bars 2 \
  --section drop \
  --energy high \
  --density medium \
  --output_dir outputs/generated
```

완료 기준:

- `outputs/generated/<job_id>.mid`
- `outputs/metrics/<job_id>.json`

## 5. Slice 3: FastAPI Inference Server

목표:

- generation contract를 HTTP로 노출한다.

Endpoint:

- `GET /health`
- `POST /infer/midi`

완료 기준:

- curl로 MIDI 생성 가능.
- validation error는 422.
- generation failure는 structured 500 또는 success with fallback flag 중 하나로 일관되게 처리.

## 6. Slice 4: Spring Boot API

목표:

- job lifecycle과 download API를 만든다.

패키지 후보:

```text
com.personalizedimproviser.generation
```

구성:

- controller
- service
- entity
- repository
- dto
- client
- exception

완료 기준:

- `POST /api/generation-jobs` creates job.
- `GET /api/generation-jobs/{id}` returns status.
- `GET /api/generation-jobs/{id}/download` returns MIDI.

## 7. Slice 5: Persistence

목표:

- PostgreSQL에 job metadata를 저장한다.

초기 저장 대상:

- request fields
- status
- result paths
- metrics JSON
- failure reason
- timestamps

완료 기준:

- 서버 재시작 후에도 job metadata 조회 가능.

## 8. Slice 6: README와 Demo

목표:

- 포트폴리오로 보여줄 수 있는 실행 경로를 정리한다.

필수 섹션:

- What it does
- Architecture
- Local setup
- Example request
- Example response
- Generated outputs
- Metrics
- Known limitations
- Future work

## 9. 작업 순서 체크리스트

- [ ] existing generator 실행 확인
- [ ] metrics 재사용 가능성 확인
- [ ] generation request/result schema 작성
- [ ] fallback MIDI generator 구현
- [ ] CLI 구현
- [ ] FastAPI 구현
- [ ] Spring Boot skeleton 구현
- [ ] DB entity 구현
- [ ] inference client 구현
- [ ] download endpoint 구현
- [ ] README 업데이트

## 10. Merge 기준

각 slice는 아래 기준을 만족해야 다음으로 넘어간다.

- 실행 명령이 문서에 있다.
- 생성 파일 경로가 명확하다.
- 실패 케이스가 JSON 또는 log에 남는다.
- 기존 Stage A 학습 스크립트를 깨지 않는다.
