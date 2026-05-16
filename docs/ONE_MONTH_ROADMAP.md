# One Month Roadmap

작성일: 2026-05-16

목표: 한 달 안에 `API request -> valid MIDI output -> job status -> download -> metrics`가 동작하는 MVP를 만든다.

## Week 1. Generation Baseline 고정

목표:

- 현재 레포의 Stage A 생성 경로를 확인한다.
- valid MIDI를 만드는 최소 CLI를 확정한다.
- 실패 output을 감지한다.

작업:

- `scripts/generate.py` 실행 확인.
- `scripts/eval_offline_metrics.py` 실행 확인.
- `outputs/` 또는 `samples/stage_a/` 기준 출력 위치 결정.
- empty MIDI, density 0, decode failure를 실패로 분류.
- fallback phrase generator 초안 작성.
- CLI 입력을 MVP request field와 맞춘다.

산출물:

- `generated.mid`
- `metrics.json`
- generation failure case 기록.
- README 실행 명령 초안.

완료 기준:

- 한 명령으로 MIDI와 metrics가 생성된다.
- 실패 시 실패 이유가 JSON에 남는다.

## Week 2. Python FastAPI Inference Server

목표:

- CLI generation pipeline을 API로 감싼다.

작업:

- `inference/app/main.py`
- `inference/app/schemas.py`
- `inference/app/generator.py`
- `inference/app/metrics.py`
- `inference/app/postprocess.py`
- `POST /infer/midi` 구현.
- request validation.
- output path 규칙 정의.
- structured error response.

산출물:

- FastAPI 서버.
- curl 예시.
- API request로 생성된 MIDI.

완료 기준:

- `curl POST /infer/midi`가 `midiPath`와 `metrics`를 반환한다.

## Week 3. Spring Boot Job API

목표:

- backend portfolio로 설명 가능한 job API를 만든다.

작업:

- `GenerationJob` entity.
- `GenerationJobStatus` enum.
- request DTO, response DTO.
- generation job service.
- Python inference client.
- job status lifecycle.
- MIDI download endpoint.
- PostgreSQL 연결.
- 로컬 개발용 Docker Compose 초안.

산출물:

- `POST /api/generation-jobs`
- `GET /api/generation-jobs/{jobId}`
- `GET /api/generation-jobs/{jobId}/download`
- DB에 저장된 job metadata.

완료 기준:

- Spring API 요청으로 Python inference를 호출하고 결과를 조회/다운로드할 수 있다.

## Week 4. Quality Gate와 Portfolio Polish

목표:

- MVP를 설명 가능한 프로젝트로 정리한다.

작업:

- metrics table 정리.
- sample MIDI 3~5개 생성.
- failure handling 정리.
- README 업데이트.
- architecture diagram 문서화.
- demo script 작성.
- 수동 QA 체크리스트 통과.

산출물:

- 최종 README.
- sample request/response.
- sample MIDI outputs.
- metrics summary.
- 한계점과 next steps.

완료 기준:

- 처음 보는 사람이 README만 보고 실행할 수 있다.
- 면접에서 architecture와 tradeoff를 설명할 수 있다.

## Cut Line

시간이 부족하면 반드시 남길 것:

1. Python CLI valid MIDI generation.
2. FastAPI inference endpoint.
3. Spring Boot job status API.
4. MIDI download.
5. README.

시간이 부족하면 미룰 것:

1. realtime DAW routing.
2. control token retraining.
3. UI.
4. Docker Compose 완성도.
5. advanced jazz theory metric.
