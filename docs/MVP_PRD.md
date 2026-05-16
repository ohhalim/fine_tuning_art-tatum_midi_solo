# MVP PRD: Personalized Live MIDI Improviser

작성일: 2026-05-16
상태: MVP 기준 문서

## 1. 제품 요약

`Personalized Live MIDI Improviser`는 house/techno groove 위에서 사용할 짧은 재즈 피아노 솔로 MIDI phrase를 생성하는 symbolic MIDI 시스템이다.

사용자는 BPM, 코드 진행, 마디 수, 섹션, energy, density를 입력하고, 시스템은 1~2마디짜리 piano solo MIDI와 metrics를 생성한다.

이 프로젝트는 raw audio generation이 아니다. 첫 MVP는 DAW plugin도 아니다. 목표는 `Spring Boot backend + Python inference server + MIDI generation pipeline`을 갖춘 모델 서빙 시스템이다.

## 2. 문제 정의

라이브 또는 제작 상황에서 drop, break, build-up 구간에 짧은 jazz piano improvisation phrase가 필요하다. 기존 DAW 작업은 직접 연주하거나 MIDI loop를 찾아야 해서 반복적이고, performance context에 즉시 반응하기 어렵다.

MVP는 완벽한 재즈 모델보다 다음 문제를 먼저 해결한다.

- 구조화된 musical input을 API로 받는다.
- 항상 열 수 있는 valid MIDI를 만든다.
- 생성 실패를 감지하고 fallback한다.
- 결과를 job 단위로 추적한다.
- metrics로 품질을 설명한다.

## 3. 목표 사용자

1차 사용자:

- 본인, 음악 제작자, DJ, live performer.

2차 관점:

- 채용 담당자 또는 면접관이 볼 수 있는 backend/model-serving portfolio.

## 4. MVP 목표

한 달 안에 아래 흐름을 완성한다.

```text
REST request
  -> Spring Boot generation job
  -> Python FastAPI inference
  -> MIDI generation pipeline
  -> generated.mid + metrics.json
  -> job status + download endpoint
```

## 5. MVP 입력

필수:

- `bpm`: integer, 예: 124
- `timeSignature`: string, 기본 `4/4`
- `chordProgression`: string array, 예: `["Cm7", "Fm7", "Bb7", "Ebmaj7"]`
- `bars`: integer, 기본 2, MVP 범위 1~4
- `section`: `intro`, `build`, `breakdown`, `drop`
- `energy`: `low`, `mid`, `high`
- `density`: `sparse`, `medium`, `dense`

선택:

- `key`: string
- `style`: string, 기본 `personal_jazz`
- `temperature`: float
- `topK`: integer
- `topP`: float
- `maxNotesPerBar`: integer
- `pitchMin`, `pitchMax`: integer MIDI pitch

## 6. MVP 출력

- generated MIDI file
- generation metadata JSON
- metrics:
  - `generationTimeMs`
  - `noteCount`
  - `noteDensity`
  - `deadAirRatio`
  - `repetitionScore`
  - `pitchMin`
  - `pitchMax`
  - `durationSec`
  - `status`
  - `failureReason`, 실패 시

## 7. 성공 기준

MVP는 아래 조건을 만족하면 완료다.

- API 요청 한 번으로 valid MIDI가 생성된다.
- 사용자는 job status를 조회할 수 있다.
- 사용자는 MIDI를 다운로드할 수 있다.
- metrics가 저장된다.
- 생성 실패 시 `FAILED`로 끝나며 이유가 남는다.
- 모델 출력이 비어 있으면 fallback phrase를 생성한다.
- README만 보고 로컬에서 실행할 수 있다.

## 8. 품질 기준

생성된 phrase accept gate:

- MIDI file이 DAW 또는 MIDI player에서 열린다.
- note count가 0이 아니다.
- 요청한 bars와 duration이 크게 어긋나지 않는다.
- dead-air ratio가 과도하지 않다.
- 같은 pitch 또는 같은 4-gram 반복이 과도하지 않다.
- pitch가 piano range 안에 있다.
- generation time이 로컬 기준 실험적으로 기록된다.

초기 숫자 기준:

- `noteCount > 0`
- `noteDensity >= 0.5` for medium/dense
- `deadAirRatio < 0.8`
- `pitchMin >= 21`
- `pitchMax <= 108`

이 기준은 초기 gate이며 음악적 품질 최종 기준은 아니다.

## 9. 비목표

MVP에서 하지 않는다.

- raw audio generation
- VST/JUCE plugin
- full DAW realtime integration
- SaaS auth/billing
- multi-user production service
- huge pretraining
- exact artist clone product wording
- complex frontend
- advanced jazz theory engine
- 완벽한 실시간 latency 달성

## 10. 포트폴리오 포지셔닝

좋은 설명:

> Built a symbolic MIDI generation backend that serves a music generation pipeline through a Spring Boot API and Python inference server. Implemented asynchronous generation jobs, MIDI file output, result download, and generation metrics such as note density, repetition, and dead-air ratio.

피해야 할 설명:

> 유명 피아니스트를 복제하는 모델.

공개 표현:

- personalized improvisation
- style-conditioned symbolic MIDI generation
- model-serving backend for music generation
- live performance MIDI phrase generation
