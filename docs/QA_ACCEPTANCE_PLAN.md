# QA and Acceptance Plan

작성일: 2026-05-16

## 1. QA 목표

MVP 품질은 “음악적으로 완벽함”이 아니라 “시스템이 안정적으로 valid MIDI를 생성하고 실패를 설명함”이다.

## 2. Acceptance Gates

### Gate A: Python CLI

통과 조건:

- CLI command가 exit code 0으로 끝난다.
- `.mid` 파일이 생성된다.
- `.json` metrics 파일이 생성된다.
- metrics에 `noteCount > 0`이 기록된다.
- `pretty_midi.PrettyMIDI(path)`로 읽힌다.

### Gate B: FastAPI

통과 조건:

- `GET /health` returns 200.
- `POST /infer/midi` returns 200 for valid request.
- invalid request returns validation error.
- generated path exists.
- metrics object exists.

### Gate C: Spring Boot

통과 조건:

- `POST /api/generation-jobs` creates job.
- job reaches `COMPLETED` or `FAILED`.
- completed job has download endpoint.
- failed job has `failureReason`.
- DB에는 request와 status가 저장된다.

### Gate D: Portfolio Demo

통과 조건:

- README command works.
- sample request/response is accurate.
- generated MIDI examples exist.
- known limitations are documented.

## 3. Test Inputs

기본 테스트:

```json
{
  "bpm": 124,
  "timeSignature": "4/4",
  "key": "C minor",
  "chordProgression": ["Cm7", "Fm7", "Bb7", "Ebmaj7"],
  "bars": 2,
  "section": "drop",
  "energy": "high",
  "density": "medium",
  "style": "personal_jazz",
  "temperature": 0.9
}
```

경계 테스트:

- bars = 1
- bars = 4
- density = sparse
- density = dense
- bpm = 80
- bpm = 140
- invalid chord string
- empty chord progression

## 4. Metrics Thresholds

초기 gate:

| Metric | Sparse | Medium | Dense |
|---|---:|---:|---:|
| noteCount | > 0 | > 0 | > 0 |
| noteDensity | >= 0.2 | >= 0.5 | >= 1.0 |
| deadAirRatio | < 0.9 | < 0.8 | < 0.7 |
| pitchMin | >= 21 | >= 21 | >= 21 |
| pitchMax | <= 108 | <= 108 | <= 108 |

이 숫자는 테스트 gate다. 최종 음악적 기준이 아니다.

## 5. Manual QA Checklist

- [ ] MIDI file opens in DAW or MIDI player.
- [ ] phrase duration roughly matches requested bars.
- [ ] output is not silent.
- [ ] output is not one repeated note only.
- [ ] output register does not collide too much with bass range.
- [ ] download endpoint returns a MIDI file.
- [ ] metrics match the generated file.
- [ ] README command is not stale.

## 6. Failure Case Expectations

실패 시 반드시 남길 것:

- `status=FAILED`
- error code or failure reason.
- request snapshot.
- timestamp.

fallback 사용 시 반드시 남길 것:

- `fallbackUsed=true`
- original failure reason if available.

## 7. Regression Rule

기존 Stage A scripts를 깨지 않는다.

보존할 명령:

```bash
python scripts/generate.py --help
python scripts/eval_offline_metrics.py --help
bash scripts/runpod_train_stage_a.sh --help
```
