# Stage B Phrase Naturalness Objectives

작성일: 2026-05-22

## 목적

Issue #103은 Issue #101의 `phrase_cadence` 후보가 scalar/chromatic flag를 줄인 대신, 큰 도약만 나열하는 exercise가 되었는지 확인하기 위한 objective metric을 추가한 단계다.

이 작업은 "재즈답다"를 자동 판정하지 않는다. 사람이 듣기 전에 다음 failure mode를 숫자로 드러내는 것이 목적이다.

> 큰 도약 뒤에 반대 방향의 작은 회복 움직임이 없으면 phrase로 들리기 어렵다.

## 구현

변경 사항:

- `scripts/review_midi_note_objectives.py`
  - `large_leap_count`
  - `resolved_large_leap_count`
  - `unresolved_large_leap_count`
  - `unresolved_large_leap_ratio`
  - `unresolved_large_leaps` objective flag
  - markdown report의 large-leap columns
- `scripts/build_listening_review_notes.py`
  - objective review metrics에 large-leap metrics 전달
- `tests/test_objective_midi_note_review.py`
  - unresolved leap 후보 flag test
  - resolved leap 후보 non-flag test
- `tests/test_listening_review_notes.py`
  - listening notes metric propagation fixture 갱신

## Metric Definition

Large leap:

- adjacent interval absolute value가 `7` semitone 이상인 경우

Resolved large leap:

- large leap 바로 다음 interval이 존재하고
- 다음 interval 방향이 large leap와 반대이며
- 다음 interval 크기가 `1-5` semitone인 경우

Flag:

- `large_leap_count >= 3`
- `unresolved_large_leap_ratio >= 0.45`

이 조건이면 `unresolved_large_leaps`를 objective flag에 추가한다.

## 하네스

실행:

```bash
bash scripts/agent_harness.sh stage-b-objective-midi-review
bash scripts/agent_harness.sh stage-b-phrase-cadence-review
```

## 결과

Issue #101 phrase/cadence review set에 새 metric을 적용한 결과:

- candidate count: `12`
- objective bucket counts:
  - warning: `12`
- objective flag counts:
  - chromatic walk: `1`
  - unresolved large leaps: `12`
- mode flag counts:
  - `data_motif`: unresolved large leaps `3`, chromatic walk `1`
  - `data_motif_guide_tones`: unresolved large leaps `3`
  - `phrase_cadence`: unresolved large leaps `3`
  - `varied_guide_tones`: unresolved large leaps `3`

기존 Issue #101 해석 보정:

- scalar/chromatic issue는 줄었다.
- 하지만 모든 review candidate가 phrase naturalness risk를 가진다.
- 따라서 clean `11`개였다는 이전 objective bucket은 새 metric 기준으로 더 이상 clean이 아니다.

## 해석

이 결과는 regression이 아니다. 이전 metric이 보지 못하던 문제를 새로 잡은 것이다.

현재 남은 핵심 문제:

- leap가 melodic phrase로 resolution되지 않는다.
- guide-tone/cadence vocabulary는 있지만 phrase-shape grammar가 부족하다.
- 다음 generation change는 큰 도약 뒤 회복 움직임을 강제하거나, data-derived contour template에서 resolution pattern을 추출하는 방향이어야 한다.

## 검증

실행한 검증:

```bash
./.venv/bin/python -m unittest tests.test_objective_midi_note_review tests.test_listening_review_notes
bash scripts/agent_harness.sh stage-b-objective-midi-review
bash scripts/agent_harness.sh stage-b-phrase-cadence-review
bash scripts/agent_harness.sh quick
```
