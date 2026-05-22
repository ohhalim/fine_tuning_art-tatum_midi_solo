# Stage B Chord Progression Coverage Audit

작성일: 2026-05-22

## 배경

Issue #75에서 reference pitch-role landing stats를 만들려고 했지만, known chord note ratio가 `0.000`이었다.

이 문제는 generator 문제가 아니다.

실제 reference tokenized records에 chord progression annotation이 없으면 다음을 판단할 수 없다.

- 어떤 음이 chord tone인지
- 어떤 음이 guide tone인지
- 어떤 음이 tension인지
- 어떤 음이 approach/outside인지

따라서 이번 단계는 현재 로컬 데이터 안에 chord progression annotation이 실제로 있는지 audit하는 것이다.

## 구현

새 스크립트:

```bash
python scripts/audit_chord_progression_coverage.py
```

새 harness:

```bash
bash scripts/agent_harness.sh chord-coverage-audit
```

Audit 대상:

- role dataset `meta.json`
- raw dataset sidecar files:
  - `.json`
  - `.csv`
  - `.tsv`
  - `.txt`
  - `.lab`
  - `.jams`
  - `.xml`
  - `.musicxml`
  - `.mxl`
- MIDI lyric/text events

탐지 기준:

- `chord_progression`
- `chords`
- `harmony`
- `changes`
- `lead_sheet`
- chord-like symbols such as `Cm7`, `F7`, `Bbmaj7`

## 결과

실행:

```bash
bash scripts/agent_harness.sh chord-coverage-audit
```

출력:

```text
outputs/chord_coverage_audit/harness_chord_coverage_audit/chord_coverage_audit.json
outputs/chord_coverage_audit/harness_chord_coverage_audit/chord_coverage_audit.md
```

요약:

| source | scanned | hits | ratio |
|---|---:|---:|---:|
| role meta | 2812 | 0 | 0.000 |
| sidecars | 0 | 0 | 0.000 |
| MIDI files scanned for text events | 120 | 0 | 0.000 |

추가 정보:

- role meta unique source MIDI count: `28`
- role meta chord fields: `0`
- sidecar files found: `0`
- MIDI files scanned for text events: `120`
- MIDI files with any text event: `0`
- MIDI chord-text candidate files: `0`
- usable chord annotation candidate: `false`

## 판단

현재 로컬 데이터셋에는 바로 사용할 수 있는 chord progression annotation이 없다.

따라서 다음 중 하나가 필요하다.

1. chord inference pipeline
2. lead-sheet / changes source alignment
3. 외부 chord-annotated jazz dataset 확보
4. 수동으로 작은 chord-labeled evaluation subset 작성

지금 generator를 더 조정하면 안 된다.

이유:

- generated 후보의 chord-tone/tension/approach 비율을 reference와 비교할 수 없다.
- reference 없는 상태에서 pitch grammar를 조정하면 "재즈스러움"이 아니라 임의의 hand-written rule을 최적화하게 된다.

## 다음 작업

다음 이슈는 다음 중 하나로 잡는다.

우선순위 1:

```text
작은 chord-labeled evaluation subset 만들기
```

이유:

- full automatic chord inference보다 작고 검증 가능하다.
- 3-5개 phrase만 chord-labeled로 만들어도 generated candidate의 pitch-role sanity check를 시작할 수 있다.

우선순위 2:

```text
chord inference / lead-sheet alignment 조사
```

이유:

- 전체 dataset에 chord labels가 없으므로 long-term에는 필요하다.
- 하지만 MVP 한 달 계획에서는 작게 잘라야 한다.

## Validation

```bash
./.venv/bin/python -m unittest tests.test_chord_progression_coverage_audit
bash scripts/agent_harness.sh chord-coverage-audit
bash scripts/agent_harness.sh quick
```
