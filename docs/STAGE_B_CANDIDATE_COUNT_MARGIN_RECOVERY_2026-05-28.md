# Stage B Candidate Count Margin Recovery

작성일: 2026-05-28

## 결론

6-file 조건에서 samples per seed를 `3`에서 `5`로 늘리면 seed `17`의 strict margin warning이 사라진다.

| 항목 | 3 samples/seed | 5 samples/seed |
|---|---:|---:|
| total samples | `9` | `15` |
| strict valid samples | `7/9` | `12/15` |
| strict pass-rate | `0.778` | `0.800` |
| grammar gate samples | `9/9` | `15/15` |
| dead-air outlier count | `1` | `2` |
| dead-air outlier rate | `0.111` | `0.133` |
| strict margin warning seeds | `17` | 없음 |
| selected best candidate | seed `23`, sample `1` | seed `23`, sample `1` |
| selected best dead-air | `0.375` | `0.375` |

후보 수 증가는 seed-level margin risk를 줄였다.
하지만 dead-air outlier 자체를 제거하지는 않았고, seed `23`에서 outlier가 하나 추가됐다.

## 실행 조건

Command:

```bash
ISSUE_NUMBER=238 MAX_FILES=6 MIN_SOURCE_FILES=6 NUM_SAMPLES=5 RUN_ID=issue_238_stage_b_candidate_count_margin_recovery bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
```

조건:

- source files: `6`
- seeds: `17, 23, 31`
- samples per seed: `5`
- epochs: `50`
- top_k: `4`
- temperature: `0.9`
- hard min strict per seed: `1`
- warning min strict per seed: `2`
- max dead-air outlier rate: `0.250`

## Seed Summary

| seed | samples | strict | margin warning | dead-air outliers | best sample | best dead-air | note range | pitch range |
|---:|---:|---:|:---:|---:|---:|---:|---|---|
| `17` | `5` | `3` | false | `1` | `3` | `0.500` | `8-20` | `2-6` |
| `23` | `5` | `4` | false | `1` | `1` | `0.375` | `7-19` | `3-7` |
| `31` | `5` | `5` | false | `0` | `5` | `0.444` | `12-19` | `4-8` |

## 해석

증명한 것:

- candidate count를 `5`로 늘리면 6-file run에서 seed strict margin warning이 사라진다.
- hard gate는 계속 통과한다.
- seed `17`은 `1/3`에서 `3/5`로 회복했다.
- selected best candidate는 3-sample baseline과 동일하게 seed `23`, sample `1`이다.

리스크:

- dead-air outlier count는 `1`에서 `2`로 늘었다.
- outlier rate는 `0.133`으로 gate `0.250` 안이지만, 후보 수 증가가 failure mode 자체를 고친 것은 아니다.
- seed `17` sample pool 안에는 여전히 dead-air failure와 unique pitch failure가 남아 있다.
- 이 결과는 objective gate 기준이며, listening quality를 증명하지 않는다.

## 결정

현재 6-file 조건에서는 `NUM_SAMPLES=5`가 `NUM_SAMPLES=3`보다 candidate selection 안정성이 높다.
다음 raw repeatability 실험의 기본 비교값은 `NUM_SAMPLES=5`로 두는 것이 합리적이다.

## 다음 작업

권장 이슈:

- `Stage B margin-recovered candidate review export`

목표:

- 5-sample run에서 seed별 best candidate를 review 대상으로 정리
- selected best와 seed별 best 후보의 objective metric 차이 기록
- MIDI 파일 자체는 commit하지 않고 report/document만 남김
