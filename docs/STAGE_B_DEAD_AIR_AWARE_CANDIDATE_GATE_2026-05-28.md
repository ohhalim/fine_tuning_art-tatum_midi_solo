# Stage B Dead-Air-Aware Candidate Gate

작성일: 2026-05-28

## 결론

| 항목 | 값 |
|---|---:|
| repeatability gate | 통과 |
| source files | `2` |
| seeds | `17, 23, 31` |
| total samples | `9` |
| strict valid samples | `8` |
| dead-air outliers | `1` |
| dead-air outlier rate | `0.111` |
| max allowed outlier rate | `0.250` |
| selected best candidate | seed `17`, sample `3` |
| selected best dead-air | `0.333` |

Issue #228은 repeatability sweep에 dead-air-aware candidate selection을 추가한 작업이다.

Outlier를 숨기지 않고 `1/9`로 기록하면서, strict-valid 후보 중 dead-air가 가장 낮은 후보를 seed별/전체 기준으로 선택한다.

## 실행 조건

Command:

```bash
ISSUE_NUMBER=228 RUN_ID=issue_228_stage_b_dead_air_candidate_gate bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
```

조건:

- max files: `2`
- seeds: `17, 23, 31`
- epochs: `50`
- samples per seed: `3`
- total samples: `9`
- top_k: `4`
- temperature: `0.9`
- overlap postprocess: enabled
- dead-air gate: `0.800`
- max dead-air outlier rate: `0.250`

## 결과

| seed | samples | strict | dead-air outliers | best sample | best dead-air | notes | pitches | phrase coverage | max removal |
|---:|---:|---:|---:|---:|---:|---|---|---|---:|
| `17` | `3` | `3` | `0` | `3` | `0.333` | `8-16` | `3-6` | `0.406-1.000` | `0.238` |
| `23` | `3` | `3` | `0` | `1` | `0.364` | `12-16` | `5-5` | `0.500-0.875` | `0.429` |
| `31` | `3` | `2` | `1` | `2` | `0.750` | `8-14` | `4-7` | `0.469-0.844` | `0.273` |

전체 selected best:

- seed: `17`
- sample: `3`
- dead-air ratio: `0.333`
- note count: `16`
- unique pitch count: `6`
- phrase coverage: `1.000`
- onset coverage: `0.500`
- sustained coverage: `0.781`
- postprocess removal ratio: `0.238`

## 해석

증명한 것:

- repeatability sweep는 outlier를 성공으로 덮지 않고 dead-air outlier count/rate를 별도 집계한다.
- seed `31`에는 outlier가 남아 있지만, 같은 seed 안에서 strict-valid 대체 후보 sample `2`를 선택할 수 있다.
- 전체 후보군 기준으로는 seed `17` sample `3`이 가장 낮은 dead-air ratio를 가진 strict-valid 후보로 선택된다.
- gate는 `strict 8/9`, `dead-air outlier rate 0.111 <= 0.250`, seed별 best 후보 존재 조건을 동시에 만족했다.

아직 아닌 것:

- dead-air outlier 생성 자체를 제거한 것은 아니다.
- source file 수는 여전히 `2`로 제한되어 있다.
- 사람이 듣는 선호나 Brad style adaptation은 검증하지 않았다.

## 다음 작업

권장 이슈:

- `Stage B broader source repeatability with candidate gate`

목표:

- source file 수를 `3+`로 늘려 candidate selection gate 유지 여부 확인
- selected best candidate의 dead-air/coverage 분포 기록
- outlier rate가 증가하는 seed/file 조건 분리
