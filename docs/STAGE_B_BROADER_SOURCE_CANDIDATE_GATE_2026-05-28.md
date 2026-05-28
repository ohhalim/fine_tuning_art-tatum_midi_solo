# Stage B Broader Source Candidate Gate

작성일: 2026-05-28

## 결론

| 항목 | 값 |
|---|---:|
| repeatability gate | 통과 |
| source files | `3` |
| seeds | `17, 23, 31` |
| total samples | `9` |
| strict valid samples | `7` |
| grammar gate samples | `9` |
| dead-air outliers | `2` |
| dead-air outlier rate | `0.222` |
| max allowed outlier rate | `0.250` |
| selected best candidate | seed `17`, sample `3` |
| selected best dead-air | `0.222` |

Issue #230은 source file 수를 `2`에서 `3`으로 늘린 상태에서 dead-air-aware candidate gate가 유지되는지 검증한 작업이다.

## 실행 조건

Command:

```bash
ISSUE_NUMBER=230 MAX_FILES=3 MIN_SOURCE_FILES=3 RUN_ID=issue_230_stage_b_broader_source_candidate_gate bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
```

조건:

- max files: `3`
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
| `17` | `3` | `2` | `1` | `3` | `0.222` | `8-10` | `3-4` | `0.437-1.000` | `0.250` |
| `23` | `3` | `3` | `0` | `1` | `0.438` | `14-17` | `5-7` | `0.687-0.875` | `0.222` |
| `31` | `3` | `2` | `1` | `1` | `0.421` | `12-20` | `4-6` | `0.812-0.937` | `0.238` |

전체 selected best:

- seed: `17`
- sample: `3`
- dead-air ratio: `0.222`
- note count: `10`
- unique pitch count: `4`
- phrase coverage: `0.469`
- onset coverage: `0.281`
- sustained coverage: `0.406`
- postprocess removal ratio: `0.091`

## 해석

증명한 것:

- 3-source-file 조건에서도 모든 seed에서 strict-valid 대체 후보가 생성된다.
- dead-air outlier는 `2/9`로 증가했지만, gate 한계 `0.250` 안에 남았다.
- selected best candidate의 dead-air ratio는 `0.222`로 2-file run의 `0.333`보다 낮다.
- grammar gate는 `9/9`로 유지된다.

남은 리스크:

- strict pass-rate는 2-file `8/9`에서 3-file `7/9`로 낮아졌다.
- seed `17`, `31`에서 각각 dead-air outlier가 1개씩 발생했다.
- source file 수는 아직 `3`으로 제한되어 있어 broad quality claim은 불가하다.
- 사람이 듣는 선호나 Brad style adaptation은 검증하지 않았다.

## 다음 작업

권장 이슈:

- `Stage B larger source repeatability risk boundary`

목표:

- source file 수를 더 늘릴 때 outlier rate가 gate를 넘는 경계 확인
- strict pass-rate와 selected best dead-air ratio의 변화 기록
- 필요 시 outlier rate gate를 report outcome으로 낮추고 retry/candidate count 확장 여부 판단
