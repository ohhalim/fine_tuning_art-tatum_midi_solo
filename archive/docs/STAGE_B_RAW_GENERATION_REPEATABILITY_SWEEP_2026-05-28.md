# Stage B Raw Generation Repeatability Sweep

작성일: 2026-05-28

## 결론

| 항목 | 판정 |
|---|---|
| repeatability gate | 통과 |
| source files | `2` |
| seeds | `17, 23, 31` |
| total samples | `9` |
| strict valid samples | `8` |
| grammar gate samples | `9` |
| dead-air outlier | `1` |

## 실행 조건

Command:

```bash
RUN_ID=issue_224_stage_b_raw_generation_repeatability_final2 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
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

## 결과

| 항목 | 값 |
|---|---:|
| total samples | `9` |
| valid sample count | `8` |
| strict valid sample count | `8` |
| grammar gate sample count | `9` |
| valid sample rate | `0.889` |
| strict valid sample rate | `0.889` |
| grammar gate sample rate | `1.000` |
| max postprocess removal ratio | `0.429` |
| allowed max postprocess removal ratio | `0.490` |

Per-seed:

| seed | files | samples | strict | grammar | note count | unique pitch | phrase coverage | max removal | failure |
|---:|---:|---:|---:|---:|---|---|---|---:|---|
| `17` | `2` | `3` | `3` | `3` | `8-16` | `3-6` | `0.406-1.000` | `0.238` | none |
| `23` | `2` | `3` | `3` | `3` | `12-16` | `5-5` | `0.500-0.875` | `0.429` | none |
| `31` | `2` | `3` | `2` | `3` | `8-14` | `4-7` | `0.469-0.844` | `0.273` | dead-air `1` |

## 해석

증명한 것:

- 1-file/1-seed가 아닌 2-file/3-seed 조건에서도 raw Stage B generation gate가 유지된다.
- 모든 seed에서 strict-valid sample이 최소 1개 이상 생성된다.
- grammar gate는 `9/9`로 통과했다.
- postprocess removal ratio는 최대 `0.429`로 strict gate 한계 `0.49` 안에 남았다.

남은 리스크:

- seed `31`에서 dead-air ratio outlier가 1개 발생했다.
- strict pass-rate가 `1.000`은 아니므로 broad quality claim은 아직 불가하다.
- source file 수는 `2`로 제한되어 있다.
- 사람이 듣는 선호나 Brad style adaptation은 검증하지 않았다.

## 다음 작업

권장 이슈:

- `Stage B raw generation dead-air outlier diagnostics`

목표:

- seed `31` 실패 sample의 dead-air 위치와 token pattern 확인
- phrase coverage는 통과하지만 dead-air가 높은 후보의 공통 원인 분리
- next gate에서 dead-air outlier 비율을 별도 추적
