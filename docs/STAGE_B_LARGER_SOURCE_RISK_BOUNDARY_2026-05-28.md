# Stage B Larger Source Risk Boundary

작성일: 2026-05-28

## 결론

| 항목 | 4 files | 5 files | 6 files |
|---|---:|---:|---:|
| repeatability gate | 통과 | 통과 | 통과 |
| total samples | `9` | `9` | `9` |
| strict valid samples | `8` | `7` | `7` |
| grammar gate samples | `9` | `9` | `9` |
| dead-air outliers | `1` | `2` | `1` |
| dead-air outlier rate | `0.111` | `0.222` | `0.111` |
| max postprocess removal | `0.400` | `0.350` | `0.429` |
| selected best dead-air | `0.438` | `0.467` | `0.375` |

4/5/6-file 조건 모두 hard gate는 통과했다.

단, 6-file 조건에서 seed `17`이 strict `1/3`까지 내려가고 `unique pitch count too low` failure가 새로 발생했다.
따라서 현재 확인된 boundary는 "hard failure"가 아니라 "seed-level strict margin 감소"다.

## 실행 조건

Commands:

```bash
ISSUE_NUMBER=232 MAX_FILES=4 MIN_SOURCE_FILES=4 RUN_ID=issue_232_stage_b_larger_source_risk_boundary_files4 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
ISSUE_NUMBER=232 MAX_FILES=5 MIN_SOURCE_FILES=5 RUN_ID=issue_232_stage_b_larger_source_risk_boundary_files5 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
ISSUE_NUMBER=232 MAX_FILES=6 MIN_SOURCE_FILES=6 RUN_ID=issue_232_stage_b_larger_source_risk_boundary_files6 bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
```

공통 조건:

- seeds: `17, 23, 31`
- epochs: `50`
- samples per seed: `3`
- top_k: `4`
- temperature: `0.9`
- overlap postprocess: enabled
- dead-air gate: `0.800`
- max dead-air outlier rate: `0.250`

## 결과

### 4 Files

| seed | strict | outliers | best sample | best dead-air |
|---:|---:|---:|---:|---:|
| `17` | `2/3` | `1` | `3` | `0.471` |
| `23` | `3/3` | `0` | `3` | `0.438` |
| `31` | `3/3` | `0` | `2` | `0.533` |

전체 selected best:

- seed `23`, sample `3`
- dead-air ratio `0.438`
- note count `17`
- phrase coverage `0.781`
- onset coverage `0.500`
- sustained coverage `0.594`

### 5 Files

| seed | strict | outliers | best sample | best dead-air |
|---:|---:|---:|---:|---:|
| `17` | `2/3` | `1` | `3` | `0.467` |
| `23` | `3/3` | `0` | `1` | `0.500` |
| `31` | `2/3` | `1` | `3` | `0.500` |

전체 selected best:

- seed `17`, sample `3`
- dead-air ratio `0.467`
- note count `16`
- phrase coverage `1.000`
- onset coverage `0.531`
- sustained coverage `0.781`

### 6 Files

| seed | strict | outliers | best sample | best dead-air | failure |
|---:|---:|---:|---:|---:|---|
| `17` | `1/3` | `1` | `3` | `0.500` | dead-air `1`, unique pitch `1` |
| `23` | `3/3` | `0` | `1` | `0.375` | none |
| `31` | `3/3` | `0` | `2` | `0.636` | none |

전체 selected best:

- seed `23`, sample `1`
- dead-air ratio `0.375`
- note count `9`
- phrase coverage `0.437`
- onset coverage `0.313`
- sustained coverage `0.438`

## 해석

증명한 것:

- source file 수를 `6`까지 늘려도 현재 hard gate는 유지된다.
- 모든 seed에서 strict-valid 대체 후보가 최소 1개 이상 남는다.
- grammar gate는 4/5/6-file 모두 `9/9`로 유지된다.
- dead-air outlier rate는 4-file `0.111`, 5-file `0.222`, 6-file `0.111`로 한계 `0.250` 안에 있다.

리스크:

- 6-file seed `17`은 strict-valid 후보가 `1/3`뿐이다.
- 6-file seed `17`에서 dead-air failure 외에 `unique pitch count too low: 2 < 3`가 발생했다.
- selected best candidate가 항상 음악적으로 좋은 후보라는 뜻은 아니다. 현재 선택 기준은 objective metric 기반이다.
- source file 수가 늘어나도 broad quality나 Brad style adaptation이 증명된 것은 아니다.

## 다음 작업

권장 이슈:

- `Stage B seed-level strict margin diagnostics`

목표:

- seed `17`의 6-file failures를 sample 단위로 분리
- dead-air failure와 unique-pitch failure가 같은 후보에서 발생하는지 확인
- candidate count 증가 또는 per-seed min strict threshold를 강화할지 판단
