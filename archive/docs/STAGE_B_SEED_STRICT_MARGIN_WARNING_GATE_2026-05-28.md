# Stage B Seed Strict Margin Warning Gate

작성일: 2026-05-28

## 결론

repeatability summary에 seed별 strict margin warning을 추가했다.
기존 hard gate는 유지하고, seed별 strict-valid 후보 수가 warning 기준보다 낮을 때만 별도 경고로 기록한다.

Issue #236 6-file 검증 결과:

| 항목 | 결과 |
|---|---|
| repeatability gate | 통과 |
| source files | `6` |
| seeds | `17, 23, 31` |
| total samples | `9` |
| strict valid samples | `7/9` |
| grammar gate samples | `9/9` |
| dead-air outlier rate | `0.111` |
| hard min strict per seed | `1` |
| warning min strict per seed | `2` |
| strict margin warning seeds | `17` |
| selected best candidate | seed `23`, sample `1`, dead-air `0.375` |

## 변경 내용

- `run_stage_b_raw_generation_repeatability_sweep.py`
  - `--warning_min_strict_samples_per_seed` 인자 추가
  - `strict_margin_warning_seed_count` 기록
  - `strict_margin_warning_seeds` 기록
  - `strict_margin_warning_rows` 기록
- `repeatability_summary.md`
  - warning 기준과 warning seed 목록 출력
  - seed table에 `margin warning` column 추가
- `agent_harness.sh`
  - `WARNING_MIN_STRICT_SAMPLES_PER_SEED` 환경 변수 연결

## 실행 조건

Commands:

```bash
bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
ISSUE_NUMBER=236 MAX_FILES=6 MIN_SOURCE_FILES=6 RUN_ID=issue_236_stage_b_seed_strict_margin_warning_gate bash scripts/agent_harness.sh stage-b-raw-generation-repeatability
```

공통 조건:

- epochs: `50`
- samples per seed: `3`
- top_k: `4`
- temperature: `0.9`
- hard min strict per seed: `1`
- warning min strict per seed: `2`
- max dead-air outlier rate: `0.250`

## 6-File Seed Summary

| seed | strict | margin warning | dead-air outliers | best sample | best dead-air |
|---:|---:|:---:|---:|---:|---:|
| `17` | `1/3` | true | `1` | `3` | `0.500` |
| `23` | `3/3` | false | `0` | `1` | `0.375` |
| `31` | `3/3` | false | `0` | `2` | `0.636` |

## 해석

증명한 것:

- hard gate와 soft warning이 분리됐다.
- 6-file run은 기존 hard gate를 계속 통과한다.
- seed `17`의 strict margin risk가 aggregate pass-rate에 묻히지 않고 summary에 직접 드러난다.
- warning은 실패 처리가 아니라 다음 실험 우선순위 신호로 남는다.

리스크:

- warning seed가 있다는 것은 candidate selection 안정성이 낮다는 뜻이다.
- warning만 추가했을 뿐, seed `17`의 실패를 줄인 것은 아니다.
- broad quality나 Brad style adaptation은 여전히 증명되지 않았다.

## 다음 작업

권장 이슈:

- `Stage B candidate count margin recovery sweep`

목표:

- 6-file 조건에서 samples per seed를 늘렸을 때 seed `17` margin warning이 줄어드는지 확인
- candidate count 증가가 hard gate 안정성과 selected best quality에 미치는 영향 기록
