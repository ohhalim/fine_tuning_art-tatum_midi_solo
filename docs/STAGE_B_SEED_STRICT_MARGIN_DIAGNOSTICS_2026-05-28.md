# Stage B Seed Strict Margin Diagnostics

작성일: 2026-05-28

## 결론

Issue #232의 6-file repeatability run에서 seed `17`만 strict margin warning에 걸렸다.

| 항목 | 결과 |
|---|---|
| source run | `issue_232_stage_b_larger_source_risk_boundary_files6` |
| hard min strict per seed | `1` |
| warning min strict per seed | `2` |
| margin warning seeds | `17` |
| dead-air + unique-pitch overlap seeds | 없음 |
| dead-air + unique-pitch separate seeds | `17` |

seed `17`의 실패는 한 후보에 겹친 문제가 아니라 서로 다른 후보에서 발생했다.

- sample `1`: dead-air failure
- sample `2`: unique pitch failure
- sample `3`: strict-valid best candidate

따라서 현재 hard gate를 바로 실패 처리할 근거는 부족하다.
다만 seed별 strict 후보가 1개만 남는 상태는 후보 선택 안정성이 낮으므로, 다음 단계에서는 per-seed strict margin을 warning 또는 soft gate로 repeatability summary에 포함시키는 것이 맞다.

## 실행 조건

Command:

```bash
bash scripts/agent_harness.sh stage-b-seed-strict-margin-diagnostics
```

내부 입력:

```bash
SUMMARY_PATH=outputs/stage_b_raw_generation_repeatability/issue_232_stage_b_larger_source_risk_boundary_files6/repeatability_summary.json
```

Gate:

- hard min strict per seed: `1`
- warning min strict per seed: `2`
- expected margin warning seeds: `17`

## Seed Summary

| seed | samples | strict | margin warning | dead-air samples | unique-pitch samples | overlap samples | best sample | best dead-air |
|---:|---:|---:|:---:|---|---|---|---:|---:|
| `17` | `3` | `1` | true | `1` | `2` | 없음 | `3` | `0.500` |
| `23` | `3` | `3` | false | 없음 | 없음 | 없음 | `1` | `0.375` |
| `31` | `3` | `3` | false | 없음 | 없음 | 없음 | `2` | `0.636` |

## Seed 17 Sample Detail

| sample | strict | notes | pitches | dead-air | phrase | onset | sustained | tail | removal | reason |
|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `1` | false | `15` | `3` | `0.857` | `1.000` | `0.312` | `0.844` | `0` | `0.250` | dead-air ratio too high |
| `2` | false | `8` | `2` | `0.714` | `0.437` | `0.250` | `0.375` | `17` | `0.200` | unique pitch count too low |
| `3` | true | `17` | `4` | `0.500` | `1.000` | `0.594` | `0.844` | `0` | `0.227` | none |

## 해석

증명한 것:

- 6-file 조건의 hard gate는 아직 유지된다.
- seed `17`도 strict-valid 후보가 하나 남는다.
- dead-air failure와 unique-pitch failure는 같은 후보에 겹친 collapse가 아니다.
- unique-pitch failure sample은 tail empty `17`로 phrase 후반이 비어 있고 pitch diversity도 낮다.

리스크:

- seed `17`은 strict-valid 후보가 `1/3`뿐이라 candidate selection 안정성이 낮다.
- aggregate strict pass-rate만 보면 seed별 margin 감소를 놓칠 수 있다.
- 현재 결과는 objective metric 진단이며, broad listening quality나 style adaptation을 증명하지 않는다.

## 다음 작업

권장 이슈:

- `Stage B per-seed strict margin warning gate`

목표:

- repeatability summary에 warning min strict per seed를 선택적으로 기록
- hard gate와 soft warning을 분리
- margin warning seed와 sample-level failure breakdown을 summary markdown에 포함
