# Stage B Dead-Air Outlier Diagnostics

작성일: 2026-05-28

## 결론

| 항목 | 값 |
|---|---:|
| source issue | `#224` |
| source seed | `31` |
| sample count | `3` |
| outlier sample | `1` |
| dead-air ratio | `0.857` |
| dead-air gate | `0.800` |
| strict valid samples | `2/3` |

Issue #224의 repeatability sweep에서 발생한 outlier는 sample `1`이었다.

원인은 collapse나 postprocess 과다 제거가 아니라, 낮은 onset/sustained coverage와 phrase 후반 공백이다.

## 실행 조건

Command:

```bash
REPORT_PATH=outputs/stage_b_generation_probe/issue_224_stage_b_raw_generation_repeatability_final2_seed31_files2/report.json RUN_ID=issue_226_stage_b_dead_air_diagnostics bash scripts/agent_harness.sh stage-b-dead-air-diagnostics
```

입력:

- source report: `outputs/stage_b_generation_probe/issue_224_stage_b_raw_generation_repeatability_final2_seed31_files2/report.json`
- dead-air threshold: `0.180s`
- dead-air gate: `0.800`
- density: `medium`
- bpm: `124`
- bars: `2`

## 결과

| sample | valid | strict | notes | pitches | dead-air | phrase | onset | sustained | span | head | tail | longest sustained empty | removed |
|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `1` | false | false | `8` | `4` | `0.857` | `0.469` | `0.250` | `0.375` | `0.469` | `6` | `11` | `10` | `3` |
| `2` | true | true | `9` | `4` | `0.750` | `0.562` | `0.281` | `0.531` | `0.562` | `6` | `8` | `8` | `2` |
| `3` | true | true | `14` | `7` | `0.769` | `0.844` | `0.469` | `0.781` | `0.844` | `5` | `0` | `5` | `4` |

Outlier sample `1`의 dead-air gap:

| from | to | start gap sec | silent gap sec |
|---:|---:|---:|---:|
| `1` | `2` | `0.363` | `0.000` |
| `2` | `3` | `0.242` | `0.121` |
| `3` | `4` | `0.242` | `0.121` |
| `5` | `6` | `0.242` | `0.121` |
| `6` | `7` | `0.242` | `0.121` |
| `7` | `8` | `0.242` | `0.000` |

## 해석

증상:

- sample `1`은 note count `8`, unique pitch count `4`로 최소 수량 조건은 통과한다.
- phrase coverage `0.469`도 medium 기준 최소 `0.350`은 넘는다.
- 하지만 onset coverage가 `0.250`, sustained coverage가 `0.375`로 낮다.
- tail empty가 `11` steps이고 longest sustained empty가 `10` steps다.
- start-to-start gap 기준 dead-air gap이 `6/7`이라 dead-air ratio가 `0.857`로 gate `0.800`을 넘는다.

아닌 것:

- collapse warning은 false다.
- repeated position/pitch pair ratio는 `0.091`로 strict 한계 안이다.
- postprocess removal ratio는 `0.273`으로 strict 한계 `0.490` 안이다.
- sample `2`, `3`도 leading duration token 때문에 `grammar_valid=false`지만, 현재 grammar gate 실패 원인은 아니다.

정리:

- 이 outlier는 "MIDI 파일 생성 실패"가 아니다.
- "전체 phrase span은 일부 확보했지만, onset 밀도와 sustained coverage가 낮아 180ms 이상 start gap 비율이 과도한 후보"다.
- 다음 개선은 모델 구조보다 candidate selection 또는 dead-air-aware sampling/retry 쪽이 작고 검증 가능하다.

## 다음 작업

권장 이슈:

- `Stage B dead-air-aware candidate selection gate`

목표:

- raw generation 후보 중 dead-air ratio가 낮은 strict-valid 후보를 우선 선택
- repeatability sweep summary에 dead-air outlier count/rate 별도 기록
- outlier가 있어도 candidate set 안에서 reviewable MIDI를 안정적으로 고르는 기준 추가
