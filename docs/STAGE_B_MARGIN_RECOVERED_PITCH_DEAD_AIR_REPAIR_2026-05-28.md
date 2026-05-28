# Stage B Margin-Recovered Pitch/Dead-Air Repair

작성일: 2026-05-28

## 목적

Issue #252에서 margin-recovered 후보 3개 모두 focused context 기준 `needs_followup`으로 남았다.

공통 blocker:

- low pitch variety: `3/3`
- dead-air issue: rank `2`, rank `3`
- focused keep fallback: `0/3`

이 작업은 broad training으로 바로 확장하지 않고, 기존 seed `31` 6-file checkpoint에서 sampling 후보 수를 늘려 pitch vocabulary와 dead-air가 같이 개선되는 후보가 있는지 확인한다.

## 입력 조건

| 항목 | 값 |
|---|---|
| checkpoint | `outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints` |
| seed | `31` |
| max files | `6` |
| max sequence | `96` |
| samples | `12` |
| temperature | `0.9` |
| top_k | `4` |
| postprocess overlap | enabled |
| baseline | `margin_recovered_rank_2_seed_31_sample_5` |

## 구현

- `scripts/select_stage_b_margin_recovered_repair_candidate.py`
  - generation report의 sample MIDI를 다시 읽어 focused solo-line metrics 계산
  - simultaneous limit `1` 기준 focused note count, unique pitch count, repeated cell, adjacent repeat 계산
  - dead-air, focused unique pitch, note count, onset/sustained coverage, repeated cell penalty를 합산해 repair candidate ranking 생성
  - baseline 대비 dead-air / unique pitch delta와 remaining flags 기록
- `stage-b-margin-recovered-pitch-dead-air-repair` harness
  - 기존 checkpoint로 12-sample top_k4 decode 실행
  - repair selector 실행
  - expected selected sample `8` 검증

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_repair_candidate_selection
bash scripts/agent_harness.sh stage-b-margin-recovered-pitch-dead-air-repair
```

## 결과

| 항목 | baseline rank 2 sample 5 | repair sample 8 |
|---|---:|---:|
| focused notes | `14` | `13` |
| focused unique pitches | `4` | `5` |
| dead-air ratio | `0.444` | `0.294` |
| onset coverage | `0.500` | `0.594` |
| sustained coverage | `0.719` | `0.781` |
| focused max active notes | `1` | `1` |
| duplicated 3-note pitch-class chunks | `0` | `0` |
| adjacent pitch repeats | `3` | `1` |
| focused keep ready | false | false |

선택 결과:

| 항목 | 값 |
|---|---|
| selected candidate | `margin_recovered_repair_seed_31_sample_8` |
| selected sample | `8` |
| candidate count | `12` |
| strict valid samples | `12/12` |
| grammar gate samples | `12/12` |
| dead-air delta | `0.150` |
| focused unique pitch delta | `+1` |
| remaining flags | `low_pitch_variety` |

## 해석

- dead-air는 baseline `0.444`에서 `0.294`로 줄었다.
- focused unique pitch는 `4`에서 `5`로 늘었다.
- focused max active notes는 `1`로 유지되어 solo-line postprocess 경계는 유지된다.
- duplicated 3-note pitch-class chunk는 `0`으로 유지된다.
- 하지만 focused unique pitch가 gate 기준 `6`에 못 미쳐 focused keep으로 승격하지 않는다.

따라서 이 결과는 partial repair다.

## 다음 작업

다음 issue는 dead-air를 유지하면서 focused unique pitch를 최소 `6` 이상으로 올리는 pitch vocabulary expansion sweep이어야 한다.

성공 기준:

- focused unique pitch count `>= 6`
- dead-air ratio `<= 0.40`
- focused note count `>= 12`
- focused max active notes `1`
- duplicated 3-note pitch-class chunks `0`
- broad model quality나 style adaptation 성공으로 표현하지 않음
