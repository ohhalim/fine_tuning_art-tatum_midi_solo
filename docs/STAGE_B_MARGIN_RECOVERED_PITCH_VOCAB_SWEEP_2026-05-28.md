# Stage B Margin-Recovered Pitch Vocabulary Sweep

작성일: 2026-05-28

## 목적

Issue #254는 dead-air를 `0.444 -> 0.294`로 줄였지만 focused unique pitch가 `4 -> 5`에 머물렀다.

이 작업은 기존 checkpoint를 그대로 두고 generation seed / top-k 후보를 넓혀 focused unique pitch `>= 6`과 dead-air `<= 0.40`을 동시에 만족하는 후보가 있는지 확인한다.

## 입력 조건

| 항목 | 값 |
|---|---|
| checkpoint | `outputs/stage_b_generation_probe/issue_238_stage_b_candidate_count_margin_recovery_seed31_files6/checkpoints` |
| seed sweep | `17`, `31` |
| top_k | `5` |
| temperature | `0.9` |
| samples per seed | `24` |
| postprocess overlap | enabled |
| focused simultaneous limit | `1` |

## 구현

- `scripts/summarize_stage_b_margin_recovered_pitch_vocab_sweep.py`
  - 여러 generation report를 하나의 sweep으로 합산
  - focused solo-line 기준 note count, unique pitch count, dead-air, max active, repeated cell 계산
  - hard gate 통과 후보를 우선 선택
  - Issue #254 후보 대비 dead-air / unique pitch delta 기록
- `stage-b-margin-recovered-pitch-vocab-sweep` harness
  - seed `17`, seed `31` 각각 top_k5 24-sample decode
  - 총 `48`개 후보 sweep summary 생성
  - expected selected source / sample 검증

## Gate

| 기준 | 값 |
|---|---:|
| focused unique pitch count | `>= 6` |
| dead-air ratio | `<= 0.40` |
| focused note count | `>= 12` |
| focused max active notes | `1` |
| duplicated 3-note pitch-class chunks | `0` |

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_pitch_vocab_sweep
bash scripts/agent_harness.sh stage-b-margin-recovered-pitch-vocab-sweep
```

## 결과

| 항목 | 값 |
|---|---|
| report count | `2` |
| candidate count | `48` |
| qualified candidate count | `1` |
| selected candidate | `margin_recovered_pitch_vocab_seed_17_topk_5_temp_09_n24_sample_4` |
| source run | `harness_stage_b_margin_recovered_pitch_vocab_seed17_topk5_temp090_n24` |
| selected sample | `4` |
| selected sample seed | `20` |
| qualified | true |
| remaining flags | `[]` |

Selected candidate metrics:

| 항목 | 값 |
|---|---:|
| focused notes | `13` |
| focused unique pitches | `6` |
| dead-air ratio | `0.400` |
| onset coverage | `0.500` |
| sustained coverage | `0.625` |
| focused max active notes | `1` |
| adjacent pitch repeats | `3` |
| duplicated 3-note pitch-class chunks | `0` |
| chord-tone ratio | `0.812` |

Issue #254 후보 대비:

| 항목 | Issue #254 sample 8 | Issue #256 selected |
|---|---:|---:|
| focused unique pitches | `5` | `6` |
| dead-air ratio | `0.294` | `0.400` |
| focused notes | `13` | `13` |
| duplicated 3-note chunks | `0` | `0` |
| adjacent pitch repeats | `1` | `3` |

## 해석

- focused unique pitch gate는 통과했다.
- dead-air는 absolute gate `<= 0.40`에는 들어왔지만, Issue #254 후보보다 `+0.106` 나빠졌다.
- adjacent pitch repeats도 `1 -> 3`으로 늘었다.
- 따라서 이 결과는 pitch vocabulary gate를 통과한 후보이며, dead-air/adjacent-repeat tradeoff가 남은 상태다.
- broad model quality, human preference, Brad style adaptation 성공으로 표현하지 않는다.

## 다음 작업

선택 후보를 focused context package로 격리해 chord context, register, repeated cell, phrase continuation을 다시 판단한다.

다음 gate:

- focused context decision이 `keep_for_focused_listening` 또는 명확한 `needs_followup`으로 분리
- dead-air `0.400`이 context에서 체감 가능한 blocker인지 기록
- adjacent pitch repeats `3`이 motif인지 반복 문제인지 기록
