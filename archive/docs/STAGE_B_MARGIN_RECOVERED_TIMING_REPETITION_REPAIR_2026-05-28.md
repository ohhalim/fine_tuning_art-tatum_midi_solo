# Stage B Margin-Recovered Timing/Repetition Repair

## 요약

Issue #264는 Issue #262 focused listening fill에서 남은 timing/repetition 문제를 좁게 repair한 작업이다.

이전 pitch-vocabulary 후보는 focused unique pitch `6`, max active `1`, duplicated 3-note chunk `0` 조건은 만족했지만, dead-air가 gate 상한 `0.400`에 걸렸고 adjacent pitch repeats가 `3`으로 남아 focused listening fill에서 `needs_followup`으로 내려갔다.

이번 작업은 같은 checkpoint에서 top-k/temperature 조건을 조정해 pitch vocabulary gate를 유지하면서 dead-air와 adjacent repeat를 동시에 줄이는 후보를 다시 선택했다.

## 변경

- timing/repetition repair sweep summary script 추가
- seed `37/41`, top_k `7`, temperature `0.86`, n `48` generation harness 추가
- qualified gate를 `dead_air < 0.400`, `adjacent repeats < 3`, `focused unique pitch >= 6`, `focused notes >= 12`, `max active = 1`, `dup3 = 0`로 고정
- 이전 Issue #262 후보 대비 delta를 summary JSON/Markdown에 기록

## 결과

| 항목 | 값 |
|---|---|
| selected candidate | `margin_recovered_timing_repetition_seed_37_topk_7_temp_086_n48_sample_39` |
| source run | `harness_stage_b_margin_recovered_timing_repetition_seed37_topk7_temp086_n48` |
| selected sample | `39` |
| selected sample seed | `75` |
| report count | `2` |
| candidate count | `96` |
| qualified candidate count | `2` |
| qualified | `true` |
| remaining flags | `[]` |
| focused note count | `14` |
| focused unique pitch count | `7` |
| focused max active notes | `1` |
| duplicated 3-note pitch-class chunks | `0` |
| dead-air ratio | `0.353` |
| adjacent pitch repeats | `2` |
| onset coverage | `0.500` |
| sustained coverage | `0.688` |

Issue #262 후보 대비:

| 항목 | 이전 | 이번 | 변화 |
|---|---:|---:|---:|
| dead-air ratio | `0.400` | `0.353` | `+0.047` 개선 |
| adjacent pitch repeats | `3` | `2` | `+1` 개선 |
| focused unique pitch count | `6` | `7` | `+1` |
| focused note count | `13` | `14` | `+1` |

## 해석

- pitch vocabulary gate를 유지하면서 timing/repetition objective는 개선됐다.
- selected candidate는 objective gate 기준 qualified candidate다.
- 아직 focused context package와 focused listening fill을 다시 통과한 것은 아니다.
- human listening preference, broad model quality, Brad style adaptation은 여전히 미검증이다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_timing_repetition_repair
bash scripts/agent_harness.sh stage-b-margin-recovered-timing-repetition-repair
```

## 산출물

- script: `scripts/summarize_stage_b_margin_recovered_timing_repetition_repair.py`
- test: `tests/test_stage_b_margin_recovered_timing_repetition_repair.py`
- harness: `stage-b-margin-recovered-timing-repetition-repair`
- summary: `outputs/stage_b_margin_recovered_timing_repetition_repair/harness_stage_b_margin_recovered_timing_repetition_repair/timing_repetition_repair_summary.json`
