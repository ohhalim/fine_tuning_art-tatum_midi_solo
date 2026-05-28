# Stage B Margin-Recovered Phrase/Vocabulary Repair

## 요약

Issue #272는 Issue #270 focused listening fill에서 남은 phrase continuation / jazz vocabulary blocker를 좁게 repair한 작업이다.

Issue #270 후보는 timing은 `acceptable`로 개선됐지만, adjacent repeats `2`와 max interval `16` 때문에 phrase continuation `weak`, jazz vocabulary `thin`, decision `needs_followup`으로 남았다.

이번 작업은 seed/top-k/temperature sweep에서 adjacent repeat와 wide interval을 동시에 낮추는 후보를 선택했다.

## 변경

- phrase/vocabulary repair summary script 추가
- seed `43/61`, top_k `7`, temperature `0.82`, n `48` generation harness 추가
- qualified gate를 `dead_air < 0.400`, `adjacent repeats < 2`, `max interval < 12`, `focused unique pitch >= 6`, `focused notes >= 12`, `max active = 1`, `dup3 = 0`로 고정
- Issue #270 후보 대비 dead-air, adjacent repeat, max interval, unique pitch, note count delta 기록

## 결과

| 항목 | 값 |
|---|---|
| selected candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| source run | `harness_stage_b_margin_recovered_phrase_vocab_seed43_topk7_temp082_n48` |
| selected sample | `43` |
| selected sample seed | `85` |
| report count | `2` |
| candidate count | `96` |
| qualified candidate count | `2` |
| qualified | `true` |
| remaining flags | `[]` |
| focused note count | `13` |
| focused unique pitch count | `8` |
| focused max active notes | `1` |
| duplicated 3-note pitch-class chunks | `0` |
| dead-air ratio | `0.333` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| onset coverage | `0.500` |
| sustained coverage | `0.594` |

Issue #270 후보 대비:

| 항목 | 이전 | 이번 | 변화 |
|---|---:|---:|---:|
| dead-air ratio | `0.353` | `0.333` | `+0.020` 개선 |
| adjacent pitch repeats | `2` | `0` | `+2` 개선 |
| max interval | `16` | `7` | `+9` 개선 |
| focused unique pitch count | `7` | `8` | `+1` |
| focused note count | `14` | `13` | `-1` |

## 해석

- Issue #270의 phrase/vocabulary blocker였던 adjacent repeats와 wide interval은 objective gate 기준으로 개선됐다.
- dead-air와 pitch vocabulary gate도 유지됐다.
- note count는 `1`개 줄었지만 focused review gate의 최소 조건은 유지한다.
- 아직 focused context package와 focused listening fill을 다시 통과한 것은 아니다.
- broad model quality나 human/audio preference proof는 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_repair
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-repair
```

## 산출물

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_repair.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_repair.py`
- harness: `stage-b-margin-recovered-phrase-vocabulary-repair`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_repair/harness_stage_b_margin_recovered_phrase_vocabulary_repair/phrase_vocabulary_repair_summary.json`
