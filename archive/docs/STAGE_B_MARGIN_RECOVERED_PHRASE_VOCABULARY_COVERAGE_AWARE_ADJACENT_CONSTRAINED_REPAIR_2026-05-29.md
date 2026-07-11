# Stage B Margin-Recovered Phrase/Vocabulary Coverage-Aware Adjacent Constrained Repair

## 요약

Issue #312는 sampling 반복 대신 constrained decoding에서 coverage와 adjacent pitch repeat를 직접 제어한 작업이다.

이 단계는 coverage-aware positions, chord-aware pitches, repeat window를 적용해 adjacent repeat blocker를 줄이면서 target-qualified 후보가 나오는지 확인하기 위한 경계다.

## 변경

- coverage-aware adjacent constrained repair harness 추가
- seed `353`, `397`, constrained decoding 조건으로 checkpoint reuse generation 실행
- seed `353`: coverage-aware positions, chord-aware pitches, repeat window `4`, groups per bar `8`
- seed `397`: 위 조건에 jazz duration tokens와 groups per bar `10` 추가
- Issue #306 target threshold로 repair summary 재평가
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| report count | `2` |
| candidate count | `48` |
| target-qualified candidate count | `0` |
| selected partial candidate | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3` |
| selected source seed | `353` |
| selected sample index | `3` |
| selected sample seed | `355` |
| selected focused note count | `12` |
| selected focused unique pitch count | `9` |
| selected dead-air ratio | `0.5714` |
| selected onset coverage | `0.4375` |
| selected sustained coverage | `0.59375` |
| selected adjacent pitch repeats | `0` |
| selected focused max interval | `7` |
| qualified | `false` |
| remaining flags | `dead_air_not_repaired` |

## 해석

- constrained decoding은 adjacent repeat를 `1 -> 0`으로 낮췄다.
- pitch variety도 `6 -> 9`로 개선됐다.
- 그러나 dead-air가 `0.5714`로 악화되어 target-qualified 후보는 없다.
- dead-air blocker는 단순 sampling/top_k 조정이나 pitch repeat window만으로 해결되지 않는다.
- 다음 경계는 duration/coverage postprocess 또는 constrained duration/onset fill을 별도 repair로 검토하는 것이다.

## 검증

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-coverage-aware-adjacent-constrained-repair
```

## 산출물

- seed353 report: `outputs/stage_b_generation_probe/harness_stage_b_margin_recovered_phrase_vocab_coverage_adjacent_seed353_groups8/report.json`
- seed397 report: `outputs/stage_b_generation_probe/harness_stage_b_margin_recovered_phrase_vocab_coverage_adjacent_seed397_groups10_duration/report.json`
- repair summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_repair/harness_stage_b_margin_recovered_phrase_vocabulary_coverage_aware_adjacent_constrained_repair/phrase_vocabulary_repair_summary.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary duration coverage fill repair`
- 목표: adjacent repeat와 pitch variety guardrail을 유지하면서 dead-air를 낮추는 duration/coverage repair
