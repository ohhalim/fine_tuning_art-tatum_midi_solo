# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Dead-Air Adjacent Repair

## 요약

Issue #310은 Issue #308 partial candidate에서 남은 dead-air와 adjacent repeat blocker를 낮추기 위한 추가 targeted sampling sweep이다.

이 단계는 pitch variety gain을 유지하면서 dead-air와 adjacent repeat를 동시에 낮출 수 있는지 확인하기 위한 경계다.

## 변경

- distinct sample-seed dead-air/adjacent repair harness 추가
- seed `269`, `311`, top_k `7`, temperature `0.80/0.78`, 각 48 samples 조건으로 checkpoint reuse generation 실행
- Issue #306 target threshold로 repair summary 재평가
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| report count | `2` |
| candidate count | `96` |
| target-qualified candidate count | `0` |
| selected partial candidate | `margin_recovered_phrase_vocab_seed_311_topk_7_temp_078_n48_sample_31` |
| selected source seed | `311` |
| selected sample index | `31` |
| selected sample seed | `341` |
| selected focused note count | `15` |
| selected focused unique pitch count | `7` |
| selected dead-air ratio | `0.3889` |
| selected onset coverage | `0.59375` |
| selected sustained coverage | `0.71875` |
| selected adjacent pitch repeats | `1` |
| selected focused max interval | `7` |
| qualified | `false` |
| remaining flags | `dead_air_not_repaired`, `adjacent_repetition_not_repaired` |

## 해석

- lower temperature/top_k 조합에서도 target-qualified 후보는 없다.
- best partial candidate는 note count `15`, unique pitch `7`, max interval `7`로 일부 guardrail은 회복했다.
- dead-air `0.3889`가 target `<= 0.376`보다 높고, adjacent repeat `1`이 남아 있다.
- 같은 checkpoint sampling 조정만으로는 dead-air와 adjacent repeat를 동시에 제거하기 어렵다.
- 다음 경계는 sampling 반복이 아니라 decoding/postprocess 또는 grammar constraint에서 adjacent repeat와 coverage를 직접 제어하는 방향이다.

## 검증

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-dead-air-adjacent-repair
```

## 산출물

- seed269 report: `outputs/stage_b_generation_probe/harness_stage_b_margin_recovered_phrase_vocab_seed269_topk7_temp080_n48/report.json`
- seed311 report: `outputs/stage_b_generation_probe/harness_stage_b_margin_recovered_phrase_vocab_seed311_topk7_temp078_n48/report.json`
- repair summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_repair/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_dead_air_adjacent_repair/phrase_vocabulary_repair_summary.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary coverage-aware adjacent-repeat constrained repair`
- 목표: post-sampling 후보 선택이 아니라 generation/decoding 단계에서 coverage와 adjacent repeat를 직접 제어
