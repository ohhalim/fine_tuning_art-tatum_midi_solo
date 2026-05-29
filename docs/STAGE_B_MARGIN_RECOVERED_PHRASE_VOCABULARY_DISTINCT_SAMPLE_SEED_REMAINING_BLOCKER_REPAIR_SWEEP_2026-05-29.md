# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Remaining Blocker Repair Sweep

## 요약

Issue #308은 Issue #306 repair target을 기준으로 checkpoint 기반 추가 sampling sweep을 실행한 작업이다.

이 단계는 sample seed `85` 중복 후보를 제외한 상태에서 phrase span, pitch variety, adjacent repeat target을 동시에 만족하는 새 후보가 있는지 확인하기 위한 경계다.

## 변경

- distinct sample-seed remaining blocker repair sweep harness 추가
- seed `181`, `223`, top_k `8`, temperature `0.90/0.86`, 각 48 samples 조건으로 checkpoint reuse generation 실행
- 기존 phrase/vocabulary repair summary에 Issue #306 target threshold 적용
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| report count | `2` |
| candidate count | `96` |
| target-qualified candidate count | `0` |
| selected partial candidate | `margin_recovered_phrase_vocab_seed_223_topk_8_temp_086_n48_sample_28` |
| selected source seed | `223` |
| selected sample index | `28` |
| selected sample seed | `250` |
| selected focused note count | `13` |
| selected focused unique pitch count | `9` |
| selected dead-air ratio | `0.3889` |
| selected onset coverage | `0.53125` |
| selected sustained coverage | `0.71875` |
| selected adjacent pitch repeats | `1` |
| selected focused max interval | `11` |
| qualified | `false` |
| remaining flags | `dead_air_not_repaired`, `adjacent_repetition_not_repaired` |

## 해석

- 추가 sweep은 pitch variety를 `6 -> 9`로 개선한 partial candidate를 찾았다.
- target-qualified 후보는 없다.
- dead-air `0.3889`가 target `<= 0.376`보다 높고, adjacent repeat `1`이 남아 있다.
- max interval `11`은 wide-interval guardrail 안에 있지만 이전 후보 `3`보다 악화됐다.
- 이 결과는 새 keep 후보가 아니라, 다음 sampling/constraint 조정이 필요하다는 evidence다.

## 검증

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker-repair-sweep
```

## 산출물

- seed181 report: `outputs/stage_b_generation_probe/harness_stage_b_margin_recovered_phrase_vocab_seed181_topk8_temp090_n48/report.json`
- seed223 report: `outputs/stage_b_generation_probe/harness_stage_b_margin_recovered_phrase_vocab_seed223_topk8_temp086_n48/report.json`
- repair summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_repair/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker_repair/phrase_vocabulary_repair_summary.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary distinct sample-seed dead-air adjacent-repeat targeted repair`
- 목표: pitch variety gain을 유지하면서 dead-air와 adjacent repeat를 동시에 낮추는 constraint/sampling 조정
