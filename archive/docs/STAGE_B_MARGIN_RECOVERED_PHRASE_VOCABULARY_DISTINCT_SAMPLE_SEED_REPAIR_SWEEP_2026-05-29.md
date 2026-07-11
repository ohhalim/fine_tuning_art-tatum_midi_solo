# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Repair Sweep

## 요약

Issue #298은 duplicate sample seed `85`와 겹치지 않는 seed range에서 phrase/vocabulary repair sweep을 다시 실행한 작업이다.

기존 checkpoint를 재사용하고 broad training은 하지 않았다. seed `109`, `157`에서 각각 48개 후보를 생성해 기존 blocker를 통과하는 distinct sample-seed 후보가 있는지 확인했다.

## 변경

- distinct sample-seed sweep summary script 추가
- 기존 checkpoint 기반 generation harness 추가
- seed `109`, `157`, top_k `7`, temperature `0.82`, 각 48 samples 조건으로 sweep 실행
- blocked sample seed `85`를 제외한 qualified 후보 집계
- README와 handoff docs 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `96` |
| qualified candidate count | `2` |
| blocked sample seeds | `85` |
| distinct sample-seed qualified count | `2` |
| qualified sample seed counts | `131: 1`, `155: 1` |
| boundary | `distinct_sample_seed_qualified_candidate_found` |
| next issue | `Stage B margin-recovered phrase/vocabulary distinct sample-seed focused context review` |

| rank | candidate | source seed | sample index | sample seed | notes | unique | dead-air | focused notes | focused unique | adjacent repeats | focused max interval |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` | `109` | `47` | `155` | `17` | `6` | `0.375` | `13` | `6` | `1` | `3` |
| 2 | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23` | `109` | `23` | `131` | `17` | `7` | `0.375` | `12` | `7` | `1` | `11` |

## 해석

- duplicate sample seed `85` 없이도 qualified 후보 `2`개가 나왔다.
- selected distinct candidate는 sample seed `155`이며, focused max interval `3`으로 기존 wide interval risk는 낮다.
- 다만 focused unique pitch는 `6`으로 gate 하한이고, dead-air ratio는 `0.375`로 gate 근처다.
- 아직 focused context decision, focused listening fill, human/audio proof는 진행하지 않았다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-sweep
```

## 산출물

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_sweep/distinct_sample_seed_sweep_summary.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary distinct sample-seed focused context review`
- distinct sample-seed 후보를 solo/context package로 격리하고 context decision을 다시 확인한다.
