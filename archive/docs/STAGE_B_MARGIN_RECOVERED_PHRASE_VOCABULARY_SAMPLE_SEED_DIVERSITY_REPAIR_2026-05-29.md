# Stage B Margin-Recovered Phrase/Vocabulary Sample-Seed Diversity Repair

## 요약

Issue #296은 duplicate sample-seed 후보를 distinct output support로 세지 않도록 claim boundary를 고친 작업이다.

Issue #294에서 selected/peer 후보가 모두 `sample_seed` `85`를 공유하고 동일 MIDI content로 수렴했으므로, peer 후보를 two-distinct-output evidence에서 제외한다.

## 변경

- sample-seed diversity repair script 추가
- repair summary와 duplicate source divergence audit를 조인하는 harness mode 추가
- qualified source seed count와 qualified sample seed count를 분리
- duplicate sample-seed peer를 distinct-output support에서 demote
- README와 handoff docs 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `96` |
| qualified candidate count | `2` |
| qualified source seed count | `2` |
| qualified sample seed count | `1` |
| duplicate sample seed counts | `85: 2` |
| distinct peer candidate count | `0` |
| boundary | `single_distinct_sample_seed_keep_support` |
| action | `demote_duplicate_peer_from_distinct_output_support` |
| claim before | `two_source_qualified_but_not_two_distinct_outputs` |
| claim after | `single_distinct_sample_seed_keep_support_until_new_sampling` |

| candidate | source seed | sample index | sample seed | rank | notes | unique | dead-air | focused unique | focused max interval |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` | `43` | `43` | `85` | `1` | `16` | `8` | `0.333` | `8` | `7` |

## 해석

- qualified source seed count는 `2`지만 qualified sample seed count는 `1`이다.
- peer 후보는 같은 sample seed와 같은 MIDI content이므로 output diversity evidence에서 제외한다.
- 현재 claim은 two-source support가 아니라 single distinct sample-seed keep support로 낮춘다.
- 다음은 sample seed가 겹치지 않는 후보를 찾는 repair sweep이다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-sample-seed-diversity
```

## 산출물

- script: `scripts/repair_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity_repair/harness_stage_b_margin_recovered_phrase_vocabulary_sample_seed_diversity/sample_seed_diversity_repair.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary distinct sample-seed repair sweep`
- sample seed가 다른 qualified 후보를 찾거나, 없으면 현재 generation/sampling 설정의 output diversity 한계를 기록한다.
