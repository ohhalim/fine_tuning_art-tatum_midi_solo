# Stage B Margin-Recovered Phrase/Vocabulary Duplicate Source Divergence Audit

## 요약

Issue #294는 Issue #292에서 확인한 selected/peer 동일 note sequence의 원인을 source divergence 관점에서 분리한 작업이다.

결론은 두 후보가 서로 다른 source run과 sample index에서 나왔지만, 같은 `sample_seed` `85`를 공유해 동일 MIDI content로 수렴했다는 것이다. 따라서 two-source qualified support는 유지되지만, 두 개의 독립적인 음악 출력이라고 주장하지 않는다.

## 변경

- duplicate source divergence audit script 추가
- repair summary와 human listening comparison boundary를 조인하는 harness mode 추가
- source seed, sample index, sample seed, note sequence, metric fingerprint 비교
- README와 handoff docs 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `96` |
| qualified candidate count | `2` |
| source seed diff | `true` |
| sample index diff | `true` |
| shared sample seed | `true` |
| note sequence match | `true` |
| metric fingerprint match | `true` |
| boundary | `shared_sample_seed_duplicate_output` |
| claim boundary | `two_source_qualified_but_not_two_distinct_outputs` |
| source diversity | `present` |
| output diversity | `absent` |

| role | candidate | source seed | sample index | sample seed | rank | qualified | notes | unique | dead-air | focused max interval |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| selected | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` | `43` | `43` | `85` | `1` | `true` | `16` | `8` | `0.333` | `7` |
| peer | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` | `61` | `25` | `85` | `2` | `true` | `16` | `8` | `0.333` | `7` |

## 해석

- seed `43` run과 seed `61` run에서 각각 qualified 후보가 나온 것은 맞다.
- 그러나 두 후보 모두 `sample_seed` `85`를 공유하고 note sequence와 metric fingerprint가 동일하다.
- 따라서 현재 결과는 two-source reproducible qualified output evidence이지, two-distinct-output diversity evidence가 아니다.
- next boundary는 sample seed가 겹치지 않도록 selection/diversity gate를 추가하는 것이다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duplicate-source-divergence
```

## 산출물

- script: `scripts/audit_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence/harness_stage_b_margin_recovered_phrase_vocabulary_duplicate_source_divergence/duplicate_source_divergence_audit.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary sample-seed diversity repair`
- candidate selection에서 sample seed duplicate를 별도 diversity warning 또는 exclusion 조건으로 분리한다.
