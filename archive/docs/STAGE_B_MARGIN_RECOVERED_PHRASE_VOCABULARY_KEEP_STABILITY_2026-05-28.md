# Stage B Margin-Recovered Phrase/Vocabulary Keep Stability

## 요약

Issue #282는 Issue #280 current margin-recovered evidence keep candidate가 단일 후보인지, 같은 phrase/vocabulary sweep 안에 qualified peer가 있는지 비교한 작업이다.

새 MIDI를 생성하지 않고, Issue #272 repair summary와 Issue #278 filled keep notes를 조인해 stability boundary를 정리했다.

## 변경

- phrase/vocabulary keep stability summary script 추가
- repair summary의 qualified candidate count, source 분포, peer candidate를 집계
- filled keep candidate와 qualified peer를 같은 metric table로 비교
- result docs와 current plan 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `96` |
| qualified candidate count | `2` |
| qualified rate | `0.020833` |
| qualified source count | `2` |
| selected candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| qualified peer | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` |
| selected source | `harness_stage_b_margin_recovered_phrase_vocab_seed43_topk7_temp082_n48` |
| peer source | `harness_stage_b_margin_recovered_phrase_vocab_seed61_topk7_temp082_n48` |
| selected metrics | notes `13`, unique `8`, dead-air `0.333`, adjacent repeat `0`, max interval `7` |
| peer metrics | notes `13`, unique `8`, dead-air `0.333`, adjacent repeat `0`, max interval `7` |
| stability boundary | `narrow_two_source_candidate_support` |
| next issue | `Stage B margin-recovered phrase/vocabulary qualified peer focused context review` |

## 해석

- current keep candidate가 완전히 단일 sample만은 아니다.
- seed `43` source와 seed `61` source에서 각각 qualified 후보가 1개씩 나왔다.
- 다만 qualified rate는 `2/96`으로 낮으므로 broad model quality나 robust repeatability로 주장하지 않는다.
- 다음 단계는 qualified peer를 focused context/listening path로 넘겨 실제 fallback 후보인지 확인하는 것이다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_keep_stability
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-keep-stability
```

## 산출물

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_keep_stability.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_keep_stability.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_keep_stability/harness_stage_b_margin_recovered_phrase_vocabulary_keep_stability/keep_stability_summary.json`
