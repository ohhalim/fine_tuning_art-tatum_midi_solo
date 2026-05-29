# Stage B Margin-Recovered Phrase/Vocabulary Two-Candidate Keep Consolidation

## 요약

Issue #290은 selected keep 후보와 qualified peer keep 후보를 하나의 evidence boundary로 묶은 작업이다.

이 단계는 단일 후보가 아니라 두 source run에서 나온 두 후보가 모두 focused context/listening evidence 기준 `keep`임을 정리한다. 다만 qualified rate가 `2/96`이므로 broad model quality나 human audio preference로 주장하지 않는다.

## 변경

- two-candidate keep summary script 추가
- selected filled notes, peer filled notes, keep stability summary를 조인하는 harness mode 추가
- selected/peer 후보의 decision field, objective metric, remaining risk를 같은 table로 정리
- README와 handoff docs 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `96` |
| qualified candidate count | `2` |
| qualified rate | `0.020833` |
| qualified source count | `2` |
| keep candidate count | `2` |
| boundary | `two_candidate_midi_context_keep_support` |
| selected candidate | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` |
| peer candidate | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` |

| role | candidate | decision | timing | phrase | vocabulary | notes | unique | dead-air | sustained | adjacent repeat | max interval | final landing | risk |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| selected | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` | `keep` | `acceptable` | `acceptable` | `acceptable` | `13` | `8` | `0.333` | `0.594` | `0` | `7` | `C5` over `Fm7`, chord tone | `sustained_coverage_review` |
| peer | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` | `keep` | `acceptable` | `acceptable` | `acceptable` | `13` | `8` | `0.333` | `0.594` | `0` | `7` | `C5` over `Fm7`, chord tone | `sustained_coverage_review` |

## 해석

- selected 후보와 peer 후보가 모두 MIDI/context evidence fill 기준 `keep`으로 정리됐다.
- 두 후보는 seed `43`, seed `61` source run에서 각각 나온 qualified 후보라 단일 sample claim보다는 강하다.
- 하지만 전체 qualified rate가 `2/96`으로 낮아 robust repeatability나 broad trained-model quality로 볼 수 없다.
- 두 후보 모두 `not_human_audio_review` boundary를 유지하므로 human/audio preference는 아직 별도 검증 대상이다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-two-candidate-keep
```

## 산출물

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep/harness_stage_b_margin_recovered_phrase_vocabulary_two_candidate_keep/two_candidate_keep_summary.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary human listening comparison boundary`
- broad training으로 넘어가기 전에 selected/peer keep 후보를 human listening 또는 audio-rendered comparison 기준으로 별도 표시한다.
