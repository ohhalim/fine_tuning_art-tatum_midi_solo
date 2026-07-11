# Stage B Margin-Recovered Phrase/Vocabulary Human Listening Comparison Boundary

## 요약

Issue #292는 selected keep 후보와 peer keep 후보를 human/audio review 대상으로 넘기기 전에, 비교 가능한 후보인지 확인하고 사람 평가 필드를 `pending`으로 분리한 작업이다.

이 단계는 사람이 들었다는 결론을 내리지 않는다. 두 후보가 실제 MIDI content 기준으로 다른지 먼저 확인하고, 동일 content라면 A/B 선호 비교를 과장하지 않는다.

## 변경

- human listening comparison boundary script 추가
- selected/peer filled notes와 two-candidate keep summary를 조인하는 harness mode 추가
- selected/peer 후보의 MIDI 경로, context MIDI 경로, pending human listening fields 기록
- note signature와 metric fingerprint 동일 여부를 objective comparison으로 기록
- README와 handoff docs 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate count | `2` |
| human listening status | `pending` |
| preference claimed | `false` |
| note sequence match | `true` |
| metric fingerprint match | `true` |
| complete note signature | `true` |
| selected note count | `13` |
| peer note count | `13` |
| boundary | `pending_human_review_same_midi_content` |
| listenability | `not_meaningful_as_ab_if_same_render` |

| role | candidate | prior decision | human status | notes | unique | dead-air | sustained | final landing | risk |
|---|---|---|---|---:|---:|---:|---:|---|---|
| selected | `margin_recovered_phrase_vocab_seed_43_topk_7_temp_082_n48_sample_43` | `keep` | `pending` | `13` | `8` | `0.333` | `0.594` | `C5` over `Fm7`, chord tone | `sustained_coverage_review` |
| peer | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` | `keep` | `pending` | `13` | `8` | `0.333` | `0.594` | `C5` over `Fm7`, chord tone | `sustained_coverage_review` |

## 해석

- selected 후보와 peer 후보는 source run과 sample index는 다르지만 note signature와 metric fingerprint가 동일하다.
- human listening field는 모두 `pending`이며 선호 판단은 기록하지 않았다.
- 동일 MIDI content를 같은 악기/템포/context로 렌더한다면 A/B 청감 비교는 의미가 약하다.
- 이 결과는 two-source support를 부정하지는 않지만, source diversity가 실제 output diversity로 이어졌는지는 별도 audit이 필요하다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-human-listening-comparison
```

## 산출물

- script: `scripts/build_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison/harness_stage_b_margin_recovered_phrase_vocabulary_human_listening_comparison/human_listening_comparison_boundary.json`

## 다음 경계

- `Stage B margin-recovered phrase/vocabulary duplicate-candidate source divergence audit`
- seed/source가 달라도 같은 MIDI content로 수렴한 원인을 분리한다.
