# Stage B Margin-Recovered Phrase/Vocabulary Qualified Peer Focused Context

## 요약

Issue #284는 Issue #282에서 확인한 qualified peer 후보를 solo/context package로 격리하고 focused context decision을 실행한 작업이다.

이 단계는 current keep candidate 외에도 같은 sweep 안의 peer 후보가 focused context 기준을 통과하는지 확인하기 위한 fallback boundary다.

## 변경

- phrase/vocabulary focused package builder에 explicit `candidate_id` 선택 옵션 추가
- qualified peer candidate를 solo/context package로 격리
- 기존 focused context decision 로직으로 final landing, context guide, max active, repeated cell 재검증
- result docs와 current plan 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` |
| copied MIDI files | `2` |
| focused context decision | `keep_for_focused_listening` |
| decision flags | `{}` |
| note count | `13` |
| unique pitch count | `8` |
| range | `G4-E5` |
| phrase span | `7.000` beats |
| max active notes | `1` |
| dead-air ratio | `0.333` |
| onset coverage | `0.500` |
| sustained coverage | `0.594` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| duplicated 3-note chunks | `0` |
| final landing | `C5` over `Fm7`, chord tone |
| context tracks | chord guide, bass root guide, solo |

## 해석

- qualified peer도 focused context blocker 없이 통과했다.
- selected keep candidate와 peer 후보의 focused context metrics가 동일 수준이다.
- 아직 peer 후보의 focused listening notes/fill은 진행하지 않았다.
- 이 결과는 fallback review evidence이며 broad model quality나 human/audio proof는 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_focused_package
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-peer-focused-context
```

## 산출물

- package: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_package/focused_review_package.json`
- decision: `outputs/stage_b_margin_recovered_focused_context_decision/harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_context_decision/focused_context_decision.json`
