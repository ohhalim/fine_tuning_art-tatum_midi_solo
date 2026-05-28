# Stage B Margin-Recovered Phrase/Vocabulary Qualified Peer Focused Listening Notes

## 요약

Issue #286은 Issue #284 focused context keep peer 후보를 focused listening review notes template으로 넘긴 작업이다.

이 단계는 peer 후보도 selected keep 후보와 같은 listening fill path로 비교하기 위한 pending notes boundary다.

## 변경

- peer focused listening notes harness 추가
- 기존 phrase/vocabulary focused listening notes builder를 peer package/context decision 경로에 재사용
- focused context metrics, prior decision, review risks를 notes candidate에 보존
- result docs와 current plan 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_61_topk_7_temp_082_n48_sample_25` |
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| prior decision | `keep_for_focused_listening` |
| listening decision | `pending` |
| note count | `13` |
| unique pitch count | `8` |
| range | `G4-E5` |
| phrase span | `7.000` beats |
| dead-air ratio | `0.333` |
| adjacent pitch repeats | `0` |
| max interval | `7` |
| final landing | `C5` over `Fm7`, chord tone |
| review risks | `sustained_coverage_review` |

## 해석

- peer 후보도 focused listening notes template으로 넘어갔다.
- 실제 청감 판단 필드는 모두 `pending`이다.
- selected keep 후보와 같은 review risk만 남아 있다.
- 아직 peer focused listening fill은 진행하지 않았다.

## 검증

```bash
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-peer-focused-listening-notes
```

## 산출물

- notes: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/harness_stage_b_margin_recovered_phrase_vocabulary_peer_focused_listening_notes/focused_listening_review_notes_template.json`
