# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Focused Listening Notes

## 요약

Issue #302는 Issue #300 focused context keep 후보를 focused listening review notes template으로 넘긴 작업이다.

이 단계는 context decision을 최종 청감 판단으로 과장하지 않고, timing, chord fit, phrase continuation, landing, jazz vocabulary, decision field를 pending 상태로 분리해 기록하기 위한 경계다.

## 변경

- distinct sample-seed focused package와 focused context decision 경로용 notes harness 추가
- focused context metrics, prior decision, review risks를 notes candidate에 보존
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` |
| candidate count | `1` |
| reviewed count | `0` |
| pending count | `1` |
| prior decision | `keep_for_focused_listening` |
| listening decision | `pending` |
| note count | `13` |
| unique pitch count | `6` |
| range | `A#4-D#5` |
| phrase span | `6.750` beats |
| dead-air ratio | `0.375` |
| onset coverage | `0.5625` |
| sustained coverage | `0.78125` |
| adjacent pitch repeats | `1` |
| max interval | `3` |
| final landing | `D5` over `Fm7`, tension |
| review risks | `dead_air_ratio_remaining`, `adjacent_pitch_repeats` |

## 해석

- distinct sample-seed focused context keep 후보를 listening notes template으로 넘겼다.
- 실제 청감 판단 필드는 모두 `pending`이다.
- wide interval risk는 해소 상태로 유지됐지만 dead-air와 adjacent repeat risk가 남아 있다.
- 아직 focused listening fill, human/audio preference, broad model quality proof는 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-notes
```

## 산출물

- notes: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_notes/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_notes/focused_listening_review_notes_template.json`
