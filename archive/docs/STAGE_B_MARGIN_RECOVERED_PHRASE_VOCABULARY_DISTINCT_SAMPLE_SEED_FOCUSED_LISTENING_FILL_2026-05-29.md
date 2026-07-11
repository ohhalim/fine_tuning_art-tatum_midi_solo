# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Focused Listening Fill

## 요약

Issue #304는 Issue #302 focused listening notes를 MIDI/context evidence 기준으로 채운 작업이다.

이 단계는 pending 상태였던 timing, chord fit, phrase continuation, landing, jazz vocabulary, decision field를 metric/context evidence로 채우고, distinct sample-seed 후보가 keep으로 올라갈 수 있는지 확인하기 위한 경계다.

## 변경

- distinct sample-seed focused listening fill harness 추가
- 기존 phrase/vocabulary focused listening fill script 재사용
- filled decision과 remaining blocker 문서화
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` |
| reviewed count | `1` |
| pending count | `0` |
| timing | `acceptable` |
| chord fit | `acceptable` |
| phrase continuation | `weak` |
| landing | `acceptable` |
| jazz vocabulary | `thin` |
| final decision | `needs_followup` |
| unique pitch count | `6` |
| phrase span | `6.750` beats |
| dead-air ratio | `0.375` |
| sustained coverage | `0.78125` |
| adjacent pitch repeats | `1` |
| max interval | `3` |
| final landing | `D5` over `Fm7`, tension |
| review risks | `dead_air_ratio_remaining`, `adjacent_pitch_repeats` |

## 해석

- timing, chord fit, landing은 blocking 수준이 아니다.
- wide interval blocker는 max interval `3`으로 해소 상태다.
- phrase continuation은 phrase span `6.750` beats로 `weak` 판정이다.
- jazz vocabulary는 unique pitch `6`과 adjacent repeat `1` 때문에 `thin` 판정이다.
- 이 후보는 distinct sample-seed evidence지만 focused listening fill 기준 keep으로 승격하지 않는다.
- 다음 repair target은 distinct sample-seed 유지 상태에서 phrase span, pitch variety, adjacent repeat를 함께 개선하는 것이다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-listening-fill
```

## 산출물

- filled notes: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_listening_fill/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_listening_fill/focused_listening_review_notes_filled.json`
