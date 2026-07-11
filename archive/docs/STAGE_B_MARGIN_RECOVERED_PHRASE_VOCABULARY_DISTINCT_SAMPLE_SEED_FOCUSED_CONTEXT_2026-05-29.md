# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Focused Context

## 요약

Issue #300은 Issue #298에서 선택한 distinct sample-seed 후보를 solo/context package로 격리하고 focused context decision을 실행한 작업이다.

이 단계는 duplicate sample seed `85`를 벗어난 후보를 바로 keep 후보로 올리지 않고, context MIDI 안에서 solo-line 조건, context guide 존재, final landing을 다시 확인하기 위한 경계다.

## 변경

- distinct sample-seed repair summary 기반 focused package harness 추가
- selected distinct sample-seed candidate의 solo MIDI와 context MIDI 복사/생성
- 기존 focused context decision 로직으로 final landing, context guide, max active, repeated cell 재검증
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` |
| source seed | `109` |
| sample index | `47` |
| sample seed | `155` |
| copied MIDI files | `2` |
| focused context decision | `keep_for_focused_listening` |
| decision flags | `{}` |
| note count | `13` |
| unique pitch count | `6` |
| range | `A#4-D#5` |
| phrase span | `6.750` beats |
| max active notes | `1` |
| dead-air ratio | `0.375` |
| onset coverage | `0.5625` |
| sustained coverage | `0.78125` |
| adjacent pitch repeats | `1` |
| max interval | `3` |
| duplicated 3-note pitch-class chunks | `0` |
| final landing | `D5` over `Fm7`, tension |
| context tracks | chord guide, bass root guide, solo |

## 해석

- focused context blocker는 발견되지 않았다.
- solo-line max active `1`, phrase span `6.75` beats, context guide 존재 조건을 만족했다.
- final note는 `Fm7` 위 tension으로 처리되어 outside landing blocker가 아니다.
- focused unique pitch count는 `6`으로 gate 하한이고 adjacent repeat `1`이 남아 있다.
- 이 결과는 distinct sample-seed 후보의 focused listening review 진입 조건이며, human/audio preference나 broad model quality proof가 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_focused_package tests.test_stage_b_margin_recovered_focused_context_decision
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-focused-context
```

## 산출물

- package: `outputs/stage_b_margin_recovered_phrase_vocabulary_focused_package/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_package/focused_review_package.json`
- decision: `outputs/stage_b_margin_recovered_focused_context_decision/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_focused_context_decision/focused_context_decision.json`
