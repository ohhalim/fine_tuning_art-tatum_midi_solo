# Stage B Margin-Recovered Phrase/Vocabulary Distinct Sample-Seed Remaining Blocker

## 요약

Issue #306은 Issue #304 `needs_followup` 결과를 다음 repair sweep의 target으로 정리한 작업이다.

이 단계는 새 MIDI 후보를 만들기 전에, distinct sample-seed 후보에서 무엇을 보존하고 무엇을 고쳐야 하는지 metric 기준으로 고정하기 위한 경계다.

## 변경

- distinct sample-seed remaining blocker summary script 추가
- filled notes 기반 repair target과 keep guardrail 분리
- unit test와 harness mode 추가
- README, CORE_PLAN, CURRENT_STATUS_AND_PLAN, AGENTS handoff 업데이트

## 결과

| 항목 | 값 |
|---|---|
| candidate | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` |
| sample seed | `155` |
| final decision | `needs_followup` |
| repair boundary | `distinct_sample_seed_candidate_needs_phrase_vocabulary_repair` |
| remaining blockers | `phrase_continuation_weak`, `jazz_vocabulary_thin`, `short_phrase_span`, `pitch_variety_floor`, `adjacent_pitch_repeats` |
| secondary risks | `dead_air_ratio_remaining` |
| next recommended issue | `Stage B margin-recovered phrase/vocabulary distinct sample-seed remaining blocker repair sweep` |

## Repair Target

| metric | current | target |
|---|---:|---:|
| phrase span beats | `6.750` | `>= 7.0` |
| unique pitch count | `6` | `>= 7` |
| adjacent pitch repeats | `1` | `0` |
| dead-air ratio | `0.375` | `<= 0.35` preferred |
| max interval | `3` | `< 12` preserve |
| max active notes | `1` | `<= 1` preserve |
| final note role | `tension` | chord tone or tension preserve |

## 해석

- 새 repair는 phrase span, pitch variety, adjacent repeat를 같이 개선해야 한다.
- timing, landing, max interval, max active notes는 guardrail로 유지한다.
- sample seed `85`는 duplicate output boundary로 제외한다.
- 이 문서는 다음 sweep 조건 정의이며 새 generated-quality claim이 아니다.

## 검증

```bash
.venv/bin/python -m unittest tests.test_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-distinct-sample-seed-remaining-blocker
```

## 산출물

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker/harness_stage_b_margin_recovered_phrase_vocabulary_distinct_sample_seed_remaining_blocker/remaining_blocker_summary.json`
