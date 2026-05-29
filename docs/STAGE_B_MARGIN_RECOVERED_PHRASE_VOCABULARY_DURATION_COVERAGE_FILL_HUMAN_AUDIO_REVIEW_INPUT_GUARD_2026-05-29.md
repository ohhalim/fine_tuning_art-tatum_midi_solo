# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Human/Audio Review Input Guard

Issue #326은 duration/coverage fill human/audio review fill에서 review input 없이 preference가 채워지는 것을 막는 작업이다.

## Context

- Issue #324 boundary: `pending_human_audio_review_source_vs_fill_distinct_midi_content`
- candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- human/audio status: `pending`
- preference claimed: `false`
- risk: review input 없이 preference/final decision 작성

## Change

- human/audio review fill guard script 추가
- review input absent 상태의 pending 유지 검증
- review input present 상태의 reviewer/audio_render/preference schema 검증
- invalid review input rejection test 추가

## Result

| item | value |
|---|---:|
| review input present | `false` |
| fill status | `pending_review_input` |
| human/audio status | `pending` |
| preference | `pending` |
| preference claimed | `false` |
| audio render used | `false` |

## Guarded Input Fields

| field | requirement |
|---|---|
| candidate_id | selected duration fill candidate match |
| reviewer | non-empty |
| audio_render_used | `true` |
| preference | `source_constrained_partial`, `duration_coverage_fill_keep`, `tie`, `reject_both` |
| timing / phrase / vocabulary | `source_constrained_partial`, `duration_coverage_fill_keep`, `tie`, `unclear` |

## Judgment

- review input absent 상태에서 preference claim 차단
- pending status 유지
- human/audio preference와 audio rendered quality는 아직 미검증
- validated review input이 들어오기 전까지 broad model quality claim 없음

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-review-input-guard
```

## Output

- script: `scripts/fill_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_fill/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_guard/duration_coverage_fill_human_audio_review_fill.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill audio review package`
- review input을 채우기 전 source/fill MIDI와 context paths를 reviewer-facing package로 정리
