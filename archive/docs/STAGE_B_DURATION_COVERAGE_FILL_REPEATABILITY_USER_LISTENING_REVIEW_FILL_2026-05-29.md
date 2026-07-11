# Stage B Duration Coverage Fill Repeatability User Listening Review Fill

Issue #359는 repeatability source WAV `2`개에 대한 사용자 청취 리뷰를 반영한 작업이다.

## Context

- Issue #357 status: `ready_for_user_listening_review`
- repeatability source WAV files: `2`
- technical WAV validation: `true`
- MIDI/dead-air gain repeatability support: `true`
- audio quality/preference claim before review: `false`

## User Review Input

- both candidates sound difficult and outside-soloing-like

## Change

- repeatability user listening review fill script 추가
- 후보 `2`개 모두 `needs_followup`으로 기록
- timing / phrase / vocabulary: `outside_or_unclear`
- human/audio keep preference claim 금지
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `repeatability_audio_review_needs_followup` |
| review status | `reviewed` |
| overall decision | `reject_all` |
| candidate decision | `needs_followup` |
| timing | `outside_or_unclear` |
| phrase | `outside_or_unclear` |
| vocabulary | `outside_or_unclear` |
| reviewed audio files | `2` |
| repeatability human/audio keep claimed | `false` |
| broad model quality claimed | `false` |

## Candidate Reviews

| sample seed | candidate | decision | assessment |
|---:|---|---|---|
| `155` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47_duration_fill_maxadd_6` | `needs_followup` | `outside-soloing-like` |
| `131` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23_duration_fill_maxadd_6` | `needs_followup` | `outside-soloing-like` |

## Judgment

- MIDI/dead-air gain repeatability는 유지
- 사용자 청취 기준 repeatability keep은 미검증
- 문제 경계: 난해함 / outside-soloing-like phrase clarity
- broad trained-model quality, Brad style adaptation, production-ready improviser claim 금지

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_repeatability_user_listening_review.py
bash scripts/agent_harness.sh stage-b-duration-coverage-repeatability-user-listening-review
```

## Output

- script: `scripts/fill_stage_b_duration_coverage_repeatability_user_listening_review.py`
- test: `tests/test_stage_b_duration_coverage_fill_repeatability_user_listening_review.py`
- summary: `outputs/stage_b_duration_coverage_fill_repeatability_user_listening_review_fill/harness_stage_b_duration_coverage_fill_repeatability_user_listening_review_fill/stage_b_duration_coverage_fill_repeatability_user_listening_review_fill.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair decision`
