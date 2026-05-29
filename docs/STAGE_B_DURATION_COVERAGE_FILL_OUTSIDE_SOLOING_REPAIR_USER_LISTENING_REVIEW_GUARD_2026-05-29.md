# Stage B Duration Coverage Fill Outside-Soloing Repair User Listening Review Guard

Issue #367은 outside-soloing repair WAV 후보 `2`개에 대한 청취 입력 부재 상태를 preference claim 없이 기록한 작업이다.

## Context

- Issue #365 status: `ready_for_user_listening_review`
- rendered WAV files: `2`
- technical WAV validation: `true`
- current missing input: user listening preference
- required boundary: no human/audio preference claim without validated review input

## Change

- outside-soloing repair user listening review fill script 추가
- review input absent 상태를 `pending_review_input`으로 기록
- candidate별 WAV path와 objective metrics 유지
- human/audio preference claim guard 유지
- objective-only follow-up 가능 여부와 preference claim 조건 분리
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `outside_soloing_repair_audio_review_pending` |
| review input present | `false` |
| fill status | `pending_review_input` |
| user listening status | `pending_review_input` |
| overall decision | `pending` |
| human/audio preference claimed | `false` |
| objective auto progress allowed | `true` |
| critical user input required | `false` |

## Reviewed Audio Files

| sample seed | role | wav |
|---:|---|---|
| `155` | `outside_repair_sample_seed_155_contour_resolution` | `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/audio/outside_repair_sample_seed_155_contour_resolution.wav` |
| `131` | `outside_repair_sample_seed_131_contour_resolution` | `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/audio/outside_repair_sample_seed_131_contour_resolution.wav` |

## Judgment

- 청취 선호는 아직 미검증
- human/audio preference, multi-reviewer preference claim 금지
- objective-only evidence consolidation은 계속 진행 가능
- broad trained-model quality, Brad style adaptation, production-ready improviser claim 금지

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-user-listening-review
```

## Output

- script: `scripts/fill_stage_b_duration_coverage_outside_soloing_repair_user_listening_review.py`
- test: `tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.json`
- markdown: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/harness_stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill/stage_b_duration_coverage_fill_outside_soloing_repair_user_listening_review_fill.md`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair objective evidence consolidation`
