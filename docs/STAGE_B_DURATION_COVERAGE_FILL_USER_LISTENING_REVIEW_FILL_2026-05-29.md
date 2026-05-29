# Stage B Duration Coverage Fill User Listening Review Fill

Issue #345는 rendered source/fill WAV에 대한 user listening review 입력을 반영한 작업이다.

## Context

- Issue #343 local audio render attempt 완료
- rendered audio file count: `2`
- source WAV: `source_constrained_partial.wav`
- fill WAV: `duration_coverage_fill_keep.wav`
- technical WAV validation: `true`
- prior MIDI evidence preference: `duration_coverage_fill_keep`

## User Review Input

- preference: `duration_coverage_fill_keep`
- source assessment: source 후보는 이해하기 어렵고 random notes처럼 들림
- fill assessment: fill 후보가 훨씬 jazz-like soloing으로 들림

## Change

- user listening review fill script 추가
- audio render report와 review input schema 검증
- single-user human/audio preference claim 기록
- source assessment / fill assessment 분리
- broad model quality claim guard 유지

## Result

| item | value |
|---|---:|
| review status | `reviewed` |
| preference | `duration_coverage_fill_keep` |
| timing | `duration_coverage_fill_keep` |
| phrase | `duration_coverage_fill_keep` |
| vocabulary | `duration_coverage_fill_keep` |
| human/audio preference claimed | `true` |
| single user review | `true` |
| broad model quality claimed | `false` |
| audio rendered quality claimed | `false` |

## Not Proven

- multi-reviewer preference
- audio rendered quality
- broad trained-model quality
- Brad style adaptation
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_user_listening_review_fill.py
bash scripts/agent_harness.sh stage-b-user-listening-review-fill
```

## Output

- script: `scripts/fill_stage_b_duration_coverage_user_listening_review.py`
- test: `tests/test_stage_b_user_listening_review_fill.py`
- summary: `outputs/stage_b_duration_coverage_fill_user_listening_review_fill/harness_stage_b_duration_coverage_fill_user_listening_review_fill/stage_b_duration_coverage_fill_user_listening_review_fill.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill user listening review consolidation`
- MIDI evidence, technical WAV validation, user listening preference를 하나의 claim boundary로 정리
