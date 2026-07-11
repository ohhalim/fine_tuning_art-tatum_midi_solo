# Stage B Duration Coverage Fill User Listening Review Consolidation

Issue #347은 MIDI evidence, WAV render validation, user listening review를 하나의 claim boundary로 정리한 작업이다.

## Context

- Issue #332 MIDI evidence consolidation: `midi_evidence_preference_support`
- Issue #343 local audio render attempt: rendered WAV files `2`, technical WAV validation `true`
- Issue #345 user listening review fill: preference `duration_coverage_fill_keep`
- user assessment: source 후보는 random-like, fill 후보는 jazz-like soloing

## Change

- user listening review consolidation script 추가
- MIDI evidence / audio render / user review reports 조인
- consolidated claim boundary 정의
- proven / not proven / next decision 정리
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `midi_evidence_and_single_user_listening_support_duration_coverage_fill_keep` |
| preferred candidate | `duration_coverage_fill_keep` |
| MIDI evidence preference | `duration_coverage_fill_keep` |
| user listening preference | `duration_coverage_fill_keep` |
| same preferred candidate | `true` |
| rendered audio file count | `2` |
| technical WAV validation | `true` |
| single user review | `true` |
| broad model quality claimed | `false` |

## Proven

- MIDI metric preference for duration/coverage fill keep
- technical WAV render validation completed
- single-user listening preference for duration/coverage fill keep

## Not Proven

- multi-reviewer preference
- audio rendered quality
- broad trained-model quality
- Brad style adaptation
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_user_listening_review_consolidation.py
bash scripts/agent_harness.sh stage-b-user-listening-review-consolidation
```

## Output

- script: `scripts/summarize_stage_b_duration_coverage_user_listening_consolidation.py`
- test: `tests/test_stage_b_user_listening_review_consolidation.py`
- summary: `outputs/stage_b_duration_coverage_fill_user_listening_review_consolidation/harness_stage_b_duration_coverage_fill_user_listening_review_consolidation/stage_b_duration_coverage_fill_user_listening_review_consolidation.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill next repair or repeatability decision`
- consolidated fill evidence를 기준으로 broader repeatability 또는 다음 repair target 분리
