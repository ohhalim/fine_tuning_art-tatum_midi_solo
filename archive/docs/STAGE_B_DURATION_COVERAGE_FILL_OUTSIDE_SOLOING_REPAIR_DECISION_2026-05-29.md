# Stage B Duration Coverage Fill Outside-Soloing Repair Decision

Issue #361은 repeatability source 청취 리뷰의 `needs_followup` 결과를 다음 repair target으로 변환한 작업이다.

## Context

- Issue #359 boundary: `repeatability_audio_review_needs_followup`
- overall decision: `reject_all`
- candidate decision: `needs_followup`
- timing / phrase / vocabulary: `outside_or_unclear`
- user review: both candidates sound difficult and outside-soloing-like
- human/audio keep claimed: `false`
- broad model quality claimed: `false`

## Change

- outside-soloing repair decision script 추가
- user review boundary를 repair target으로 변환
- auto progress 가능 여부와 critical user input 필요 여부 분리
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| input boundary | `repeatability_audio_review_needs_followup` |
| next boundary | `outside_soloing_pitch_role_phrase_clarity_repair` |
| auto progress allowed | `true` |
| critical user input required | `false` |
| repair target count | `5` |
| human/audio keep claimed | `false` |
| broad model quality claimed | `false` |

## Repair Targets

- `reduce_outside_sounding_pitch_choices`
- `increase_chord_tone_or_guide_tone_landing`
- `limit_non_chord_tone_run_length`
- `penalize_large_interval_after_fill`
- `prefer_phrase_contour_resolution_over_density`

## Judgment

- MIDI/dead-air repeatability는 유지
- 청취 기준 문제는 density 자체보다 pitch-role / chord-fit / phrase clarity 축으로 분리
- 다음 repair는 dead-air gain과 monophonic gate를 유지하면서 outside-sounding pitch 선택을 제한
- repair 후 audio review 필요
- broad trained-model quality, Brad style adaptation, production-ready improviser claim 금지

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_outside_soloing_repair_decision.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-decision
```

## Output

- script: `scripts/decide_stage_b_duration_coverage_outside_soloing_repair.py`
- test: `tests/test_stage_b_duration_coverage_outside_soloing_repair_decision.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_decision/harness_stage_b_duration_coverage_fill_outside_soloing_repair_decision/stage_b_duration_coverage_fill_outside_soloing_repair_decision.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair sweep`
