# Stage B Duration Coverage Fill Outside-Soloing Repair Broader Repeatability Sweep

Issue #373은 outside-soloing repair policy variant 전체를 source별로 집계해 objective repeatability를 확인한 작업이다.

## Context

- Issue #371 next boundary: `outside_soloing_repair_broader_repeatability_sweep`
- auto progress allowed: `true`
- critical user input required: `false`
- human/audio preference claimed: `false`
- repair sweep source candidates: `2`

## Change

- outside-soloing repair broader repeatability sweep script 추가
- `chord_tone_snap`, `guide_tone_landing`, `contour_resolution` policy별 source repeatability 집계
- dead-air preservation, chord-tone ratio, non-chord run, interval gate 반복성 측정
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `outside_soloing_repair_policy_repeatability_support` |
| source candidates | `2` |
| repair policies | `3` |
| supported repair policies | `3` |
| total variants | `6` |
| qualified variants | `6` |
| selected min chord-tone ratio | `1.000` |
| selected max non-chord run | `0` |
| selected max interval | `7` |
| human/audio preference claimed | `false` |
| broad model quality claimed | `false` |

## Policy Summary

| policy | sources | qualified | chord-tone min | non-chord max | interval max | supported |
|---|---:|---:|---:|---:|---:|:---:|
| `chord_tone_snap` | `2` | `2/2` | `1.000` | `0` | `7` | `true` |
| `contour_resolution` | `2` | `2/2` | `1.000` | `0` | `7` | `true` |
| `guide_tone_landing` | `2` | `2/2` | `1.000` | `0` | `7` | `true` |

## Judgment

- repair policy `3/3`에서 source `2/2` objective repeatability support
- total variants `6/6` qualified
- dead-air preservation, chord-tone ratio, non-chord run, interval gate 반복성 확인
- human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-broader-repeatability
```

## Output

- script: `scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.py`
- test: `tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/harness_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.json`
- markdown: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/harness_stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep/stage_b_duration_coverage_fill_outside_soloing_repair_broader_repeatability_sweep.md`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair repeatability consolidation`
