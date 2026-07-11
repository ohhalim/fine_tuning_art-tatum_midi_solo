# Stage B Duration Coverage Fill Outside-Soloing Repair Sweep

Issue #363은 repeatability source 후보 `2`개에 pitch-role / chord-fit 보정 sweep을 적용한 작업이다.

## Context

- Issue #361 next boundary: `outside_soloing_pitch_role_phrase_clarity_repair`
- Issue #353 source boundary: `qualified_gate_repeatability_with_dead_air_gain`
- user listening review: repeatability WAV 후보 `2`개 모두 difficult / outside-soloing-like
- 기존 유지 조건: dead-air gain, monophonic gate
- 제외 claim: human/audio preference, broad trained-model quality, Brad style adaptation, production-ready improviser

## Change

- outside-soloing repair sweep script 추가
- `chord_tone_snap`, `guide_tone_landing`, `contour_resolution` policy 비교
- chord-tone ratio, guide-tone landing, max non-chord run, max interval 측정
- source별 selected repaired MIDI 후보 기록
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `outside_soloing_pitch_role_repair_candidates` |
| source candidates | `2` |
| repaired source candidates | `2` |
| dead-air preserved source candidates | `2` |
| total variants | `6` |
| qualified variants | `6` |
| selected policy | `contour_resolution` |
| selected min chord-tone ratio | `1.000` |
| selected max non-chord run | `0` |
| selected max interval | `7` |
| broad model quality claimed | `false` |

## Selected Sources

| sample seed | dead-air | unique pitch | max interval | chord-tone ratio | selected |
|---:|---:|---:|---:|---:|---|
| `155` | `0.3333` | `10` | `7` | `1.000` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47_duration_fill_maxadd_6_outside_repair_contour_resolution` |
| `131` | `0.3529` | `9` | `5` | `1.000` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23_duration_fill_maxadd_6_outside_repair_contour_resolution` |

## Judgment

- outside-soloing 청취 문제를 pitch-role objective repair 후보로 1차 보정
- dead-air gain preserved source candidates: `2/2`
- selected candidates: chord-tone ratio `1.000`, max non-chord run `0`
- sample seed `131` max interval: `11 -> 5`
- sample seed `155` max interval: `6 -> 7`
- 이번 결과는 MIDI objective repair 후보이며 청취 선호 proof가 아님
- audio review 필요

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_sweep.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-sweep
```

## Output

- script: `scripts/summarize_stage_b_duration_coverage_fill_outside_soloing_repair_sweep.py`
- test: `tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_sweep.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep/stage_b_duration_coverage_fill_outside_soloing_repair_sweep.json`
- markdown: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_sweep/harness_stage_b_duration_coverage_fill_outside_soloing_repair_sweep/stage_b_duration_coverage_fill_outside_soloing_repair_sweep.md`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair audio review package`
