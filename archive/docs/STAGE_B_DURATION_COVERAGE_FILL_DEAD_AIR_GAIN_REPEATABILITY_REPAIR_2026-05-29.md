# Stage B Duration Coverage Fill Dead-Air Gain Repeatability Repair

Issue #353은 duration coverage fill 반복성 sweep에서 dead-air gain이 부분적으로만 관측된 원인을 selected variant 기준으로 보정한 작업이다.

## Context

- Issue #351 boundary: `qualified_gate_repeatability_with_partial_dead_air_gain`
- source candidates: `2`
- qualified source candidates: `2`
- dead-air improved source candidates: `1`
- total variants: `8`
- qualified variants: `7`
- broad model quality claimed: `false`

## Change

- dead-air gain repeatability repair summary script 추가
- selection rule: `qualified_dead_air_gain_then_min_fill_additions`
- source별 full fill variant report 저장
- selected variant 기준 dead-air gain 재측정
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| previous boundary | `qualified_gate_repeatability_with_partial_dead_air_gain` |
| repaired boundary | `qualified_gate_repeatability_with_dead_air_gain` |
| source candidates | `2` |
| qualified source candidates | `2` |
| dead-air gain source candidates | `2` |
| total variants | `8` |
| qualified variants | `7` |
| dead-air gain variants | `6` |
| selected fill additions | `[6]` |
| broad model quality claimed | `false` |

## Source Sweep

| source | sample seed | selected | variants | dead-air gain variants | dead-air | unique | adj repeat | max interval |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` | `155` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47_duration_fill_maxadd_6` | `4/4` | `3` | `0.3750 -> 0.3333` | `12` | `0` | `6` |
| `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23` | `131` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23_duration_fill_maxadd_6` | `3/4` | `3` | `0.3750 -> 0.3529` | `13` | `0` | `11` |

## Judgment

- 이전 partial boundary의 원인: qualified variant 중 fill addition 최소값 우선 선택
- repair 기준: qualified + dead-air gain 후보만 우선 선택
- selected distinct source `2/2`에서 dead-air gain 관측
- new source human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.py
bash scripts/agent_harness.sh stage-b-duration-coverage-dead-air-gain-repeatability-repair
```

## Output

- script: `scripts/summarize_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.py`
- test: `tests/test_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.py`
- summary: `outputs/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/harness_stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair/stage_b_duration_coverage_fill_dead_air_gain_repeatability_repair.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill repeatability consolidation`
