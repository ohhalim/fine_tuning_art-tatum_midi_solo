# Stage B Duration Coverage Fill Broader Repeatability Sweep

Issue #351은 duration coverage fill 후보의 반복성 경계를 distinct sample-seed 후보 기준으로 재검토한 작업이다.

## Context

- Issue #349 next boundary: `broader_repeatability_sweep`
- current keep anchor: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- current anchor variants: `2/4` qualified
- current anchor dead-air: `0.5714 -> 0.2941`
- current anchor single-user listening preference: `true`
- broad model quality claimed: `false`

## Change

- broader repeatability sweep summary script 추가
- distinct sample-seed qualified 후보 `2`개에 duration/coverage fill gate 재적용
- current keep anchor와 distinct source sweep 결과 분리
- uniform dead-air gain 여부와 qualified MIDI gate 여부 분리
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| boundary | `qualified_gate_repeatability_with_partial_dead_air_gain` |
| source candidates | `2` |
| distinct sample seeds | `2` |
| qualified source candidates | `2` |
| dead-air improved source candidates | `1` |
| total variants | `8` |
| qualified variants | `7` |
| broad model quality claimed | `false` |

## Source Sweep

| source | sample seed | selected | variants | dead-air | unique | adj repeat | max interval | improved |
|---|---:|---|---:|---:|---:|---:|---:|:---:|
| `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47` | `155` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_47_duration_fill_maxadd_4` | `4/4` | `0.3750 -> 0.3750` | `10` | `0` | `4` | `false` |
| `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23` | `131` | `margin_recovered_phrase_vocab_seed_109_topk_7_temp_082_n48_sample_23_duration_fill_maxadd_6` | `3/4` | `0.3750 -> 0.3529` | `13` | `0` | `11` | `true` |

## Judgment

- distinct sample-seed 후보 `2/2`에서 selected fill 후보가 MIDI gate 통과
- total variants `7/8` qualified
- dead-air gain은 `1/2` 후보에서만 관측
- 반복성 경계: qualified gate 반복성은 관측, uniform dead-air gain은 미검증
- new source human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_broader_repeatability_sweep.py
bash scripts/agent_harness.sh stage-b-duration-coverage-broader-repeatability-sweep
```

## Output

- script: `scripts/summarize_stage_b_duration_coverage_fill_broader_repeatability_sweep.py`
- test: `tests/test_stage_b_duration_coverage_fill_broader_repeatability_sweep.py`
- summary: `outputs/stage_b_duration_coverage_fill_broader_repeatability_sweep/harness_stage_b_duration_coverage_fill_broader_repeatability_sweep/stage_b_duration_coverage_fill_broader_repeatability_sweep.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill dead-air gain repeatability repair`
