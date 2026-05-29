# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Repair

Issue #314는 Issue #312 partial candidate에서 남은 `dead_air_not_repaired` blocker를 postprocess duration/coverage fill로 검토한 작업이다.

## Context

- Issue #312 coverage-aware constrained decoding result
- target-qualified candidate count: `0/48`
- partial candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3`
- focused unique pitch count: `9`
- focused note count: `12`
- dead-air ratio: `0.5714`
- adjacent pitch repeats: `0`
- focused max interval: `7`
- remaining flag: `dead_air_not_repaired`

## Change

- source candidate polyphony postprocess: simultaneous limit `1`
- large onset-gap 기준 fill note 삽입
- fill pitch selection: chord tone + passing-tone bridge
- monophonic note end 보정
- fill variant summary/report 추가
- harness mode 추가: `stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-repair`

## Result

| item | value |
|---|---:|
| variant count | `4` |
| qualified variant count | `2` |
| selected fill additions | `6` |
| baseline dead-air | `0.5714` |
| selected dead-air | `0.2941` |
| dead-air delta | `0.2773` |
| selected focused note count | `18` |
| selected focused unique pitch count | `15` |
| selected adjacent pitch repeats | `0` |
| selected duplicated 3-note pitch-class chunks | `0` |
| selected max interval | `7` |
| remaining flags | `[]` |

Selected candidate:

```text
margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6
```

Variant boundary:

| rank | additions | qualified | dead-air | unique | adjacent repeat | dup3 | max interval | flags |
|---:|---:|:---:|---:|---:|---:|---:|---:|---|
| `1` | `6` | `true` | `0.2941` | `15` | `0` | `0` | `7` | `[]` |
| `2` | `10` | `true` | `0.1429` | `19` | `0` | `0` | `10` | `[]` |
| `3` | `4` | `false` | `0.4667` | `13` | `0` | `0` | `7` | `dead_air_not_repaired` |
| `4` | `8` | `false` | `0.2105` | `17` | `0` | `1` | `10` | `repeated_pitch_class_cell` |

## Judgment

- dead-air blocker repaired within objective gate.
- adjacent repeat, duplicated pitch-class cell, max interval guardrail 유지.
- selected variant는 threshold 통과에 필요한 최소 addition count 기준.
- claim boundary: `postprocess_duration_coverage_fill_candidate`.
- broad trained-model quality 또는 Brad style adaptation 성공 근거 아님.

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_repair.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-repair
```

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill focused context review`
- repaired candidate solo/context package 검토
- focused listening notes boundary 이동 여부 판단
