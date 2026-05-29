# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Focused Context

Issue #316은 Issue #314 selected duration/coverage fill candidate를 solo/context package로 격리하고 focused context decision을 검토한 작업이다.

## Context

- source candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- source boundary: `postprocess_duration_coverage_fill_candidate`
- fill additions: `6`
- dead-air ratio: `0.2941`
- focused note count: `18`
- focused unique pitch count: `15`
- adjacent pitch repeats: `0`
- duplicated 3-note pitch-class chunks: `0`
- max interval: `7`
- remaining flags: `[]`

## Change

- phrase/vocabulary focused package builder의 `review_files.report_path` 지원
- duration/coverage fill summary의 `duration_coverage_gate` 지원
- duration/coverage fill candidate focused package harness 추가
- focused context decision harness 추가
- unit test coverage 추가

## Result

| item | value |
|---|---:|
| candidate count | `1` |
| decision | `keep_for_focused_listening` |
| decision flags | `[]` |
| note count | `18` |
| unique pitch count | `15` |
| range | `D#4-G#5` |
| phrase span | `7.000` beats |
| max active notes | `1` |
| dead-air ratio | `0.2941` |
| max interval | `7` |
| adjacent pitch repeats | `0` |
| duplicated 3-note pitch-class chunks | `0` |
| final note | `F4` |
| final chord | `Fm7` |
| final note role | `chord_tone` |

Context MIDI:

- chord guide: `present`
- bass guide: `present`
- solo track: `present`
- context total beats: `8.0`
- solo max end beats: `7.25`

## Judgment

- duration/coverage fill candidate is ready for focused listening notes.
- focused context blocker not observed.
- human/audio preference and broad trained-model quality remain unverified.
- Brad style adaptation success is not claimed.

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_focused_package.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-focused-context
```

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill focused listening notes`
- focused listening fields remain pending until evidence fill or human review
