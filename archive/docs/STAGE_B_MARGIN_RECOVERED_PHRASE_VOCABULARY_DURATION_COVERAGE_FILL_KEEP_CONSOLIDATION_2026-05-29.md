# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Keep Consolidation

Issue #322는 Issue #320의 duration/coverage fill `keep` 결과를 claim boundary 기준으로 정리한 작업이다.

## Context

- source candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3`
- selected candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- previous issue: focused listening fill decision `keep`
- review boundary: MIDI/context evidence fill
- claim risk: 단일 postprocess 후보의 `keep`을 broad model quality로 과장할 가능성

## Change

- duration fill repair summary와 focused listening filled notes 조인
- single keep candidate 검증
- postprocess claim boundary 검증
- proven / not proven boundary 분리
- harness mode 추가

## Result

| item | value |
|---|---:|
| decision | `keep` |
| boundary | `single_postprocess_candidate_keep_support` |
| postprocess claim boundary | `postprocess_duration_coverage_fill_candidate` |
| variant count | `4` |
| qualified variant count | `2` |
| fill additions | `6` |
| dead-air | `0.5714 -> 0.2941` |
| onset coverage | `0.5625` |
| sustained coverage | `0.6250` |

Focused metrics:

| item | value |
|---|---:|
| note count | `18` |
| unique pitch count | `15` |
| range | `D#4-G#5` |
| phrase span | `7.000` beats |
| max active notes | `1` |
| adjacent pitch repeats | `0` |
| duplicated 3-note pitch-class chunks | `0` |
| max interval | `7` |
| final note | `F4` over `Fm7`, chord tone |
| review risks | `[]` |

## Proven

- MIDI/context evidence 기준 `keep`
- constrained partial 후보 대비 dead-air 감소
- adjacent repeat blocker repair
- wide interval blocker repair
- solo/context review artifact 확보
- final landing chord tone

## Not Proven

- human/audio preference
- broad trained-model quality
- Brad style adaptation
- broad repeatability
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-keep-consolidation
```

## Output

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_keep_consolidation/duration_coverage_fill_keep_consolidation.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio comparison boundary`
- MIDI/context evidence keep과 human/audio review boundary 분리
