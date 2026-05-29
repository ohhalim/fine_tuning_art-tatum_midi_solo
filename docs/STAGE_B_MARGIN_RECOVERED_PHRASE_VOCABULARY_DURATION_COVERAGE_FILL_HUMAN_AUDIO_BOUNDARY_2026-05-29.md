# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Human/Audio Boundary

Issue #324는 duration/coverage fill keep 후보의 human/audio review boundary를 정의한 작업이다.

## Context

- source candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3`
- selected fill candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- previous boundary: `single_postprocess_candidate_keep_support`
- claim risk: MIDI/context evidence keep을 human/audio preference로 과장할 가능성

## Change

- source constrained partial MIDI와 duration fill keep MIDI 비교
- note sequence / metric summary match 여부 기록
- human/audio review field pending 유지
- preference claim 차단

## Result

| item | value |
|---|---:|
| review item count | `2` |
| human/audio status | `pending` |
| boundary | `pending_human_audio_review_source_vs_fill_distinct_midi_content` |
| preference claimed | `false` |
| note sequence match | `false` |
| metric summary match | `false` |
| fill additions | `6` |
| dead-air delta | `0.2773` |

| role | candidate | prior decision | note signature | notes | focused notes | unique | focused unique | dead-air | max active |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| source_constrained_partial | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3` | `needs_duration_coverage_fill` | `15` | `15` | `12` | `10` | `9` | `0.5714` | `2` |
| duration_coverage_fill_keep | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6` | `keep` | `18` | `18` | `18` | `15` | `15` | `0.2941` | `1` |

## Judgment

- source vs fill MIDI content distinct
- human/audio review status pending
- preference claim 없음
- audio render quality 미검증
- broad trained-model quality, Brad style adaptation 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-human-audio-boundary
```

## Output

- script: `scripts/build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_boundary/duration_coverage_fill_human_audio_boundary.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill human/audio review fill`
- audio render 또는 human review 입력 전까지 preference field pending 유지
