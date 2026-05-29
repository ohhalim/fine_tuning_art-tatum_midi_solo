# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill Audio Review Package

Issue #328은 duration/coverage fill 후보의 외부 review input 전 package를 만든 작업이다.

## Context

- Issue #326 result: review input absent, preference pending
- candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- source vs fill MIDI content distinct
- next need: reviewer-facing manifest and input template

## Change

- source/fill MIDI path manifest 생성
- selected fill context MIDI path 포함
- required file existence and checksum validation
- review input template export
- preference claim remains false

## Result

| item | value |
|---|---:|
| review item count | `2` |
| package status | `ready_for_external_review_input` |
| audio render status | `not_rendered_by_harness` |
| preference claimed | `false` |
| required file count | `3` |

| role | candidate | MIDI exists | context exists | notes | focused notes | unique | focused unique | dead-air | sha256 |
|---|---|:---:|:---:|---:|---:|---:|---:|---:|---|
| source_constrained_partial | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3` | `true` | `false` | `15` | `12` | `10` | `9` | `0.5714` | `8429ccb789ba` |
| duration_coverage_fill_keep | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6` | `true` | `true` | `18` | `18` | `15` | `15` | `0.2941` | `b517b822a919` |

## Review Input Template

- schema: `stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_human_audio_review_input_v1`
- candidate_id: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- required before preference claim: reviewer, audio_render_used `true`, preference, timing, phrase, vocabulary
- preference values: `source_constrained_partial`, `duration_coverage_fill_keep`, `tie`, `reject_both`
- timing/phrase/vocabulary values: `source_constrained_partial`, `duration_coverage_fill_keep`, `tie`, `unclear`

## Judgment

- review package ready for external input
- harness did not render audio
- preference remains unclaimed
- human/audio preference and audio rendered quality remain unverified

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-audio-review-package
```

## Output

- script: `scripts/build_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package/duration_coverage_fill_audio_review_package.json`
- template: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_audio_review_package/duration_coverage_fill_human_audio_review_input_template.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill external review input fill`
- requires external human/audio review input
