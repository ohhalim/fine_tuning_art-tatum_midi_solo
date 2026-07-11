# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill MIDI Evidence Review

Issue #330은 source constrained partial과 duration/coverage fill 후보를 MIDI evidence 기준으로 비교한 작업이다.

## Context

- Issue #328 package status: `ready_for_external_review_input`
- source candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3`
- fill candidate: `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6`
- user request: local MIDI evidence review/report
- boundary: human/audio preference는 미검증

## Change

- MIDI metric and note-structure review script 추가
- source vs fill score 비교
- MIDI evidence preference 기록
- human/audio preference claim 차단

## Result

| item | value |
|---|---:|
| review basis | `midi_metric_and_note_structure` |
| MIDI evidence preference | `duration_coverage_fill_keep` |
| score delta fill-source | `79.7311` |
| dead-air delta fill-source | `-0.2773` |
| focused note count delta | `+6` |
| focused unique pitch count delta | `+6` |
| max simultaneous notes delta | `-1` |
| human/audio preference claimed | `false` |
| audio render used | `false` |

| role | candidate | focused notes | focused unique | dead-air | max active | score |
|---|---|---:|---:|---:|---:|---:|
| source_constrained_partial | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3` | `12` | `9` | `0.5714` | `2` | `91.857` |
| duration_coverage_fill_keep | `margin_recovered_phrase_vocab_seed_353_topk_7_temp_082_n24_sample_3_duration_fill_maxadd_6` | `18` | `15` | `0.2941` | `1` | `171.588` |

## Judgment

- MIDI evidence 기준 fill 후보 우세
- fill 후보는 dead-air 감소, focused note count 증가, focused unique pitch 증가, max active note 감소
- adjacent repeat, duplicated 3-note pitch-class chunk, max interval guardrail 유지
- audio render 미사용
- human/audio preference, audio rendered quality, broad trained-model quality, Brad style adaptation 미검증

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-midi-evidence-review
```

## Output

- script: `scripts/review_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_midi_evidence_review/duration_coverage_fill_midi_evidence_review.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill MIDI evidence review consolidation`
- MIDI evidence preference와 human/audio proof boundary 분리 유지
