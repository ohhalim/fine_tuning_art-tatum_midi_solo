# Stage B Margin-Recovered Phrase/Vocabulary Duration Coverage Fill External Human/Audio Boundary

Issue #335는 Issue #332 MIDI evidence consolidation 이후 human/audio review claim 경계를 정리한 작업이다.

## Context

- Issue #332 boundary: `midi_evidence_preference_support`
- MIDI evidence preference: `duration_coverage_fill_keep`
- score delta fill-source: `+79.7311`
- dead-air delta fill-source: `-0.2773`
- focused note count delta: `+6`
- focused unique pitch count delta: `+6`
- max simultaneous notes delta: `-1`
- human/audio preference claimed: `false`
- audio rendered quality: 미검증

## Change

- external human/audio review boundary summary 추가
- required external review input schema 정리
- MIDI evidence preference와 human/audio preference claim 분리
- pending external review 상태 검증
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| source boundary | `midi_evidence_preference_support` |
| external boundary | `external_human_audio_review_required_for_human_preference_claim` |
| external review status | `pending_external_review_input` |
| MIDI evidence preference | `duration_coverage_fill_keep` |
| score delta fill-source | `+79.7311` |
| human/audio preference claimed | `false` |
| audio render used | `false` |

## Required External Review Input

- `reviewer`
- `audio_render_used`
- `preference`
- `timing`
- `phrase`
- `vocabulary`
- `notes`

## Not Proven

- human/audio preference
- audio rendered quality
- broad trained-model quality
- Brad style adaptation
- production-ready improviser

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary.py
bash scripts/agent_harness.sh stage-b-margin-recovered-phrase-vocabulary-duration-coverage-fill-external-human-audio-boundary
```

## Output

- script: `scripts/summarize_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary.py`
- test: `tests/test_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary.py`
- summary: `outputs/stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary/harness_stage_b_margin_recovered_phrase_vocabulary_duration_coverage_fill_external_human_audio_boundary/duration_coverage_fill_external_human_audio_boundary.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill local audio render package`
- 외부 review input 확보 시 `Stage B margin-recovered phrase/vocabulary duration coverage fill external review input fill`
