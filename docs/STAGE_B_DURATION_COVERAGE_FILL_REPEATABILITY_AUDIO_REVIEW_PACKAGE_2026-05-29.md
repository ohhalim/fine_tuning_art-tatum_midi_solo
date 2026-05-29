# Stage B Duration Coverage Fill Repeatability Audio Review Package

Issue #357은 repeatability source 후보 `2`개를 WAV로 렌더하고 사용자 청취 review 입력 전 기술 검증 경계를 정리한 작업이다.

## Context

- Issue #355 boundary: `current_keep_and_distinct_source_dead_air_gain_midi_support`
- distinct source MIDI/dead-air gain support: `true`
- new source human/audio preference claimed: `false`
- broad model quality claimed: `false`
- renderer: FluidSynth
- soundfont: GeneralUser GS `v1.471.sf2`

## Change

- repeatability audio review package render script 추가
- distinct source selected fill MIDI `2`개 WAV 렌더
- WAV sample rate, duration, size, sha256 검증
- audio quality/preference claim guard 유지
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| candidate | `duration_coverage_fill_repeatability_sources` |
| status | `ready_for_user_listening_review` |
| rendered audio file count | `2` |
| technical WAV validation | `true` |
| audio rendered quality claimed | `false` |
| human/audio preference claimed | `false` |
| broad model quality claimed | `false` |

## Review Files

| role | sample seed | WAV | duration | sample rate | size | dead-air | unique |
|---|---:|---|---:|---:|---:|---:|---:|
| `repeatability_sample_seed_155_duration_fill` | `155` | `outputs/stage_b_duration_coverage_fill_repeatability_audio_review_package/harness_stage_b_duration_coverage_fill_repeatability_audio_review_package/audio/repeatability_sample_seed_155_duration_fill.wav` | `6.622s` | `44100` | `1168172` | `0.3333` | `12` |
| `repeatability_sample_seed_131_duration_fill` | `131` | `outputs/stage_b_duration_coverage_fill_repeatability_audio_review_package/harness_stage_b_duration_coverage_fill_repeatability_audio_review_package/audio/repeatability_sample_seed_131_duration_fill.wav` | `6.866s` | `44100` | `1211180` | `0.3529` | `13` |

## Judgment

- WAV 파일 `2`개 생성 및 technical validation 완료
- 이 결과는 청취 가능한 artifact 준비이며 음악적 품질 proof가 아님
- human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증
- generated WAV files는 commit 대상에서 제외

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_repeatability_audio_review_package.py
bash scripts/agent_harness.sh stage-b-duration-coverage-repeatability-audio-review-package
```

## Output

- script: `scripts/render_stage_b_duration_coverage_fill_repeatability_audio_review_package.py`
- test: `tests/test_stage_b_duration_coverage_fill_repeatability_audio_review_package.py`
- summary: `outputs/stage_b_duration_coverage_fill_repeatability_audio_review_package/harness_stage_b_duration_coverage_fill_repeatability_audio_review_package/stage_b_duration_coverage_fill_repeatability_audio_review_package.json`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill repeatability user listening review fill`
