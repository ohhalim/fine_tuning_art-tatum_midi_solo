# Stage B Duration Coverage Fill Outside-Soloing Repair Audio Review Package

Issue #365는 outside-soloing repair 후보 `2`개를 WAV로 렌더하고 청취 리뷰 준비 상태를 검증한 작업이다.

## Context

- Issue #363 boundary: `outside_soloing_pitch_role_repair_candidates`
- repaired source candidates: `2/2`
- qualified variants: `6/6`
- selected policy: `contour_resolution`
- selected min chord-tone ratio: `1.000`
- selected max non-chord run: `0`
- 남은 검증: audio review / human preference

## Change

- outside-soloing repair audio review package script 추가
- selected repaired MIDI `2`개 WAV 렌더
- file count, sample rate, frame count, duration, size, checksum 검증
- audio quality / human preference claim guard 유지
- 전용 harness와 unit test 추가

## Result

| item | value |
|---|---:|
| status | `ready_for_user_listening_review` |
| rendered audio file count | `2` |
| technical WAV validation | `true` |
| sample rate | `44100` |
| audio rendered quality claimed | `false` |
| human/audio preference claimed | `false` |
| broad model quality claimed | `false` |

## Rendered WAV

| sample seed | duration | size | chord-tone ratio | non-chord run | wav |
|---:|---:|---:|---:|---:|---|
| `155` | `6.621s` | `1167916` | `1.000` | `0` | `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/audio/outside_repair_sample_seed_155_contour_resolution.wav` |
| `131` | `6.866s` | `1211180` | `1.000` | `0` | `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/audio/outside_repair_sample_seed_131_contour_resolution.wav` |

## Judgment

- WAV `2`개 생성 및 technical validation 완료
- 이 결과는 청취 가능한 artifact 준비이며 음악적 품질 proof가 아님
- human/audio preference, multi-reviewer preference, broad trained-model quality는 미검증
- generated WAV files는 commit 대상에서 제외

## Validation

```bash
.venv/bin/python -m unittest tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.py
bash scripts/agent_harness.sh stage-b-duration-coverage-outside-soloing-repair-audio-review-package
```

## Output

- script: `scripts/render_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.py`
- test: `tests/test_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.py`
- summary: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.json`
- markdown: `outputs/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/harness_stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package/stage_b_duration_coverage_fill_outside_soloing_repair_audio_review_package.md`

## Next

- `Stage B margin-recovered phrase/vocabulary duration coverage fill outside-soloing repair user listening review fill`
